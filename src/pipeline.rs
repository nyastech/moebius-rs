use crate::ddim::{ddim_step_into, make_ddim};
use crate::imaging::{
    LAT, chw_to_rgba, make_masked_chw, mask_rgba_to_binary, mask_to_latent, paste_back, randn,
    rgba_to_chw,
};
use crate::model::{CandleMoebius, ModelError, ModelId};
use gloo_timers::future::TimeoutFuture;
use serde::{Deserialize, Serialize};

const NOISE_OFFSET: f32 = 0.0357;

/// Runtime options for one Moebius inpainting request.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunOptions {
    pub steps: usize,
    pub guidance: f32,
    pub seed: u32,
    pub paste: bool,
}

/// RGBA image and mask bytes for a fixed 512x512 inpainting run.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunInput {
    pub image_rgba: Vec<u8>,
    pub mask_rgba: Vec<u8>,
    pub options: RunOptions,
}

/// RGBA result bytes returned by the worker.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunOutput {
    pub rgba: Vec<u8>,
    pub elapsed_ms: f64,
}

/// Coarse-grained progress for one browser inpainting request.
pub struct PipelineProgress {
    pub stage: &'static str,
    pub current: usize,
    pub total: usize,
}

/// Candle-backed Moebius pipeline.
pub struct MoebiusPipeline {
    model: CandleMoebius,
}

impl MoebiusPipeline {
    /// Loads the selected model and constructs the pipeline.
    pub async fn load(model_id: ModelId, progress: impl FnMut(String)) -> Result<Self, ModelError> {
        let model = CandleMoebius::load(model_id, progress).await?;
        Ok(Self { model })
    }

    /// Runs one fixed-resolution inpainting request.
    pub async fn run(
        &mut self,
        input: RunInput,
        mut progress: impl FnMut(PipelineProgress),
    ) -> Result<RunOutput, ModelError> {
        let started = js_sys::Date::now();
        let total = input.options.steps.max(1) + 4;
        progress(PipelineProgress {
            stage: "Preparing tensors",
            current: 1,
            total,
        });
        yield_to_browser().await;

        let image_chw = rgba_to_chw(&input.image_rgba);
        let mask = mask_rgba_to_binary(&input.mask_rgba);
        let masked_chw = make_masked_chw(&image_chw, &mask);
        progress(PipelineProgress {
            stage: "Encoding masked image on CPU",
            current: 2,
            total,
        });
        yield_to_browser().await;
        let masked_latent = self
            .model
            .encode_masked_image(&masked_chw)?
            .flatten_all()
            .map_err(|error| ModelError::Candle(error.to_string()))?
            .to_vec1::<f32>()
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        let mask64 = mask_to_latent(&mask);

        // Keep deterministic sampling setup next to the model boundary so the real UNet
        // implementation can be dropped in without changing the worker/UI contract.
        let ddim = make_ddim(input.options.steps);
        let total = ddim.timesteps.len() + 4;
        let mut latents = randn(4 * LAT * LAT, input.options.seed);
        let mut next_latents = vec![0.0; latents.len()];
        let mut latent9 = vec![0.0; 2 * 9 * LAT * LAT];
        let offsets = randn(4, input.options.seed ^ 0x9e37_79b9);
        for channel in 0..4 {
            for pixel in 0..LAT * LAT {
                latents[channel * LAT * LAT + pixel] += NOISE_OFFSET * offsets[channel];
            }
        }

        for (index, timestep) in ddim.timesteps.iter().copied().enumerate() {
            progress(PipelineProgress {
                stage: "Denoising",
                current: index + 3,
                total,
            });
            yield_to_browser().await;
            assemble_cfg_input(&mut latent9, &latents, &mask64, &masked_latent);
            let eps = self
                .model
                .predict_noise(&latent9, timestep)?
                .flatten_all()
                .map_err(|error| ModelError::Candle(error.to_string()))?
                .to_vec1::<f32>()
                .map_err(|error| ModelError::Candle(error.to_string()))?;
            let prev_t = ddim.timesteps.get(index + 1).copied();
            ddim_step_into(&eps, &latents, timestep, prev_t, &ddim, &mut next_latents);
            std::mem::swap(&mut latents, &mut next_latents);
        }

        progress(PipelineProgress {
            stage: "Decoding image",
            current: total - 1,
            total,
        });
        yield_to_browser().await;
        let decoded = self
            .model
            .decode_latent(&latents)?
            .flatten_all()
            .map_err(|error| ModelError::Candle(error.to_string()))?
            .to_vec1::<f32>()
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        let decoded = chw_to_rgba(&decoded);

        progress(PipelineProgress {
            stage: "Compositing result",
            current: total,
            total,
        });
        yield_to_browser().await;
        let result = if input.options.paste {
            paste_back(&decoded, &input.image_rgba, &mask)
        } else {
            decoded
        };

        Ok(RunOutput {
            rgba: result,
            elapsed_ms: js_sys::Date::now() - started,
        })
    }
}

#[inline]
async fn yield_to_browser() {
    TimeoutFuture::new(0).await;
}

fn assemble_cfg_input(cfg: &mut [f32], latents: &[f32], mask64: &[f32], masked_latent: &[f32]) {
    let plane = LAT * LAT;
    let (nine, copy) = cfg.split_at_mut(9 * plane);
    nine[..4 * plane].copy_from_slice(&latents[..4 * plane]);
    nine[4 * plane..5 * plane].copy_from_slice(mask64);
    nine[5 * plane..9 * plane].copy_from_slice(&masked_latent[..4 * plane]);
    copy.copy_from_slice(nine);
}
