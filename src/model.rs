use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};
use gloo_net::http::Request;
use thiserror::Error;

use crate::imaging::{IMG, LAT};
use crate::moebius_unet::MoebiusUnet;

const VAE_SCALING_FACTOR: f64 = 0.13025;

/// Stable model identifier used by the UI and worker protocol.
#[derive(Clone, Copy, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
pub enum ModelId {
    #[serde(rename = "ft-places2")]
    FtPlaces2,
}

impl ModelId {
    /// Returns the stable string representation for worker messages.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FtPlaces2 => "ft-places2",
        }
    }
}

impl Default for ModelId {
    #[inline]
    fn default() -> Self {
        DEFAULT_MODEL_ID
    }
}

/// Default model for the first browser port.
pub const DEFAULT_MODEL_ID: ModelId = ModelId::FtPlaces2;

/// Downloadable model artifact configuration.
pub struct ModelConfig {
    pub id: ModelId,
    pub label: &'static str,
    pub base_url: &'static str,
    pub files: &'static [&'static str],
}

const FT_PLACES2_FILES: &[&str] = &["moebius.safetensors", "vae.safetensors"];

const MODEL_VARIANTS: &[ModelConfig] = &[ModelConfig {
    id: ModelId::FtPlaces2,
    label: "Moebius ft_places2",
    base_url: "models/moebius-ft-places2",
    files: FT_PLACES2_FILES,
}];

/// Returns every model variant supported by the browser app.
#[inline]
pub fn model_variants() -> &'static [ModelConfig] {
    MODEL_VARIANTS
}

/// Looks up model configuration for a selected model id.
#[inline]
pub fn model_config(id: ModelId) -> &'static ModelConfig {
    MODEL_VARIANTS
        .iter()
        .find(|config| config.id == id)
        .unwrap_or(&MODEL_VARIANTS[0])
}

/// Errors raised while loading or running the Candle Moebius model.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("failed to download {file}: {message}")]
    Download { file: String, message: String },
    #[error("failed to parse {file} as safetensors: {message}")]
    Safetensors { file: String, message: String },
    #[error("Candle error: {0}")]
    Candle(String),
    #[error("{component} is not implemented yet")]
    ArchitecturePending { component: &'static str },
}

/// Candle-backed Moebius model container.
pub struct CandleMoebius {
    device: Device,
    vae: AutoEncoderKL,
    unet: MoebiusUnet,
    cfg_input_ids: Tensor,
}

impl CandleMoebius {
    /// Downloads safetensors artifacts and prepares the Candle model container.
    pub async fn load(
        model_id: ModelId,
        mut progress: impl FnMut(String),
    ) -> Result<Self, ModelError> {
        let config = model_config(model_id);
        let device = Device::Cpu;

        progress(format!("Downloading {}", config.files[0]));
        let unet_bytes = fetch_artifact(config.base_url, config.files[0]).await?;
        progress(format!("Initializing {}", config.files[0]));
        let unet = MoebiusUnet::from_safetensors(unet_bytes, &device)?;

        progress(format!("Downloading {}", config.files[1]));
        let vae_bytes = fetch_artifact(config.base_url, config.files[1]).await?;
        progress(format!("Initializing {}", config.files[1]));
        let vae = build_vae(vae_bytes, &device)?;
        let cfg_input_ids = Tensor::from_vec(cfg_input_ids(), (2, 10), &device)
            .map_err(|error| ModelError::Candle(error.to_string()))?;

        Ok(Self {
            device,
            vae,
            unet,
            cfg_input_ids,
        })
    }

    /// Runs the VAE encoder. This is the explicit integration point for the Rust VAE port.
    pub fn encode_masked_image(&self, _masked_chw: &[f32]) -> Result<Tensor, ModelError> {
        let image = Tensor::from_vec(_masked_chw.to_vec(), (1, 3, IMG, IMG), &self.device)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        let latent = self
            .vae
            .encode(&image)
            .map_err(|error| ModelError::Candle(error.to_string()))?
            .sample()
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        (&latent * VAE_SCALING_FACTOR).map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the UNet denoiser. This is the explicit integration point for the Rust UNet port.
    pub fn predict_noise(
        &mut self,
        latent9: &[f32],
        timestep: usize,
    ) -> Result<Tensor, ModelError> {
        let latent9 = Tensor::from_vec(latent9.to_vec(), (2, 9, LAT, LAT), &self.device)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        let timesteps = Tensor::new(&[timestep as u32, timestep as u32], &self.device)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        self.unet.forward(&latent9, &timesteps, &self.cfg_input_ids)
    }

    /// Runs the VAE decoder. This is the explicit integration point for the Rust VAE port.
    pub fn decode_latent(&self, latent: &[f32]) -> Result<Tensor, ModelError> {
        let latent = Tensor::from_vec(latent.to_vec(), (1, 4, LAT, LAT), &self.device)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        let latent = (&latent / VAE_SCALING_FACTOR)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        self.vae
            .decode(&latent)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Returns the active Candle device.
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

async fn fetch_artifact(base_url: &str, file: &str) -> Result<Vec<u8>, ModelError> {
    let url = resolve_artifact_url(base_url, file)?;
    Request::get(&url)
        .send()
        .await
        .map_err(|error| ModelError::Download {
            file: file.to_string(),
            message: error.to_string(),
        })?
        .binary()
        .await
        .map_err(|error| ModelError::Download {
            file: file.to_string(),
            message: error.to_string(),
        })
}

fn resolve_artifact_url(base_url: &str, file: &str) -> Result<String, ModelError> {
    let path = format!("{}/{file}", base_url.trim_end_matches('/'));
    if path.starts_with("http://") || path.starts_with("https://") || path.starts_with('/') {
        return Ok(path);
    }

    let base = js_sys::Reflect::get(&js_sys::global(), &"__MOEBIUS_BASE_URL".into())
        .ok()
        .and_then(|value| value.as_string())
        .or_else(browser_base_url)
        .unwrap_or_else(|| "./".to_string());
    web_sys::Url::new_with_base(&path, &base)
        .map(|url| url.href())
        .map_err(|error| ModelError::Download {
            file: file.to_string(),
            message: format!("failed to resolve model URL: {error:?}"),
        })
}

#[inline]
fn browser_base_url() -> Option<String> {
    web_sys::window()
        .and_then(|window| window.location().href().ok())
        .and_then(|href| web_sys::Url::new_with_base("./", &href).ok())
        .map(|url| url.href())
}

fn build_vae(bytes: Vec<u8>, device: &Device) -> Result<AutoEncoderKL, ModelError> {
    let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, device)
        .map_err(|error| ModelError::Candle(error.to_string()))?;
    let config = AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
        use_quant_conv: true,
        use_post_quant_conv: true,
    };
    AutoEncoderKL::new(vb, 3, 3, config).map_err(|error| ModelError::Candle(error.to_string()))
}

fn cfg_input_ids() -> Vec<u32> {
    let mut ids = Vec::with_capacity(20);
    ids.extend(10..20);
    ids.extend(0..10);
    ids
}
