use serde::{Deserialize, Serialize};

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;

/// A DDIM scheduler configured to match the validated Moebius browser port.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Ddim {
    pub alphas_cumprod: Vec<f64>,
    pub timesteps: Vec<usize>,
}

/// Builds the deterministic DDIM schedule used by the TypeScript reference.
#[inline]
pub fn make_ddim(num_steps: usize) -> Ddim {
    make_ddim_with_strength(num_steps, 0.99)
}

/// Builds the deterministic DDIM schedule for an explicit inpaint strength.
pub fn make_ddim_with_strength(num_steps: usize, strength: f64) -> Ddim {
    let mut alphas_cumprod = Vec::with_capacity(NUM_TRAIN_TIMESTEPS);
    let mut acc = 1.0;
    let beta_a = BETA_START.sqrt();
    let beta_b = BETA_END.sqrt();

    for index in 0..NUM_TRAIN_TIMESTEPS {
        let pos = index as f64 / (NUM_TRAIN_TIMESTEPS - 1) as f64;
        let beta = (beta_a + (beta_b - beta_a) * pos).powi(2);
        acc *= 1.0 - beta;
        alphas_cumprod.push(acc);
    }

    let step_ratio = NUM_TRAIN_TIMESTEPS / num_steps.max(1);
    let mut timesteps = (0..num_steps)
        .map(|index| index * step_ratio)
        .collect::<Vec<_>>();
    timesteps.reverse();

    let init_timestep = ((num_steps as f64 * strength).floor() as usize).min(num_steps);
    let t_start = num_steps.saturating_sub(init_timestep);
    timesteps.drain(0..t_start);

    Ddim {
        alphas_cumprod,
        timesteps,
    }
}

/// Applies one eta=0 DDIM update to a latent sample.
#[inline]
pub fn ddim_step(
    eps: &[f32],
    sample: &[f32],
    t: usize,
    prev_t: Option<usize>,
    ddim: &Ddim,
) -> Vec<f32> {
    let mut output = vec![0.0; sample.len()];
    ddim_step_into(eps, sample, t, prev_t, ddim, &mut output);
    output
}

/// Applies one eta=0 DDIM update into an existing latent buffer.
#[inline]
pub fn ddim_step_into(
    eps: &[f32],
    sample: &[f32],
    t: usize,
    prev_t: Option<usize>,
    ddim: &Ddim,
    output: &mut [f32],
) {
    let ac_t = ddim.alphas_cumprod[t];
    let ac_prev = prev_t.map_or(1.0, |index| ddim.alphas_cumprod[index]);
    let sqrt_ac_t = ac_t.sqrt();
    let sqrt_beta_t = (1.0 - ac_t).sqrt();
    let sqrt_ac_prev = ac_prev.sqrt();
    let sqrt_one_minus_ac_prev = (1.0 - ac_prev).sqrt();

    for ((output, eps), sample) in output.iter_mut().zip(eps).zip(sample) {
        let pred_x0 = (*sample as f64 - sqrt_beta_t * *eps as f64) / sqrt_ac_t;
        *output = (sqrt_ac_prev * pred_x0 + sqrt_one_minus_ac_prev * *eps as f64) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::{ddim_step, make_ddim};

    #[test]
    fn builds_reference_timesteps() {
        let ddim = make_ddim(20);

        assert_eq!(
            ddim.timesteps,
            vec![
                900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150,
                100, 50, 0
            ]
        );
    }

    #[test]
    fn updates_sample_without_changing_input() {
        let ddim = make_ddim(20);
        let sample = vec![0.25, -0.5, 1.0];
        let eps = vec![0.1, -0.2, 0.3];

        let out = ddim_step(&eps, &sample, 900, Some(850), &ddim);

        assert_eq!(sample, vec![0.25, -0.5, 1.0]);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.292_210_16).abs() < 1e-6);
    }
}
