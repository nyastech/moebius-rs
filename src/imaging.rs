use serde::{Deserialize, Serialize};

pub const IMG: usize = 512;
pub const LAT: usize = 64;
pub const RGBA_CHANNELS: usize = 4;
pub const RGB_CHANNELS: usize = 3;

/// Rectangle occupied by an image after contain-fitting into the fixed canvas.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FitRect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// Computes the letterboxed 512x512 content rectangle for a source image.
#[inline]
pub fn fit_rect(src_w: usize, src_h: usize) -> FitRect {
    let scale = (IMG as f64 / src_w.max(1) as f64).min(IMG as f64 / src_h.max(1) as f64);
    let w = (src_w as f64 * scale).round() as usize;
    let h = (src_h as f64 * scale).round() as usize;
    FitRect {
        x: (IMG - w) / 2,
        y: (IMG - h) / 2,
        w,
        h,
    }
}

/// Converts 512x512 RGBA bytes into CHW RGB floats in [-1, 1].
pub fn rgba_to_chw(rgba: &[u8]) -> Vec<f32> {
    let plane = IMG * IMG;
    let mut out = vec![0.0; RGB_CHANNELS * plane];

    for pixel in 0..plane {
        out[pixel] = rgba[pixel * RGBA_CHANNELS] as f32 / 127.5 - 1.0;
        out[plane + pixel] = rgba[pixel * RGBA_CHANNELS + 1] as f32 / 127.5 - 1.0;
        out[2 * plane + pixel] = rgba[pixel * RGBA_CHANNELS + 2] as f32 / 127.5 - 1.0;
    }

    out
}

/// Extracts a binary inpaint mask from RGBA mask bytes, using alpha as paint coverage.
pub fn mask_rgba_to_binary(rgba: &[u8]) -> Vec<f32> {
    (0..IMG * IMG)
        .map(|pixel| {
            if rgba[pixel * RGBA_CHANNELS + 3] >= 128 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Applies the binary mask to CHW image data in the same way as the TypeScript reference.
pub fn make_masked_chw(image_chw: &[f32], mask: &[f32]) -> Vec<f32> {
    let plane = IMG * IMG;
    let mut out = vec![0.0; image_chw.len()];

    for channel in 0..RGB_CHANNELS {
        for pixel in 0..plane {
            out[channel * plane + pixel] = image_chw[channel * plane + pixel] * (1.0 - mask[pixel]);
        }
    }

    out
}

/// Downsamples a 512x512 binary mask to 64x64 using PyTorch nearest semantics.
pub fn mask_to_latent(mask: &[f32]) -> Vec<f32> {
    let ratio = IMG / LAT;
    let mut out = vec![0.0; LAT * LAT];

    for y in 0..LAT {
        for x in 0..LAT {
            out[y * LAT + x] = mask[y * ratio * IMG + x * ratio];
        }
    }

    out
}

/// Converts CHW RGB floats in [-1, 1] into opaque RGBA bytes.
pub fn chw_to_rgba(chw: &[f32]) -> Vec<u8> {
    let plane = IMG * IMG;
    let mut out = vec![255; IMG * IMG * RGBA_CHANNELS];

    for pixel in 0..plane {
        for channel in 0..RGB_CHANNELS {
            let normalized = ((chw[channel * plane + pixel] + 1.0) * 0.5).clamp(0.0, 1.0);
            out[pixel * RGBA_CHANNELS + channel] = (normalized * 255.0).round() as u8;
        }
    }

    out
}

/// Reuses the original image outside the inpaint mask.
pub fn paste_back(result: &[u8], original: &[u8], mask: &[f32]) -> Vec<u8> {
    let mut out = result.to_vec();
    for (pixel, mask_value) in mask.iter().enumerate().take(IMG * IMG) {
        if *mask_value < 0.5 {
            let base = pixel * RGBA_CHANNELS;
            out[base..base + RGBA_CHANNELS].copy_from_slice(&original[base..base + RGBA_CHANNELS]);
        }
    }
    out
}

/// Deterministic Mulberry32 PRNG used to match the TypeScript implementation.
#[derive(Clone, Copy, Debug)]
pub struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    /// Creates a PRNG from the given seed.
    #[inline]
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Returns the next uniform sample in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x6d2b_79f5);
        let mut value = self.state;
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        value = value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61)) ^ value;
        ((value ^ (value >> 14)) as f64 / 4_294_967_296.0) as f32
    }
}

/// Generates deterministic Gaussian noise with Box-Muller.
pub fn randn(len: usize, seed: u32) -> Vec<f32> {
    let mut rng = Mulberry32::new(seed);
    let mut out = Vec::with_capacity(len);

    for _ in 0..len {
        let mut u = 0.0;
        let mut v = 0.0;
        while u == 0.0 {
            u = rng.next_f32();
        }
        while v == 0.0 {
            v = rng.next_f32();
        }
        out.push((-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos());
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{FitRect, IMG, fit_rect, mask_to_latent, randn};

    #[test]
    fn fits_wide_image_into_square_canvas() {
        assert_eq!(
            fit_rect(1024, 512),
            FitRect {
                x: 0,
                y: 128,
                w: 512,
                h: 256,
            }
        );
    }

    #[test]
    fn downscales_mask_by_top_left_sample() {
        let mut mask = vec![0.0; IMG * IMG];
        mask[8 * IMG + 16] = 1.0;

        let latent = mask_to_latent(&mask);

        assert_eq!(latent[64 + 2], 1.0);
    }

    #[test]
    fn generates_reference_noise_prefix() {
        let values = randn(3, 42);

        assert!((values[0] - -0.956_162_2).abs() < 1e-6);
        assert!((values[1] - -0.273_026_1).abs() < 1e-6);
        assert!((values[2] - -1.841_627_2).abs() < 1e-6);
    }
}
