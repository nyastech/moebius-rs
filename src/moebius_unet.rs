use candle_core::{Device, Tensor};
use candle_nn::{
    Embedding, GroupNorm, Linear, Module, VarBuilder, embedding, group_norm, linear, ops,
};

use crate::model::ModelError;
use crate::moebius_layers::{
    DWMixTFDownBlock2D, DWMixTFDownBlockConfig, DWMixTFUpBlock2D, DWMixTFUpBlockConfig,
    DWResnetBlock2D, DepthwiseSeparableConv, Downsample2D, GLUMBConv, MixTransformer2DModel,
    MixTransformerConfig, MultiQueryCrossLambda, MultiQuerySelfLambda, TimestepEmbedding,
    Timesteps,
};

/// Fixed Moebius UNet dimensions for the first ft_places2 browser port.
pub const NUM_EMBEDDINGS: usize = 20;
pub const EMBEDDING_DIM: usize = 3072;
pub const LATENT_CHANNELS: usize = 4;
pub const INPAINT_CHANNELS: usize = 9;
pub const TIME_PROJ_CHANNELS: usize = 320;
pub const TIME_EMBED_DIM: usize = 1280;

/// Candle-side container for the custom Moebius lambda UNet.
///
/// The upstream architecture is not a stock Stable Diffusion UNet: it uses
/// depthwise separable convolutions, `DWResnetBlock2D`, `MixTransformer2DModel`,
/// and lambda self/cross attention.  This type owns the converted weights now so
/// the rest of the pipeline can be wired and parity-tested while the custom
/// blocks are filled in module by module.
pub struct MoebiusUnet {
    _weights: Vec<u8>,
    _device: Device,
    conv_in: DepthwiseSeparableConv,
    embedding_layer: Embedding,
    encoder_hid_proj: Linear,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    down0_resnet0: DWResnetBlock2D,
    down0_attn0_attn1: MultiQuerySelfLambda,
    down0_attn0_attn2: MultiQueryCrossLambda,
    down0_attn0_ff: GLUMBConv,
    down0_attn0: MixTransformer2DModel,
    down0_resnet1: DWResnetBlock2D,
    down0_attn1: MixTransformer2DModel,
    down0_downsample: Downsample2D,
    down1: DWMixTFDownBlock2D,
    down2: DWMixTFDownBlock2D,
    up0: DWMixTFUpBlock2D,
    up1: DWMixTFUpBlock2D,
    up2: DWMixTFUpBlock2D,
    conv_norm_out: GroupNorm,
    conv_out: DepthwiseSeparableConv,
}

impl MoebiusUnet {
    /// Parses the converted Moebius safetensors and constructs the UNet container.
    pub fn from_safetensors(bytes: Vec<u8>, device: &Device) -> Result<Self, ModelError> {
        let vb =
            VarBuilder::from_buffered_safetensors(bytes.clone(), candle_core::DType::F32, device)
                .map_err(|error| ModelError::Candle(error.to_string()))?;
        let conv_in = DepthwiseSeparableConv::new(9, 320, 3, vb.pp("diff_model.conv_in"))
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        let embedding_layer = embedding(NUM_EMBEDDINGS, EMBEDDING_DIM, vb.pp("embedding_layer"))
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        let encoder_hid_proj = linear(EMBEDDING_DIM, 768, vb.pp("diff_model.encoder_hid_proj"))
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        let time_embedding = TimestepEmbedding::new(
            TIME_PROJ_CHANNELS,
            TIME_EMBED_DIM,
            vb.pp("diff_model.time_embedding"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let down0_resnet0 = DWResnetBlock2D::new(
            320,
            320,
            TIME_EMBED_DIM,
            vb.pp("diff_model.down_blocks.0.resnets.0"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let down0_attn0_attn1 = MultiQuerySelfLambda::new(
            320,
            40,
            8,
            vb.pp("diff_model.down_blocks.0.attentions.0.transformer_blocks.0.attn1"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let down0_attn0_attn2 = MultiQueryCrossLambda::new(
            320,
            768,
            40,
            8,
            4096,
            10,
            vb.pp("diff_model.down_blocks.0.attentions.0.transformer_blocks.0.attn2"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let down0_attn0_ff = GLUMBConv::new(
            320,
            800,
            320,
            vb.pp("diff_model.down_blocks.0.attentions.0.transformer_blocks.0.ff"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let down0_attn0 = MixTransformer2DModel::new(
            MixTransformerConfig {
                dim: 320,
                cross_dim: 768,
                dim_k: 40,
                heads: 8,
                tokens: 4096,
                context_tokens: 10,
                ff_hidden_dim: 800,
            },
            vb.pp("diff_model.down_blocks.0.attentions.0"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let down0_resnet1 = DWResnetBlock2D::new(
            320,
            320,
            TIME_EMBED_DIM,
            vb.pp("diff_model.down_blocks.0.resnets.1"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let down0_attn1 = MixTransformer2DModel::new(
            MixTransformerConfig {
                dim: 320,
                cross_dim: 768,
                dim_k: 40,
                heads: 8,
                tokens: 4096,
                context_tokens: 10,
                ff_hidden_dim: 800,
            },
            vb.pp("diff_model.down_blocks.0.attentions.1"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let down0_downsample =
            Downsample2D::new(320, vb.pp("diff_model.down_blocks.0.downsamplers.0"))
                .map_err(|error| ModelError::Candle(error.to_string()))?;
        let down1 = DWMixTFDownBlock2D::new(
            DWMixTFDownBlockConfig {
                in_channels: 320,
                out_channels: 640,
                temb_channels: TIME_EMBED_DIM,
                tokens: 1024,
                add_downsample: true,
            },
            vb.pp("diff_model.down_blocks.1"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let down2 = DWMixTFDownBlock2D::new(
            DWMixTFDownBlockConfig {
                in_channels: 640,
                out_channels: 1280,
                temb_channels: TIME_EMBED_DIM,
                tokens: 256,
                add_downsample: false,
            },
            vb.pp("diff_model.down_blocks.2"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let up0 = DWMixTFUpBlock2D::new(
            DWMixTFUpBlockConfig {
                in_channels: 640,
                out_channels: 1280,
                prev_output_channel: 1280,
                temb_channels: TIME_EMBED_DIM,
                tokens: 256,
                add_upsample: true,
            },
            vb.pp("diff_model.up_blocks.0"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let up1 = DWMixTFUpBlock2D::new(
            DWMixTFUpBlockConfig {
                in_channels: 320,
                out_channels: 640,
                prev_output_channel: 1280,
                temb_channels: TIME_EMBED_DIM,
                tokens: 1024,
                add_upsample: true,
            },
            vb.pp("diff_model.up_blocks.1"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let up2 = DWMixTFUpBlock2D::new(
            DWMixTFUpBlockConfig {
                in_channels: 320,
                out_channels: 320,
                prev_output_channel: 640,
                temb_channels: TIME_EMBED_DIM,
                tokens: 4096,
                add_upsample: false,
            },
            vb.pp("diff_model.up_blocks.2"),
        )
        .map_err(|error| ModelError::Candle(error.to_string()))?;
        let conv_norm_out = group_norm(32, 320, 1e-5, vb.pp("diff_model.conv_norm_out"))
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        let conv_out =
            DepthwiseSeparableConv::new(320, LATENT_CHANNELS, 3, vb.pp("diff_model.conv_out"))
                .map_err(|error| ModelError::Candle(error.to_string()))?;
        Ok(Self {
            _weights: bytes,
            _device: device.clone(),
            conv_in,
            embedding_layer,
            encoder_hid_proj,
            time_proj: Timesteps::new(TIME_PROJ_CHANNELS, true, 0.0),
            time_embedding,
            down0_resnet0,
            down0_attn0_attn1,
            down0_attn0_attn2,
            down0_attn0_ff,
            down0_attn0,
            down0_resnet1,
            down0_attn1,
            down0_downsample,
            down1,
            down2,
            up0,
            up1,
            up2,
            conv_norm_out,
            conv_out,
        })
    }

    /// Looks up Moebius condition tokens and projects them to cross-attention width.
    pub fn forward_encoder_hidden_states(&self, input_ids: &Tensor) -> Result<Tensor, ModelError> {
        let hidden = self
            .embedding_layer
            .forward(input_ids)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        self.encoder_hid_proj
            .forward(&hidden)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Builds the learned Moebius timestep embedding.
    pub fn forward_time_embedding(&self, timesteps: &Tensor) -> Result<Tensor, ModelError> {
        let projected = self
            .time_proj
            .forward(timesteps)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        self.time_embedding
            .forward(&projected)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the first UNet convolution block.
    pub fn forward_conv_in(&self, latent9: &Tensor) -> Result<Tensor, ModelError> {
        self.conv_in
            .forward(latent9)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the first residual block in the first down block.
    pub fn forward_down0_resnet0(
        &self,
        latent9: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let hidden = self.forward_conv_in(latent9)?;
        let emb = self.forward_time_embedding(timesteps)?;
        self.down0_resnet0
            .forward(&hidden, &emb)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the first MixTransformer feed-forward block.
    pub fn forward_down0_attn0_ff(&self, hidden: &Tensor) -> Result<Tensor, ModelError> {
        self.down0_attn0_ff
            .forward(hidden)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the first MixTransformer Lambda self-attention block.
    pub fn forward_down0_attn0_attn1(&self, hidden: &Tensor) -> Result<Tensor, ModelError> {
        self.down0_attn0_attn1
            .forward(hidden)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the first MixTransformer Lambda cross-attention block.
    pub fn forward_down0_attn0_attn2(
        &self,
        hidden: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let encoder_hidden_states = self.forward_encoder_hidden_states(input_ids)?;
        self.down0_attn0_attn2
            .forward(hidden, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the first full MixTransformer2D attention module.
    pub fn forward_down0_attn0(
        &self,
        hidden: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let encoder_hidden_states = self.forward_encoder_hidden_states(input_ids)?;
        self.down0_attn0
            .forward(hidden, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the first full down block through its downsampler output.
    pub fn forward_down0(
        &self,
        latent9: &Tensor,
        timesteps: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let emb = self.forward_time_embedding(timesteps)?;
        let encoder_hidden_states = self.forward_encoder_hidden_states(input_ids)?;
        let mut hidden = self.forward_conv_in(latent9)?;
        hidden = self
            .down0_resnet0
            .forward(&hidden, &emb)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        hidden = self
            .down0_attn0
            .forward(&hidden, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        hidden = self
            .down0_resnet1
            .forward(&hidden, &emb)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        hidden = self
            .down0_attn1
            .forward(&hidden, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        self.down0_downsample
            .forward(&hidden)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the second down block from a block-0 output sample.
    pub fn forward_down1(
        &self,
        hidden: &Tensor,
        timesteps: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let emb = self.forward_time_embedding(timesteps)?;
        let encoder_hidden_states = self.forward_encoder_hidden_states(input_ids)?;
        self.down1
            .forward(hidden, &emb, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the third down block from a block-1 output sample.
    pub fn forward_down2(
        &self,
        hidden: &Tensor,
        timesteps: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let emb = self.forward_time_embedding(timesteps)?;
        let encoder_hidden_states = self.forward_encoder_hidden_states(input_ids)?;
        self.down2
            .forward(hidden, &emb, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the first up block from traced skip samples.
    pub fn forward_up0(
        &self,
        hidden: &Tensor,
        skips: &[Tensor],
        timesteps: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let emb = self.forward_time_embedding(timesteps)?;
        let encoder_hidden_states = self.forward_encoder_hidden_states(input_ids)?;
        self.up0
            .forward(hidden, skips, &emb, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the second up block from traced skip samples.
    pub fn forward_up1(
        &self,
        hidden: &Tensor,
        skips: &[Tensor],
        timesteps: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let emb = self.forward_time_embedding(timesteps)?;
        let encoder_hidden_states = self.forward_encoder_hidden_states(input_ids)?;
        self.up1
            .forward(hidden, skips, &emb, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the third up block from traced skip samples.
    pub fn forward_up2(
        &self,
        hidden: &Tensor,
        skips: &[Tensor],
        timesteps: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let emb = self.forward_time_embedding(timesteps)?;
        let encoder_hidden_states = self.forward_encoder_hidden_states(input_ids)?;
        self.up2
            .forward(hidden, skips, &emb, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Runs the final UNet normalization, activation, and output convolution.
    pub fn forward_post_process(&self, hidden: &Tensor) -> Result<Tensor, ModelError> {
        let hidden = self
            .conv_norm_out
            .forward(hidden)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        let hidden = ops::silu(&hidden).map_err(|error| ModelError::Candle(error.to_string()))?;
        self.conv_out
            .forward(&hidden)
            .map_err(|error| ModelError::Candle(error.to_string()))
    }

    /// Predicts diffusion noise for a CFG batch.
    pub fn forward(
        &mut self,
        latent9: &Tensor,
        timesteps: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let emb = self.forward_time_embedding(timesteps)?;
        let encoder_hidden_states = self.forward_encoder_hidden_states(input_ids)?;
        let mut hidden = self.forward_conv_in(latent9)?;
        let mut residuals = vec![hidden.clone()];

        hidden = self
            .down0_resnet0
            .forward(&hidden, &emb)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        hidden = self
            .down0_attn0
            .forward(&hidden, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        residuals.push(hidden.clone());
        hidden = self
            .down0_resnet1
            .forward(&hidden, &emb)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        hidden = self
            .down0_attn1
            .forward(&hidden, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        residuals.push(hidden.clone());
        hidden = self
            .down0_downsample
            .forward(&hidden)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        residuals.push(hidden.clone());

        let (down1, mut states) = self
            .down1
            .forward_with_states(&hidden, &emb, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        residuals.append(&mut states);
        let (down2, mut states) = self
            .down2
            .forward_with_states(&down1, &emb, &encoder_hidden_states)
            .map_err(|error| ModelError::Candle(error.to_string()))?;
        residuals.append(&mut states);

        hidden = self.run_up_blocks(&down2, residuals, &emb, &encoder_hidden_states)?;
        self.forward_post_process(&hidden)
    }

    fn run_up_blocks(
        &self,
        hidden: &Tensor,
        mut residuals: Vec<Tensor>,
        emb: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let mut hidden = hidden.clone();
        for up_block in [&self.up0, &self.up1, &self.up2] {
            let skips = residuals.split_off(residuals.len() - 3);
            hidden = up_block
                .forward(&hidden, &skips, emb, encoder_hidden_states)
                .map_err(|error| ModelError::Candle(error.to_string()))?;
        }
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use ndarray::ArrayD;
    use ndarray_npy::read_npy;

    #[test]
    fn conv_in_matches_python_trace_shape() {
        let device = Device::Cpu;
        let weights = std::fs::read("public/models/moebius-ft-places2/moebius.safetensors")
            .expect("converted Moebius weights exist");
        let unet = MoebiusUnet::from_safetensors(weights, &device).expect("UNet loads conv_in");
        let input = read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_input_latent9.npy");
        let expected =
            read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__conv_in.npy");
        let actual = unet.forward_conv_in(&input).expect("conv_in runs");
        assert_eq!(actual.shape(), expected.shape());
        let diff = (&actual - &expected)
            .expect("shapes match")
            .abs()
            .expect("absolute diff works")
            .max_all()
            .expect("max diff works")
            .to_scalar::<f32>()
            .expect("scalar f32");
        assert!(diff < 1e-4, "conv_in max_abs_diff={diff}");
    }

    #[test]
    fn time_embedding_matches_python_trace_shape() {
        let unet = load_unet();
        let timesteps = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_timesteps.npy");
        let expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__time_embedding.npy",
        );
        let actual = unet
            .forward_time_embedding(&timesteps)
            .expect("time embedding runs");
        assert_close("time_embedding", &actual, &expected, 1e-4);
    }

    #[test]
    fn encoder_hidden_states_match_python_trace_shape() {
        let unet = load_unet();
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__encoder_hid_proj.npy",
        );
        let actual = unet
            .forward_encoder_hidden_states(&input_ids)
            .expect("encoder hidden projection runs");
        assert_close("encoder_hid_proj", &actual, &expected, 1e-4);
    }

    #[test]
    fn first_down_resnet_matches_python_trace_shape() {
        let unet = load_unet();
        let input = read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_input_latent9.npy");
        let timesteps = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_timesteps.npy");
        let expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__resnets__0.npy",
        );
        let actual = unet
            .forward_down0_resnet0(&input, &timesteps)
            .expect("first down resnet runs");
        assert_close("down0_resnet0", &actual, &expected, 1e-4);
    }

    #[test]
    fn first_mix_ffn_matches_python_trace_shape() {
        let unet = load_unet();
        let input = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__0__transformer_blocks__0__ff__input0.npy",
        );
        let expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__0__transformer_blocks__0__ff.npy",
        );
        let actual = unet
            .forward_down0_attn0_ff(&input)
            .expect("first MixFFN runs");
        assert_close("down0_attn0_ff", &actual, &expected, 1e-4);
    }

    #[test]
    fn first_lambda_self_attention_matches_python_trace_shape() {
        let unet = load_unet();
        let input = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__0__transformer_blocks__0__attn1__input0.npy",
        );
        let expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__0__transformer_blocks__0__attn1.npy",
        );
        let actual = unet
            .forward_down0_attn0_attn1(&input)
            .expect("first Lambda self-attention runs");
        assert_close("down0_attn0_attn1", &actual, &expected, 1e-4);
    }

    #[test]
    fn first_lambda_cross_attention_matches_python_trace_shape() {
        let unet = load_unet();
        let input = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__0__transformer_blocks__0__attn2__input0.npy",
        );
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__0__transformer_blocks__0__attn2.npy",
        );
        let actual = unet
            .forward_down0_attn0_attn2(&input, &input_ids)
            .expect("first Lambda cross-attention runs");
        assert_close("down0_attn0_attn2", &actual, &expected, 1e-4);
    }

    #[test]
    fn first_mix_transformer_2d_matches_python_trace_shape() {
        let unet = load_unet();
        let input = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__resnets__0.npy",
        );
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__0.npy",
        );
        let actual = unet
            .forward_down0_attn0(&input, &input_ids)
            .expect("first MixTransformer2D runs");
        assert_close("down0_attn0", &actual, &expected, 1e-4);
    }

    #[test]
    fn first_down_block_second_pair_matches_python_trace_shape() {
        let unet = load_unet();
        let emb = unet
            .forward_time_embedding(&read_i64_npy(
                "tmp/fixtures/moebius-ft-places2/unet_input_timesteps.npy",
            ))
            .expect("time embedding runs");
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let encoder_hidden_states = unet
            .forward_encoder_hidden_states(&input_ids)
            .expect("encoder hidden states run");
        let attention0 = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__0.npy",
        );
        let resnet1_expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__resnets__1.npy",
        );
        let resnet1 = unet
            .down0_resnet1
            .forward(&attention0, &emb)
            .expect("second resnet runs");
        assert_close("down0_resnet1", &resnet1, &resnet1_expected, 1e-4);

        let attention1_expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__1.npy",
        );
        let attention1 = unet
            .down0_attn1
            .forward(&resnet1, &encoder_hidden_states)
            .expect("second attention runs");
        assert_close("down0_attn1", &attention1, &attention1_expected, 1e-4);

        let downsample_expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__downsamplers__0.npy",
        );
        let downsample = unet
            .down0_downsample
            .forward(&attention1)
            .expect("downsampler runs");
        assert_close("down0_downsample", &downsample, &downsample_expected, 1e-4);
    }

    #[test]
    fn first_down_block_matches_python_trace_shape() {
        let unet = load_unet();
        let input = read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_input_latent9.npy");
        let timesteps = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_timesteps.npy");
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0.npy",
        );
        let actual = unet
            .forward_down0(&input, &timesteps, &input_ids)
            .expect("first down block runs");
        assert_close("down0", &actual, &expected, 3e-4);
    }

    #[test]
    fn second_down_block_matches_python_trace_shape() {
        let unet = load_unet();
        let input = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0.npy",
        );
        let timesteps = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_timesteps.npy");
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__1.npy",
        );
        let actual = unet
            .forward_down1(&input, &timesteps, &input_ids)
            .expect("second down block runs");
        assert_close("down1", &actual, &expected, 3e-4);
    }

    #[test]
    fn third_down_block_matches_python_trace_shape() {
        let unet = load_unet();
        let input = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__1.npy",
        );
        let timesteps = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_timesteps.npy");
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let expected = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__2.npy",
        );
        let actual = unet
            .forward_down2(&input, &timesteps, &input_ids)
            .expect("third down block runs");
        assert_close("down2", &actual, &expected, 3e-4);
    }

    #[test]
    fn first_up_block_matches_python_trace_shape() {
        let unet = load_unet();
        let input = read_f32_npy(
            "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__2.npy",
        );
        let skips = vec![
            read_f32_npy(
                "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__1.npy",
            ),
            read_f32_npy(
                "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__2__attentions__0.npy",
            ),
            read_f32_npy(
                "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__2__attentions__1.npy",
            ),
        ];
        let timesteps = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_timesteps.npy");
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let expected =
            read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__up_blocks__0.npy");
        let actual = unet
            .forward_up0(&input, &skips, &timesteps, &input_ids)
            .expect("first up block runs");
        assert_close("up0", &actual, &expected, 4e-4);
    }

    #[test]
    fn second_up_block_matches_python_trace_shape() {
        let unet = load_unet();
        let input =
            read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__up_blocks__0.npy");
        let skips = vec![
            read_f32_npy(
                "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0.npy",
            ),
            read_f32_npy(
                "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__1__attentions__0.npy",
            ),
            read_f32_npy(
                "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__1__attentions__1.npy",
            ),
        ];
        let timesteps = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_timesteps.npy");
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let expected =
            read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__up_blocks__1.npy");
        let actual = unet
            .forward_up1(&input, &skips, &timesteps, &input_ids)
            .expect("second up block runs");
        assert_close("up1", &actual, &expected, 5e-4);
    }

    #[test]
    fn third_up_block_matches_python_trace_shape() {
        let unet = load_unet();
        let input =
            read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__up_blocks__1.npy");
        let skips = vec![
            read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__conv_in.npy"),
            read_f32_npy(
                "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__0.npy",
            ),
            read_f32_npy(
                "tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__down_blocks__0__attentions__1.npy",
            ),
        ];
        let timesteps = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_timesteps.npy");
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let expected =
            read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__up_blocks__2.npy");
        let actual = unet
            .forward_up2(&input, &skips, &timesteps, &input_ids)
            .expect("third up block runs");
        assert_close("up2", &actual, &expected, 2e-3);
    }

    #[test]
    fn post_process_matches_python_trace_shape() {
        let unet = load_unet();
        let input =
            read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__up_blocks__2.npy");
        let expected =
            read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_trace/diff_model__conv_out.npy");
        let actual = unet
            .forward_post_process(&input)
            .expect("post process runs");
        assert_close("post_process", &actual, &expected, 1e-4);
    }

    #[test]
    fn full_unet_matches_python_trace_shape() {
        let mut unet = load_unet();
        let input = read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_input_latent9.npy");
        let timesteps = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_timesteps.npy");
        let input_ids = read_i64_npy("tmp/fixtures/moebius-ft-places2/unet_input_ids.npy");
        let expected = read_f32_npy("tmp/fixtures/moebius-ft-places2/unet_noise.npy");
        let actual = unet
            .forward(&input, &timesteps, &input_ids)
            .expect("full UNet runs");
        assert_close("full_unet", &actual, &expected, 3e-3);
    }

    fn read_f32_npy(path: &str) -> Tensor {
        let array: ArrayD<f32> = read_npy(path).expect("fixture npy exists");
        let shape = array.shape().to_vec();
        let data = array.as_slice().expect("npy array is contiguous").to_vec();
        Tensor::from_vec(data, shape, &Device::Cpu)
            .expect("fixture tensor")
            .to_dtype(DType::F32)
            .expect("f32 tensor")
    }

    fn read_i64_npy(path: &str) -> Tensor {
        let array: ArrayD<i64> = read_npy(path).expect("fixture npy exists");
        let shape = array.shape().to_vec();
        let data = array.as_slice().expect("npy array is contiguous").to_vec();
        Tensor::from_vec(data, shape, &Device::Cpu).expect("fixture tensor")
    }

    fn load_unet() -> MoebiusUnet {
        let device = Device::Cpu;
        let weights = std::fs::read("public/models/moebius-ft-places2/moebius.safetensors")
            .expect("converted Moebius weights exist");
        MoebiusUnet::from_safetensors(weights, &device).expect("UNet loads traced layers")
    }

    fn assert_close(name: &str, actual: &Tensor, expected: &Tensor, threshold: f32) {
        assert_eq!(actual.shape(), expected.shape());
        let diff = (actual - expected)
            .expect("shapes match")
            .abs()
            .expect("absolute diff works")
            .max_all()
            .expect("max diff works")
            .to_scalar::<f32>()
            .expect("scalar f32");
        assert!(diff < threshold, "{name} max_abs_diff={diff}");
    }
}
