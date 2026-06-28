use candle_core::{D, DType, Result, Tensor, bail};
use candle_nn::{
    BatchNorm, Conv2d, Conv2dConfig, GroupNorm, LayerNorm, Linear, Module, ModuleT, VarBuilder,
    batch_norm, conv2d, conv2d_no_bias, group_norm, layer_norm, linear, ops,
};

/// Sinusoidal timestep projection used before Moebius' learned time MLP.
pub struct Timesteps {
    num_channels: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
}

impl Timesteps {
    /// Creates the fixed positional timestep projection.
    pub fn new(num_channels: usize, flip_sin_to_cos: bool, downscale_freq_shift: f64) -> Self {
        Self {
            num_channels,
            flip_sin_to_cos,
            downscale_freq_shift,
        }
    }

    /// Projects integer scheduler timesteps into sinusoidal features.
    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let half_dim = (self.num_channels / 2) as u32;
        let exponent = Tensor::arange(0, half_dim, timesteps.device())?.to_dtype(DType::F32)?;
        let exponent = (exponent * -f64::ln(10000.))?;
        let exponent = (exponent / (half_dim as f64 - self.downscale_freq_shift))?;
        let emb = exponent.exp()?;
        let timesteps = timesteps.to_dtype(DType::F32)?;
        let emb = timesteps
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&emb.unsqueeze(0)?)?;
        let (cos, sin) = (emb.cos()?, emb.sin()?);
        let emb = if self.flip_sin_to_cos {
            Tensor::cat(&[&cos, &sin], D::Minus1)?
        } else {
            Tensor::cat(&[&sin, &cos], D::Minus1)?
        };
        if self.num_channels % 2 == 1 {
            emb.pad_with_zeros(D::Minus1, 0, 1)
        } else {
            Ok(emb)
        }
    }
}

/// Learned Moebius timestep embedding MLP.
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    /// Loads the learned timestep MLP from converted Moebius weights.
    pub fn new(channel: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear(channel, time_embed_dim, vb.pp("linear_1"))?,
            linear_2: linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?,
        })
    }

    /// Applies the timestep MLP.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = ops::silu(&self.linear_1.forward(input)?)?;
        self.linear_2.forward(&hidden)
    }
}

/// Depthwise-separable convolution block used by the Moebius UNet.
///
/// The upstream block comes from Timm's `DepthwiseSeparableConv`: depthwise
/// convolution, BatchNorm+activation, pointwise convolution, then BatchNorm.
/// Moebius uses it for `conv_in`, `conv_out`, and every DW residual block.
pub struct DepthwiseSeparableConv {
    conv_dw: Conv2d,
    bn1: BatchNorm,
    conv_pw: Conv2d,
    bn2: BatchNorm,
    has_skip: bool,
    bn1_has_relu: bool,
}

impl DepthwiseSeparableConv {
    /// Loads a depthwise-separable block from the converted Moebius weights.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let depthwise_cfg = Conv2dConfig {
            padding: kernel_size / 2,
            groups: in_channels,
            ..Default::default()
        };
        let pointwise_cfg = Conv2dConfig::default();
        Ok(Self {
            conv_dw: conv2d_no_bias(
                in_channels,
                in_channels,
                kernel_size,
                depthwise_cfg,
                vb.pp("conv_dw"),
            )?,
            bn1: batch_norm(in_channels, 1e-5, vb.pp("bn1"))?,
            conv_pw: conv2d_no_bias(
                in_channels,
                out_channels,
                1,
                pointwise_cfg,
                vb.pp("conv_pw"),
            )?,
            bn2: batch_norm(out_channels, 1e-5, vb.pp("bn2"))?,
            has_skip: in_channels == out_channels,
            bn1_has_relu: true,
        })
    }

    /// Applies the block in inference mode.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shortcut = input;
        let mut hidden = self.conv_dw.forward(input)?;
        hidden = self.bn1.forward_t(&hidden, false)?;
        if self.bn1_has_relu {
            hidden = hidden.relu()?;
        }
        hidden = self.conv_pw.forward(&hidden)?;
        hidden = self.bn2.forward_t(&hidden, false)?;
        if self.has_skip {
            hidden = (hidden + shortcut)?;
        }
        Ok(hidden)
    }
}

/// Moebius residual block built from depthwise-separable convolutions.
pub struct DWResnetBlock2D {
    norm1: GroupNorm,
    conv1: DepthwiseSeparableConv,
    time_emb_proj: Linear,
    norm2: GroupNorm,
    conv2: DepthwiseSeparableConv,
    conv_shortcut: Option<Conv2d>,
}

impl DWResnetBlock2D {
    /// Loads a Moebius `DWResnetBlock2D`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_shortcut = load_shortcut(in_channels, out_channels, vb.pp("conv_shortcut"))?;
        Ok(Self {
            norm1: group_norm(32, in_channels, 1e-5, vb.pp("norm1"))?,
            conv1: DepthwiseSeparableConv::new(in_channels, out_channels, 3, vb.pp("conv1"))?,
            time_emb_proj: linear(temb_channels, out_channels, vb.pp("time_emb_proj"))?,
            norm2: group_norm(32, out_channels, 1e-5, vb.pp("norm2"))?,
            conv2: DepthwiseSeparableConv::new(out_channels, out_channels, 3, vb.pp("conv2"))?,
            conv_shortcut,
        })
    }

    /// Applies the residual block in inference mode.
    pub fn forward(&self, input: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let hidden = self.norm1.forward(input)?;
        let hidden = self.conv1.forward(&ops::silu(&hidden)?)?;
        let temb = self.time_emb_proj.forward(&ops::silu(temb)?)?;
        let hidden = hidden.broadcast_add(&temb.unsqueeze(2)?.unsqueeze(3)?)?;
        let hidden = self.norm2.forward(&hidden)?;
        let hidden = self.conv2.forward(&ops::silu(&hidden)?)?;
        let input = match &self.conv_shortcut {
            Some(shortcut) => shortcut.forward(input)?,
            None => input.clone(),
        };
        input + hidden
    }
}

#[inline]
fn load_shortcut(
    in_channels: usize,
    out_channels: usize,
    vb: VarBuilder,
) -> Result<Option<Conv2d>> {
    if in_channels == out_channels {
        return Ok(None);
    }
    conv2d(in_channels, out_channels, 1, Conv2dConfig::default(), vb).map(Some)
}

/// MixTransformer feed-forward block used by Moebius attention layers.
pub struct GLUMBConv {
    inverted_conv: Conv2d,
    depth_conv: Conv2d,
    point_conv: Conv2d,
}

impl GLUMBConv {
    /// Loads a GLUMBConv/MixFFN block from converted Moebius weights.
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let depth_cfg = Conv2dConfig {
            padding: 1,
            groups: hidden_features * 2,
            ..Default::default()
        };
        Ok(Self {
            inverted_conv: conv2d(
                in_features,
                hidden_features * 2,
                1,
                Default::default(),
                vb.pp("inverted_conv.conv"),
            )?,
            depth_conv: conv2d(
                hidden_features * 2,
                hidden_features * 2,
                3,
                depth_cfg,
                vb.pp("depth_conv.conv"),
            )?,
            point_conv: conv2d_no_bias(
                hidden_features,
                out_features,
                1,
                Default::default(),
                vb.pp("point_conv.conv"),
            )?,
        })
    }

    /// Applies the feed-forward block to `[batch, tokens, channels]` hidden states.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, tokens, channels) = input.dims3()?;
        let side = square_side(tokens)?;
        let hidden = input
            .reshape((batch, side, side, channels))?
            .permute((0, 3, 1, 2))?;
        let hidden = ops::silu(&self.inverted_conv.forward(&hidden)?)?;
        let hidden = self.depth_conv.forward(&hidden)?;
        let chunks = hidden.chunk(2, 1)?;
        let hidden = chunks[0].broadcast_mul(&ops::silu(&chunks[1])?)?;
        let hidden = self.point_conv.forward(&hidden)?;
        hidden
            .reshape((batch, channels, tokens))?
            .permute((0, 2, 1))
    }
}

#[inline]
fn square_side(tokens: usize) -> Result<usize> {
    let side = (tokens as f64).sqrt() as usize;
    if side * side != tokens {
        bail!("expected square token count, got {tokens}");
    }
    Ok(side)
}

/// Multi-query Lambda self-attention used inside Moebius MixTransformer blocks.
pub struct MultiQuerySelfLambda {
    to_q: Conv2d,
    to_k: Conv2d,
    to_v: Conv2d,
    norm_q: BatchNorm,
    norm_v: BatchNorm,
    pos_conv: Conv2d,
    heads: usize,
    dim_k: usize,
    dim_v: usize,
}

impl MultiQuerySelfLambda {
    /// Loads local-window Lambda self-attention from converted Moebius weights.
    pub fn new(dim: usize, dim_k: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let dim_v = dim / heads;
        Ok(Self {
            to_q: conv2d_no_bias(dim, dim_k * heads, 1, Default::default(), vb.pp("to_q"))?,
            to_k: conv2d_no_bias(dim, dim_k, 1, Default::default(), vb.pp("to_k"))?,
            to_v: conv2d_no_bias(dim, dim_v, 1, Default::default(), vb.pp("to_v"))?,
            norm_q: batch_norm(dim_k * heads, 1e-5, vb.pp("norm_q"))?,
            norm_v: batch_norm(dim_v, 1e-5, vb.pp("norm_v"))?,
            pos_conv: load_lambda_pos_conv(dim_k, vb.pp("pos_conv"))?,
            heads,
            dim_k,
            dim_v,
        })
    }

    /// Applies self-attention to `[batch, tokens, channels]` hidden states.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, tokens, channels) = input.dims3()?;
        let side = square_side(tokens)?;
        let image = input
            .reshape((batch, side, side, channels))?
            .permute((0, 3, 1, 2))?;
        let q = self.norm_q.forward_t(&self.to_q.forward(&image)?, false)?;
        let k = self.to_k.forward(&image)?;
        let v = self.norm_v.forward_t(&self.to_v.forward(&image)?, false)?;
        let q = q.reshape((batch, self.heads, self.dim_k, tokens))?;
        let k = ops::softmax(&k.reshape((batch, self.dim_k, tokens))?, D::Minus1)?;
        let v_flat = v.reshape((batch, self.dim_v, tokens))?;
        let lambda_c = k.broadcast_matmul(&v_flat.transpose(1, 2)?)?;
        let yc = q
            .transpose(2, 3)?
            .broadcast_matmul(&lambda_c.unsqueeze(1)?)?
            .permute((0, 1, 3, 2))?;
        let yp = self.local_position_term(&q, &v, side, tokens)?;
        let y = (yc + yp)?;
        y.reshape((batch, channels, side, side))?
            .permute((0, 2, 3, 1))?
            .reshape((batch, tokens, channels))
    }

    fn local_position_term(
        &self,
        q: &Tensor,
        v: &Tensor,
        side: usize,
        tokens: usize,
    ) -> Result<Tensor> {
        let batch = v.dim(0)?;
        let pos = v
            .reshape((batch * self.dim_v, 1, side, side))?
            .apply(&self.pos_conv)?
            .reshape((batch, self.dim_v, self.dim_k, tokens))?
            .permute((0, 2, 1, 3))?;
        q.unsqueeze(3)?.broadcast_mul(&pos.unsqueeze(1)?)?.sum(2)
    }
}

#[inline]
fn load_lambda_pos_conv(dim_k: usize, vb: VarBuilder) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding: 7,
        ..Default::default()
    };
    let weight = vb.get((dim_k, 1, 1, 15, 15), "weight")?.squeeze(2)?;
    let bias = vb.get(dim_k, "bias")?;
    Ok(Conv2d::new(weight, Some(bias), cfg))
}

/// Multi-query Lambda cross-attention used inside Moebius MixTransformer blocks.
pub struct MultiQueryCrossLambda {
    to_q: Conv2d,
    to_k: Linear,
    to_v: Linear,
    norm_q: BatchNorm,
    norm_v: BatchNorm,
    rel_pos_emb: Tensor,
    heads: usize,
    dim_k: usize,
    dim_v: usize,
}

impl MultiQueryCrossLambda {
    /// Loads global Lambda cross-attention from converted Moebius weights.
    pub fn new(
        dim: usize,
        dim_cross: usize,
        dim_k: usize,
        heads: usize,
        tokens: usize,
        context_tokens: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dim_v = dim / heads;
        Ok(Self {
            to_q: conv2d_no_bias(dim, dim_k * heads, 1, Default::default(), vb.pp("to_q"))?,
            to_k: linear_no_bias(dim_cross, dim_k, vb.pp("to_k"))?,
            to_v: linear_no_bias(dim_cross, dim_v, vb.pp("to_v"))?,
            norm_q: batch_norm(dim_k * heads, 1e-5, vb.pp("norm_q"))?,
            norm_v: batch_norm(dim_v, 1e-5, vb.pp("norm_v"))?,
            rel_pos_emb: vb
                .get((tokens, context_tokens, dim_k, 1), "rel_pos_emb")?
                .squeeze(3)?,
            heads,
            dim_k,
            dim_v,
        })
    }

    /// Applies cross-attention to hidden states and projected condition tokens.
    pub fn forward(&self, input: &Tensor, encoder_hidden_states: &Tensor) -> Result<Tensor> {
        let (batch, tokens, channels) = input.dims3()?;
        let side = square_side(tokens)?;
        let context_tokens = encoder_hidden_states.dim(1)?;
        let image = input
            .reshape((batch, side, side, channels))?
            .permute((0, 3, 1, 2))?;
        let q = self.norm_q.forward_t(&self.to_q.forward(&image)?, false)?;
        let k = self.to_k.forward(encoder_hidden_states)?.transpose(1, 2)?;
        let v = self.to_v.forward(encoder_hidden_states)?.transpose(1, 2)?;
        let q = q.reshape((batch, self.heads, self.dim_k, tokens))?;
        let k = ops::softmax(&k.reshape((batch, self.dim_k, context_tokens))?, D::Minus1)?;
        let v = self
            .norm_v
            .forward_t(&v.reshape((batch, self.dim_v, context_tokens))?, false)?;
        let lambda_c = k.broadcast_matmul(&v.transpose(1, 2)?)?;
        let yc = q
            .transpose(2, 3)?
            .broadcast_matmul(&lambda_c.unsqueeze(1)?)?
            .permute((0, 1, 3, 2))?;
        let yp = self.global_position_term(&q, &v)?;
        let y = (yc + yp)?;
        y.reshape((batch, channels, side, side))?
            .permute((0, 2, 3, 1))?
            .reshape((batch, tokens, channels))
    }

    fn global_position_term(&self, q: &Tensor, v: &Tensor) -> Result<Tensor> {
        let lambda_p = self
            .rel_pos_emb
            .permute((0, 2, 1))?
            .unsqueeze(0)?
            .broadcast_matmul(&v.transpose(1, 2)?.unsqueeze(1)?)?
            .permute((0, 2, 1, 3))?;
        q.unsqueeze(4)?
            .broadcast_mul(&lambda_p.unsqueeze(1)?)?
            .sum(2)?
            .permute((0, 1, 3, 2))
    }
}

#[inline]
fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

/// One Moebius MixTransformer block: self-lambda, cross-lambda, and MixFFN.
pub struct MixTransformerBlock {
    norm1: LayerNorm,
    attn1: MultiQuerySelfLambda,
    norm2: LayerNorm,
    attn2: MultiQueryCrossLambda,
    norm3: LayerNorm,
    ff: GLUMBConv,
}

impl MixTransformerBlock {
    /// Loads a single Moebius MixTransformer block.
    pub fn new(config: MixTransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(config.dim, 1e-5, vb.pp("norm1"))?,
            attn1: MultiQuerySelfLambda::new(
                config.dim,
                config.dim_k,
                config.heads,
                vb.pp("attn1"),
            )?,
            norm2: layer_norm(config.dim, 1e-5, vb.pp("norm2"))?,
            attn2: MultiQueryCrossLambda::new(
                config.dim,
                config.cross_dim,
                config.dim_k,
                config.heads,
                config.tokens,
                config.context_tokens,
                vb.pp("attn2"),
            )?,
            norm3: layer_norm(config.dim, 1e-5, vb.pp("norm3"))?,
            ff: GLUMBConv::new(config.dim, config.ff_hidden_dim, config.dim, vb.pp("ff"))?,
        })
    }

    /// Applies the transformer block to tokenized hidden states.
    pub fn forward(&self, input: &Tensor, encoder_hidden_states: &Tensor) -> Result<Tensor> {
        let attn1 = self.attn1.forward(&self.norm1.forward(input)?)?;
        let hidden = (input + attn1)?;
        let attn2 = self
            .attn2
            .forward(&self.norm2.forward(&hidden)?, encoder_hidden_states)?;
        let hidden = (hidden + attn2)?;
        let ff = self.ff.forward(&self.norm3.forward(&hidden)?)?;
        hidden + ff
    }
}

/// Fixed dimensions for a Moebius MixTransformer layer instance.
#[derive(Clone, Copy)]
pub struct MixTransformerConfig {
    /// Token/channel width inside the transformer.
    pub dim: usize,
    /// Cross-attention condition width.
    pub cross_dim: usize,
    /// Lambda key width per head group.
    pub dim_k: usize,
    /// Number of Lambda query heads.
    pub heads: usize,
    /// Spatial token count.
    pub tokens: usize,
    /// Conditioning token count.
    pub context_tokens: usize,
    /// Hidden channel count in the MixFFN block.
    pub ff_hidden_dim: usize,
}

/// Continuous-input MixTransformer2D wrapper around a single Moebius block.
pub struct MixTransformer2DModel {
    norm: GroupNorm,
    proj_in: Conv2d,
    block: MixTransformerBlock,
    proj_out: Conv2d,
}

impl MixTransformer2DModel {
    /// Loads a one-layer Moebius MixTransformer2D model.
    pub fn new(config: MixTransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: group_norm(32, config.dim, 1e-6, vb.pp("norm"))?,
            proj_in: conv2d(
                config.dim,
                config.dim,
                1,
                Default::default(),
                vb.pp("proj_in"),
            )?,
            block: MixTransformerBlock::new(config, vb.pp("transformer_blocks.0"))?,
            proj_out: conv2d(
                config.dim,
                config.dim,
                1,
                Default::default(),
                vb.pp("proj_out"),
            )?,
        })
    }

    /// Applies the 2D transformer and returns an image-shaped residual output.
    pub fn forward(&self, input: &Tensor, encoder_hidden_states: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = input.dims4()?;
        let residual = input;
        let hidden = self.proj_in.forward(&self.norm.forward(input)?)?;
        let hidden = hidden
            .permute((0, 2, 3, 1))?
            .reshape((batch, height * width, channels))?;
        let hidden = self.block.forward(&hidden, encoder_hidden_states)?;
        let hidden = hidden
            .reshape((batch, height, width, channels))?
            .permute((0, 3, 1, 2))?;
        let hidden = self.proj_out.forward(&hidden)?;
        hidden + residual
    }
}

/// Diffusers-style stride-2 convolutional downsampler.
pub struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    /// Loads a 3x3 stride-2 downsampler.
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d(channels, channels, 3, cfg, vb.pp("conv"))?,
        })
    }

    /// Applies the downsampler.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.conv.forward(input)
    }
}

/// A Moebius down block with two DW residual/attention pairs and optional downsample.
pub struct DWMixTFDownBlock2D {
    resnets: Vec<DWResnetBlock2D>,
    attentions: Vec<MixTransformer2DModel>,
    downsampler: Option<Downsample2D>,
}

impl DWMixTFDownBlock2D {
    /// Loads one Moebius down block.
    pub fn new(config: DWMixTFDownBlockConfig, vb: VarBuilder) -> Result<Self> {
        let resnets = vec![
            DWResnetBlock2D::new(
                config.in_channels,
                config.out_channels,
                config.temb_channels,
                vb.pp("resnets.0"),
            )?,
            DWResnetBlock2D::new(
                config.out_channels,
                config.out_channels,
                config.temb_channels,
                vb.pp("resnets.1"),
            )?,
        ];
        let attention_config = config.attention_config();
        let attentions = vec![
            MixTransformer2DModel::new(attention_config, vb.pp("attentions.0"))?,
            MixTransformer2DModel::new(attention_config, vb.pp("attentions.1"))?,
        ];
        let downsampler = if config.add_downsample {
            Some(Downsample2D::new(
                config.out_channels,
                vb.pp("downsamplers.0"),
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            attentions,
            downsampler,
        })
    }

    /// Applies the down block and returns its final sample.
    pub fn forward(
        &self,
        input: &Tensor,
        temb: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let (hidden, _) = self.forward_with_states(input, temb, encoder_hidden_states)?;
        Ok(hidden)
    }

    /// Applies the down block and returns both final sample and skip states.
    pub fn forward_with_states(
        &self,
        input: &Tensor,
        temb: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut hidden = input.clone();
        let mut states = Vec::with_capacity(3);
        for (resnet, attention) in self.resnets.iter().zip(self.attentions.iter()) {
            hidden = resnet.forward(&hidden, temb)?;
            hidden = attention.forward(&hidden, encoder_hidden_states)?;
            states.push(hidden.clone());
        }
        if let Some(downsampler) = &self.downsampler {
            hidden = downsampler.forward(&hidden)?;
            states.push(hidden.clone());
        }
        Ok((hidden, states))
    }
}

/// Dimensions for a two-layer Moebius down block.
#[derive(Clone, Copy)]
pub struct DWMixTFDownBlockConfig {
    /// Input image-channel count.
    pub in_channels: usize,
    /// Output image-channel count.
    pub out_channels: usize,
    /// Time embedding width.
    pub temb_channels: usize,
    /// Spatial token count for each attention block.
    pub tokens: usize,
    /// Whether the block ends with a stride-2 downsample.
    pub add_downsample: bool,
}

impl DWMixTFDownBlockConfig {
    #[inline]
    fn attention_config(self) -> MixTransformerConfig {
        MixTransformerConfig {
            dim: self.out_channels,
            cross_dim: 768,
            dim_k: self.out_channels / 8,
            heads: 8,
            tokens: self.tokens,
            context_tokens: 10,
            ff_hidden_dim: self.out_channels * 5 / 2,
        }
    }
}

/// Diffusers-style nearest-neighbor upsampler followed by a convolution.
pub struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    /// Loads an upsampler convolution.
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d(channels, channels, 3, cfg, vb.pp("conv"))?,
        })
    }

    /// Applies nearest-neighbor 2x upsampling and the learned convolution.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, _, height, width) = input.dims4()?;
        let hidden = input.upsample_nearest2d(height * 2, width * 2)?;
        self.conv.forward(&hidden)
    }
}

/// A Moebius up block with three DW residual/attention pairs and optional upsample.
pub struct DWMixTFUpBlock2D {
    resnets: Vec<DWResnetBlock2D>,
    attentions: Vec<MixTransformer2DModel>,
    upsampler: Option<Upsample2D>,
}

impl DWMixTFUpBlock2D {
    /// Loads one Moebius up block.
    pub fn new(config: DWMixTFUpBlockConfig, vb: VarBuilder) -> Result<Self> {
        let attention_config = config.attention_config();
        let mut resnets = Vec::with_capacity(3);
        let mut attentions = Vec::with_capacity(3);
        for index in 0..3 {
            resnets.push(DWResnetBlock2D::new(
                config.resnet_input_channels(index),
                config.out_channels,
                config.temb_channels,
                vb.pp(format!("resnets.{index}")),
            )?);
            attentions.push(MixTransformer2DModel::new(
                attention_config,
                vb.pp(format!("attentions.{index}")),
            )?);
        }
        let upsampler = if config.add_upsample {
            Some(Upsample2D::new(config.out_channels, vb.pp("upsamplers.0"))?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            attentions,
            upsampler,
        })
    }

    /// Applies the up block using skip samples in pop order.
    pub fn forward(
        &self,
        input: &Tensor,
        skips: &[Tensor],
        temb: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        if skips.len() != self.resnets.len() {
            bail!(
                "expected {} up-block skips, got {}",
                self.resnets.len(),
                skips.len()
            );
        }
        let mut hidden = input.clone();
        for ((resnet, attention), skip) in self
            .resnets
            .iter()
            .zip(self.attentions.iter())
            .zip(skips.iter().rev())
        {
            hidden = Tensor::cat(&[&hidden, skip], 1)?;
            hidden = resnet.forward(&hidden, temb)?;
            hidden = attention.forward(&hidden, encoder_hidden_states)?;
        }
        match &self.upsampler {
            Some(upsampler) => upsampler.forward(&hidden),
            None => Ok(hidden),
        }
    }
}

/// Dimensions for a three-layer Moebius up block.
#[derive(Clone, Copy)]
pub struct DWMixTFUpBlockConfig {
    /// Skip-channel count from the next lower-resolution down block.
    pub in_channels: usize,
    /// Output image-channel count.
    pub out_channels: usize,
    /// Previous decoder output-channel count.
    pub prev_output_channel: usize,
    /// Time embedding width.
    pub temb_channels: usize,
    /// Spatial token count before any optional upsample.
    pub tokens: usize,
    /// Whether the block ends with 2x upsample.
    pub add_upsample: bool,
}

impl DWMixTFUpBlockConfig {
    #[inline]
    fn attention_config(self) -> MixTransformerConfig {
        MixTransformerConfig {
            dim: self.out_channels,
            cross_dim: 768,
            dim_k: self.out_channels / 8,
            heads: 8,
            tokens: self.tokens,
            context_tokens: 10,
            ff_hidden_dim: self.out_channels * 5 / 2,
        }
    }

    #[inline]
    fn resnet_input_channels(self, index: usize) -> usize {
        let skip_channels = if index == 2 {
            self.in_channels
        } else {
            self.out_channels
        };
        let hidden_channels = if index == 0 {
            self.prev_output_channel
        } else {
            self.out_channels
        };
        hidden_channels + skip_channels
    }
}
