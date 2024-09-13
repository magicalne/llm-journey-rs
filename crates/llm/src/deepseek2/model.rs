use crate::deepseek2::config::ModelConfig;
use anyhow::{anyhow, Result};
use candle_core::{self, quantized::gguf_file, DType, Device, Module, Tensor, D};
use candle_nn::{Embedding, LayerNorm};
use candle_transformers::{models::quantized_llama, quantized_var_builder::VarBuilder};

pub struct Model {
    token_embeddings: Embedding,
}

struct TransformerLayer {
    attention: MultiHeadAttention,
    ln_1: LayerNorm,
    ln_2: LayerNorm,
}

struct MultiHeadAttention {
    // Implement multi-head attention structure
}

impl Model {
    pub fn from_gguf(config: &ModelConfig, device: &Device, vb: VarBuilder) -> Result<Self> {
        let weights = vb.get_no_shape("token_embd.weight")?;
        let weights = weights.dequantize(device)?;
        let token_embeddings = Embedding::new(weights, config.hidden_size);
        Ok(Self { token_embeddings })
    }
}

#[derive(Debug, Clone)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
    //span: tracing::Span,
}

impl T5LayerNorm {
    fn load(h: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(h, "weight")?.dequantize(vb.device())?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
            //span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }
}

impl Module for T5LayerNorm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        //let _enter = self.span.enter();
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        // variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &ModelConfig, dev: &Device) -> Result<Self> {
        let rope_theta = cfg.rope_theta as f32;
        // TODO: migrate from mixtral to deepseek
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

pub fn read_gguf_file<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<VarBuilder> {
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(path, device)?;
    Ok(vb)
}
