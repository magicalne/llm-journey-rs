// ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};

fn find_correction_dim(
    num_rotations: f64,
    dim: usize,
    base: f64,
    max_position_embeddings: usize,
) -> f64 {
    (dim as f64
        * (max_position_embeddings as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
        / (2.0 * base.ln())
}

fn find_correction_range(
    low_rot: f64,
    high_rot: f64,
    dim: usize,
    base: f64,
    max_position_embeddings: usize,
) -> (usize, usize) {
    let low = find_correction_dim(low_rot, dim, base, max_position_embeddings).floor() as usize;
    let high = find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil() as usize;
    (low.max(0), high.min(dim - 1))
}

//def linear_ramp_mask(min, max, dim):
//    if min == max:
//        max += 0.001  # Prevent singularity
//
//    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
//    ramp_func = torch.clamp(linear_func, 0, 1)
//    return ramp_func
fn linear_ramp_mask(min: usize, max: usize, dim: usize) -> Result<Tensor> {
    let delta = match min == max {
        true => 0.001,
        false => 0.,
    };
    let min = min as f64;
    let max = max as f64 + delta;
    let linear_func = ((Tensor::arange(0., dim as f64, &Device::Cpu)? - min)? / (max - min))?;
    linear_func.clamp(0.0, 1.0)
}

pub fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

#[derive(Debug, Clone)]
pub struct LlamaYaRNScaledRotaryEmbedding {
    rotary_dim: usize,
    max_position_embeddings: usize,
    base: f64,
    scaling_factor: f64,
    original_max_position_embeddings: usize,
    extrapolation_factor: f64,
    // attn_factor: f64, DeepSeek doesn't use this.
    beta_fast: f64,
    beta_slow: f64,
    inv_freq: Tensor,
    mscale: f64,
    max_seq_len_cached: usize,
    cache: Tensor,
    head_size: usize,
}

impl LlamaYaRNScaledRotaryEmbedding {
    pub fn new(
        head_size: usize,
        rotary_dim: usize,
        max_position_embeddings: usize,
        base: f64,
        scaling_factor: f64,
        mscale: f64,
        mscale_all_dim: f64,
        original_max_position_embeddings: usize,
        extrapolation_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
    ) -> Result<Self> {
        // TODO: how about f32?
        let dtype = DType::F64;
        let device = Device::Cpu;
        let step_tensor =
            Tensor::arange_step(0., rotary_dim as f64, 2., &device)?.to_dtype(dtype)?;
        let shape = step_tensor.shape();
        let base_tensor = Tensor::full(base as f32, shape, &device)?;

        //pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        //inv_freq_extrapolation = 1.0 / pos_freqs
        //inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)
        let pos_freqs = base_tensor.pow(&(step_tensor / (rotary_dim as f64))?)?;
        let inv_freq_extrapolation = (1. / &pos_freqs)?;
        let inv_freq_interpolation = (1. / (scaling_factor * &pos_freqs)?)?;

        //low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        //inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
        //inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        let (low, high) = find_correction_range(
            beta_fast,
            beta_slow,
            rotary_dim,
            base,
            original_max_position_embeddings,
        );
        let inv_freq_mask =
            ((1. - linear_ramp_mask(low, high, rotary_dim / 2)?)? * extrapolation_factor)?;
        let inv_freq = (inv_freq_interpolation * (1. - &inv_freq_mask)?
            + (inv_freq_extrapolation * &inv_freq_mask)?)?;

        let mscale = yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim);
        let t = Tensor::arange(
            0.,
            (max_position_embeddings as f64 * scaling_factor) as f64,
            &Device::Cpu,
        )?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = (freqs.cos()? * mscale)?;
        let sin = (freqs.sin()? * mscale)?;
        let cache = Tensor::cat(&[&cos, &sin], D::Minus1)?;

        Ok(Self {
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            scaling_factor,
            original_max_position_embeddings,
            extrapolation_factor,
            beta_fast,
            beta_slow,
            inv_freq,
            mscale,
            cache,
            max_seq_len_cached: max_position_embeddings,
        })
    }

    pub fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // x: [bs, num_attention_heads, seq_len, head_size]
        let inv_freq_expanded = self.inv_freq.unsqueeze(0)?.unsqueeze(2)?;
        let dims = inv_freq_expanded.dims();
        let inv_freq_expanded =
            inv_freq_expanded.expand(&[position_ids.shape().dims()[0], dims[1], 1])?;
        let position_ids_expanded = position_ids.unsqueeze(1)?.unsqueeze(2)?;
        // Force float32 since bfloat16 loses precision on long contexts
        let freqs = inv_freq_expanded
            .to_dtype(DType::F32)?
            .matmul(&position_ids_expanded.to_dtype(DType::F32)?)?
            .transpose(1, 2)?;
        let emb = Tensor::cat(&[freqs.clone(), freqs], D::Minus1)?;
        let cos = (emb.cos()? * self.mscale)?;
        let sin = (emb.sin()? * self.mscale)?;
        Ok((cos.to_dtype(x.dtype())?, sin.to_dtype(x.dtype())?))
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(x.dims().len() - 1)?;
    let half_dim = last_dim / 2;
    let x1 = x.i((.., ..half_dim))?;
    let x2 = x.i((.., half_dim..))?;
    Tensor::cat(&[&x2.neg()?, &x1], x.dims().len() - 1)
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    unsqueeze_dim: Option<usize>,
) -> Result<(Tensor, Tensor)> {
    let unsqueeze_dim = unsqueeze_dim.unwrap_or(1);
    let cos = cos.unsqueeze(unsqueeze_dim)?;
    let sin = sin.unsqueeze(unsqueeze_dim)?;

    //b, h, s, d = q.shape
    //q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    //b, h, s, d = k.shape
    //k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    let (b, h, s, d) = q.dims4()?;
    let q = q
        .reshape((b, h, s, d / 2, 2))?
        .transpose(4, 3)?
        .reshape((b, h, s, d))?;
    let (b, h, s, d) = k.dims4()?;
    let k = k
        .reshape((b, h, s, d / 2, 2))?
        .transpose(4, 3)?
        .reshape((b, h, s, d))?;

    let q_embed = (&q * &cos)? + &(rotate_half(&q)? * &sin)?;
    let k_embed = (&k * &cos)? + &(rotate_half(&k)? * &sin)?;

    Ok((q_embed?, k_embed?))
}

#[cfg(test)]
mod tests {
    use std::borrow::BorrowMut;

    use candle_core::{Device, Result, Tensor};

    use super::LlamaYaRNScaledRotaryEmbedding;

    #[test]
    fn test() -> Result<()> {
        //dim = 128
        //max_position_embeddings = 2048
        //base = 10000
        //scale = 1
        //original_max_position_embeddings = 2048
        //extrapolation_factor = 1
        //attn_factor = 1
        //beta_fast = 32
        //beta_slow = 1
        //finetuned = False
        //device = None
        //
        //rotary_emb = LlamaYaRNScaledRotaryEmbedding(dim, max_position_embeddings, base, scale, original_max_position_embeddings, extrapolation_factor, attn_factor, beta_fast, beta_slow, finetuned, device)
        //
        //# Mock data for forward pass
        //x = torch.randn(2, 8, 16, 64)  # Example input: [bs, num_attention_heads, seq_len, head_size]
        //seq_len = 16
        //
        //cos, sin = rotary_emb(x, seq_len)
        //
        //print(cos.shape, sin.shape)
        //
        let dim = 128;
        let max_position_embeddings = 2048;
        let base = 10000.;
        let scaling_factor = 40.;
        let mscale = 0.707;
        let mscale_all_dim = 0.707;
        let original_max_position_embeddings = 2048;
        let extrapolation_factor = 1.;
        let beta_fast = 32.;
        let beta_slow = 1.;

        let mut rotary_emb = LlamaYaRNScaledRotaryEmbedding::new(
            dim,
            dim,
            max_position_embeddings,
            base,
            scaling_factor,
            mscale,
            mscale_all_dim,
            original_max_position_embeddings,
            extrapolation_factor,
            beta_fast,
            beta_slow,
        )?;
        //x = torch.randn(2, 8, 16, 64)  # Example input: [bs, num_attention_heads, seq_len, head_size]
        //seq_len = 16
        //cos, sin = rotary_emb(x, seq_len)
        let x = Tensor::randn(5., 1., (2, 8, 16, 64), &Device::Cpu)?;
        let positions = Tensor::arange(0, 16, &Device::Cpu)?;
        let (sin, cos) = rotary_emb.borrow_mut().forward(&x, &positions)?;
        println!("sin: {:?}", sin);
        println!("cos: {:?}", cos);
        Ok(())
    }
}
