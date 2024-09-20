use candle_core::{DType, Device, Result, Tensor, D};

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

fn get_mscale(scale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * scale.ln() + 1.0
    }
}

pub struct LlamaYaRNScaledRotaryEmbedding {
    dim: usize,
    max_position_embeddings: usize,
    base: f64,
    scale: f64,
    original_max_position_embeddings: usize,
    extrapolation_factor: f64,
    attn_factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    inv_freq: Tensor,
    mscale: f64,
    cos_cached: Option<Tensor>,
    sin_cached: Option<Tensor>,
    max_seq_len_cached: usize,
}

impl LlamaYaRNScaledRotaryEmbedding {
    fn new(
        dim: usize,
        max_position_embeddings: usize,
        base: f64,
        scale: f64,
        original_max_position_embeddings: usize,
        extrapolation_factor: f64,
        attn_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
    ) -> Result<Self> {
        // TODO: how about f32?
        let dtype = DType::F64;
        let device = Device::Cpu;
        let step_tensor = Tensor::arange_step(0., dim as f64, 2., &device)?.to_dtype(dtype)?;
        let shape = step_tensor.shape();
        let base_tensor = Tensor::full(base as f32, shape, &device)?;

        //pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        //inv_freq_extrapolation = 1.0 / pos_freqs
        //inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)
        let pos_freqs = base_tensor.pow(&(step_tensor / (dim as f64))?)?;
        let inv_freq_extrapolation = (1. / &pos_freqs)?;
        let inv_freq_interpolation = (1. / (scale * &pos_freqs)?)?;

        //low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        //inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
        //inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        let (low, high) = find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        );
        let inv_freq_mask = ((1. - linear_ramp_mask(low, high, dim / 2)?)? * extrapolation_factor)?;
        let inv_freq = (inv_freq_interpolation * (1. - &inv_freq_mask)?
            + (inv_freq_extrapolation * &inv_freq_mask)?)?;

        //self.register_buffer("inv_freq", inv_freq)
        //self.mscale = float(get_mscale(self.scale) * self.attn_factor) # Get n-d magnitude scaling corrected for interpolation
        let mscale = get_mscale(scale) * attn_factor;

        Ok(Self {
            dim,
            max_position_embeddings,
            base,
            scale,
            original_max_position_embeddings,
            extrapolation_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            inv_freq,
            mscale,
            cos_cached: None,
            sin_cached: None,
            max_seq_len_cached: max_position_embeddings,
        })
    }

    //def forward(self, x, seq_len=None):
    //    # x: [bs, num_attention_heads, seq_len, head_size]
    //    # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
    //    if seq_len > self.max_seq_len_cached:
    //        self.max_seq_len_cached = seq_len

    //        t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
    //        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    //        # Different from paper, but it uses a different permutation in order to obtain the same calculation
    //        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

    //        self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(x.dtype), persistent=False)
    //        self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(x.dtype), persistent=False)
    //    return (
    //        self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
    //        self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
    //    )
    pub fn forward(&mut self, x: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        if seq_len > self.max_seq_len_cached {
            self.max_seq_len_cached = seq_len;
            let t = Tensor::arange(0., self.max_seq_len_cached as f64, &Device::Cpu)?;
            let freqs = t.unsqueeze(1)?.matmul(&self.inv_freq.unsqueeze(0)?)?;

            let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
            self.cos_cached = Some((emb.cos()? * self.mscale)?.unsqueeze(0)?.unsqueeze(0)?);
            self.sin_cached = Some((emb.sin()? * self.mscale)?.unsqueeze(0)?.unsqueeze(0)?);
        }
        let cos = self
            .cos_cached
            .as_ref()
            .unwrap()
            .narrow(2, 0, seq_len)?
            .to_dtype(x.dtype())?;
        let sin = self
            .sin_cached
            .as_ref()
            .unwrap()
            .narrow(2, 0, seq_len)?
            .to_dtype(x.dtype())?;
        Ok((cos, sin))
    }
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
        let scale = 1.;
        let original_max_position_embeddings = 2048;
        let extrapolation_factor = 1.;
        let attn_factor = 1.;
        let beta_fast = 32.;
        let beta_slow = 1.;

        let mut rotary_emb = LlamaYaRNScaledRotaryEmbedding::new(
            dim,
            max_position_embeddings,
            base,
            scale,
            original_max_position_embeddings,
            extrapolation_factor,
            attn_factor,
            beta_fast,
            beta_slow,
        )?;
        //x = torch.randn(2, 8, 16, 64)  # Example input: [bs, num_attention_heads, seq_len, head_size]
        //seq_len = 16
        //cos, sin = rotary_emb(x, seq_len)
        let x = Tensor::randn(5., 1., (2, 8, 16, 64), &Device::Cpu)?;
        let (sin, cos) = rotary_emb.borrow_mut().forward(&x, 16)?;
        println!("sin: {:?}", sin);
        println!("cos: {:?}", cos);
        Ok(())
    }
}
