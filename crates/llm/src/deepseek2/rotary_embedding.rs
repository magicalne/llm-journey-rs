use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};

#[derive(Debug, Clone)]
pub struct DeepseekScalingRotaryEmbedding {
    head_size: usize,
    rotary_dim: usize,
    max_position_embeddings: usize,
    base: f64,
    is_neox_style: bool,
    scaling_factor: f64,
    extrapolation_factor: f64,
    attn_factor: f64,
    beta_fast: usize,
    beta_slow: usize,
    mscale: f64,
    mscale_all_dim: f64,
    cos_sin_cache: Tensor,
}

impl DeepseekScalingRotaryEmbedding {
    pub fn new(
        head_size: usize,
        rotary_dim: usize,
        max_position_embeddings: usize,
        base: f64,
        is_neox_style: bool,
        scaling_factor: f64,
        dtype: DType,
        extrapolation_factor: f64,
        attn_factor: f64,
        beta_fast: usize,
        beta_slow: usize,
        mscale: f64,
        mscale_all_dim: f64,
        device: &Device,
    ) -> Result<Self> {
        let mscale = yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim)
            * attn_factor;
        let cos_sin_cache = Self::compute_cos_sin_cache(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            scaling_factor,
            extrapolation_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            mscale,
            dtype,
            device,
        )?;

        Ok(Self {
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            scaling_factor,
            extrapolation_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            mscale,
            mscale_all_dim,
            cos_sin_cache,
        })
    }

    fn compute_cos_sin_cache(
        head_size: usize,
        rotary_dim: usize,
        max_position_embeddings: usize,
        base: f64,
        is_neox_style: bool,
        scaling_factor: f64,
        extrapolation_factor: f64,
        attn_factor: f64,
        beta_fast: usize,
        beta_slow: usize,
        mscale: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let inv_freq = Self::compute_inv_freq(
            base,
            rotary_dim,
            scaling_factor,
            extrapolation_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            max_position_embeddings,
            dtype,
        )?;
        let t = Tensor::arange(
            0,
            max_position_embeddings as u32 * scaling_factor as u32,
            device,
        )?
        .to_dtype(dtype)?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = (freqs.cos()? * mscale)?;
        let sin = (freqs.sin()? * mscale)?;
        Ok(Tensor::cat(&[&cos, &sin], D::Minus1)?)
    }

    fn compute_inv_freq(
        base: f64,
        rotary_dim: usize,
        scaling_factor: f64,
        extrapolation_factor: f64,
        attn_factor: f64,
        beta_fast: usize,
        beta_slow: usize,
        max_position_embeddings: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let pos_freqs = Tensor::from_slice(
            &(0..rotary_dim)
                .step_by(2)
                .map(|i| base.powf(i as f64 / rotary_dim as f64))
                .collect::<Vec<f64>>(),
            (rotary_dim / 2,),
            &Device::Cpu,
        )?;
        let inv_freq_extrapolation =
            (Tensor::ones((rotary_dim / 2,), dtype, &Device::Cpu)? / &pos_freqs)?;
        let inv_freq_interpolation = (Tensor::ones((rotary_dim / 2,), dtype, &Device::Cpu)?
            / (&pos_freqs * scaling_factor))?;

        let (low, high) = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            rotary_dim,
            base,
            max_position_embeddings,
        );
        let inv_freq_mask = (1 - yarn_linear_ramp_mask(low, high, rotary_dim / 2, dtype)?)?;
        let inv_freq = (inv_freq_interpolation * (1.0 - &inv_freq_mask)
            + inv_freq_extrapolation * &inv_freq_mask)?;
        Ok(inv_freq)
    }

    pub fn forward(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
        offsets: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor), candle::Error> {
        let query_rot = query.narrow(-1, 0, self.rotary_dim)?;
        let key_rot = key.narrow(-1, 0, self.rotary_dim)?;
        let query_pass = if self.rotary_dim < self.head_size {
            Some(query.narrow(-1, self.rotary_dim, self.head_size - self.rotary_dim)?)
        } else {
            None
        };
        let key_pass = if self.rotary_dim < self.head_size {
            Some(key.narrow(-1, self.rotary_dim, self.head_size - self.rotary_dim)?)
        } else {
            None
        };

        let cos_sin = if let Some(offsets) = offsets {
            self.cos_sin_cache
                .narrow(0, positions + offsets, positions.shape()[0])?
        } else {
            self.cos_sin_cache
                .narrow(0, positions, positions.shape()[0])?
        };
        let (cos, sin) = cos_sin.chunk(2, -1)?;

        let rotate_fn = if self.is_neox_style {
            rotate_neox
        } else {
            rotate_gptj
        };
        let query_rot = query_rot * &cos + rotate_fn(&query_rot) * &sin;
        let key_rot = key_rot * &cos + rotate_fn(&key_rot) * &sin;

        let query = if let Some(query_pass) = query_pass {
            Tensor::cat(&[&query_rot, &query_pass], -1)?
        } else {
            query_rot
        };
        let key = if let Some(key_pass) = key_pass {
            Tensor::cat(&[&key_rot, &key_pass], -1)?
        } else {
            key_rot
        };

        Ok((query, key))
    }
}

fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

fn yarn_find_correction_dim(
    num_rotations: usize,
    dim: usize,
    base: f64,
    max_position_embeddings: usize,
) -> f64 {
    (dim * (max_position_embeddings as f64 / (num_rotations as f64 * 2.0 * std::f64::consts::PI))
        .ln())
        / (2.0 * base.ln())
}

fn yarn_find_correction_range(
    low_rot: usize,
    high_rot: usize,
    dim: usize,
    base: f64,
    max_position_embeddings: usize,
) -> (usize, usize) {
    let low =
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor() as usize;
    let high =
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil() as usize;
    (low.max(0), high.min(dim - 1))
}

fn yarn_linear_ramp_mask(
    low: usize,
    high: usize,
    dim: usize,
    dtype: DType,
) -> Result<Tensor, candle_core::Error> {
    let linear_func = (Tensor::arange(0, dim, &Device::Cpu)?.to_dtype(dtype)? - low)?;
    let ramp_func = linear_func.clamp(0.0, 1.0)?;
    ramp_func / (high - low)
}

fn rotate_neox(x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let last_dim = x.dim(D::Minus1)?;
    let xs1 = x.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = x.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn rotate_gptj(x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let dims = x.dims();
    let last_dim = *dims.last().unwrap() as u32;

    // Create index tensors for even and odd indices
    let even_indices: Vec<u32> = (0..last_dim).step_by(2).collect();
    let odd_indices: Vec<u32> = (1..last_dim).step_by(2).collect();

    let even_tensor = Tensor::from_vec(even_indices, (last_dim as usize / 2,), &Device::Cpu)?;
    let odd_tensor = Tensor::from_vec(odd_indices, (last_dim as usize / 2,), &Device::Cpu)?;

    // Use index_select to get even and odd elements
    let x1 = x.index_select(&even_tensor, dims.len() - 1)?;
    let x2 = x.index_select(&odd_tensor, dims.len() - 1)?;

    // Negate x2
    let neg_x2 = x2.neg()?;

    // Stack -x2 and x1
    let stacked = Tensor::stack(&[neg_x2, x1], dims.len() - 1)?;

    // Flatten the last two dimensions
    let new_shape: Vec<usize> = dims[..dims.len() - 1]
        .iter()
        .chain(&[last_dim as usize])
        .copied()
        .collect();

    stacked.reshape(new_shape)
}
