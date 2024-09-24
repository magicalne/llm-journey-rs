use candle_core::{IndexOp, Result, Tensor, D};

pub fn repeat_interleave(tensor: &Tensor, repeats: usize, dim: D) -> Result<Tensor> {
    let shape = tensor.shape();
    let mut new_shape = shape.clone();
    new_shape[dim as usize] *= repeats;

    let mut new_tensor = Tensor::zeros(new_shape, tensor.dtype(), tensor.device())?;

    for i in 0..shape[dim as usize] {
        let indices = (0..repeats).map(|j| i * repeats + j).collect::<Vec<_>>();
        let slice = new_tensor.i((.., indices.as_slice()))?;
        slice.copy_(&tensor.i((.., i..i + 1))?)?;
    }

    Ok(new_tensor)
}
