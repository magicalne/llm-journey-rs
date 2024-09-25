use candle_core::{DType, Device, Result, Tensor, WithDType, D};

pub fn topk(tensor: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    // Sorted descending
    let sorted_indices = tensor.arg_sort_last_dim(false)?;
    let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
    let values = tensor.gather(&topk_indices, D::Minus1)?;
    Ok((values, topk_indices))
}

pub fn scatter(tensor: &Tensor, dim: usize, index: &Tensor, value: f64) -> Result<Tensor> {
    let mut result = tensor.clone();

    // Create a mask tensor with the same shape as the input tensor
    let mask = tensor.zeros_like()?;

    // Set the values in the mask tensor to 1 where index specifies
    mask.index_add(
        index,
        &Tensor::ones(index.shape(), index.dtype(), index.device())?,
        dim,
    )?;

    // Multiply the mask by the value to get the scatter values
    let scatter_values = (mask * value)?;

    // Add the scatter values to the result tensor
    result = result.add(&scatter_values)?;

    Ok(result)
}

// https://github.com/mokeyish/candle-ext/blob/main/src/masked_fill.rs
/// xs are on false (0), value is on true (1)
pub fn masked_fill<D: WithDType>(xs: &Tensor, mask: &Tensor, value: D) -> Result<Tensor> {
    let on_true = Tensor::full(value, xs.shape(), xs.device())?.to_dtype(xs.dtype())?;
    let on_false = xs;
    let res = mask
        .broadcast_as(xs.shape())?
        .where_cond(&on_true, on_false)?;
    Ok(res)
}
