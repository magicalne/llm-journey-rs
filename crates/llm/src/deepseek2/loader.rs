use candle_core::{quantized::gguf_file, Device};
use candle_transformers::models::quantized_phi::ModelWeights as Qwen2;

fn load_quantized_model(model_path: &str) -> Result<Qwen2, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(model_path)?;
    let start = std::time::Instant::now();

    let mut model = {
        let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &total_size_in_bytes,
            start.elapsed().as_secs_f32(),
        );
        // FIXME: hardcoded cpu device
        Qwen2::from_gguf(model, &mut file, &Device::Cpu)?
    };

    Ok(model)
}
