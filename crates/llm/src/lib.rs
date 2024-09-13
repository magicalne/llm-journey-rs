pub mod config;
pub mod deepseek2;

#[cfg(test)]
mod tests {

    use anyhow::Result;
    use candle_core::Device;
    use candle_transformers::models::quantized_llama;
    use gguf_rs::{get_gguf_container, GGUFModel};
    use tokenizers::Tokenizer;

    use crate::deepseek2::{
        config::ModelConfig,
        model::{self, read_gguf_file, Model},
    };

    #[test]
    fn model_test() -> Result<()> {
        let path = "/run/media/magicalne/ssd1/data/gguf/DeepSeek-Coder-V2-Lite-Instruct.Q4_K.gguf";
        let device = Device::Cpu;
        let vb = read_gguf_file(path, &device)?;
        let mut container = get_gguf_container(path)?;
        let model = container.decode()?;
        // dbg!(model.metadata());
        let api = hf_hub::api::sync::Api::new()?;
        let repo = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct";
        let api = api.model(repo.to_string());
        let tokenizer_path = api.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        let tokens = tokenizer
            .encode("are you ok?", true)
            .map_err(anyhow::Error::msg)?;

        let config = ModelConfig::deepseek_coder_v2_lite_instruct();
        let _model = Model::from_gguf(&config, &device, vb)?;
        Ok(())
    }
}
