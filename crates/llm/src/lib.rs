pub mod config;
pub mod model;

#[cfg(test)]
mod tests {
    use std::any::Any;

    use crate::model;
    use snafu::{prelude::*, ResultExt, Whatever};

    #[test]
    fn model_test() -> Result<(), Whatever> {
        let path = "/run/media/magicalne/ssd1/data/gguf/DeepSeek-Coder-V2-Lite-Instruct.Q4_K.gguf";
        //let model = model::GgufFile::from_path(path)
        //    .with_whatever_context(|err| format!("Read model failed: {:?}", err))?;

        let mut container = gguf_rs::get_gguf_container(path)
            .with_whatever_context(|err| format!("Read model failed: {:?}", err))?;
        let model = container.decode().expect("decode");
        println!("version: {:?}", &model.tensors()[0..19]);
        Ok(())
    }
}
