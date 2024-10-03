use std::io::Write;
use std::time::Instant;
use std::{mem, sync::Arc};

use anyhow::Context;
use candle_core::{quantized::QTensor, Device, Tensor};
use candle_core::{CpuStorage, Storage};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::{quantized_nn::Linear, quantized_var_builder::VarBuilder};
use log::info;
use rayon::prelude::*;
use tokenizers::Tokenizer;

use llm::deepseek2::{
    config::ModelConfig,
    model::{print_size, Model},
};

const MAX_SEQ_LEN: usize = 8192;

// RUST_LOG=info cargo test --release --test deepseek_lite_test -- --nocapture
#[test]
fn test_lite() -> anyhow::Result<()> {
    env_logger::init();
    let path = "/run/media/magicalne/ssd1/data/gguf/DeepSeek-Coder-V2-Lite-Instruct.Q4_K.gguf";
    let device = Device::Cpu;

    //let mut file = std::fs::File::open(path)?;
    //let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    //content
    //    .tensor_infos
    //    .iter()
    //    .for_each(|(key, value)| println!("{}: {:?}", key, value));

    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()?;
    let ts = Instant::now();
    let mut vb = VarBuilder::from_gguf(path, &device)?;
    //info!("{:?}", vb.tensor_infos());
    info!("Loading gguf file cost: {}s", ts.elapsed().as_secs());
    //let mut container = get_gguf_container(path)?;
    //let model = container.decode()?;
    let api = hf_hub::api::sync::Api::new()?;
    let repo = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct";
    let api = api.model(repo.to_string());
    let tokenizer_path = api.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    let tokens = tokenizer
        .encode("are you ok?", true)
        .map_err(anyhow::Error::msg)?;
    for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
        let token = token.replace('▁', " ").replace("<0x0A>", "\n");
        println!("{id:7} -> '{token}'");
    }

    let config = ModelConfig::deepseek_coder_v2_lite_instruct();
    let ts = Instant::now();
    let mut model = Model::from_gguf(&config, &device, vb)?;
    info!("Loading model weights costs: {}s", ts.elapsed().as_secs());

    let mut pre_prompt_tokens = vec![];
    let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
    let to_sample = 1000 - 1;
    let prompt_tokens = if prompt_tokens.len() + to_sample > MAX_SEQ_LEN - 10 {
        let to_remove = prompt_tokens.len() + to_sample + 10 - MAX_SEQ_LEN;
        prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
    } else {
        prompt_tokens
    };
    let mut all_tokens = vec![];
    let top_k = None;
    let top_p = None;
    let seed = 299792458;
    let temperature = 0.2;
    let split_prompt = false;
    let mut logits_processor = {
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (top_k, top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(seed, sampling)
    };

    let start_prompt_processing = std::time::Instant::now();
    let mut next_token = if !split_prompt {
        let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;

        let logits = model.forward(&input, None, None, false, false, None)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    } else {
        let mut next_token = 0;
        for (pos, token) in prompt_tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, None, None, false, false, None)?;
            let logits = logits.squeeze(0)?;
            next_token = logits_processor.sample(&logits)?
        }
        next_token
    };
    let prompt_dt = start_prompt_processing.elapsed();
    all_tokens.push(next_token);
    let t = tokenizer
        .decode(&all_tokens, true)
        .map_err(anyhow::Error::msg)?;
    print!("{t}");
    std::io::stdout().flush()?;

    //let eos_token = match args.which {
    //    Which::L8b => "<|end_of_text|>",
    //    _ => match args.which.is_open_chat() {
    //        true => "<|end_of_turn|>",
    //        false => "</s>",
    //    },
    //};

    //let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
    let start_post_prompt = std::time::Instant::now();
    let mut sampled = 0;
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        //let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = model.forward(&input, None, None, false, false, None)?;
        let logits = logits.squeeze(0)?;
        //let logits = if args.repeat_penalty == 1. {
        //    logits
        //} else {
        //    let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
        //    candle_transformers::utils::apply_repeat_penalty(
        //        &logits,
        //        args.repeat_penalty,
        //        &all_tokens[start_at..],
        //    )?
        //};
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        let t = tokenizer
            .decode(&all_tokens, false)
            .map_err(anyhow::Error::msg)?;
        println!("{t}");
        //if let Some(t) = tos.next_token(next_token)? {
        //    print!("{t}");
        //    std::io::stdout().flush()?;
        //}
        sampled += 1;
    }
    std::io::stdout().flush()?;
    let dt = start_post_prompt.elapsed();
    println!(
        "\n\n{:4} prompt tokens processed: {:.2} token/s",
        prompt_tokens.len(),
        prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
    );
    println!(
        "{sampled:4} tokens generated: {:.2} token/s",
        sampled as f64 / dt.as_secs_f64(),
    );

    Ok(())
}
