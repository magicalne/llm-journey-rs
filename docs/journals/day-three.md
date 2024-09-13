---
title: "Day Three"
date: 2024-09-13
---

# Day Three

Today I decide to write more code. The target of today is making `DeepSeek-V2-Lite (MoE-16B)` start to generate.

## Steps

### Config file

Copy config from https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/blob/main/config.json

### Load model file

There are two methods to load GGUF files:
- ggml_file::Content
- quantized_var_builder::VarBuilder

The `ggml_file::Content` doesn't read the whole file into memory. It just stores the key information with `offset`. So that people can access specific part in the model file with the offset.
The `VarBuilder` uses `ggml_file::Content`, so I should stick with `VarBuilder`.

### Tokenizer and embeddings

**Tokenizer** is tricky. The GGUF file includes the tokenizer json content, I guess? But there is no easy way to get it.
I found a solution is reading `tokenizer.json` form hf-hub with api. The api will store the tokenizer.json in /tmp. So I can load tokenizer with `tokenizers`:

```rust
let repo = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct";
let api = api.model(repo.to_string());
let tokenizer_path = api.get("tokenizer.json")?;
let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
```

To get token embeddings:
```rust
let weights = vb.get_no_shape("token_embd.weight")?;
let weights = weights.dequantize(device)?;
let token_embeddings = Embedding::new(weights, config.hidden_size);
```

When reading the code, I can see a lot of `norm` and I have no idea what it is. Must be something about normalization. But normalization of what?
So I need to read the paper: [Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization](https://arxiv.org/pdf/1607.06450)
Turns out Layer normalization is alternative of batch normalization. 

> on, all the hidden units in a layer share the same normalization terms µ and σ, but different training cases have different normalization terms.

And what is batch normalization doesn't matter now.
