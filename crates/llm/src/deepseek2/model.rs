use std::{ops::Mul, sync::Arc};

use crate::deepseek2::config::ModelConfig;
use anyhow::{anyhow, Result};
use candle_core::{self, quantized::gguf_file, DType, Device, Module, Tensor, D};
use candle_nn::{rotary_emb, Activation, Embedding, LayerNorm};
use candle_transformers::{
    models::quantized_llama,
    quantized_nn::{linear_no_bias, Linear, RmsNorm},
    quantized_var_builder::VarBuilder,
    utils::repeat_kv,
};

#[derive(Debug, Clone)]
struct DeepSeekV2MLP {
    config: ModelConfig,
    hidden_size: usize,
    intermediate_size: usize,
    gate_proj: Linear,
    down_proj: Linear,
    up_proj: Linear,
    act_fn: Activation,
}

impl DeepSeekV2MLP {
    fn new(cfg: &ModelConfig, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        Ok(Self {
            config: cfg.clone(),
            act_fn: cfg.hidden_act,
            hidden_size,
            intermediate_size,
            gate_proj,
            down_proj,
            up_proj,
        })
    }
}

impl Module for DeepSeekV2MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        //gate_up, _ = self.gate_up_proj(x)
        //x = self.act_fn(gate_up)
        //x, _ = self.down_proj(x)

        xs.apply(&self.gate_proj)?
            .apply(&self.down_proj)
            .mul(xs.apply(&self.up_proj)?)
    }
}

#[derive(Debug, Clone)]
struct DeepseekV2Moe {
    config: ModelConfig,
    num_experts_per_tok: usize,
    gate: Linear,
    experts: Vec<DeepSeekV2MLP>,
    shared_experts: DeepSeekV2MLP,
    num_experts_per_tok: usize,
}

impl DeepseekV2Moe {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let ep_size = 1;
        let expert_per_rank = cfg.n_routed_experts;
        let ep_rank = 0;
        let gate = linear_no_bias(cfg.hidden_size, cfg.n_routed_experts, vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(cfg.n_routed_experts);
        let vb = vb.pp("experts");
        for idx in 0..cfg.n_routed_experts {
            let expert = DeepSeekV2MLP::new(cfg, cfg.moe_intermediate_size, vb.pp(idx))?;
            experts.push(expert)
        }
        let vb = vb.pp("shared_experts");
        let intermediate_size = cfg.moe_intermediate_size * cfg.n_shared_experts;
        let shared_experts = DeepSeekV2MLP::new(cfg, intermediate_size, vb);
        Ok(DeepseekV2Moe {
            config: cfg.clone(),
            num_experts_per_tok: cfg.num_experts_per_tok,
            gate,
            experts,
            shared_experts,
        })
    }
}

impl Module for DeepseekV2Moe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let router_logits = xs.apply(&self.gate)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // In order to extract topk, we extract the data from the tensor and manipulate it
        // directly. Maybe we will want to use some custom ops instead at some point.
        let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        // top_x contains the row indexes to evaluate for each expert.
        let mut top_x = vec![vec![]; self.experts.len()];
        let mut selected_rws = vec![vec![]; self.experts.len()];
        for (row_idx, rw) in routing_weights.iter().enumerate() {
            let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
            dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
            let mut sum_routing_weights = 0f32;
            for &expert_idx in dst.iter().take(self.num_experts_per_tok) {
                let expert_idx = expert_idx as usize;
                let routing_weight = rw[expert_idx];
                sum_routing_weights += routing_weight;
                top_x[expert_idx].push(row_idx as u32);
            }
            for &expert_idx in dst.iter().take(self.num_experts_per_tok) {
                let expert_idx = expert_idx as usize;
                let routing_weight = rw[expert_idx];
                selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
            }
        }

        // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert_layer) in self.experts.iter().enumerate() {
            let top_x = &top_x[expert_idx];
            if top_x.is_empty() {
                continue;
            }
            let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
            let selected_rws =
                Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?.reshape(((), 1))?;
            // Index the correct hidden states and compute the expert hidden state for
            // the current expert. We need to make sure to multiply the output hidden
            // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
            // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
            let current_hidden_states = expert_layer.forward(&current_state)?;
            let current_hidden_states = current_hidden_states.broadcast_mul(&selected_rws)?;
            ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
        }

        let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
        Ok(ys)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    use_flash_attn: bool,
}

impl Attention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = repeat_kv(key_states, self.num_kv_groups)?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?;

        let attn_output = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = query_states.transpose(1, 2)?;
            let k = key_states.transpose(1, 2)?;
            let v = value_states.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, q_len > 1)?.transpose(1, 2)?
        } else {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}
#[derive(Debug, Clone)]
struct DeepseekV2DecoderLayer {
    hidden_size: usize,
    self_attn: Attention, // TODO
    mlp: dyn Module,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DeepseekV2DecoderLayer {
    fn new(cfg: &Config, layer_id: usize, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let mlp_vb = vb.pp("mlp");
        let mlp = if cfg.n_routed_experts > 0
            && layer_id > cfg.first_k_dense_replace
            && layer_id % cfg.moe_layer_freq == 0
        {
            DeepseekV2Moe::new(cfg, vb)
        } else {
            DeepSeekV2MLP::new(cfg, cfg.intermediate_size, vb)
        };
        Ok(Self {
            hidden_size: cfg.hidden_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

pub struct Model {
    token_embeddings: Embedding,
    norm: RmsNorm,
    layers: Vec<DeepseekV2DecoderLayer>,
}

struct MultiHeadAttention {
    // Implement multi-head attention structure
}

impl Model {
    pub fn from_gguf(config: &ModelConfig, device: &Device, vb: VarBuilder) -> Result<Self> {
        let weights = vb.get_no_shape("token_embd.weight")?;
        let weights = weights.dequantize(device)?;
        let token_embeddings = Embedding::new(weights, config.hidden_size);
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        //
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = DeepseekV2DecoderLayer::new(cfg, layer_idx, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        //

        Ok(Self {
            token_embeddings,
            norm,
            layers,
        })
    }
}

#[derive(Debug, Clone)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
    //span: tracing::Span,
}

impl T5LayerNorm {
    fn load(h: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(h, "weight")?.dequantize(vb.device())?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
            //span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }
}

impl Module for T5LayerNorm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        //let _enter = self.span.enter();
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        // variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &ModelConfig, dev: &Device) -> candle_core::Result<Self> {
        let dim = cfg.qk_rope_head_dim;
        let rope_theta = cfg.rope_theta as f32; // base
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

pub fn read_gguf_file<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<VarBuilder> {
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(path, device)?;
    Ok(vb)
}
