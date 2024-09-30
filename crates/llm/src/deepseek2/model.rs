use std::{any::Any, collections::HashMap, ops::Mul, sync::Arc};

use crate::{
    deepseek2::{config::ModelConfig, rotary_embedding::apply_rotary_pos_emb},
    utils::{masked_fill, scatter, topk},
};
use candle_core::{
    self,
    quantized::{gguf_file, QMatMul, QTensor},
    Context, DType, Device, IndexOp, Module, Result, Tensor, D,
};
use candle_nn::{
    ops::{softmax_last_dim, Identity},
    rotary_emb, Activation, Dropout, Embedding, LayerNorm,
};
use candle_transformers::{
    models::quantized_llama,
    quantized_nn::{linear_b, linear_no_bias, Linear, RmsNorm},
    quantized_var_builder::VarBuilder,
    utils::repeat_kv,
};

use super::{
    config::TopkMethod,
    rotary_embedding::{yarn_get_mscale, LlamaYaRNScaledRotaryEmbedding},
};

type Cache = (Tensor, Tensor);

#[derive(Debug, Clone)]
struct DeepSeekV2MLP {
    gate_proj: Linear,
    down_proj: Linear,
    up_proj: Linear,
    act_fn: Activation,
}

impl DeepSeekV2MLP {
    fn new(
        down_proj: Linear,
        gate_proj: Linear,
        up_proj: Linear,
        act_fn: Activation,
    ) -> Result<Self> {
        //let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        //let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        //let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            act_fn,
            gate_proj,
            down_proj,
            up_proj,
        })
    }
}

impl Module for DeepSeekV2MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        //act = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        //down_proj = self.down_proj(act)
        //return down_proj
        let act = self
            .act_fn
            .forward(&(self.gate_proj.forward(xs)? * self.up_proj.forward(xs)?)?)?;
        self.down_proj.forward(&act)
    }
}

struct MoEGateConfig {
    num_experts_per_tok: usize,
    n_routed_experts: usize,
    routed_scaling_factor: f32,
    scoring_func: String,
    aux_loss_alpha: f32,
    seq_aux: bool,
    topk_method: String,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    hidden_size: usize,
}

#[derive(Debug, Clone)]
struct MoEGate {
    weight: Linear,
}

impl MoEGate {
    fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let gating_dim = config.hidden_size;
        let weight = linear_no_bias(gating_dim, config.n_routed_experts, vb.pp("ffn_gate_inp"))?;
        let gate = MoEGate { weight };
        Ok(gate)
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let logits = self.weight.forward(hidden_states)?;
        let scores = softmax_last_dim(&logits)?;
        Ok(scores)
    }
}

#[derive(Debug, Clone)]
struct DeepseekV2Moe {
    config: ModelConfig,
    num_experts_per_tok: usize,
    gate: MoEGate,
    experts: Vec<DeepSeekV2MLP>,
    shared_experts: DeepSeekV2MLP,
}

impl DeepseekV2Moe {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        // let ep_size = 1;
        // let expert_per_rank = cfg.n_routed_experts;
        // let ep_rank = 0;
        let gate = MoEGate::new(cfg, vb.clone()).context("MoeGate")?; // gate

        let mut experts = Vec::with_capacity(cfg.n_routed_experts);
        let s = (
            cfg.n_routed_experts,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
        );
        let device = &Device::Cpu;
        let ffn_down_exps = vb.get(s, "ffn_down_exps.weight")?;
        let ffn_down_exps_dtype = ffn_down_exps.dtype();
        let ffn_down_exps = ffn_down_exps
            .dequantize(device)?
            .chunk(cfg.n_routed_experts, 0)?;
        let s = (
            cfg.n_routed_experts,
            cfg.moe_intermediate_size,
            cfg.hidden_size,
        );
        let ffn_gate_exps = vb.get(s, "ffn_gate_exps.weight")?;
        let ffn_gate_exps_dtype = ffn_gate_exps.dtype();
        let ffn_gate_exps = ffn_gate_exps
            .dequantize(device)?
            .chunk(cfg.n_routed_experts, 0)?;
        let ffn_up_exps = vb.get(s, "ffn_up_exps.weight")?;
        let ffn_up_exps_dtype = ffn_up_exps.dtype();
        let ffn_up_exps = ffn_up_exps
            .dequantize(device)?
            .chunk(cfg.n_routed_experts, 0)?;
        dbg!(1);

        let hidden_act = cfg.hidden_act;
        for idx in 0..cfg.n_routed_experts {
            dbg!(idx);
            let down =
                QTensor::quantize(&ffn_down_exps[idx], ffn_down_exps_dtype).context("down")?;
            let gate =
                QTensor::quantize(&ffn_gate_exps[idx], ffn_gate_exps_dtype).context("gate")?;
            let up = QTensor::quantize(&ffn_up_exps[idx], ffn_up_exps_dtype).context("up")?;
            let gate_proj = Linear::from_arc(Arc::new(gate), None).context("gate_proj")?;
            let down_proj = Linear::from_arc(Arc::new(down), None).context("down_proj")?;
            let up_proj = Linear::from_arc(Arc::new(up), None).context("up_proj")?;
            let expert = DeepSeekV2MLP::new(down_proj, gate_proj, up_proj, hidden_act)?;
            experts.push(expert)
        }

        dbg!(2);
        let vb = vb.pp("shared_experts");
        let intermediate_size = cfg.moe_intermediate_size * cfg.n_shared_experts;
        let ffn_down_shexp = vb.get(
            (intermediate_size, cfg.hidden_size),
            "ffn_down_shexp.weight",
        )?;
        let ffn_down_shexp_dtype = ffn_down_shexp.dtype();
        let ffn_down_shexp = ffn_down_shexp.dequantize(device)?;
        let s = (cfg.hidden_size, intermediate_size);
        let ffn_gate_shexp = vb.get(s, "ffn_gate_shexp.weight")?;
        let ffn_gate_shexp_dtype = ffn_gate_shexp.dtype();
        let ffn_gate_shexp = ffn_gate_shexp.dequantize(device)?;
        let ffn_up_shexp = vb.get(s, "ffn_up_shexp.weight")?;
        let ffn_up_shexp_dtype = ffn_up_shexp.dtype();
        let ffn_up_shexp = ffn_up_shexp.dequantize(device)?;
        let ffn_down_shexp = QTensor::quantize(&ffn_down_shexp, ffn_down_shexp_dtype)?;
        let ffn_gate_shexp = QTensor::quantize(&ffn_gate_shexp, ffn_gate_shexp_dtype)?;
        let ffn_up_shexp = QTensor::quantize(&ffn_up_shexp, ffn_up_shexp_dtype)?;
        let gate_proj = Linear::from_arc(Arc::new(ffn_gate_shexp), None)?;
        let down_proj = Linear::from_arc(Arc::new(ffn_down_shexp), None)?;
        let up_proj = Linear::from_arc(Arc::new(ffn_up_shexp), None)?;
        dbg!(3);

        let shared_experts = DeepSeekV2MLP::new(down_proj, gate_proj, up_proj, hidden_act)?;
        Ok(DeepseekV2Moe {
            config: cfg.clone(),
            num_experts_per_tok: cfg.num_experts_per_tok,
            gate,
            experts,
            shared_experts,
        })
    }

    fn moe_infer(
        &self,
        hidden_states: &Tensor,
        topk_ids: &Tensor,
        topk_weight: &Tensor,
    ) -> Result<Tensor> {
        let device = hidden_states.device();

        let cnts = Tensor::zeros((topk_ids.dim(0)?, self.experts.len()), DType::U32, device)?;
        let cnts = scatter(&cnts, 1, topk_ids, 1.)?;

        let tokens_per_expert = cnts.sum_keepdim(0)?;
        let idxs = topk_ids.arg_sort_last_dim(false)?;
        let idxs = (idxs.to_dtype(DType::F64)? / (topk_ids.dims2()?.1 as f64))?;
        let idxs = idxs.to_dtype(DType::U32)?;

        let sorted_tokens = hidden_states.i(&idxs)?;
        let tokens_per_expert = tokens_per_expert.to_vec1::<u32>()?; // TODO: is this dtype right?

        let mut outputs = Vec::new();
        let mut start_idx = 0;

        for (i, &num_tokens) in tokens_per_expert.iter().enumerate() {
            let end_idx = start_idx + num_tokens as usize;
            if num_tokens == 0 {
                continue;
            }
            let expert = &self.experts[i];
            let tokens_for_this_expert = sorted_tokens.i(start_idx..end_idx)?;
            let expert_out = expert.forward(&tokens_for_this_expert)?;
            outputs.push(expert_out);
            start_idx = end_idx;
        }

        let outs = if !outputs.is_empty() {
            Tensor::cat(&outputs, 0)?
        } else {
            sorted_tokens.zeros_like()?
            //unsafe { sorted_tokens.empty_like()? }
        };

        let new_x = outs.zeros_like()?;
        //let new_x = unsafe { Tensor::empty_like(&outs)? };

        new_x.scatter_add(&idxs, &outs, 0)?;

        let final_out = new_x
            .reshape((topk_ids.dim(0)?, topk_ids.dim(1)?, ()))?
            .to_dtype(topk_weight.dtype())?
            .mul(topk_weight.unsqueeze(D::Minus1)?)?
            .sum(1)?
            .to_dtype(new_x.dtype())?;

        Ok(final_out)
    }
}

impl Module for DeepseekV2Moe {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let identity = hidden_states;
        let (b_size, seq_len, hidden_dim) = hidden_states.dims3()?;
        let hidden_states = hidden_states.reshape(((), hidden_dim))?;
        let scores = self.gate.forward(&hidden_states)?;

        // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        // top_x contains the row indexes to evaluate for each expert.
        let (topk_weights, topk_idx) = match self.config.topk_method {
            TopkMethod::Greedy => topk(&scores, self.config.num_experts_per_tok)?,
            TopkMethod::GroupLimitedGreedy => {
                let group_scores = scores
                    .reshape((b_size * seq_len, self.config.n_group, ()))?
                    .max(D::Minus1)?; // [n, n_group]
                let (group_idx, _) = topk(&group_scores, self.config.topk_group)?; // [n, top_k_group]
                let group_mask = &group_scores.zeros_like()?; // [n, n_group]

                let group_mask = scatter(group_mask, 1, &group_idx, 1.)?;
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .expand((
                        b_size * seq_len,
                        self.config.n_group,
                        self.config.n_routed_experts / self.config.n_group,
                    ))?
                    .reshape((b_size * seq_len, ()))?; // [n, e]

                let tmp_scores = masked_fill(&scores, &score_mask, 0.)?;
                topk(&tmp_scores, self.config.num_experts_per_tok)?
            }
        };
        let topk_weight = if self.config.num_experts_per_tok > 1 && self.config.norm_topk_prob {
            let denominator = (topk_weights.sum_keepdim(D::Minus1)? + 1e-20)?;
            (topk_weights / denominator)?
        } else {
            (topk_weights * self.config.routed_scaling_factor)?
        };
        let y = self.moe_infer(&hidden_states, &topk_idx, &topk_weight)?;
        let y = (y + self.shared_experts.forward(identity)?)?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
enum AttentionType {
    Normal {
        q_a_norm: RmsNorm,
        q_a_proj: Linear,
        q_b_proj: Linear,
    },
    Lite {
        q_proj: Linear,
    },
}

impl AttentionType {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        Ok(match cfg.q_lora_rank {
            Some(q_lora_rank) => {
                let q_a_norm = RmsNorm::new(q_lora_rank, cfg.rms_norm_eps, vb.pp("q_a_layernorm"))?;
                let q_a_proj = linear_no_bias(cfg.hidden_size, q_lora_rank, vb.pp("q_a_proj"))?;

                let q_b_proj = linear_no_bias(
                    q_lora_rank,
                    cfg.num_attention_heads * qk_head_dim,
                    vb.pp("q_a_proj"),
                )?;
                Self::Normal {
                    q_a_norm,
                    q_a_proj,
                    q_b_proj,
                }
            }
            None => {
                let q_proj = linear_no_bias(
                    cfg.hidden_size,
                    cfg.num_attention_heads * qk_head_dim,
                    vb.pp("attn_q"), // q_proj
                )?;
                Self::Lite { q_proj }
            }
        })
    }

    fn get_q_proj(&self) -> Result<&Linear> {
        match self {
            AttentionType::Normal {
                q_a_norm: _,
                q_a_proj: _,
                q_b_proj: _,
            } => Err(candle_core::Error::Msg(
                "q_proj is in LITE model only.".to_string(),
            )),
            AttentionType::Lite { q_proj } => Ok(q_proj),
        }
    }

    fn get_q_a_layernorm(&self) -> Result<&RmsNorm> {
        match self {
            AttentionType::Normal {
                q_a_norm,
                q_a_proj: _,
                q_b_proj: _,
            } => Ok(q_a_norm),
            AttentionType::Lite { q_proj: _ } => Err(candle_core::Error::Msg(
                "invalid q_a_layernorm in LITE model".to_string(),
            )),
        }
    }
    fn get_q_a_proj(&self) -> Result<&Linear> {
        match self {
            AttentionType::Normal {
                q_a_norm: _,
                q_a_proj,
                q_b_proj: _,
            } => Ok(q_a_proj),
            AttentionType::Lite { q_proj: _ } => Err(candle_core::Error::Msg(
                "invalid q_a_proj in LITE model".to_string(),
            )),
        }
    }
    fn get_q_b_proj(&self) -> Result<&Linear> {
        match self {
            AttentionType::Normal {
                q_a_norm: _,
                q_a_proj: _,
                q_b_proj,
            } => Ok(q_b_proj),
            AttentionType::Lite { q_proj: _ } => Err(candle_core::Error::Msg(
                "invalid q_b_proj in LITE model".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
struct Attention {
    layer_id: usize,
    q_type: AttentionType,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    o_proj: Linear,
    rotary_emb: LlamaYaRNScaledRotaryEmbedding,
    config: ModelConfig,
    q_head_dim: usize,
    softmax_scale: f64,
}

impl Attention {
    fn new(cfg: &ModelConfig, layer_id: usize, vb: VarBuilder, _device: &Device) -> Result<Self> {
        let q_type = AttentionType::new(cfg, vb.clone())?;
        let q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        let kv_a_proj_with_mqa = linear_no_bias(
            cfg.hidden_size,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            vb.pp("attn_kv_a_mqa"), //kv_a_proj_with_mqa
        )?;
        let kv_a_layernorm =
            RmsNorm::new(cfg.kv_lora_rank, cfg.rms_norm_eps, vb.pp("attn_kv_a_norm"))?; //kv_a_layernorm
        let out_dim = cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);
        let kv_b_proj = linear_b(
            cfg.kv_lora_rank,
            out_dim,
            cfg.attention_bias,
            vb.pp("attn_kv_b"), //kv_b_proj
        )?;
        let o_proj = linear_b(
            cfg.num_attention_heads * cfg.v_head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("attn_output"), //o_proj
        )?;
        let rope_scaling = &cfg.rope_scaling;
        let rotary_emb = LlamaYaRNScaledRotaryEmbedding::new(
            cfg.qk_rope_head_dim,
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            rope_scaling.factor,
            rope_scaling.mscale,
            rope_scaling.mscale_all_dim,
            rope_scaling.original_max_position_embeddings,
            1.,
            rope_scaling.beta_fast,
            rope_scaling.beta_slow,
        )?;
        let mut softmax_scale = (q_head_dim as f64).powf(-0.5);
        let mscale_all_dim = cfg.rope_scaling.mscale_all_dim;
        let scaling_factor = cfg.rope_scaling.factor;
        let mscale = yarn_get_mscale(scaling_factor, mscale_all_dim);
        softmax_scale *= mscale * mscale;

        Ok(Self {
            config: cfg.clone(),
            layer_id,
            q_type,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            q_head_dim,
            softmax_scale,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        _past_key_value: Option<&Cache>,
        output_attentions: bool,
        _use_cache: bool,
        _cache_position: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>, Cache)> {
        let dims = hidden_states.dims();
        let bsz = dims[0];
        let q_len = dims[1];

        let q = if self.config.q_lora_rank.is_none() {
            self.q_type.get_q_proj()?.forward(hidden_states)?
        } else {
            self.q_type.get_q_b_proj()?.forward(
                &self
                    .q_type
                    .get_q_a_layernorm()?
                    .forward(&self.q_type.get_q_a_proj()?.forward(hidden_states)?)?,
            )?
        };

        let q = q
            .reshape((bsz, q_len, self.config.num_attention_heads, self.q_head_dim))?
            .transpose(1, 2)?;
        let q_nope = q.i((.., .., .., ..self.config.qk_nope_head_dim))?;
        let q_pe = q.i((.., .., .., self.config.qk_nope_head_dim..))?;
        assert_eq!(
            q_pe.dim(D::Minus1)?,
            self.config.qk_rope_head_dim,
            "split q"
        );

        let compressed_kv = self.kv_a_proj_with_mqa.forward(hidden_states)?;
        let compressed_kv = compressed_kv.i((.., .., ..self.config.kv_lora_rank))?;
        let k_pe = compressed_kv
            .i((.., .., self.config.kv_lora_rank..))?
            .reshape((bsz, q_len, 1, self.config.qk_rope_head_dim))?
            .transpose(1, 2)?;

        let kv = self
            .kv_b_proj
            .forward(&self.kv_a_layernorm.forward(&compressed_kv)?)?
            .reshape((
                bsz,
                q_len,
                self.config.num_attention_heads,
                self.config.qk_nope_head_dim + self.config.v_head_dim,
            ))?
            .transpose(1, 2)?;

        let k_nope = kv.i((.., .., .., ..self.config.qk_nope_head_dim))?;
        let value_states = kv.i((.., .., .., self.config.qk_nope_head_dim..))?;
        let _kv_seq_len = value_states.dim(2)?;

        /*
        if let Some(past_kv) = past_key_value {
            if self.layer_id.is_none() {
                return Err(candle_core::Error::Msg(
                    "Layer index is required for caching".into(),
                ));
            }
            kv_seq_len += past_kv.get_usable_length(kv_seq_len, self.layer_idx.unwrap())?;
        }
        */

        let (cos, sin) = self.rotary_emb.forward(&q_pe, position_ids)?;
        let (q_pe, k_pe) = apply_rotary_pos_emb(&q_pe, &k_pe, &cos, &sin, None)?;

        let query_states = Tensor::zeros(
            (bsz, self.config.num_attention_heads, q_len, self.q_head_dim),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;
        query_states.slice_assign(
            &[(..), (..), (..), (..self.config.qk_nope_head_dim)],
            &q_nope,
        )?;
        query_states.slice_assign(&[(..), (..), (..), (self.config.qk_nope_head_dim..)], &q_pe)?;

        let key_states = Tensor::zeros(
            (bsz, self.config.num_attention_heads, q_len, self.q_head_dim),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;
        key_states.slice_assign(
            &[(..), (..), (..), (..self.config.qk_nope_head_dim)],
            &k_nope,
        )?;
        key_states.slice_assign(&[(..), (..), (..), (self.config.qk_nope_head_dim..)], &k_pe)?;

        /*
         * TODO: kvcache
        if let Some(past_kv) = past_key_value {
            let cache_kwargs = CacheKwargs {
                sin: sin,
                cos: cos,
                cache_position: cache_position.cloned(),
            };
            let (new_key_states, new_value_states) =
                past_kv.update(&key_states, &value_states, self.layer_id, cache_kwargs)?;
            key_states = new_key_states;
            value_states = new_value_states;
        }
        */

        let attn_weights =
            (query_states.matmul(&key_states.transpose(2, 3)?)? * self.softmax_scale)?;

        let attn_weights = if let Some(mask) = attention_mask {
            (&attn_weights + mask)?
        } else {
            attn_weights
        };

        let attn_weights =
            candle_nn::ops::softmax(&attn_weights, D::Minus1)?.to_dtype(query_states.dtype())?;
        let attn_weights =
            Dropout::new(self.config.attention_dropout as f32).forward(&attn_weights, false)?;

        let attn_output = attn_weights.matmul(&value_states)?;

        if attn_output.dims()
            != [
                bsz,
                self.config.num_attention_heads,
                q_len,
                self.config.v_head_dim,
            ]
        {
            return Err(candle_core::Error::Msg(
                "Unexpected attention output size".into(),
            ));
        }

        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            bsz,
            q_len,
            self.config.num_attention_heads * self.config.v_head_dim,
        ))?;

        let attn_output = self.o_proj.forward(&attn_output)?;

        let attn_weights = if output_attentions {
            Some(attn_weights)
        } else {
            None
        };
        // temp kv cache
        let past_key_value = (key_states, value_states);

        Ok((attn_output, attn_weights, past_key_value))
    }
}
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> candle_core::Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[derive(Debug, Clone)]
enum MLP {
    Moe(DeepseekV2Moe),
    Mlp(DeepSeekV2MLP),
}

impl MLP {
    fn new_moe(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Moe(DeepseekV2Moe::new(cfg, vb)?))
    }

    fn new_mlp(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let intermediate_size = cfg.intermediate_size;

        let gate = linear_no_bias(cfg.hidden_size, intermediate_size, vb.pp("ffn_gate"))
            .context("ffn_gate")?;
        let up = linear_no_bias(cfg.hidden_size, intermediate_size, vb.pp("ffn_up"))
            .context("ffn_up")?;
        let down = linear_no_bias(intermediate_size, cfg.hidden_size, vb.pp("ffn_down"))
            .context("ffn_down")?;
        Ok(Self::Mlp(DeepSeekV2MLP::new(
            down,
            gate,
            up,
            cfg.hidden_act,
        )?))
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            MLP::Moe(moe) => moe.forward(xs),
            MLP::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct DeepseekV2DecoderLayer {
    hidden_size: usize,
    self_attn: Attention, // TODO
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DeepseekV2DecoderLayer {
    fn new(cfg: &ModelConfig, layer_id: usize, vb: VarBuilder, device: &Device) -> Result<Self> {
        // model.layer.index.self_attn
        let self_attn = Attention::new(cfg, layer_id, vb.clone(), device)?;
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("attn_norm"))?; //input_layernorm
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("ffn_norm"), //post_attention_layernorm
        )
        .context("post_attention_layernorm")?;

        let mlp_vb = vb.clone(); //mlp
        let mlp = if cfg.n_routed_experts > 0
            && layer_id >= cfg.first_k_dense_replace
            && layer_id % cfg.moe_layer_freq == 0
        {
            MLP::new_moe(cfg, mlp_vb).context("new_moe")?
        } else {
            MLP::new_mlp(cfg, mlp_vb).context("new_mlp")?
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
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        past_key_value: Option<&Cache>,
        output_attentions: bool,
        use_cache: bool,
        cache_position: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>, Option<Cache>)> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let (hidden_states, self_attn_weights, present_key_value) = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        )?;
        let hidden_states = (residual + hidden_states)?;
        // Fully Connected
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;

        let hidden_states = (residual + hidden_states)?;
        let cache = match use_cache {
            true => Some(present_key_value),
            false => None,
        };
        Ok((hidden_states, self_attn_weights, cache))
    }
}

pub struct Model {
    config: ModelConfig,
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DeepseekV2DecoderLayer>,
}

impl Model {
    pub fn from_gguf(config: &ModelConfig, device: &Device, vb: VarBuilder) -> Result<Self> {
        let weights = vb.get_no_shape("token_embd.weight")?;
        let _dtype = weights.dtype();
        let weights = weights.dequantize(device)?;
        let embed_tokens = Embedding::new(weights, config.hidden_size);
        let norm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("output_norm"),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_l = vb.pp("blk");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = DeepseekV2DecoderLayer::new(config, layer_idx, vb_l.pp(layer_idx), device)?;
            layers.push(layer)
        }

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            norm,
            layers,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        past_key_value: Option<&Cache>,
        use_cache: bool,
        output_attentions: bool,
        cache_position: Option<&Tensor>,
    ) -> Result<Tensor> {
        let inputs_embeds = self.embed_tokens.forward(input_ids)?;
        // TODO: attention_mask

        // embed positions
        let mut hidden_states = inputs_embeds;

        // decoder layers
        for decoder_layer in &self.layers {
            let layer_outputs = decoder_layer.forward(
                &hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
            )?;
            hidden_states = layer_outputs.0;
        }
        hidden_states = self.norm.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

pub struct CausalLM {
    model: Model,
    config: ModelConfig,
    lm_head: Linear,
}

impl CausalLM {
    pub fn new(config: &ModelConfig, device: &Device, vb: VarBuilder) -> Result<Self> {
        let model = Model::from_gguf(config, device, vb.clone())?;
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            model,
            lm_head,
            config: config.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        past_key_value: Option<&Cache>,
        use_cache: bool,
        output_attentions: bool,
        cache_position: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hidden_states = self.model.forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
            output_attentions,
            cache_position,
        )?;
        let logits = self.lm_head.forward(&hidden_states)?;
        let dim = logits.dim(1)?;
        let logits = logits
            .i((.., dim - 1, ..))?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?;
        Ok(logits)
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

pub fn read_gguf_file<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<VarBuilder> {
    VarBuilder::from_gguf(path, device)
}
