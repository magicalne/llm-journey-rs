use std::{collections::HashMap, ops::Mul, sync::Arc};

use crate::{
    deepseek2::{config::ModelConfig, rotary_embedding::apply_rotary_pos_emb},
    utils::{masked_fill, scatter, topk},
};
use anyhow::Context;
use candle_core::{
    self, backend::BackendDevice, quantized::QTensor, CudaDevice, DType, Device, Error, IndexOp,
    Module, Result, Tensor, D,
};
use candle_nn::{ops::softmax_last_dim, rotary_emb, Activation, Dropout, Embedding};
use candle_transformers::{
    quantized_nn::{linear_b, linear_no_bias, Linear, RmsNorm},
    quantized_var_builder::VarBuilder,
};
use log::{info, trace};
use rayon::prelude::*;

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
    gate: MoEGate,
    experts: Vec<DeepSeekV2MLP>,
    shared_experts: DeepSeekV2MLP,
}

impl DeepseekV2Moe {
    fn new(cfg: &ModelConfig, mut vb: VarBuilder) -> anyhow::Result<Self> {
        // let ep_size = 1;
        // let expert_per_rank = cfg.n_routed_experts;
        // let ep_rank = 0;
        let gate = MoEGate::new(cfg, vb.clone()).context("MoeGate")?; // gate
        trace!("gate loaded");

        let s = (
            cfg.n_routed_experts,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
        );
        let device = &Device::Cpu;
        //let device = &Device::Cuda(CudaDevice::new(0)?);

        let qffn_down_exps = vb.get(s, "ffn_down_exps.weight")?;
        let ffn_down_exps_dtype = qffn_down_exps.dtype();
        let ffn_down_exps = qffn_down_exps
            .dequantize(device)?
            .chunk(cfg.n_routed_experts, 0)?;
        let s = (
            cfg.n_routed_experts,
            cfg.moe_intermediate_size,
            cfg.hidden_size,
        );
        let qffn_gate_exps = vb.get(s, "ffn_gate_exps.weight")?;
        let ffn_gate_exps_dtype = qffn_gate_exps.dtype();
        let ffn_gate_exps = qffn_gate_exps
            .dequantize(device)?
            .chunk(cfg.n_routed_experts, 0)?;
        let qffn_up_exps = vb.get(s, "ffn_up_exps.weight")?;
        let ffn_up_exps_dtype = qffn_up_exps.dtype();
        let ffn_up_exps = qffn_up_exps
            .dequantize(device)?
            .chunk(cfg.n_routed_experts, 0)?;
        {
            drop(qffn_down_exps);
            drop(qffn_gate_exps);
            drop(qffn_up_exps);
        }

        let hidden_act = cfg.hidden_act;
        // speed up with rayon
        let experts = (0..cfg.n_routed_experts)
            //.into_par_iter()
            .map(|idx| {
                trace!("experts: {}", idx);
                let gate = QTensor::quantize(
                    &ffn_gate_exps[idx].force_contiguous()?.get(0)?,
                    ffn_gate_exps_dtype,
                )
                .context("gate")?;
                let down = QTensor::quantize(
                    &ffn_down_exps[idx].force_contiguous()?.get(0)?,
                    ffn_down_exps_dtype,
                )
                .context("down")?;
                let up = QTensor::quantize(
                    &ffn_up_exps[idx].force_contiguous()?.get(0)?,
                    ffn_up_exps_dtype,
                )
                .context("up")?;
                let gate_proj = Linear::from_arc(Arc::new(gate), None).context("gate_proj")?;
                let down_proj = Linear::from_arc(Arc::new(down), None).context("down_proj")?;
                let up_proj = Linear::from_arc(Arc::new(up), None).context("up_proj")?;
                let expert = DeepSeekV2MLP::new(down_proj, gate_proj, up_proj, hidden_act)?;
                Ok(expert)
            })
            .collect::<Vec<anyhow::Result<DeepSeekV2MLP>>>()
            .into_iter()
            .collect::<anyhow::Result<Vec<DeepSeekV2MLP>>>()?;

        {
            drop(ffn_down_exps);
            drop(ffn_gate_exps);
            drop(ffn_up_exps);
        }

        let intermediate_size = cfg.moe_intermediate_size * cfg.n_shared_experts;
        let qffn_down_shexp = vb.get(
            (cfg.hidden_size, intermediate_size),
            "ffn_down_shexp.weight",
        )?;
        let ffn_down_shexp_dtype = qffn_down_shexp.dtype();
        let ffn_down_shexp = qffn_down_shexp.dequantize(device)?;
        let s = (intermediate_size, cfg.hidden_size);
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

        let shared_experts = DeepSeekV2MLP::new(down_proj, gate_proj, up_proj, hidden_act)?;
        Ok(DeepseekV2Moe {
            config: cfg.clone(),
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
    q_type: AttentionType,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    o_proj: Linear,
    rotary_emb: Arc<LlamaYaRNScaledRotaryEmbedding>,
    config: ModelConfig,
    q_head_dim: usize,
    softmax_scale: f64,
}

impl Attention {
    fn new(
        cfg: &ModelConfig,
        q_head_dim: usize,
        rotary_emb: Arc<LlamaYaRNScaledRotaryEmbedding>,
        q_type: AttentionType,
        kv_a_layernorm: RmsNorm,
        kv_a_proj_with_mqa: Linear,
        kv_b_proj: Linear,
        o_proj: Linear,
    ) -> Result<Self> {
        let rope_scaling = &cfg.rope_scaling;
        let mut softmax_scale = (q_head_dim as f64).powf(-0.5);
        let mscale_all_dim = cfg.rope_scaling.mscale_all_dim;
        let scaling_factor = cfg.rope_scaling.factor;
        let mscale = yarn_get_mscale(scaling_factor, mscale_all_dim);
        softmax_scale *= mscale * mscale;

        Ok(Self {
            config: cfg.clone(),
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
        // [batch_size, seq_len, hidden_size]
        dbg!(hidden_states.shape());
        let (bsz, q_len, _hidde_size) = hidden_states.dims3()?;

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

        dbg!(q.shape());

        //q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        //q_nope, q_pe = torch.split(
        //    q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        //)
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

        //compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        //compressed_kv, k_pe = torch.split(
        //    compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        //)
        //k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        //
        // better option
        //let input_tensor = ...; // assume a tensor of shape (batch_size, seq_len, features)
        //let split_indices = vec![10, 20]; // define the split indices
        //let split_tensors: Vec<_> = input_tensor.split_indices.iter().map(|&i| input_tensor.slice(s![.., .., i..])).collect();
        let compressed_kv_ = self.kv_a_proj_with_mqa.forward(hidden_states)?;
        let compressed_kv = compressed_kv_
            .i((.., .., ..self.config.kv_lora_rank))?
            .contiguous()?;
        let k_pe = compressed_kv_
            .i((.., .., self.config.kv_lora_rank..))?
            .reshape((bsz, q_len, 1, self.config.qk_rope_head_dim))?
            .transpose(1, 2)?;
        dbg!(q_pe.shape(), k_pe.shape());
        //kv = (
        //    self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        //    .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        //    .transpose(1, 2)
        //)
        let norm = self.kv_a_layernorm.forward(&compressed_kv)?;
        let kv = self
            .kv_b_proj
            .forward(&norm)?
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
        dbg!(3);

        let query_states = Tensor::zeros(
            (bsz, self.config.num_attention_heads, q_len, self.q_head_dim),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;
        let (d1, d2, d3, _d4) = query_states.dims4()?;
        query_states.slice_assign(
            &[(..d1), (..d2), (..d3), ..self.config.qk_nope_head_dim],
            &q_nope,
        )?;
        query_states.slice_assign(
            &[(0..), (0..), (0..), (self.config.qk_nope_head_dim..)],
            &q_pe,
        )?;

        let key_states = Tensor::zeros(
            (bsz, self.config.num_attention_heads, q_len, self.q_head_dim),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;
        let (d1, d2, d3, _d4) = key_states.dims4()?;
        key_states.slice_assign(
            &[(..d1), (..d2), (..d3), (..self.config.qk_nope_head_dim)],
            &k_nope,
        )?;
        key_states.slice_assign(
            &[(0..), (0..), (0..), (self.config.qk_nope_head_dim..)],
            &k_pe,
        )?;

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
enum Mlp {
    Moe(DeepseekV2Moe),
    Mlp(DeepSeekV2MLP),
}

impl Mlp {
    fn new_moe(cfg: &ModelConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        Ok(Self::Moe(DeepseekV2Moe::new(cfg, vb)?))
    }

    fn new_mlp(cfg: &ModelConfig, vb: VarBuilder) -> anyhow::Result<Self> {
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

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Mlp::Moe(moe) => moe.forward(xs),
            Mlp::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct DeepseekV2DecoderLayer {
    self_attn: Attention, // TODO
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DeepseekV2DecoderLayer {
    //fn new(
    //    cfg: &ModelConfig,
    //    layer_id: usize,
    //    vb: VarBuilder,
    //    device: &Device,
    //) -> anyhow::Result<Self> {
    //    // model.layer.index.self_attn
    //    let self_attn = Attention::new(cfg, layer_id, vb.clone(), device)?;
    //    let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("attn_norm"))?; //input_layernorm
    //    let post_attention_layernorm = RmsNorm::new(
    //        cfg.hidden_size,
    //        cfg.rms_norm_eps,
    //        vb.pp("ffn_norm"), //post_attention_layernorm
    //    )
    //    .context("post_attention_layernorm")?;

    //    let mlp_vb = vb.clone(); //mlp
    //    let mlp = if cfg.n_routed_experts > 0
    //        && layer_id >= cfg.first_k_dense_replace
    //        && layer_id % cfg.moe_layer_freq == 0
    //    {
    //        Mlp::new_moe(cfg, mlp_vb).context("new_moe")?
    //    } else {
    //        Mlp::new_mlp(cfg, mlp_vb).context("new_mlp")?
    //    };
    //    Ok(Self {
    //        self_attn,
    //        mlp,
    //        input_layernorm,
    //        post_attention_layernorm,
    //    })
    //}

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        past_key_value: Option<&Cache>,
        output_attentions: bool,
        use_cache: bool,
        cache_position: Option<&Tensor>,
    ) -> anyhow::Result<(Tensor, Option<Tensor>, Option<Cache>)> {
        let residual = hidden_states;
        let hidden_states = self
            .input_layernorm
            .forward(hidden_states)
            .context("input_layernorm")?;
        dbg!(hidden_states.shape());
        let (hidden_states, self_attn_weights, present_key_value) = self
            .self_attn
            .forward(
                &hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
            )
            .context("self_attn")?;
        let hidden_states = (residual + hidden_states)?;
        // Fully Connected
        let residual = &hidden_states;
        let hidden_states = self
            .post_attention_layernorm
            .forward(&hidden_states)
            .context("post_attention_layernorm")?;
        let hidden_states = self.mlp.forward(&hidden_states).context("ffn")?;

        let hidden_states = (residual + hidden_states)?;
        let cache = match use_cache {
            true => Some(present_key_value),
            false => None,
        };
        Ok((hidden_states, self_attn_weights, cache))
    }
}

pub struct Model {
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DeepseekV2DecoderLayer>,
    masks: HashMap<usize, Tensor>,
}

impl Model {
    pub fn from_gguf(
        config: &ModelConfig,
        device: &Device,
        mut vb: VarBuilder,
    ) -> anyhow::Result<Self> {
        let qweights = vb.get_no_shape("token_embd.weight")?;
        let _dtype = qweights.dtype();
        let weights = qweights.dequantize(device)?;
        drop(qweights);
        info!("token_embd.weight loaded");
        let embed_tokens = Embedding::new(weights, config.hidden_size);
        let norm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("output_norm"),
        )?;
        // Parallelize loading weights
        let vb = vb.pp("blk");
        let post_attention_layernorms = (0..config.num_hidden_layers)
            .into_par_iter()
            .map(|layer_id| {
                let output_layernorm = RmsNorm::new(
                    config.hidden_size,
                    config.rms_norm_eps,
                    vb.pp(layer_id).pp("ffn_norm"),
                )?;
                Ok(output_layernorm)
            })
            .collect::<Vec<anyhow::Result<RmsNorm>>>()
            .into_iter()
            .collect::<anyhow::Result<Vec<RmsNorm>>>()?;
        info!("post_attention_layernorms loaded");
        let input_layernorms = (0..config.num_hidden_layers)
            .into_par_iter()
            .map(|layer_id| {
                let input_layernorm = RmsNorm::new(
                    config.hidden_size,
                    config.rms_norm_eps,
                    vb.pp(layer_id).pp("attn_norm"),
                )?;
                Ok(input_layernorm)
            })
            .collect::<Vec<anyhow::Result<RmsNorm>>>()
            .into_iter()
            .collect::<anyhow::Result<Vec<RmsNorm>>>()?;
        info!("input_layernorms loaded");

        let rope_scaling = &config.rope_scaling;
        let rotary_emb = Arc::new(LlamaYaRNScaledRotaryEmbedding::new(
            config.qk_rope_head_dim,
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            rope_scaling.factor,
            rope_scaling.mscale,
            rope_scaling.mscale_all_dim,
            rope_scaling.original_max_position_embeddings,
            1.,
            rope_scaling.beta_fast,
            rope_scaling.beta_slow,
        )?);
        let attns = load_attentions(config, rotary_emb, vb.clone())?;
        info!("attentions loaded");

        let mlps = (0..config.num_hidden_layers)
            .into_par_iter()
            .map(|layer_id| {
                let mlp_vb = vb.pp(layer_id); //mlp
                let mlp = if config.n_routed_experts > 0
                    && layer_id >= config.first_k_dense_replace
                    && layer_id % config.moe_layer_freq == 0
                {
                    Mlp::new_moe(config, mlp_vb).context("new_moe")?
                } else {
                    Mlp::new_mlp(config, mlp_vb).context("new_mlp")?
                };
                Ok(mlp)
            })
            .collect::<Vec<anyhow::Result<Mlp>>>()
            .into_iter()
            .collect::<anyhow::Result<Vec<Mlp>>>()?;
        info!("mlps loaded");

        let layers = attns
            .into_iter()
            .zip(
                input_layernorms
                    .into_iter()
                    .zip(post_attention_layernorms.into_iter().zip(mlps)),
            )
            .map(
                |(self_attn, (input_layernorm, (post_attention_layernorm, mlp)))| {
                    DeepseekV2DecoderLayer {
                        self_attn,
                        input_layernorm,
                        post_attention_layernorm,
                        mlp,
                    }
                },
            )
            .collect::<Vec<DeepseekV2DecoderLayer>>();

        Ok(Self {
            embed_tokens,
            norm,
            layers,
            masks: HashMap::default(),
        })
    }

    fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
    pub fn forward(
        &mut self,
        input_ids: &Tensor, // [batch_size, seq_len]
        // position_ids: &Tensor,
        position_index: u32,
        past_key_value: Option<&Cache>,
        use_cache: bool,
        output_attentions: bool,
        cache_position: Option<&Tensor>,
    ) -> anyhow::Result<Tensor> {
        info!("input_ids shape: {:?}", input_ids.shape());
        let (b_sz, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, input_ids.device())?)
        };
        // [batch_size, seq_len, hidden_size]
        let inputs_embeds = self.embed_tokens.forward(input_ids).context("embedding")?;
        let dev = inputs_embeds.device();
        let position_ids = Tensor::arange(position_index, position_index + seq_len as u32, dev)?;
        let position_ids = position_ids.unsqueeze(0)?.broadcast_as((b_sz, seq_len))?;

        // embed positions
        let mut hidden_states = inputs_embeds;

        // decoder layers
        for decoder_layer in &self.layers {
            let layer_outputs = decoder_layer
                .forward(
                    &hidden_states,
                    attention_mask.as_ref(),
                    &position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
                .context("decoder_layer")?;
            hidden_states = layer_outputs.0;
        }
        hidden_states = self.norm.forward(&hidden_states).context("norm")?;
        Ok(hidden_states)
    }
}

fn load_attentions(
    cfg: &ModelConfig,
    rotary_emb: Arc<LlamaYaRNScaledRotaryEmbedding>,
    vb: VarBuilder,
) -> anyhow::Result<Vec<Attention>> {
    //let q_type = AttentionType::new(cfg, vb.clone())?;
    let q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    //let kv_a_layernorm = RmsNorm::new(cfg.kv_lora_rank, cfg.rms_norm_eps, vb.pp("attn_kv_a_norm"))?; //kv_a_layernorm
    let out_dim = cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);
    //let kv_a_proj_with_mqa = linear_no_bias(
    //    cfg.hidden_size,
    //    cfg.kv_lora_rank + cfg.qk_rope_head_dim,
    //    vb.pp("attn_kv_a_mqa"), //kv_a_proj_with_mqa
    //)?;
    //    let kv_b_proj = linear_b(
    //        cfg.kv_lora_rank,
    //        out_dim,
    //        cfg.attention_bias,
    //        vb.pp("attn_kv_b"), //kv_b_proj
    //    )?;
    //let o_proj = linear_b(
    //    cfg.num_attention_heads * cfg.v_head_dim,
    //    cfg.hidden_size,
    //    cfg.attention_bias,
    //    vb.pp("attn_output"), //o_proj
    //)?;
    let q_types = (0..cfg.num_hidden_layers)
        .into_par_iter()
        .map(|layer_id| {
            trace!("q_type: {layer_id}");
            let vb = vb.pp(layer_id);
            let q_type = AttentionType::new(cfg, vb.clone())?;
            Ok(q_type)
        })
        .collect::<Vec<anyhow::Result<AttentionType>>>()
        .into_iter()
        .collect::<anyhow::Result<Vec<AttentionType>>>()?;
    let kv_a_layernorms = (0..cfg.num_hidden_layers)
        .into_par_iter()
        .map(|layer_id| {
            trace!("kv_a_layernorm: {layer_id}");
            let kv_a_layernorm = RmsNorm::new(
                cfg.kv_lora_rank,
                cfg.rms_norm_eps,
                vb.pp(layer_id).pp("attn_kv_a_norm"),
            )?; //kv_a_layernorm
            Ok(kv_a_layernorm)
        })
        .collect::<Vec<anyhow::Result<RmsNorm>>>()
        .into_iter()
        .collect::<anyhow::Result<Vec<RmsNorm>>>()?;
    let kv_a_proj_with_mqas = (0..cfg.num_hidden_layers)
        .into_par_iter()
        .map(|layer_id| {
            trace!("kv_a_proj_with_mqas: {layer_id}");
            let kv_a_proj_with_mqa = linear_no_bias(
                cfg.hidden_size,
                cfg.kv_lora_rank + cfg.qk_rope_head_dim,
                vb.pp(layer_id).pp("attn_kv_a_mqa"), //kv_a_proj_with_mqa
            )?;
            Ok(kv_a_proj_with_mqa)
        })
        .collect::<Vec<anyhow::Result<Linear>>>()
        .into_iter()
        .collect::<anyhow::Result<Vec<Linear>>>()?;

    let kv_b_projs = (0..cfg.num_hidden_layers)
        .into_par_iter()
        .map(|layer_id| {
            trace!("kv_b_projs: {layer_id}");
            let kv_b_proj = linear_b(
                cfg.kv_lora_rank,
                out_dim,
                cfg.attention_bias,
                vb.pp(layer_id).pp("attn_kv_b"), //kv_b_proj
            )?;
            Ok(kv_b_proj)
        })
        .collect::<Vec<anyhow::Result<Linear>>>()
        .into_iter()
        .collect::<anyhow::Result<Vec<Linear>>>()?;

    let o_projs = (0..cfg.num_hidden_layers)
        .into_par_iter()
        .map(|layer_id| {
            trace!("o_proj: {layer_id}");
            let o_proj = linear_b(
                cfg.num_attention_heads * cfg.v_head_dim,
                cfg.hidden_size,
                cfg.attention_bias,
                vb.pp(layer_id).pp("attn_output"), //o_proj
            )?;
            Ok(o_proj)
        })
        .collect::<Vec<anyhow::Result<Linear>>>()
        .into_iter()
        .collect::<anyhow::Result<Vec<Linear>>>()?;
    q_types
        .into_iter()
        .zip(
            kv_a_layernorms.into_iter().zip(
                kv_a_proj_with_mqas
                    .into_iter()
                    .zip(kv_b_projs.into_iter().zip(o_projs)),
            ),
        )
        .map(
            |(q_type, (kv_a_layernorm, (kv_a_proj_with_mqa, (kv_b_proj, o_proj))))| {
                let attn = Attention::new(
                    cfg,
                    q_head_dim,
                    rotary_emb.clone(),
                    q_type,
                    kv_a_layernorm,
                    kv_a_proj_with_mqa,
                    kv_b_proj,
                    o_proj,
                )?;
                Ok(attn)
            },
        )
        .collect()
}

pub fn read_gguf_file<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<VarBuilder> {
    VarBuilder::from_gguf(path, device)
}

pub fn print_size(name: &str, tensor: &QTensor) {
    info!(
        "tensor: {}, size: {}MB",
        name,
        tensor.storage_size_in_bytes() as f64 / 1024. / 1024.
    )
}
