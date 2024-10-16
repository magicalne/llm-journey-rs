use candle_nn::Activation;

#[derive(Debug, Clone)]
pub enum TopkMethod {
    Greedy,
    GroupLimitedGreedy,
}
// https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/blob/main/config.json
#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub factor: f64,
    pub mscale: f64,
    pub mscale_all_dim: f64,
    pub original_max_position_embeddings: usize,
    pub typ: String,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub aux_loss_alpha: f64,
    pub pad_token_id: Option<usize>,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub first_k_dense_replace: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub kv_lora_rank: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub moe_intermediate_size: usize,
    pub moe_layer_freq: usize,
    pub n_group: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub norm_topk_prob: bool,
    pub num_attention_heads: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pretraining_tp: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: RopeScaling,
    pub rope_theta: f64,
    pub routed_scaling_factor: f64,
    pub scoring_func: String,
    pub seq_aux: bool,
    pub tie_word_embeddings: bool,
    pub topk_group: usize,
    pub topk_method: TopkMethod,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub v_head_dim: usize,
    pub vocab_size: usize,
    pub use_flash_attn: bool,

    // training configs
    pub ep_size: usize,
}

impl ModelConfig {
    pub fn deepseek_coder_v2_lite_instruct() -> Self {
        Self {
            architectures: vec!["DeepseekV2ForCausalLM".to_string()],
            attention_bias: false,
            attention_dropout: 0.0,
            aux_loss_alpha: 0.001,
            pad_token_id: None,
            bos_token_id: 100000,
            eos_token_id: 100001,
            first_k_dense_replace: 1,
            hidden_act: Activation::Silu,
            hidden_size: 2048,
            initializer_range: 0.02,
            intermediate_size: 10944,
            kv_lora_rank: 512,
            max_position_embeddings: 163840,
            model_type: "deepseek_v2".to_string(),
            moe_intermediate_size: 1408,
            moe_layer_freq: 1,
            n_group: 1,
            n_routed_experts: 64,
            n_shared_experts: 2,
            norm_topk_prob: false,
            num_attention_heads: 16,
            num_experts_per_tok: 6,
            num_hidden_layers: 27,
            num_key_value_heads: 16,
            pretraining_tp: 1,
            q_lora_rank: None,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            rms_norm_eps: 1e-06,
            rope_scaling: RopeScaling {
                beta_fast: 32.,
                beta_slow: 1.,
                factor: 40.,
                mscale: 0.707,
                mscale_all_dim: 0.707,
                original_max_position_embeddings: 4096,
                typ: "yarn".to_string(),
            },
            rope_theta: 10000.,
            routed_scaling_factor: 1.0,
            scoring_func: "softmax".to_string(),
            seq_aux: true,
            tie_word_embeddings: false,
            topk_group: 1,
            topk_method: TopkMethod::Greedy,
            torch_dtype: "bfloat16".to_string(),
            transformers_version: "4.39.3".to_string(),
            use_cache: true,
            v_head_dim: 128,
            vocab_size: 102400,
            use_flash_attn: true,
            ep_size: 1,
        }
    }
}
