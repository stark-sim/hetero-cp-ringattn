use crate::model::{
    backend::LocalAttentionBackend,
    cache::{create_kv_caches, KvCaches},
    config::ModelConfig,
    layers::{DecoderLayer, GqaAttention, Mlp, RmsNorm, RotaryEmbedding},
    ModelError, ModelWeights, WeightNames,
};

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

/// A Llama-family causal language model.
///
/// Supports Llama, Mistral, Qwen2, and compatible architectures.
#[cfg(feature = "tch-backend")]
pub struct LlamaModel {
    pub config: ModelConfig,
    pub embedding: Tensor,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub lm_head: Option<Tensor>,
    /// Global sequence offset for distributed domains.
    /// Domain 0 uses 0; domain 1 uses seq_len / num_domains, etc.
    pub seq_offset: i64,
    /// Number of distributed domains.
    #[allow(dead_code)]
    pub num_domains: usize,
    /// Global sequence length (prefill + generated tokens).
    /// Used for correct position_ids in distributed decode.
    pub global_seq_len: usize,
}

#[cfg(feature = "tch-backend")]
impl LlamaModel {
    /// Build model from loaded safetensors weights and config.
    pub fn from_weights(config: ModelConfig, weights: &ModelWeights, device: Device, num_domains: usize) -> Result<Self, ModelError> {
        let embedding = weights.get(WeightNames::embedding())?.shallow_clone();

        let norm = RmsNorm::from_weights(weights, WeightNames::layer_norm(), config.rms_norm_eps)?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(weights.get_lm_head(&config)?.shallow_clone())
        };

        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings.unwrap_or(4096),
            config.rope_theta,
            device,
        );

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let input_ln = RmsNorm::from_weights(
                weights,
                &WeightNames::rms_norm_weight(layer_idx),
                config.rms_norm_eps,
            )?;
            let post_attn_ln = RmsNorm::from_weights(
                weights,
                &WeightNames::post_attn_norm_weight(layer_idx),
                config.rms_norm_eps,
            )?;

            let attention = GqaAttention::from_weights(weights, layer_idx, &config, &rope)?;
            let mlp = Mlp::from_weights(weights, layer_idx)?;

            let backend: Box<dyn crate::model::backend::AttentionBackend> = if num_domains > 1 {
                Box::new(crate::model::backend::HcpRingAttentionBackend::from_weights(
                    weights, layer_idx, &config, &rope, num_domains,
                )?)
            } else {
                Box::new(LocalAttentionBackend { attention })
            };

            layers.push(DecoderLayer {
                input_layernorm: input_ln,
                post_attention_layernorm: post_attn_ln,
                attention: backend,
                mlp,
            });
        }

        Ok(Self { config, embedding, layers, norm, lm_head, seq_offset: 0, num_domains, global_seq_len: 0 })
    }

    /// Configure distributed transport for all attention backends in this model.
    /// Only affects layers using `HcpRingAttentionBackend`.
    #[cfg(feature = "tch-backend")]
    #[allow(dead_code)]
    pub fn setup_distributed_domain<F>(&mut self, domain_id: usize, seq_offset: i64, mut transport_factory: F)
    where
        F: FnMut(usize) -> Option<Box<dyn crate::model::KvTransport>>,
    {
        self.seq_offset = seq_offset;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let transport = transport_factory(layer_idx);
            layer.attention.set_distributed(domain_id, seq_offset as usize, transport);
        }
    }

    /// Full forward pass (prefill or single step).
    ///
    /// `input_ids`: `[batch, seq_len]` (Int64)
    /// `kv_caches`: per-layer KV caches; `None` means no caching for that layer
    ///
    /// Returns logits: `[batch, seq_len, vocab_size]`
    pub fn forward(&mut self, input_ids: &Tensor, kv_caches: &mut KvCaches) -> Result<Tensor, ModelError> {
        let batch = input_ids.size()[0];
        let seq_len = input_ids.size()[1];
        let device = input_ids.device();

        // Embedding lookup
        let mut hidden_states = Tensor::embedding(&self.embedding, input_ids, -1, false, false);

        // Position IDs: [batch, seq_len]
        let position_ids = if seq_len > 1 {
            // Prefill: sequential positions [seq_offset, seq_offset+1, ..., seq_offset+seq_len-1]
            // global_seq_len tracks the rightmost global position this domain has processed.
            // For distributed decode, all domains must agree on the global prompt length.
            self.global_seq_len = (self.seq_offset + seq_len) as usize;
            let base = Tensor::arange(seq_len, (Kind::Int64, device)) + self.seq_offset;
            base.unsqueeze(0).repeat([batch, 1])
        } else {
            // Decode: position = global_seq_len (same across all distributed domains).
            // In distributed CP, each domain only holds local KV cache, so cache_len
            // would be local length. We must use the global position instead.
            let pos = self.global_seq_len as i64;
            Tensor::from_slice(&[pos])
                .to_device(device)
                .unsqueeze(0)
                .repeat([batch, 1])
        };

        // Causal mask for prefill (not needed for single-token decode)
        let attention_mask = if seq_len > 1 {
            Some(Self::create_causal_mask(seq_len, device))
        } else {
            None
        };

        // Layer stack
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let kv_cache = kv_caches.get_mut(layer_idx).and_then(|c| c.as_mut());
            hidden_states = layer.forward(&hidden_states, &position_ids, kv_cache, attention_mask.as_ref())?;
        }

        // Increment global_seq_len after decode step
        if seq_len == 1 {
            self.global_seq_len += 1;
        }

        // Final norm
        hidden_states = self.norm.forward(&hidden_states);

        // LM Head
        let logits = if let Some(ref lm_head) = self.lm_head {
            hidden_states.matmul(&lm_head.transpose(0, 1))
        } else {
            hidden_states.matmul(&self.embedding.transpose(0, 1))
        };

        Ok(logits)
    }

    /// Create a causal attention mask for prefill.
    ///
    /// Shape: `[1, 1, seq_len, seq_len]` — broadcasts over batch and heads.
    fn create_causal_mask(seq_len: i64, device: Device) -> Tensor {
        let mask = Tensor::ones([seq_len, seq_len], (Kind::Float, device))
            .triu(1)
            .to_kind(Kind::Bool);
        Tensor::zeros([seq_len, seq_len], (Kind::Float, device))
            .masked_fill(&mask, f64::NEG_INFINITY)
            .unsqueeze(0)
            .unsqueeze(0)
    }

    /// Create fresh KV caches for all layers.
    pub fn create_kv_caches(&self) -> KvCaches {
        create_kv_caches(self.config.num_layers)
    }
}

#[cfg(test)]
#[cfg(feature = "tch-backend")]
mod tests {
    use super::*;
    use crate::model::weights::{ModelWeights, WeightNames};
    use tch::{Device, Kind, Tensor};

    fn create_synthetic_weights(config: &ModelConfig, device: Device) -> ModelWeights {
        let mut tensors = std::collections::HashMap::new();
        let hidden = config.hidden_size as i64;
        let vocab = config.vocab_size as i64;
        let intermediate = config.intermediate_size as i64;
        let head_dim = (config.hidden_size / config.num_heads) as i64;
        let num_heads = config.num_heads as i64;
        let num_kv_heads = config.num_kv_heads.unwrap_or(config.num_heads) as i64;

        tensors.insert(WeightNames::embedding().to_string(), Tensor::randn([vocab, hidden], (Kind::Float, device)));
        tensors.insert(WeightNames::layer_norm().to_string(), Tensor::ones([hidden], (Kind::Float, device)));
        tensors.insert(WeightNames::lm_head().to_string(), Tensor::randn([vocab, hidden], (Kind::Float, device)));

        for layer in 0..config.num_layers {
            tensors.insert(WeightNames::rms_norm_weight(layer), Tensor::ones([hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::post_attn_norm_weight(layer), Tensor::ones([hidden], (Kind::Float, device)));
            // HF format: q/k/v/o proj weights are [out_features, in_features]
            tensors.insert(WeightNames::q_proj_weight(layer), Tensor::randn([num_heads * head_dim, hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::k_proj_weight(layer), Tensor::randn([num_kv_heads * head_dim, hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::v_proj_weight(layer), Tensor::randn([num_kv_heads * head_dim, hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::o_proj_weight(layer), Tensor::randn([hidden, num_heads * head_dim], (Kind::Float, device)));
            // HF format: gate/up/down proj weights follow nn.Linear(out_features, in_features)
            tensors.insert(WeightNames::gate_proj_weight(layer), Tensor::randn([intermediate, hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::up_proj_weight(layer), Tensor::randn([intermediate, hidden], (Kind::Float, device)));
            tensors.insert(WeightNames::down_proj_weight(layer), Tensor::randn([hidden, intermediate], (Kind::Float, device)));
        }

        ModelWeights { tensors }
    }

    /// 【分布式 LlamaModel prefill 端到端测试】
    /// 
    /// 测试目标：验证把 16 个 token 的序列拆成两个 domain（各 8 个 token）做分布式 prefill，
    /// 最终拼接的 logits 与单进程参考模型的 logits 一致。
    /// 
    /// 场景设计：
    /// - 参考模型（ref_model）：num_domains=1，处理完整序列 [0,1,...,15]，使用标准 GQA。
    /// - domain0：num_domains=2，domain_id=0，处理前半段 [0,1,...,7]，seq_offset=0。
    /// - domain1：num_domains=2，domain_id=1，处理后半段 [8,9,...,15]，seq_offset=8。
    /// 
    /// domain0 的 attention 只能看到 K/V [0..8)（自己的），看不到 domain1 的 K/V。
    /// 这没问题，因为对于 token 0..7 来说，token 8..15 都是"未来"，因果 mask 本来就看不到。
    /// 
    /// domain1 的 attention 需要看到 K/V [0..16)。
    /// - 本地 K/V [8..16) 由 domain1 自己计算。
    /// - peer K/V [0..8) 需要 domain0 通过网络发送过来。
    #[test]
    fn test_distributed_llama_model_prefill() {
        let device = Device::Cpu;

        // ====== 模型配置 ======
        // 为了测试速度快，用小模型：
        // - hidden_size=32: 每个 token 的向量维度是 32
        // - num_layers=2: 只有 2 层 transformer
        // - num_heads=4: 4 个 attention head
        // - num_kv_heads=1: GQA，4 个 query head 共享 1 个 key/value head（节省显存）
        // - intermediate_size=64: MLP 中间层维度
        // - vocab_size=100: 词表大小
        let config = ModelConfig {
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: Some(1),
            intermediate_size: 64,
            vocab_size: 100,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            torch_dtype: Some("float32".to_string()),
            hidden_act: "silu".to_string(),
            max_position_embeddings: Some(128),
            attention_dropout: 0.0,
            bos_token_id: None,
            eos_token_id: None,
            use_cache: true,
            sliding_window: None,
            use_sliding_window: None,
            partial_rotary_factor: 1.0,
        };

        // 生成随机权重（测试用，不需要加载真实模型）
        let weights = create_synthetic_weights(&config, device);

        // ====== 创建三个模型实例 ======
        // 三个模型共享同一组权重，所以它们的参数完全相同。
        // 
        // ref_model: num_domains=1，使用 LocalAttentionBackend（标准 GQA）。
        // domain0/domain1: num_domains=2，使用 HcpRingAttentionBackend（ring attention）。
        // Use HcpRingAttentionBackend with num_domains=1 as reference to isolate
        // any differences between LocalAttentionBackend and HcpRingAttentionBackend.
        let mut ref_model = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut domain0 = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
        let mut domain1 = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();

        // ====== 设置传输层：每个 layer 独立的一对 transport ======
        // 【重要】每个 layer 必须有自己独立的 transport pair，
        // 否则 layer0 和 layer1 会共享同一个队列，导致跨层污染。
        //
        // 具体做法：
        // - 为每个 layer 创建一对 LinkedMockKvTransport (t0, t1)。
        // - t0 给 domain0 用，t1 给 domain1 用。
        // - domain0.setup_distributed_domain: 遍历所有 layer，把对应 layer 的 transport 注入进去。
        // - closure |layer_idx| 根据 layer 索引返回对应的 transport。
        let num_layers = config.num_layers;
        let mut transports0 = Vec::with_capacity(num_layers);
        let mut transports1 = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let (t0, t1) = crate::model::kv_transport::LinkedMockKvTransport::create_pair();
            transports0.push(t0);
            transports1.push(t1);
        }
        domain0.setup_distributed_domain(0, 0, |layer_idx| Some(Box::new(transports0[layer_idx].clone())));
        domain1.setup_distributed_domain(1, 8, |layer_idx| Some(Box::new(transports1[layer_idx].clone())));

        // ====== 构造输入数据 ======
        // input_ids = [0, 1, 2, ..., 15]，shape [1, 16]。
        // half = 8，拆成两段：
        // - input_ids0 = [0, 1, ..., 7]（domain0）
        // - input_ids1 = [8, 9, ..., 15]（domain1）
        let seq_len = 16i64;
        let half = seq_len / 2;
        let input_ids = Tensor::arange(seq_len, (Kind::Int64, device))
            .unsqueeze(0); // [1, 16]

        // ====== 参考模型前向传播 ======
        // 单进程处理完整序列，输出 logits shape: [1, 16, vocab_size]
        let mut ref_caches = ref_model.create_kv_caches();
        let ref_logits = ref_model.forward(&input_ids, &mut ref_caches).unwrap();

        // ====== domain0 前向传播 ======
        // domain0 处理前半段 [0..8)。
        // narrow(1, 0, 8): 在第 1 维（seq_len 维）上从索引 0 开始取 8 个元素。
        let input_ids0 = input_ids.narrow(1, 0, half);
        let mut caches0 = domain0.create_kv_caches();
        let logits0 = domain0.forward(&input_ids0, &mut caches0).unwrap();

        // ====== domain1 前向传播 ======
        // domain1 处理后半段 [8..16)。
        // 注意：domain0.forward 先执行，domain1.forward 后执行。
        // domain0 发送的 KV block 会先进入 domain1 的收件箱，domain1 在 forward 中就能 recv 到。
        let input_ids1 = input_ids.narrow(1, half, half);
        let mut caches1 = domain1.create_kv_caches();
        let logits1 = domain1.forward(&input_ids1, &mut caches1).unwrap();

        // ====== 拼接分布式输出并比较 ======
        // cat(&[logits0, logits1], 1): 在第 1 维（seq_len 维）上拼接，恢复 [1, 16, vocab_size]。
        let dist_logits = Tensor::cat(&[logits0, logits1], 1);

        // 计算整体平均误差。
        let diff = (&ref_logits - &dist_logits).abs().mean(Kind::Float).double_value(&[]);
        println!("Distributed LlamaModel prefill diff = {}", diff);
        
        // 逐 token 比较误差，方便定位问题。
        for i in 0..seq_len {
            let ref_token = ref_logits.narrow(1, i, 1);
            let dist_token = dist_logits.narrow(1, i, 1);
            let token_diff = (&ref_token - &dist_token).abs().mean(Kind::Float).double_value(&[]);
            println!("token {} diff = {}", i, token_diff);
        }
        
        // End-to-end tolerance tier: mean absolute error threshold for multi-layer model output.
        const END_TO_END_MEAN_ABS_ERR_TOL: f64 = 1e-3;
        assert!(
            diff < END_TO_END_MEAN_ABS_ERR_TOL,
            "Distributed LlamaModel prefill differs from reference: {}",
            diff
        );
    }

    /// 【分布式 LlamaModel decode 端到端测试】
    ///
    /// 测试目标：验证 prefill 后的分布式 decode（单 token 自回归生成）与单进程参考模型一致。
    ///
    /// 场景设计：
    /// - 先执行和 test_distributed_llama_model_prefill 相同的 prefill
    /// - 从参考模型 prefill 输出的最后一个位置取 logits，argmax 得到 next_token
    /// - 用 next_token 作为输入，分别对参考模型和分布式模型做 decode forward
    /// - 比较各模型的 decode 输出 logits
    ///
    /// 关键验证点：
    /// - decode 阶段（seq_len=1）必须走 ring_attention 路径（而非 local_attention_scores 回退）
    /// - 各节点的 KV cache 通过 ring 交换，计算完整 attention
    /// - 所有分布式节点的输出与单进程参考一致
    #[test]
    fn test_distributed_llama_model_decode() {
        let device = Device::Cpu;

        let config = ModelConfig {
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: Some(1),
            intermediate_size: 64,
            vocab_size: 100,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            torch_dtype: Some("float32".to_string()),
            hidden_act: "silu".to_string(),
            max_position_embeddings: Some(128),
            attention_dropout: 0.0,
            bos_token_id: None,
            eos_token_id: None,
            use_cache: true,
            sliding_window: None,
            use_sliding_window: None,
            partial_rotary_factor: 1.0,
        };

        let weights = create_synthetic_weights(&config, device);

        let mut ref_model = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut ring_ref = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
        let mut domain0 = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();
        let mut domain1 = LlamaModel::from_weights(config.clone(), &weights, device, 2).unwrap();

        let num_layers = config.num_layers;
        let mut transports0 = Vec::with_capacity(num_layers);
        let mut transports1 = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let (t0, t1) = crate::model::kv_transport::LinkedMockKvTransport::create_pair();
            transports0.push(t0);
            transports1.push(t1);
        }
        // ring_ref: num_domains=2 but no transport → uses HcpRingAttentionBackend with local KV only
        ring_ref.setup_distributed_domain(0, 0, |_layer_idx| None::<Box<dyn crate::model::KvTransport>>);
        domain0.setup_distributed_domain(0, 0, |layer_idx| Some(Box::new(transports0[layer_idx].clone())));
        domain1.setup_distributed_domain(1, 8, |layer_idx| Some(Box::new(transports1[layer_idx].clone())));

        let seq_len = 16i64;
        let half = seq_len / 2;
        let input_ids = Tensor::arange(seq_len, (Kind::Int64, device))
            .unsqueeze(0); // [1, 16]

        // ====== Prefill ======
        let mut ref_caches = ref_model.create_kv_caches();
        let ref_logits = ref_model.forward(&input_ids, &mut ref_caches).unwrap();

        let mut ring_ref_caches = ring_ref.create_kv_caches();
        let _ring_ref_logits = ring_ref.forward(&input_ids, &mut ring_ref_caches).unwrap();

        let input_ids0 = input_ids.narrow(1, 0, half);
        let mut caches0 = domain0.create_kv_caches();
        let _logits0_prefill = domain0.forward(&input_ids0, &mut caches0).unwrap();

        let input_ids1 = input_ids.narrow(1, half, half);
        let mut caches1 = domain1.create_kv_caches();
        let _logits1_prefill = domain1.forward(&input_ids1, &mut caches1).unwrap();

        // Synchronize global_seq_len across domains for decode.
        // In a real multi-process setup, the coordinator broadcasts the global prompt length.
        let global_prompt_len = domain1.global_seq_len; // domain1 has the rightmost position
        domain0.global_seq_len = global_prompt_len;

        // ====== Decode: sample next token from reference model's last position ======
        let last_logits = ref_logits.narrow(1, seq_len - 1, 1).squeeze();
        let next_token = last_logits.argmax(-1, false).int64_value(&[]) as i64;
        println!("next_token = {}", next_token);

        let decode_input = Tensor::from_slice(&[next_token])
            .unsqueeze(0)
            .to_device(device); // [1, 1]

        // ====== Reference model decode ======
        let ref_decode_logits = ref_model.forward(&decode_input, &mut ref_caches).unwrap();
        let ring_ref_decode_logits = ring_ref.forward(&decode_input, &mut ring_ref_caches).unwrap();
        println!("ref_decode_logits shape = {:?}", ref_decode_logits.size());

        // ====== Distributed model decode ======
        let decode_logits0 = domain0.forward(&decode_input, &mut caches0).unwrap();
        let decode_logits1 = domain1.forward(&decode_input, &mut caches1).unwrap();

        // ====== Compare ======
        let diff_ring = (&ref_decode_logits - &ring_ref_decode_logits).abs().mean(Kind::Float).double_value(&[]);
        let diff0 = (&ref_decode_logits - &decode_logits0).abs().mean(Kind::Float).double_value(&[]);
        let diff1 = (&ref_decode_logits - &decode_logits1).abs().mean(Kind::Float).double_value(&[]);
        let diff01 = (&decode_logits0 - &decode_logits1).abs().mean(Kind::Float).double_value(&[]);

        println!("Decode diff ref-vs-ring_ref (no transport) = {}", diff_ring);
        println!("Decode diff ref-vs-domain0 = {}", diff0);
        println!("Decode diff ref-vs-domain1 = {}", diff1);
        println!("Decode diff domain0-vs-domain1 = {}", diff01);

        const DECODE_TOL: f64 = 1e-3;
        assert!(
            diff_ring < DECODE_TOL,
            "Ring reference (no transport) differs from local reference: {}",
            diff_ring
        );
        assert!(
            diff0 < DECODE_TOL,
            "Distributed decode domain0 differs from reference: {}",
            diff0
        );
        assert!(
            diff1 < DECODE_TOL,
            "Distributed decode domain1 differs from reference: {}",
            diff1
        );
        assert!(
            diff01 < DECODE_TOL,
            "Distributed decode domain0 differs from domain1: {}",
            diff01
        );
    }
}
