use crate::model::{
    // LocalAttentionBackend removed: all paths now use HcpRingAttentionBackend
    // with fixed chunk-size上限 to avoid O(seq²) scores materialization.
    cache::{create_kv_caches, KvCaches},
    config::ModelConfig,
    layers::{DecoderLayer, GqaAttention, Mlp, RmsNorm, RotaryEmbedding},
    ModelError, ModelWeights, WeightNames,
};

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

/// 【Llama 家族因果语言模型】
///
/// 支持 Llama、Mistral、Qwen2 等同家族架构。
/// 这是 HCP 的核心模型结构，负责完整的 transformer forward pass。
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
    /// Whether the first forward (prefill) has been completed.
    /// Distinguishes prefill from decode even when prefill chunk length is 1.
    pub is_prefill_done: bool,
}

#[cfg(feature = "tch-backend")]
impl LlamaModel {
    /// 【从已加载的 safetensors 权重和配置构建模型】
    ///
    /// 遍历所有 layer，为每个 layer 创建：
    /// - 输入 RMSNorm（attention 之前）
    /// - Attention 后端（HcpRingAttentionBackend）
    /// - MLP（SwiGLU）
    /// - 输出 RMSNorm（attention 之后、MLP 之前）
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

            let _attention = GqaAttention::from_weights(weights, layer_idx, &config, &rope)?;
            let mlp = Mlp::from_weights(weights, layer_idx)?;

            // Always use HcpRingAttentionBackend (even for single-node).
            // It implements online softmax with fixed chunk-size上限,
            // avoiding the O(seq²) scores materialization in GqaAttention::forward.
            let backend: Box<dyn crate::model::attention::AttentionBackend> =
                Box::new(crate::model::attention::HcpRingAttentionBackend::from_weights(
                    weights, layer_idx, &config, &rope, num_domains,
                )?);

            layers.push(DecoderLayer {
                input_layernorm: input_ln,
                post_attention_layernorm: post_attn_ln,
                attention: backend,
                mlp,
            });
        }

        Ok(Self { config, embedding, layers, norm, lm_head, seq_offset: 0, num_domains, global_seq_len: 0, is_prefill_done: false })
    }

    /// 【为模型中所有 attention 后端配置分布式传输】
    ///
    /// 遍历所有 layer，为每个 layer 的 attention 注入对应的 KvTransport。
    /// `transport_factory` 是一个闭包，根据 layer_idx 返回对应的 transport。
    ///
    /// 注意：每个 layer 必须有独立的 transport，layer 之间不能共享。
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

    /// 【完整前向传播】支持 prefill（多 token）和 decode（单 token）两种模式。
    ///
    /// 流程：
    /// 1. Embedding lookup
    /// 2. 生成 position_ids（prefill 用递增序列，decode 用 global_seq_len）
    /// 3. 生成 attention mask（prefill 需要 causal mask，decode 不需要）
    /// 4. 逐层 forward（每层 = Norm → Attention → Norm → MLP）
    /// 5. 最终 Norm + LM Head
    ///
    /// 【内存优化】
    /// - no_grad_guard: 禁用梯度计算，避免推理时保留计算图
    /// - LM Head chunked: 长序列时只计算最后一个位置的 logits
    ///   （避免分配 [batch, seq_len, vocab_size] 的完整 logits 张量）
    ///
    /// `input_ids`: `[batch, seq_len]` (Int64)
    /// `kv_caches`: per-layer KV caches; `None` means no caching for that layer
    ///
    /// Returns logits: `[batch, seq_len, vocab_size]`
    pub fn forward(&mut self, input_ids: &Tensor, kv_caches: &mut KvCaches) -> Result<Tensor, ModelError> {
        // Disable gradient computation for inference. Without this, PyTorch
        // retains the entire computation graph across all 24 layers, which
        // balloons memory usage by several GB (especially for long sequences).
        let _no_grad = tch::no_grad_guard();

        let batch = input_ids.size()[0];
        let seq_len = input_ids.size()[1];
        let device = input_ids.device();

        // Embedding lookup
        let mut hidden_states = Tensor::embedding(&self.embedding, input_ids, -1, false, false);

        // Guard: prevent position_ids from exceeding RoPE cache / model capacity.
        if let Some(max_pos) = self.config.max_position_embeddings {
            let max_pos = max_pos as i64;
            if self.seq_offset + seq_len > max_pos {
                return Err(ModelError::Generation(format!(
                    "sequence length {} + offset {} exceeds max_position_embeddings {}; prompt too long",
                    seq_len, self.seq_offset, max_pos
                )));
            }
        }

        // Position IDs: [batch, seq_len]
        let is_prefill = !self.is_prefill_done;
        let position_ids = if is_prefill {
            // Prefill: sequential positions [seq_offset, seq_offset+1, ..., seq_offset+seq_len-1]
            // global_seq_len tracks the rightmost global position this domain has processed.
            // For distributed decode, all domains must agree on the global prompt length.
            self.global_seq_len = (self.seq_offset + seq_len) as usize;
            self.is_prefill_done = true;
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
        // For distributed inference, ring_attention only checks is_some() and
        // implements causality via global position comparison; it never reads
        // the dense mask tensor. Use a tiny dummy to avoid O(seq_len²) allocation.
        let attention_mask = if seq_len > 1 {
            // For long sequences or distributed mode, skip O(seq_len²) dense mask.
            // HcpRingAttentionBackend implements causality via position comparison
            // and never reads the mask tensor data; it only checks is_some().
            if self.num_domains > 1 || seq_len > 8192 {
                Some(Tensor::zeros([1, 1, 1, 1], (Kind::Float, device)))
            } else {
                Some(Self::create_causal_mask(seq_len, device))
            }
        } else {
            None
        };

        // Layer stack
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let kv_cache: Option<&mut dyn crate::model::cache::KvCache> = kv_caches
                .get_mut(layer_idx)
                .and_then(|c| c.as_mut().map(|c| c as &mut dyn crate::model::cache::KvCache));
            hidden_states = layer.forward(&hidden_states, &position_ids, kv_cache, attention_mask.as_ref())?;
        }

        // Increment global_seq_len after decode step only
        if !is_prefill {
            self.global_seq_len += 1;
        }

        // Final norm
        hidden_states = self.norm.forward(&hidden_states);

        // LM Head — chunked for long sequences to avoid OOM.
        // Avoid pre-allocating a full [batch, seq_len, vocab_size] buffer;
        // instead compute chunks and cat at the end. This keeps peak memory
        // at ~5GB (8K * vocab_size * 4B) instead of ~20GB (32K * vocab_size * 4B).
        const LM_HEAD_CHUNK_SIZE: i64 = 8192;
        let seq_len = hidden_states.size()[1];

        let logits = if seq_len > LM_HEAD_CHUNK_SIZE {
            // Long prefill: only compute logits for the last position.
            // All callers (Generator, distributed::worker) only use the last
            // token's logits for sampling. Avoids ~20GB peak.
            let last_hidden = hidden_states.narrow(1, seq_len - 1, 1);
            if let Some(ref lm_head) = self.lm_head {
                last_hidden.matmul(&lm_head.transpose(0, 1))
            } else {
                last_hidden.matmul(&self.embedding.transpose(0, 1))
            }
        } else if let Some(ref lm_head) = self.lm_head {
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

    /// Forward pass that also exports per-layer hidden states.
    ///
    /// Runs the same computation as `forward`, but after each decoder layer
    /// and after the final norm, saves the hidden states to `{export_dir}/layer_{i}.bin`.
    /// Each file format: `[batch: u64 LE][seq_len: u64 LE][hidden_size: u64 LE][f32 data...]`
    ///
    /// This is intended for debug correctness validation only.
    pub fn forward_with_hidden_state_export(
        &mut self,
        input_ids: &Tensor,
        kv_caches: &mut KvCaches,
        export_dir: &str,
    ) -> Result<Tensor, ModelError> {
        let _no_grad = tch::no_grad_guard();

        let batch = input_ids.size()[0];
        let seq_len = input_ids.size()[1];
        let device = input_ids.device();

        let mut hidden_states = Tensor::embedding(&self.embedding, input_ids, -1, false, false);

        if let Some(max_pos) = self.config.max_position_embeddings {
            let max_pos = max_pos as i64;
            if self.seq_offset + seq_len > max_pos {
                return Err(ModelError::Generation(format!(
                    "sequence length {} + offset {} exceeds max_position_embeddings {}; prompt too long",
                    seq_len, self.seq_offset, max_pos
                )));
            }
        }

        let is_prefill = !self.is_prefill_done;
        let position_ids = if is_prefill {
            self.global_seq_len = (self.seq_offset + seq_len) as usize;
            self.is_prefill_done = true;
            let base = Tensor::arange(seq_len, (Kind::Int64, device)) + self.seq_offset;
            base.unsqueeze(0).repeat([batch, 1])
        } else {
            let pos = self.global_seq_len as i64;
            Tensor::from_slice(&[pos])
                .to_device(device)
                .unsqueeze(0)
                .repeat([batch, 1])
        };

        let attention_mask = if seq_len > 1 {
            if self.num_domains > 1 || seq_len > 8192 {
                Some(Tensor::zeros([1, 1, 1, 1], (Kind::Float, device)))
            } else {
                Some(Self::create_causal_mask(seq_len, device))
            }
        } else {
            None
        };

        std::fs::create_dir_all(export_dir)
            .map_err(|e| ModelError::Generation(format!("failed to create export dir: {}", e)))?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let kv_cache: Option<&mut dyn crate::model::cache::KvCache> = kv_caches
                .get_mut(layer_idx)
                .and_then(|c| c.as_mut().map(|c| c as &mut dyn crate::model::cache::KvCache));
            hidden_states = layer.forward(&hidden_states, &position_ids, kv_cache, attention_mask.as_ref())?;

            // Export hidden state after this layer
            let file_path = std::path::Path::new(export_dir).join(format!("layer_{}.bin", layer_idx));
            Self::write_tensor_as_binary(&hidden_states, &file_path)
                .map_err(|e| ModelError::Generation(format!("failed to write layer {} hidden state: {}", layer_idx, e)))?;
        }

        if !is_prefill {
            self.global_seq_len += 1;
        }

        hidden_states = self.norm.forward(&hidden_states);

        // Export final norm output
        let file_path = std::path::Path::new(export_dir).join("final_norm.bin");
        Self::write_tensor_as_binary(&hidden_states, &file_path)
            .map_err(|e| ModelError::Generation(format!("failed to write final norm hidden state: {}", e)))?;

        let seq_len = hidden_states.size()[1];
        const LM_HEAD_CHUNK_SIZE: i64 = 8192;
        let logits = if seq_len > LM_HEAD_CHUNK_SIZE {
            let last_hidden = hidden_states.narrow(1, seq_len - 1, 1);
            if let Some(ref lm_head) = self.lm_head {
                last_hidden.matmul(&lm_head.transpose(0, 1))
            } else {
                last_hidden.matmul(&self.embedding.transpose(0, 1))
            }
        } else if let Some(ref lm_head) = self.lm_head {
            hidden_states.matmul(&lm_head.transpose(0, 1))
        } else {
            hidden_states.matmul(&self.embedding.transpose(0, 1))
        };

        Ok(logits)
    }

    /// Write a tensor to a binary file: [ndims: u64 LE][dim0: u64 LE]...[dimN: u64 LE][f32 data...]
    fn write_tensor_as_binary(tensor: &Tensor, path: &std::path::Path) -> Result<(), String> {
        use std::io::Write;
        let flat = tensor.view(-1);
        let data: Vec<f32> = Vec::try_from(&flat).map_err(|e| format!("tensor to vec: {}", e))?;
        let shape = tensor.size();
        let mut file = std::fs::File::create(path)
            .map_err(|e| format!("create file: {}", e))?;
        let ndims = shape.len() as u64;
        file.write_all(&ndims.to_le_bytes()).map_err(|e| e.to_string())?;
        for &dim in &shape {
            file.write_all(&(dim as u64).to_le_bytes()).map_err(|e| e.to_string())?;
        }
        for &val in &data {
            file.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "tch-backend")]
pub(crate) fn create_synthetic_weights(config: &ModelConfig, device: Device) -> ModelWeights {
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

#[cfg(test)]
#[cfg(feature = "tch-backend")]
mod tests {
    use super::*;
    use crate::model::weights::{ModelWeights, WeightNames};
    use tch::{Device, Kind, Tensor};

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
        // ref_model: num_domains=1，使用 HcpRingAttentionBackend（分块 online softmax）。
        // domain0/domain1: num_domains=2，使用 HcpRingAttentionBackend（ring attention）。
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
            let (t0, t1) = crate::model::transport::LinkedMockKvTransport::create_pair();
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
            let (t0, t1) = crate::model::transport::LinkedMockKvTransport::create_pair();
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

    /// 【多步分布式 decode 端到端测试】
    ///
    /// 验证 prefill 后连续生成多个 token，每一步分布式模型都与单节点参考一致。
    /// 这是 `test_distributed_llama_model_decode` 的自然延伸：单步对 ≠ 多步对，
    /// 因为 KV cache 和 position_ids 会在每一步累积和变化。
    #[test]
    fn test_distributed_llama_model_multi_step_decode() {
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
            let (t0, t1) = crate::model::transport::LinkedMockKvTransport::create_pair();
            transports0.push(t0);
            transports1.push(t1);
        }
        ring_ref.setup_distributed_domain(0, 0, |_layer_idx| None::<Box<dyn crate::model::KvTransport>>);
        domain0.setup_distributed_domain(0, 0, |layer_idx| Some(Box::new(transports0[layer_idx].clone())));
        domain1.setup_distributed_domain(1, 8, |layer_idx| Some(Box::new(transports1[layer_idx].clone())));

        let seq_len = 16i64;
        let half = seq_len / 2;
        let input_ids = Tensor::arange(seq_len, (Kind::Int64, device)).unsqueeze(0);

        // Prefill all models
        let mut ref_caches = ref_model.create_kv_caches();
        let ref_logits = ref_model.forward(&input_ids, &mut ref_caches).unwrap();

        let mut ring_ref_caches = ring_ref.create_kv_caches();
        let _ = ring_ref.forward(&input_ids, &mut ring_ref_caches).unwrap();

        let input_ids0 = input_ids.narrow(1, 0, half);
        let mut caches0 = domain0.create_kv_caches();
        let _ = domain0.forward(&input_ids0, &mut caches0).unwrap();

        let input_ids1 = input_ids.narrow(1, half, half);
        let mut caches1 = domain1.create_kv_caches();
        let _ = domain1.forward(&input_ids1, &mut caches1).unwrap();

        // Synchronize global_seq_len for decode
        let global_prompt_len = domain1.global_seq_len;
        domain0.global_seq_len = global_prompt_len;

        // Sample first token from ref prefill output
        let mut next_token = ref_logits.narrow(1, seq_len - 1, 1).squeeze()
            .argmax(-1, false).int64_value(&[]) as i64;
        println!("multi-step decode first_token = {}", next_token);

        const DECODE_TOL: f64 = 1e-3;
        const NUM_DECODE_STEPS: usize = 4;

        for step in 0..NUM_DECODE_STEPS {
            let decode_input = Tensor::from_slice(&[next_token])
                .unsqueeze(0)
                .to_device(device);

            let ref_decode_logits = ref_model.forward(&decode_input, &mut ref_caches).unwrap();
            let ring_ref_decode_logits = ring_ref.forward(&decode_input, &mut ring_ref_caches).unwrap();
            let decode_logits0 = domain0.forward(&decode_input, &mut caches0).unwrap();
            let decode_logits1 = domain1.forward(&decode_input, &mut caches1).unwrap();

            let diff_ring = (&ref_decode_logits - &ring_ref_decode_logits).abs().mean(Kind::Float).double_value(&[]);
            let diff0 = (&ref_decode_logits - &decode_logits0).abs().mean(Kind::Float).double_value(&[]);
            let diff1 = (&ref_decode_logits - &decode_logits1).abs().mean(Kind::Float).double_value(&[]);
            let diff01 = (&decode_logits0 - &decode_logits1).abs().mean(Kind::Float).double_value(&[]);

            println!(
                "step {} diff ring={:.2e} ref0={:.2e} ref1={:.2e} d01={:.2e}",
                step, diff_ring, diff0, diff1, diff01
            );

            assert!(diff_ring < DECODE_TOL, "step {} ring_ref diff too large: {}", step, diff_ring);
            assert!(diff0 < DECODE_TOL, "step {} domain0 diff too large: {}", step, diff0);
            assert!(diff1 < DECODE_TOL, "step {} domain1 diff too large: {}", step, diff1);
            assert!(diff01 < DECODE_TOL, "step {} domain0-vs-domain1 diff too large: {}", step, diff01);

            // Sample next token from ref for the following step
            next_token = ref_decode_logits.squeeze().argmax(-1, false).int64_value(&[]) as i64;
        }
    }

    /// 【Batch forward correctness 验证】
    ///
    /// 测试目标：验证 LlamaModel::forward 在 batch > 1 时，
    /// 每个 sample 的输出与单独 batch=1 forward 的结果在数值上一致。
    ///
    /// 这是 batching 的 correctness 基线：如果 batch > 1 的结果与
    /// 逐个处理 batch=1 的结果不一致，说明模型层存在 batch 相关的 bug
    ///（如 position_ids、attention_mask、KV cache 等未正确处理 batch 维度）。
    ///
    /// 验证内容：
    /// 1. Prefill batch=2 vs 两个独立的 batch=1 prefill
    /// 2. Decode batch=2 vs 两个独立的 batch=1 decode（各一步）
    /// 3. 多步 decode batch=2 vs 独立的 batch=1 decode（4 步）
    #[test]
    fn test_batch_forward_correctness() {
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

        // 创建三个独立的模型实例，各自有独立的 KV cache
        let mut model_batch = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut model_a = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut model_b = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();

        // 准备两个不同的 prompt（相同长度，避免 padding 复杂度）
        let seq_len = 12i64;
        let prompt_a: Vec<i64> = (0..seq_len).collect();
        let prompt_b: Vec<i64> = (10..10 + seq_len).collect();

        // ====== Prefill correctness ======
        let input_a = Tensor::from_slice(&prompt_a).unsqueeze(0).to_device(device); // [1, 12]
        let input_b = Tensor::from_slice(&prompt_b).unsqueeze(0).to_device(device); // [1, 12]
        let input_batch = Tensor::cat(&[input_a.shallow_clone(), input_b.shallow_clone()], 0); // [2, 12]

        let mut caches_a = model_a.create_kv_caches();
        let mut caches_b = model_b.create_kv_caches();
        let mut caches_batch = model_batch.create_kv_caches();

        let logits_a = model_a.forward(&input_a, &mut caches_a).unwrap();
        let logits_b = model_b.forward(&input_b, &mut caches_b).unwrap();
        let logits_batch = model_batch.forward(&input_batch, &mut caches_batch).unwrap();

        // logits_batch shape: [2, 12, vocab_size]
        let batch_a = logits_batch.narrow(0, 0, 1); // [1, 12, vocab]
        let batch_b = logits_batch.narrow(0, 1, 1); // [1, 12, vocab]

        let diff_a = (&logits_a - &batch_a).abs().mean(Kind::Float).double_value(&[]);
        let diff_b = (&logits_b - &batch_b).abs().mean(Kind::Float).double_value(&[]);

        println!("Prefill batch correctness: diff_a={:.2e}, diff_b={:.2e}", diff_a, diff_b);

        // CPU BLAS non-determinism can cause ~1.5e-5 diff between batched and
        // single-path. Use relaxed tolerance + token agreement as the true
        // correctness signal.
        const BATCH_TOL: f64 = 1e-4;
        assert!(diff_a < BATCH_TOL, "Prefill batch sample 0 differs: {}", diff_a);
        assert!(diff_b < BATCH_TOL, "Prefill batch sample 1 differs: {}", diff_b);

        // ====== Single-step decode correctness ======
        // 从 batch 模型的 prefill 输出采样两个 token
        let last_logits_a = logits_a.narrow(1, seq_len - 1, 1).squeeze();
        let last_logits_b = logits_b.narrow(1, seq_len - 1, 1).squeeze();
        let token_a = last_logits_a.argmax(-1, false).int64_value(&[]) as i64;
        let token_b = last_logits_b.argmax(-1, false).int64_value(&[]) as i64;

        let decode_a = Tensor::from_slice(&[token_a]).unsqueeze(0).to_device(device);
        let decode_b = Tensor::from_slice(&[token_b]).unsqueeze(0).to_device(device);
        let decode_batch = Tensor::cat(&[decode_a.shallow_clone(), decode_b.shallow_clone()], 0);

        let d_logits_a = model_a.forward(&decode_a, &mut caches_a).unwrap();
        let d_logits_b = model_b.forward(&decode_b, &mut caches_b).unwrap();
        let d_logits_batch = model_batch.forward(&decode_batch, &mut caches_batch).unwrap();

        let d_batch_a = d_logits_batch.narrow(0, 0, 1);
        let d_batch_b = d_logits_batch.narrow(0, 1, 1);

        let d_diff_a = (&d_logits_a - &d_batch_a).abs().mean(Kind::Float).double_value(&[]);
        let d_diff_b = (&d_logits_b - &d_batch_b).abs().mean(Kind::Float).double_value(&[]);

        println!("Decode batch correctness: diff_a={:.2e}, diff_b={:.2e}", d_diff_a, d_diff_b);

        assert!(d_diff_a < BATCH_TOL, "Decode batch sample 0 differs: {}", d_diff_a);
        assert!(d_diff_b < BATCH_TOL, "Decode batch sample 1 differs: {}", d_diff_b);

        // Token agreement is the true correctness signal.
        let d_token_batch_a = d_batch_a.squeeze().argmax(-1, false).int64_value(&[]);
        let d_token_batch_b = d_batch_b.squeeze().argmax(-1, false).int64_value(&[]);
        let d_token_a = d_logits_a.squeeze().argmax(-1, false).int64_value(&[]);
        let d_token_b = d_logits_b.squeeze().argmax(-1, false).int64_value(&[]);
        assert_eq!(d_token_a, d_token_batch_a, "single-decode token mismatch for sample 0");
        assert_eq!(d_token_b, d_token_batch_b, "single-decode token mismatch for sample 1");

        // ====== Multi-step decode correctness ======
        // Logits may diverge slightly due to floating-point non-determinism
        // in batched vs single-path BLAS kernels (~1e-5). What matters for
        // correctness is that the *sampled tokens* are identical. We check
        // both: logits diff must stay below a generous tolerance, and argmax
        // must agree exactly.
        const NUM_DECODE_STEPS: usize = 4;
        const LOGITS_TOL: f64 = 1e-3;
        let mut next_token_a = token_a;
        let mut next_token_b = token_b;

        for step in 0..NUM_DECODE_STEPS {
            let da = Tensor::from_slice(&[next_token_a]).unsqueeze(0).to_device(device);
            let db = Tensor::from_slice(&[next_token_b]).unsqueeze(0).to_device(device);
            let dbatch = Tensor::cat(&[da.shallow_clone(), db.shallow_clone()], 0);

            let la = model_a.forward(&da, &mut caches_a).unwrap();
            let lb = model_b.forward(&db, &mut caches_b).unwrap();
            let lbatch = model_batch.forward(&dbatch, &mut caches_batch).unwrap();

            let lbatch_a = lbatch.narrow(0, 0, 1);
            let lbatch_b = lbatch.narrow(0, 1, 1);

            let step_diff_a = (&la - &lbatch_a).abs().mean(Kind::Float).double_value(&[]);
            let step_diff_b = (&lb - &lbatch_b).abs().mean(Kind::Float).double_value(&[]);

            let token_batch_a = lbatch_a.squeeze().argmax(-1, false).int64_value(&[]);
            let token_batch_b = lbatch_b.squeeze().argmax(-1, false).int64_value(&[]);
            let token_ref_a = la.squeeze().argmax(-1, false).int64_value(&[]);
            let token_ref_b = lb.squeeze().argmax(-1, false).int64_value(&[]);

            println!(
                "Multi-step decode step {}: diff_a={:.2e}, diff_b={:.2e}, tokens=[{},{}] vs ref=[{},{}]",
                step, step_diff_a, step_diff_b, token_batch_a, token_batch_b, token_ref_a, token_ref_b
            );

            assert!(step_diff_a < LOGITS_TOL, "step {} batch sample 0 logits diff too large: {}", step, step_diff_a);
            assert!(step_diff_b < LOGITS_TOL, "step {} batch sample 1 logits diff too large: {}", step, step_diff_b);
            assert_eq!(token_batch_a, token_ref_a, "step {} batch sample 0 token mismatch", step);
            assert_eq!(token_batch_b, token_ref_b, "step {} batch sample 1 token mismatch", step);

            next_token_a = token_ref_a as i64;
            next_token_b = token_ref_b as i64;
        }
    }

    /// 【BlockTableKvCache 集成测试】验证 BlockTable 缓存通过 LlamaModel::forward 的
    /// 完整 prefill + decode 路径时，输出与 ContiguousKvCache 完全一致。
    ///
    /// 使用 block_size=4 确保测试跨越 block 边界（prefill 8 tokens 跨越 2 个 block）。
    #[test]
    fn test_block_table_through_model_forward() {
        use crate::model::cache::{BlockTableKvCache, KvCacheImpl};

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
        let mut model_contiguous = LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut model_block = LlamaModel::from_weights(config, &weights, device, 1).unwrap();

        // Prefill with 8 tokens (crosses block boundary when block_size=4)
        let prompt: Vec<i64> = (0..8).collect();
        let input_ids = Tensor::from_slice(&prompt).unsqueeze(0).to_device(device);

        let mut caches_contiguous = model_contiguous.create_kv_caches();
        let mut caches_block: KvCaches = (0..model_block.layers.len())
            .map(|_| Some(KvCacheImpl::BlockTable(BlockTableKvCache::new(4))))
            .collect();

        let logits_contiguous = model_contiguous.forward(&input_ids, &mut caches_contiguous).unwrap();
        let logits_block = model_block.forward(&input_ids, &mut caches_block).unwrap();

        let prefill_diff = (&logits_contiguous - &logits_block).abs().mean(Kind::Float).double_value(&[]);
        println!("Prefill diff (Contiguous vs BlockTable): {:.2e}", prefill_diff);
        assert!(prefill_diff < 1e-6, "Prefill logits differ: {}", prefill_diff);

        // Decode 3 steps
        let mut next_token_contiguous = logits_contiguous
            .narrow(1, 7, 1)
            .squeeze()
            .argmax(-1, false)
            .int64_value(&[]) as i64;
        let mut next_token_block = logits_block
            .narrow(1, 7, 1)
            .squeeze()
            .argmax(-1, false)
            .int64_value(&[]) as i64;

        // argmax should agree on prefill
        assert_eq!(next_token_contiguous, next_token_block);

        for step in 0..3 {
            let input_c = Tensor::from_slice(&[next_token_contiguous]).unsqueeze(0).to_device(device);
            let input_b = Tensor::from_slice(&[next_token_block]).unsqueeze(0).to_device(device);

            let logit_c = model_contiguous.forward(&input_c, &mut caches_contiguous).unwrap();
            let logit_b = model_block.forward(&input_b, &mut caches_block).unwrap();

            let step_diff = (&logit_c - &logit_b).abs().mean(Kind::Float).double_value(&[]);
            println!("Decode step {} diff: {:.2e}", step, step_diff);
            assert!(step_diff < 1e-6, "Decode step {} logits differ: {}", step, step_diff);

            next_token_contiguous = logit_c.squeeze().argmax(-1, false).int64_value(&[]) as i64;
            next_token_block = logit_b.squeeze().argmax(-1, false).int64_value(&[]) as i64;
            assert_eq!(next_token_contiguous, next_token_block,
                "Decode step {} sampled token mismatch", step);
        }

        println!("✅ BlockTableKvCache integration test passed");
    }
}
