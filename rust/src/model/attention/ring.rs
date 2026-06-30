use crate::model::ModelError;
use super::backend::AttentionBackend;
use super::strategy::RingSchedulingStrategy;

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};
#[cfg(feature = "tch-backend")]
use crate::model::transport::{KvBlock, KvTransport};

#[cfg(feature = "tch-backend")]
use std::time::Instant;
#[cfg(feature = "tch-backend")]
use std::io::Write;

/// Ring-attention backend that splits sequence into chunks and computes
/// attention via online softmax over K/V blocks.
///
/// This is a single-process simulation of multi-domain ring attention.
/// In Phase 3 it will be extended to true multi-process / multi-node.
#[cfg(feature = "tch-backend")]
pub struct HcpRingAttentionBackend {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    rope: crate::model::layers::RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    num_domains: usize,
    layer_idx: usize,
    #[cfg(feature = "tch-backend")]
    kv_transport: Option<Box<dyn KvTransport>>,
    #[allow(dead_code)]
    local_domain_id: usize,
    seq_offset: usize,
    /// Length of KV cache produced during prefill. In decode phase we only send
    /// the prefill partition (not decode-appended tokens) to peers.
    prefill_kv_len: usize,
    /// Whether the first forward (prefill) has completed.
    is_prefill_done: bool,
    /// 【micro KV block 大小】用于跨 domain KV 传输的粒度控制。
    /// 大 seq 时整个 KV block 可能几十 MB，小 domain 无法一次性接收。
    /// 把 KV 切成 micro blocks 后，可以流水化传输和逐块处理。
    /// 默认 0 表示使用 kv_chunk_size 作为 micro block 大小。
    micro_kv_block_size: usize,
    /// 【禁用 overlap】当设置 HCP_DISABLE_OVERLAP=1 时，回到串行模式
    ///（先全部 exchange 完，再统一 compute），用于对比测试。
    disable_overlap: bool,
    /// 【本地序列的原始位置 id】用于 Striped / 非连续分片场景。
    /// 如果为 None，则默认假设本地 chunk 是原始序列中的连续段。
    position_ids: Option<Tensor>,
    /// 【调度策略】仅影响输入分片方式，不影响 online softmax 数学。
    #[allow(dead_code)]
    strategy: RingSchedulingStrategy,
    /// 【本地 KV cache 的全局起始位置】用于 micro-block 元数据。
    /// 对于非连续分片，等于 position_ids 的第一个元素。
    kv_base_global_start: usize,
}

#[cfg(feature = "tch-backend")]
impl Default for HcpRingAttentionBackend {
    fn default() -> Self {
        let device = Device::Cpu;
        Self {
            q_proj: Tensor::zeros([1, 1], (Kind::Float, device)),
            k_proj: Tensor::zeros([1, 1], (Kind::Float, device)),
            v_proj: Tensor::zeros([1, 1], (Kind::Float, device)),
            o_proj: Tensor::zeros([1, 1], (Kind::Float, device)),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            rope: crate::model::layers::RotaryEmbedding::new(1, 1, 1.0, device),
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            scale: 1.0,
            num_domains: 1,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
            micro_kv_block_size: 0,
            disable_overlap: false,
            position_ids: None,
            strategy: RingSchedulingStrategy::default(),
            kv_base_global_start: 0,
        }
    }
}

#[cfg(feature = "tch-backend")]
impl HcpRingAttentionBackend {
    pub fn from_weights(
        weights: &crate::model::ModelWeights,
        layer: usize,
        config: &crate::model::ModelConfig,
        rope: &crate::model::layers::RotaryEmbedding,
        num_domains: usize,
    ) -> Result<Self, ModelError> {
        let q_bias = weights.get(&crate::model::WeightNames::q_proj_bias(layer)).ok().map(|t| t.shallow_clone());
        let k_bias = weights.get(&crate::model::WeightNames::k_proj_bias(layer)).ok().map(|t| t.shallow_clone());
        let v_bias = weights.get(&crate::model::WeightNames::v_proj_bias(layer)).ok().map(|t| t.shallow_clone());
        Ok(Self {
            q_proj: weights.get(&crate::model::WeightNames::q_proj_weight(layer))?.shallow_clone(),
            k_proj: weights.get(&crate::model::WeightNames::k_proj_weight(layer))?.shallow_clone(),
            v_proj: weights.get(&crate::model::WeightNames::v_proj_weight(layer))?.shallow_clone(),
            o_proj: weights.get(&crate::model::WeightNames::o_proj_weight(layer))?.shallow_clone(),
            q_bias,
            k_bias,
            v_bias,
            rope: rope.clone(),
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads(),
            head_dim: config.head_dim(),
            scale: 1.0 / (config.head_dim() as f64).sqrt(),
            num_domains: num_domains.max(1),
            layer_idx: layer,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
            micro_kv_block_size: std::env::var("HCP_MICRO_KV_BLOCK_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            disable_overlap: std::env::var("HCP_DISABLE_OVERLAP")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false),
            position_ids: None,
            strategy: RingSchedulingStrategy::default(),
            kv_base_global_start: 0,
        })
    }

    #[cfg(feature = "tch-backend")]
    #[allow(dead_code)]
    pub fn set_transport(&mut self, transport: Box<dyn KvTransport>) {
        self.kv_transport = Some(transport);
    }

    #[allow(dead_code)]
    pub fn set_local_domain_id(&mut self, id: usize) {
        self.local_domain_id = id;
    }

    #[allow(dead_code)]
    pub fn set_seq_offset(&mut self, offset: usize) {
        self.seq_offset = offset;
    }

    /// 【处理单个 KV block】用一组 Q 去和一组 K/V 做 attention，然后更新 online softmax 状态。
    /// 
    /// 所有位置参数都必须是【全局位置】，不能是本地索引。
    /// 例如 domain1 的本地索引 0 对应全局位置 8。
    #[allow(clippy::too_many_arguments)]
    fn process_kv_block(
        &self,
        q_chunk: &Tensor,            // 当前 Q chunk，shape: [batch, num_heads, q_chunk_len, head_dim]
        q_pos: &Tensor,              // 当前 Q chunk 的原始全局位置，shape: [q_chunk_len] (Int64)
        k_chunk: &Tensor,            // 当前 K block，shape: [batch, num_heads, kv_chunk_len, head_dim]
        v_chunk: &Tensor,            // 当前 V block，shape: [batch, num_heads, kv_chunk_len, head_dim]
        k_pos: &Tensor,              // 当前 K/V block 的原始全局位置，shape: [kv_chunk_len] (Int64)
        rm: &mut Tensor,             // 【running max】当前见过的最大 score（可变的引用）
        rs: &mut Tensor,             // 【running sum】当前 softmax 分母的累加和
        obh: &mut Tensor,            // 【output buffer】当前加权累加的输出
        apply_causal_mask: bool,     // 【是否应用因果掩码】true=因果路径, false=非因果路径
    ) {
        let kv_chunk_len = k_pos.size()[0];

        // 空 block 跳过。
        if kv_chunk_len <= 0 {
            return;
        }

        // Early Return 优化（仅因果路径）：如果 KV block 的最小位置严格大于 Q chunk 的最大位置，
        // 说明所有 KV 都在 Q 的"未来"，会被 causal mask 完全屏蔽，直接跳过可省一次矩阵乘法。
        if apply_causal_mask {
            let k_min = k_pos.min().int64_value(&[]);
            let q_max = q_pos.max().int64_value(&[]);
            if k_min > q_max {
                return;
            }
        }

        // ====== 第一步：计算 Attention Scores ======
        // Attention 的核心公式：score = Q @ K^T / sqrt(head_dim)
        // 
        // q_chunk shape: [batch, num_heads, q_chunk_len, head_dim]
        // k_chunk shape: [batch, num_heads, kv_chunk_len, head_dim]
        // k_chunk.transpose(2, 3) → [batch, num_heads, head_dim, kv_chunk_len]
        // matmul 结果 shape: [batch, num_heads, q_chunk_len, kv_chunk_len]
        // 最后乘 self.scale（即 1/sqrt(head_dim)）做缩放，防止数值过大导致 softmax 梯度消失。
        let mut scores = q_chunk.matmul(&k_chunk.transpose(2, 3)) * self.scale;

        // 用全局位置构造因果掩码，确保当前 token 看不到未来的 token。
        // 
        // 原理：对于每个 query 位置 i 和 key 位置 j，如果 i >= j 则允许 attention，否则 mask 为 -inf。
        // 
        // q_pos: 当前 Q chunk 的全局位置，shape [1, 1, q_chunk_len, 1]
        // k_pos: 当前 K/V block 的全局位置，shape [1, 1, 1, kv_chunk_len]
        // causal = q_pos.ge_tensor(&k_pos): element-wise 比较，返回 bool tensor。
        // masked_fill(&causal.logical_not(), NEG_INFINITY):
        //   把 causal=False 的位置填入负无穷，softmax 后权重变为 0。
        if apply_causal_mask {
            // MPS backend has bugs with masked_fill and arange on device.
            // Build position tensors on CPU then move to device; use add+mul instead of masked_fill.
            let q_pos_t = q_pos
                .unsqueeze(1)
                .unsqueeze(0)
                .unsqueeze(0)
                .to_device(q_chunk.device());
            let k_pos_t = k_pos
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to_device(q_chunk.device());
            let causal = q_pos_t.ge_tensor(&k_pos_t);
            // MPS masked_fill bug workaround: use where_self instead of add+mul.
            // add+mul (logical_not().to_kind(Float) * NEG_INFINITY) produces NaN
            // because 0.0 * NEG_INFINITY = NaN in IEEE 754.
            let q_kind = q_chunk.kind();
            let neg_inf = Tensor::from(f64::NEG_INFINITY)
                .to_kind(q_kind)
                .to_device(q_chunk.device());
            let zero = Tensor::zeros(1, (q_kind, q_chunk.device()));
            let neg_inf_mask = neg_inf.where_self(&causal.logical_not(), &zero);
            scores += neg_inf_mask;
        }

        // ====== 第三步：Online Softmax 更新 ======
        // Ring Attention 的核心技巧：不需要等所有 KV block 都处理完再算 softmax，
        // 而是每处理一个 KV block 就更新一次结果，始终保持正确的 softmax 输出。
        //
        // 标准 softmax 公式：
        //   softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
        //
        // 但如果直接对很多个 block 分别做 softmax 再平均，结果是错的。
        // Online softmax 的关键：维护一个"全局最大值"和"全局累加和"，
        // 每来一个新 block，就根据新的最大值重新调整之前的累加和。
        //
        // 数学推导：
        // 假设之前处理了一些 KV，得到：
        //   prev_max = max(prev_scores)
        //   prev_sum = sum(exp(prev_scores - prev_max))
        //   prev_out = sum(exp(prev_scores - prev_max) * V_prev) / prev_sum
        //
        // 现在来了一个新 block，得到：
        //   local_max = max(local_scores)
        //   local_sum = sum(exp(local_scores - local_max))
        //   local_pv = sum(exp(local_scores - local_max) * V_local)
        //
        // 合并后的全局最大值：new_max = max(prev_max, local_max)
        //
        // 为了合并，需要把之前的值"对齐"到新的最大值：
        //   exp_prev = exp(prev_max - new_max)   （之前的值需要缩小多少）
        //   exp_local = exp(local_max - new_max) （新值需要缩小多少）
        //
        // 新的累加和：new_sum = exp_prev * prev_sum + exp_local * local_sum
        //
        // 新的输出：
        //   new_out = (exp_prev * prev_sum * prev_out + exp_local * local_pv) / new_sum
        //           = (exp_prev.unsqueeze(3) * rs.unsqueeze(3) * obh + exp_local.unsqueeze(3) * local_pv) / new_sum.unsqueeze(3)

        // local_max: 当前 KV block 中每个 query 的最大 score。
        // amax([3], false): 在第 3 维（kv_chunk_len 维）上取最大值，不保留维度。
        // 使用 amax 代替 max_dim 避免 MPS 后端对 argmax 的 bug。
        // shape: [batch, num_heads, q_chunk_len]
        let local_max = scores.amax(&[3i64][..], false);

        // 处理因果掩码下的"全屏蔽 query"：如果某个 query 在当前 block 中所有 KV
        // 都位于未来（scores 全为 -inf），local_max 会是 -inf。直接计算
        // exp(-inf - (-inf)) 会得到 NaN，进而污染整个状态。这里把 local_max 中的
        // -inf 替换为 0，使得 weights 在该 query 上全为 0，local_sum/local_pv 也为 0，
        // 同时保留原始 local_max = -inf 用于后续 exp_local = 0 的缩放。
        let q_kind = q_chunk.kind();
        let neg_inf_t = Tensor::from(f64::NEG_INFINITY)
            .to_kind(q_kind)
            .to_device(q_chunk.device());
        let all_masked = local_max.eq_tensor(&neg_inf_t);
        let local_max_safe = local_max.masked_fill(&all_masked, 0.0);

        // weights: 当前 block 的 score 减去最大值后取指数。
        // 减最大值是为了数值稳定性（防止 exp 爆炸）。
        // shape: [batch, num_heads, q_chunk_len, kv_chunk_len]
        let weights = (&scores - local_max_safe.unsqueeze(3)).exp();

        // local_sum: 当前 block 的权重之和（softmax 的分母的一部分）。
        // sum_dim_intlist(&[3i64][..], false, q_chunk.kind()): 在第 3 维求和。
        // shape: [batch, num_heads, q_chunk_len]
        let local_sum = weights.sum_dim_intlist(&[3i64][..], false, q_kind);

        // local_pv: 当前 block 的加权 Value 之和（softmax 的分子的一部分）。
        // weights [batch, num_heads, q_chunk_len, kv_chunk_len] @ V [batch, num_heads, kv_chunk_len, head_dim]
        // = [batch, num_heads, q_chunk_len, head_dim]
        let local_pv = weights.matmul(v_chunk);

        // new_max: 全局最大值 = max(之前见过的最大值, 当前 block 的最大值)
        let new_max = rm.max_other(&local_max);

        // exp_prev: 把之前的状态缩放到新的最大值下。
        // &*rm 中的 * 是解引用（因为 rm 是 &mut Tensor，需要解引用才能做减法）。
        let exp_prev = (&*rm - &new_max).exp();

        // exp_local: 把当前 block 的状态缩放到新的最大值下。
        let exp_local = (&local_max - &new_max).exp();

        // new_sum: 合并后的 softmax 分母。
        // &*rs 同理，解引用 rs（&mut Tensor）得到 Tensor。
        let new_sum = &exp_prev * &*rs + &exp_local * &local_sum;
        // 当某个 query 在当前及之前所有 block 中都被屏蔽时，new_sum 可能为 0。
        // 直接除以 0 会产生 NaN；将该位置替换为 1，此时分子也为 0，输出保持 0。
        let new_sum_safe = new_sum.where_self(&new_sum.ne(0.0), &Tensor::ones_like(&new_sum));

        // 更新输出 obh。
        // 公式：new_out = (exp_prev * prev_sum * prev_out + exp_local * local_pv) / new_sum
        // unsqueeze(3) 把 [batch, num_heads, q_chunk_len] 变成 [batch, num_heads, q_chunk_len, 1]，
        // 以便和 [batch, num_heads, q_chunk_len, head_dim] 做 element-wise 乘法（广播）。
        *obh = (&exp_prev.unsqueeze(3) * &rs.unsqueeze(3) * &*obh
            + &exp_local.unsqueeze(3) * &local_pv)
            / &new_sum_safe.unsqueeze(3);

        // 更新 running max 和 running sum，供下一个 KV block 使用。
        *rm = new_max;
        *rs = new_sum;
    }

    /// 将单个 perf event 以 JSONL 格式追加到 `HCP_PERF_LOG` 指定的文件。
    #[cfg(feature = "tch-backend")]
    fn emit_perf_event(&self, event_type: &str, fields: &[(&str, &str)]) {
        if let Ok(path) = std::env::var("HCP_PERF_LOG") {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            let ts = format!("{}.{:03}Z", now.as_secs(), now.subsec_millis());
            let mut line = format!(
                "{{\"ts\":\"{}\",\"event\":\"{}\",\"domain\":{},\"layer\":{}",
                ts, event_type, self.local_domain_id, self.layer_idx
            );
            for (k, v) in fields {
                line.push_str(&format!(",\"{}\":{}", k, v));
            }
            line.push_str("}\n");
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                let _ = file.write_all(line.as_bytes());
            }
        }
    }

    /// 【Ring Attention 核心算法 — Split-Phase Pipeline + micro_KV_block 版】
    ///
    /// 【micro_KV_block 设计动机】
    /// 在异构分布式场景中（如 Mac MPS + RTX 4090），各 domain 的 seq 长度差异巨大：
    /// - 大 domain 可能产生 50MB+ 的 KV block
    /// - 小 domain 显存/内存有限，无法一次性缓冲这么大的 block
    /// - 把整个 KV block 切成 micro blocks（如 512/1024/2048 tokens）后：
    ///   1. 小 domain 可以逐个接收 micro block，不需要一次性缓冲整个大 block
    ///   2. 传输可以流水化：发送 micro block 0 → 接收方处理 → 发送 micro block 1 → ...
    ///   3. 提升并发性：多个 micro blocks 可以交错传输
    ///
    /// 【两种运行模式】
    /// - Pipeline 模式（默认）：Phase 0 submit_send → Phase 1 本地 compute → Phase 2 循环 recv→process→转发
    /// - 串行模式（HCP_DISABLE_OVERLAP=1）：先全部 exchange 完，再统一 compute，用于 baseline 对比测试
    fn ring_attention(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        global_seq_start: usize,
    ) -> Result<Tensor, ModelError> {
        let batch = q.size()[0];
        let num_heads = q.size()[1];
        let seq_len = q.size()[2];
        let head_dim = q.size()[3];

        let ring_start = Instant::now();
        let mut perf_local_compute_ms = 0.0_f64;
        let mut perf_peer_compute_ms = 0.0_f64;
        let mut perf_recv_ms = 0.0_f64;
        let mut perf_send_ms = 0.0_f64;
        let mut perf_forward_ms = 0.0_f64;
        let mut perf_flush_ms = 0.0_f64;
        let mut perf_kv_sent_bytes: usize = 0;
        let mut perf_kv_recv_bytes: usize = 0;
        let mut perf_micro_block_count: usize = 0;

        // ====== 确定 chunk 大小 ======
        let q_chunk_size = (seq_len as usize)
            .div_ceil(self.num_domains)
            .clamp(1, 2048);
        let kv_chunk_size = if seq_len == 1 { 2048 } else { q_chunk_size };

        // micro KV block 大小：默认使用 kv_chunk_size，可通过 HCP_MICRO_KV_BLOCK_SIZE 覆盖
        // 弱网络下小 micro block 的 QUIC round-trip overhead 很大，默认至少 1024 tokens
        // 在保持一定流水粒度的同时减少往返次数。
        let micro_kv_block_size = if self.micro_kv_block_size > 0 {
            self.micro_kv_block_size
        } else {
            kv_chunk_size.max(1024)
        };

        let local_kv_len = k.size()[2] as usize;
        let kv_chunks: Vec<(usize, usize)> = (0..local_kv_len)
            .step_by(kv_chunk_size)
            .map(|start| (start, (start + kv_chunk_size).min(local_kv_len)))
            .collect();

        let apply_causal = attention_mask.is_some();

        // ====== 预创建所有 Q chunk 的状态 ======
        struct QChunkState {
            q_chunk: Tensor,
            q_pos: Tensor, // [q_chunk_len] Int64，原始全局位置
            rm: Tensor,
            rs: Tensor,
            obh: Tensor,
        }

        let mut q_states: Vec<QChunkState> = Vec::new();
        for q_start in (0..seq_len as usize).step_by(q_chunk_size) {
            let q_end = (q_start + q_chunk_size).min(seq_len as usize);
            let q_chunk_len = (q_end - q_start) as i64;
            let q_chunk = q.narrow(2, q_start as i64, q_chunk_len);
            let q_pos = match self.position_ids.as_ref() {
                Some(pos) => pos.narrow(0, q_start as i64, q_chunk_len),
                None => Tensor::arange_start(
                    (global_seq_start + q_start) as i64,
                    (global_seq_start + q_end) as i64,
                    (Kind::Int64, Device::Cpu),
                ),
            };
            let q_kind = q.kind();
            q_states.push(QChunkState {
                q_chunk,
                q_pos,
                // 初始 running max 用一个有限大负数而非 -inf。
                // 原因：Striped / 非连续分片下，某个 query 可能在第一个 block 就全被因果掩码屏蔽；
                // 若 rm = -inf，online softmax 的 exp(-inf - (-inf)) 会产生 NaN。
                // -1e4 对 Half/BFloat16/Float/Double 都可表示，且 exp(-1e4) 在所有浮点格式下都为 0。
                rm: Tensor::full([batch, num_heads, q_chunk_len], -1e4_f64, (q_kind, q.device())),
                rs: Tensor::zeros([batch, num_heads, q_chunk_len], (q_kind, q.device())),
                obh: Tensor::zeros([batch, num_heads, q_chunk_len, head_dim], (q_kind, q.device())),
            });
        }

        // ====== 准备本地 KV micro blocks ======
        let has_transport = self.kv_transport.is_some();
        let local_micro_blocks: Vec<KvBlock> = if has_transport {
            let (k_to_send, v_to_send, _send_seq_end) = if seq_len == 1 {
                let history_len = self.prefill_kv_len as i64;
                (
                    k.narrow(2, 0, history_len),
                    v.narrow(2, 0, history_len),
                    global_seq_start + self.prefill_kv_len,
                )
            } else {
                (k.shallow_clone(), v.shallow_clone(), global_seq_start + k.size()[2] as usize)
            };
            let send_kv_len = k_to_send.size()[2] as usize;
            let num_micro = send_kv_len.div_ceil(micro_kv_block_size);
            (0..num_micro)
                .map(|i| {
                    let start = i * micro_kv_block_size;
                    let end = ((i + 1) * micro_kv_block_size).min(send_kv_len);
                    let len_i64 = (end - start) as i64;
                    let pos_ids = self.position_ids.as_ref().map(|pos| pos.narrow(0, start as i64, len_i64));
                    KvBlock {
                        layer_idx: self.layer_idx,
                        global_seq_start: global_seq_start + start,
                        global_seq_end: global_seq_start + end,
                        k: k_to_send.narrow(2, start as i64, len_i64),
                        v: v_to_send.narrow(2, start as i64, len_i64),
                        micro_block_idx: i,
                        total_micro_blocks: num_micro,
                        position_ids: pos_ids,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // 构造本地 KV chunk 的原始位置 id（连续或非连续分片）
        let build_k_pos = |start: usize, end: usize| -> Tensor {
            match self.position_ids.as_ref() {
                Some(pos) => pos.narrow(0, start as i64, (end - start) as i64),
                None => Tensor::arange_start(
                    (global_seq_start + start) as i64,
                    (global_seq_start + end) as i64,
                    (Kind::Int64, Device::Cpu),
                ),
            }
        };

        // 构造 peer KvBlock 的原始位置 id
        let peer_k_pos = |block: &KvBlock| -> Tensor {
            match block.position_ids.as_ref() {
                Some(pos) => pos.shallow_clone(),
                None => Tensor::arange_start(
                    block.global_seq_start as i64,
                    block.global_seq_end as i64,
                    (Kind::Int64, Device::Cpu),
                ),
            }
        };

        // ====== 接收一个 micro block（poll_recv + recv_kv_block fallback）======
        let recv_micro_block = |transport: &mut dyn KvTransport| -> Result<Option<KvBlock>, String> {
            match transport.poll_recv() {
                Ok(Some(block)) => Ok(Some(block)),
                Ok(None) => match transport.recv_kv_block() {
                    Ok(Some(block)) => Ok(Some(block)),
                    Ok(None) => Ok(None),
                    Err(e) => Err(e),
                },
                Err(e) => Err(e),
            }
        };

        if has_transport {
            let num_rounds = self.num_domains.saturating_sub(1);

            // 统计本地待发送 KV bytes（发送端+转发端都会复用）
            let local_elem_bytes = match k.kind() {
                Kind::Float => 4,
                Kind::Half => 2,
                Kind::BFloat16 => 2,
                Kind::Double => 8,
                _ => 4,
            };
            perf_kv_sent_bytes = local_micro_blocks.iter()
                .map(|b| b.k.numel() * local_elem_bytes + b.v.numel() * local_elem_bytes)
                .sum();

            if self.disable_overlap {
                // ====== 【串行模式】先全部 exchange，再统一 compute ======
                let all_peer_micro_blocks: Vec<Vec<KvBlock>> = {
                    let transport = self.kv_transport.as_mut().unwrap();

                    // 1. 发送所有本地 micro blocks
                    let send_start = Instant::now();
                    for micro_block in &local_micro_blocks {
                        transport.submit_send(micro_block).map_err(|e| ModelError::Backend(format!("submit_send: {e}")))?;
                    }
                    transport.flush_send().map_err(|e| ModelError::Backend(format!("flush_send: {e}")))?;
                    perf_send_ms = send_start.elapsed().as_secs_f64() * 1000.0;

                    // 2. 收集所有 peer micro blocks（所有 rounds）
                    let recv_start = Instant::now();
                    let mut all_blocks: Vec<Vec<KvBlock>> = Vec::new();
                    for round in 0..num_rounds {
                        let mut round_blocks = Vec::new();
                        while let Some(block) = recv_micro_block(transport.as_mut()).map_err(|e| ModelError::Backend(format!("recv_micro_block: {e}")))? {
                            let elem_bytes = match block.k.kind() {
                                Kind::Float => 4,
                                Kind::Half => 2,
                                Kind::BFloat16 => 2,
                                Kind::Double => 8,
                                _ => 4,
                            };
                            perf_kv_recv_bytes += block.k.numel() * elem_bytes + block.v.numel() * elem_bytes;
                            let is_last = block.micro_block_idx + 1 == block.total_micro_blocks;
                            round_blocks.push(block);
                            if is_last { break; }
                        }
                        all_blocks.push(round_blocks);

                        // 转发（最后一轮不需要）
                        if round < num_rounds - 1 {
                            let fwd_start = Instant::now();
                            for block in &all_blocks[round] {
                                let elem_bytes = match block.k.kind() {
                                    Kind::Float => 4,
                                    Kind::Half => 2,
                                    Kind::BFloat16 => 2,
                                    Kind::Double => 8,
                                    _ => 4,
                                };
                                perf_kv_sent_bytes += block.k.numel() * elem_bytes + block.v.numel() * elem_bytes;
                                transport.submit_send(block).map_err(|e| ModelError::Backend(format!("forward submit_send: {e}")))?;
                            }
                            transport.flush_send().map_err(|e| ModelError::Backend(format!("flush_send: {e}")))?;
                            perf_forward_ms += fwd_start.elapsed().as_secs_f64() * 1000.0;
                        }
                    }
                    perf_recv_ms = recv_start.elapsed().as_secs_f64() * 1000.0;
                    perf_micro_block_count = all_blocks.iter().map(|r| r.len()).sum();
                    all_blocks
                }; // transport borrow 结束

                // 3. 本地 KV compute
                let local_compute_start = Instant::now();
                for state in q_states.iter_mut() {
                    for (kv_start, kv_end) in &kv_chunks {
                        let kv_chunk_len = (*kv_end - *kv_start) as i64;
                        let k_chunk = k.narrow(2, *kv_start as i64, kv_chunk_len);
                        let v_chunk = v.narrow(2, *kv_start as i64, kv_chunk_len);
                        let k_pos = build_k_pos(*kv_start, *kv_end);
                        self.process_kv_block(
                            &state.q_chunk, &state.q_pos,
                            &k_chunk, &v_chunk, &k_pos,
                            &mut state.rm, &mut state.rs, &mut state.obh,
                            apply_causal,
                        );
                    }
                }
                perf_local_compute_ms = local_compute_start.elapsed().as_secs_f64() * 1000.0;

                // 4. Peer KV compute（按 round 顺序，每个 round 内按 seq 排序）
                let peer_compute_start = Instant::now();
                for round_blocks in &all_peer_micro_blocks {
                    let mut sorted = round_blocks.clone();
                    sorted.sort_by_key(|b| b.global_seq_start);
                    for micro_block in &sorted {
                        let k_pos = peer_k_pos(micro_block);
                        for state in q_states.iter_mut() {
                            self.process_kv_block(
                                &state.q_chunk, &state.q_pos,
                                &micro_block.k, &micro_block.v, &k_pos,
                                &mut state.rm, &mut state.rs, &mut state.obh,
                                apply_causal,
                            );
                        }
                    }
                }
                perf_peer_compute_ms = peer_compute_start.elapsed().as_secs_f64() * 1000.0;
            } else {
                // ====== 【Pipeline 模式】compute-communication overlap ======
                {
                    let transport = self.kv_transport.as_mut().unwrap();
                    // Phase 0: 逐个 submit_send 本地 micro blocks（send task 后台传输）
                    let phase0_start = Instant::now();
                    for micro_block in &local_micro_blocks {
                        transport.submit_send(micro_block).map_err(|e| ModelError::Backend(format!("submit_send: {e}")))?;
                    }
                    perf_send_ms = phase0_start.elapsed().as_secs_f64() * 1000.0;
                } // transport borrow 结束

                // Phase 1: 本地 KV compute（与 Phase 0 的网络传输重叠）
                let phase1_start = Instant::now();
                for state in q_states.iter_mut() {
                    for (kv_start, kv_end) in &kv_chunks {
                        let kv_chunk_len = (*kv_end - *kv_start) as i64;
                        let k_chunk = k.narrow(2, *kv_start as i64, kv_chunk_len);
                        let v_chunk = v.narrow(2, *kv_start as i64, kv_chunk_len);
                        let k_pos = build_k_pos(*kv_start, *kv_end);
                        self.process_kv_block(
                            &state.q_chunk, &state.q_pos,
                            &k_chunk, &v_chunk, &k_pos,
                            &mut state.rm, &mut state.rs, &mut state.obh,
                            apply_causal,
                        );
                    }
                }
                perf_local_compute_ms = phase1_start.elapsed().as_secs_f64() * 1000.0;

                // Phase 2: 循环接收 peer micro blocks，逐个 process，转发
                // 【streaming compute】收到一个 block 就立刻处理，不等全部收完。
                // 通过 inner scope 释放 transport borrow，让 process_kv_block 可以获取 &self。
                // 原因：process_kv_block 需要 &self，而 kv_transport 持有 &mut self，
                // 两者不能同时存在。inner scope 每轮释放 transport borrow。
                let mut total_recv_time = std::time::Duration::ZERO;
                let mut total_compute_time = std::time::Duration::ZERO;
                let mut micro_block_count = 0usize;
                for round in 0..num_rounds {
                    let mut expected_micro_idx = 0;
                    loop {
                        // Step 1: 接收一个 micro block（inner scope 释放 transport borrow）
                        let recv_start = std::time::Instant::now();
                        let block = {
                            let transport = self.kv_transport.as_mut().unwrap();
                            match recv_micro_block(transport.as_mut()).map_err(|e| ModelError::Backend(format!("recv_micro_block: {e}")))? {
                                Some(block) => block,
                                None => break,
                            }
                        }; // transport borrow 结束
                        let recv_elapsed = recv_start.elapsed();
                        total_recv_time += recv_elapsed;

                        // 动态计算 element size（支持 Float/Half/BFloat16/Double）
                        let elem_bytes = match block.k.kind() {
                            Kind::Float => 4,
                            Kind::Half => 2,
                            Kind::BFloat16 => 2,
                            Kind::Double => 8,
                            _ => 4,
                        };
                        let recv_bytes = block.k.numel() * elem_bytes + block.v.numel() * elem_bytes;
                        perf_kv_recv_bytes += recv_bytes;
                        let is_last = block.micro_block_idx + 1 == block.total_micro_blocks;
                        // 防御性断言：micro blocks 必须按顺序到达（QUIC stream 保证有序）
                        debug_assert_eq!(block.micro_block_idx, expected_micro_idx, "micro blocks must arrive in order; expected {expected_micro_idx}, got {}", block.micro_block_idx);
                        expected_micro_idx += 1;
                        if is_last {
                            println!("[ring_attention] round {round} layer {}: received micro_block {}/{}, {recv_bytes} bytes (last)",
                                self.layer_idx, block.micro_block_idx + 1, block.total_micro_blocks);
                        }

                        // Step 2: 立刻处理这个 block（transport borrow 已释放）
                        let compute_start = std::time::Instant::now();
                        let k_pos = peer_k_pos(&block);
                        for state in q_states.iter_mut() {
                            self.process_kv_block(
                                &state.q_chunk, &state.q_pos,
                                &block.k, &block.v, &k_pos,
                                &mut state.rm, &mut state.rs, &mut state.obh,
                                apply_causal,
                            );
                        }
                        let compute_elapsed = compute_start.elapsed();
                        total_compute_time += compute_elapsed;
                        micro_block_count += 1;

                        // Step 3: 转发（最后一轮不需要）
                        if round < num_rounds.saturating_sub(1) {
                            let fwd_start = std::time::Instant::now();
                            let transport = self.kv_transport.as_mut().unwrap();
                            transport.submit_send(&block).map_err(|e| ModelError::Backend(format!("forward submit_send: {e}")))?;
                            perf_forward_ms += fwd_start.elapsed().as_secs_f64() * 1000.0;
                            perf_kv_sent_bytes += recv_bytes;
                        }

                        if is_last { break; }
                    }
                }
                if micro_block_count > 0 {
                    let avg_recv_ms = total_recv_time.as_secs_f64() * 1000.0 / micro_block_count as f64;
                    let avg_compute_ms = total_compute_time.as_secs_f64() * 1000.0 / micro_block_count as f64;
                    // overlap hint: 如果 compute > recv 说明网络没拖后腿；反之说明网络是瓶颈
                    let overlap_ratio = if total_compute_time.as_secs_f64() > 0.0 {
                        total_recv_time.as_secs_f64() / total_compute_time.as_secs_f64()
                    } else { 0.0 };
                    println!("[ring_attention] layer {} Phase 2 summary: {micro_block_count} micro blocks, avg recv={avg_recv_ms:.2}ms, avg compute={avg_compute_ms:.2}ms, recv/compute={overlap_ratio:.2}x",
                        self.layer_idx);
                }
                perf_recv_ms = total_recv_time.as_secs_f64() * 1000.0;
                perf_peer_compute_ms = total_compute_time.as_secs_f64() * 1000.0;
                perf_micro_block_count = micro_block_count;

                // Phase 3: Flush
                {
                    let flush_start = Instant::now();
                    let transport = self.kv_transport.as_mut().unwrap();
                    transport.flush_send().map_err(|e| ModelError::Backend(format!("flush_send: {e}")))?;
                    perf_flush_ms = flush_start.elapsed().as_secs_f64() * 1000.0;
                }
            }
        } else {
            // 无 transport：只做本地 KV compute
            for state in q_states.iter_mut() {
                for (kv_start, kv_end) in &kv_chunks {
                    let kv_chunk_len = (*kv_end - *kv_start) as i64;
                    let k_chunk = k.narrow(2, *kv_start as i64, kv_chunk_len);
                    let v_chunk = v.narrow(2, *kv_start as i64, kv_chunk_len);
                    let k_pos = build_k_pos(*kv_start, *kv_end);
                    self.process_kv_block(
                        &state.q_chunk, &state.q_pos,
                        &k_chunk, &v_chunk, &k_pos,
                        &mut state.rm, &mut state.rs, &mut state.obh,
                        apply_causal,
                    );
                }
            }
        }

        // ====== Phase 4: 提取输出 & perf 上报 ======
        let total_ms = ring_start.elapsed().as_secs_f64() * 1000.0;
        if has_transport {
            self.emit_perf_event(
                "ring_attention",
                &[
                    ("total_ms", &format!("{:.3}", total_ms)),
                    ("local_compute_ms", &format!("{:.3}", perf_local_compute_ms)),
                    ("peer_compute_ms", &format!("{:.3}", perf_peer_compute_ms)),
                    ("recv_ms", &format!("{:.3}", perf_recv_ms)),
                    ("send_ms", &format!("{:.3}", perf_send_ms)),
                    ("forward_ms", &format!("{:.3}", perf_forward_ms)),
                    ("flush_ms", &format!("{:.3}", perf_flush_ms)),
                    ("kv_sent_bytes", &perf_kv_sent_bytes.to_string()),
                    ("kv_recv_bytes", &perf_kv_recv_bytes.to_string()),
                    ("micro_blocks", &perf_micro_block_count.to_string()),
                    ("num_domains", &self.num_domains.to_string()),
                    ("seq_len", &seq_len.to_string()),
                    ("overlap", &if self.disable_overlap { "0" } else { "1" }),
                ],
            );
        }
        let outputs: Vec<Tensor> = q_states.into_iter().map(|s| s.obh).collect();
        Ok(Tensor::cat(&outputs, 2))
    }

    /// Standard local attention for short sequences or single-token decode.
    #[allow(dead_code)]
    fn local_attention_scores(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Tensor {
        let scores = q.matmul(&k.transpose(2, 3)) * self.scale;

        let scores = if let Some(mask) = attention_mask {
            scores + mask
        } else {
            scores
        };

        let attn_weights = scores.softmax(-1, scores.kind());
        attn_weights.matmul(v)
    }
}

#[cfg(feature = "tch-backend")]
impl HcpRingAttentionBackend {
    /// 【Debug forward】逐层导出 attention 内部中间结果，用于定位数值 divergence。
    ///
    /// 与 `forward` 计算完全一致，但在每个关键步骤后导出 tensor 到 `export_dir`：
    /// - `q_proj_layer_{i}.bin` / `k_proj_layer_{i}.bin` / `v_proj_layer_{i}.bin`
    ///   —— linear projection 后、RoPE 前的 Q/K/V
    /// - `q_rope_layer_{i}.bin` / `k_rope_layer_{i}.bin`
    ///   —— RoPE 后的 Q/K
    /// - `k_cache_layer_{i}.bin` / `v_cache_layer_{i}.bin`
    ///   —— KV cache update 后的 K/V（即写入 cache 前的值）
    /// - `attn_out_layer_{i}.bin`
    ///   —— ring_attention 输出（O-projection 前）
    /// - `attn_final_layer_{i}.bin`
    ///   —— O-projection 后的最终 attention 输出
    ///
    /// 所有 tensor 使用与 `LlamaModel::write_tensor_as_binary` 相同的二进制格式。
    pub fn forward_debug(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut dyn crate::model::cache::KvCache>,
        attention_mask: Option<&Tensor>,
        export_dir: &std::path::Path,
    ) -> Result<Tensor, ModelError> {
        let batch = hidden_states.size()[0];
        let seq_len = hidden_states.size()[1];
        let hidden_size = hidden_states.size()[2];

        // Step 1: Linear projection
        let mut q = hidden_states.matmul(&self.q_proj.transpose(0, 1));
        if let Some(ref bias) = self.q_bias { q += bias; }
        let mut k = hidden_states.matmul(&self.k_proj.transpose(0, 1));
        if let Some(ref bias) = self.k_bias { k += bias; }
        let mut v = hidden_states.matmul(&self.v_proj.transpose(0, 1));
        if let Some(ref bias) = self.v_bias { v += bias; }

        crate::model::model::LlamaModel::write_tensor_as_binary(&q, &export_dir.join(format!("q_proj_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export q_proj: {}", e)))?;
        crate::model::model::LlamaModel::write_tensor_as_binary(&k, &export_dir.join(format!("k_proj_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export k_proj: {}", e)))?;
        crate::model::model::LlamaModel::write_tensor_as_binary(&v, &export_dir.join(format!("v_proj_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export v_proj: {}", e)))?;

        // Step 2: Reshape
        let q = q.view([batch, seq_len, self.num_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let k = k.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let v = v.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);

        // Step 3: RoPE
        let (q, k) = self.rope.apply(&q, &k, Some(position_ids));

        crate::model::model::LlamaModel::write_tensor_as_binary(&q, &export_dir.join(format!("q_rope_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export q_rope: {}", e)))?;
        crate::model::model::LlamaModel::write_tensor_as_binary(&k, &export_dir.join(format!("k_rope_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export k_rope: {}", e)))?;

        // Step 4: KV Cache update
        let (k_cached, v_cached) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?
        } else {
            (k.shallow_clone(), v.shallow_clone())
        };

        if !self.is_prefill_done {
            self.prefill_kv_len = k_cached.size()[2] as usize;
            self.is_prefill_done = true;
        }

        crate::model::model::LlamaModel::write_tensor_as_binary(&k_cached, &export_dir.join(format!("k_cache_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export k_cache: {}", e)))?;
        crate::model::model::LlamaModel::write_tensor_as_binary(&v_cached, &export_dir.join(format!("v_cache_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export v_cache: {}", e)))?;

        // Step 5: GQA head repeat
        let num_rep = self.num_heads / self.num_kv_heads;
        let k_cached = if num_rep > 1 {
            let shape = k_cached.size();
            k_cached.unsqueeze(2)
                .expand([shape[0], shape[1], num_rep as i64, shape[2], shape[3]], false)
                .reshape([shape[0], shape[1] * num_rep as i64, shape[2], shape[3]])
        } else {
            k_cached
        };
        let v_cached = if num_rep > 1 {
            let shape = v_cached.size();
            v_cached.unsqueeze(2)
                .expand([shape[0], shape[1], num_rep as i64, shape[2], shape[3]], false)
                .reshape([shape[0], shape[1] * num_rep as i64, shape[2], shape[3]])
        } else {
            v_cached
        };

        // Step 6: Ring Attention
        let global_seq_start = self.seq_offset;
        let attn_output = self.ring_attention(&q, &k_cached, &v_cached, attention_mask, global_seq_start)?;

        crate::model::model::LlamaModel::write_tensor_as_binary(&attn_output, &export_dir.join(format!("attn_out_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export attn_out: {}", e)))?;

        // Step 7: O-projection
        let attn_output = attn_output.transpose(1, 2).contiguous().view([batch, seq_len, hidden_size]);
        let result = attn_output.matmul(&self.o_proj.transpose(0, 1));

        crate::model::model::LlamaModel::write_tensor_as_binary(&result, &export_dir.join(format!("attn_final_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export attn_final: {}", e)))?;

        Ok(result)
    }

    /// 【Debug forward with injected Q/K】
    ///
    /// Same as `forward_debug`, but loads Q and K projections from Python-exported files
    /// instead of computing them via matmul. V projection is still computed normally.
    ///
    /// This is to verify whether the Q/K projection BLAS difference is the root cause
    /// of downstream divergence.
    ///
    /// `qk_dir`: directory containing `q_proj_layer_{i}.bin` and `k_proj_layer_{i}.bin`
    ///           in the standard binary format [ndims][dims...][f32 data...]
    pub fn forward_debug_with_injected_qk(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut dyn crate::model::cache::KvCache>,
        attention_mask: Option<&Tensor>,
        export_dir: &std::path::Path,
        qk_dir: &std::path::Path,
    ) -> Result<Tensor, ModelError> {
        let batch = hidden_states.size()[0];
        let seq_len = hidden_states.size()[1];
        let hidden_size = hidden_states.size()[2];

        // Step 1: Load Q and K from Python export, compute V normally
        let q_path = qk_dir.join(format!("q_proj_layer_{}.bin", self.layer_idx));
        let k_path = qk_dir.join(format!("k_proj_layer_{}.bin", self.layer_idx));

        let q = Self::read_tensor_as_binary(&q_path, hidden_states.device())
            .map_err(|e| ModelError::Generation(format!("failed to read injected q_proj: {}", e)))?;
        let k = Self::read_tensor_as_binary(&k_path, hidden_states.device())
            .map_err(|e| ModelError::Generation(format!("failed to read injected k_proj: {}", e)))?;

        // Compute V normally
        let mut v = hidden_states.matmul(&self.v_proj.transpose(0, 1));
        if let Some(ref bias) = self.v_bias { v += bias; }

        crate::model::model::LlamaModel::write_tensor_as_binary(&q, &export_dir.join(format!("q_proj_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export q_proj: {}", e)))?;
        crate::model::model::LlamaModel::write_tensor_as_binary(&k, &export_dir.join(format!("k_proj_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export k_proj: {}", e)))?;
        crate::model::model::LlamaModel::write_tensor_as_binary(&v, &export_dir.join(format!("v_proj_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export v_proj: {}", e)))?;

        // Step 2: Reshape (if needed)
        // Python exports q/k as [batch, num_heads, seq_len, head_dim] (already transposed)
        // Rust computes them as [batch, seq_len, num_heads * head_dim] then reshapes
        let q = if q.size().len() == 4 {
            q  // Already [batch, num_heads, seq_len, head_dim]
        } else {
            q.view([batch, seq_len, self.num_heads as i64, self.head_dim as i64])
                .transpose(1, 2)
        };
        let k = if k.size().len() == 4 {
            k
        } else {
            k.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
                .transpose(1, 2)
        };
        let v = v.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);

        // Step 3: RoPE
        let (q, k) = self.rope.apply(&q, &k, Some(position_ids));

        crate::model::model::LlamaModel::write_tensor_as_binary(&q, &export_dir.join(format!("q_rope_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export q_rope: {}", e)))?;
        crate::model::model::LlamaModel::write_tensor_as_binary(&k, &export_dir.join(format!("k_rope_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export k_rope: {}", e)))?;

        // Step 4: KV Cache update
        let (k_cached, v_cached) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?
        } else {
            (k.shallow_clone(), v.shallow_clone())
        };

        if !self.is_prefill_done {
            self.prefill_kv_len = k_cached.size()[2] as usize;
            self.is_prefill_done = true;
        }

        crate::model::model::LlamaModel::write_tensor_as_binary(&k_cached, &export_dir.join(format!("k_cache_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export k_cache: {}", e)))?;
        crate::model::model::LlamaModel::write_tensor_as_binary(&v_cached, &export_dir.join(format!("v_cache_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export v_cache: {}", e)))?;

        // Step 5: GQA head repeat
        let num_rep = self.num_heads / self.num_kv_heads;
        let k_cached = if num_rep > 1 {
            let shape = k_cached.size();
            k_cached.unsqueeze(2)
                .expand([shape[0], shape[1], num_rep as i64, shape[2], shape[3]], false)
                .reshape([shape[0], shape[1] * num_rep as i64, shape[2], shape[3]])
        } else {
            k_cached
        };
        let v_cached = if num_rep > 1 {
            let shape = v_cached.size();
            v_cached.unsqueeze(2)
                .expand([shape[0], shape[1], num_rep as i64, shape[2], shape[3]], false)
                .reshape([shape[0], shape[1] * num_rep as i64, shape[2], shape[3]])
        } else {
            v_cached
        };

        // Step 6: Ring Attention
        let global_seq_start = self.seq_offset;
        let attn_output = self.ring_attention(&q, &k_cached, &v_cached, attention_mask, global_seq_start)?;

        crate::model::model::LlamaModel::write_tensor_as_binary(&attn_output, &export_dir.join(format!("attn_out_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export attn_out: {}", e)))?;

        // Step 7: O-projection
        let attn_output = attn_output.transpose(1, 2).contiguous().view([batch, seq_len, hidden_size]);
        let result = attn_output.matmul(&self.o_proj.transpose(0, 1));

        crate::model::model::LlamaModel::write_tensor_as_binary(&result, &export_dir.join(format!("attn_final_layer_{}.bin", self.layer_idx)))
            .map_err(|e| ModelError::Generation(format!("debug export attn_final: {}", e)))?;

        Ok(result)
    }

    /// Read a tensor from a binary file: [ndims: u64 LE][dim0: u64 LE]...[dimN: u64 LE][f32 data...]
    fn read_tensor_as_binary(path: &std::path::Path, device: Device) -> Result<Tensor, String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("open file: {}", e))?;
        let mut ndims_buf = [0u8; 8];
        file.read_exact(&mut ndims_buf).map_err(|e| e.to_string())?;
        let ndims = u64::from_le_bytes(ndims_buf) as i64;
        let mut shape = Vec::new();
        for _ in 0..ndims {
            let mut dim_buf = [0u8; 8];
            file.read_exact(&mut dim_buf).map_err(|e| e.to_string())?;
            shape.push(i64::from_le_bytes(dim_buf));
        }
        let numel: i64 = shape.iter().product();
        let mut data_buf = vec![0u8; (numel * 4) as usize];
        file.read_exact(&mut data_buf).map_err(|e| e.to_string())?;
        let data: Vec<f32> = data_buf.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(Tensor::from_slice(&data).to_device(device).reshape(&shape))
    }
}

#[cfg(feature = "tch-backend")]
impl AttentionBackend for HcpRingAttentionBackend {
    fn set_distributed(&mut self, domain_id: usize, seq_offset: usize, transport: Option<Box<dyn KvTransport>>) {
        self.local_domain_id = domain_id;
        self.seq_offset = seq_offset;
        self.kv_base_global_start = seq_offset;
        // Only replace transport if explicitly provided (None means "keep existing").
        if let Some(t) = transport {
            self.kv_transport = Some(t);
        }
        // Reset per-request state when seq_offset is explicitly updated (new prefill).
        // Without this, subsequent requests reuse stale prefill_kv_len from previous
        // requests, causing narrow() to fail when the new KV cache is shorter.
        self.is_prefill_done = false;
        self.prefill_kv_len = 0;
    }

    fn set_strategy(&mut self, strategy: RingSchedulingStrategy) {
        self.strategy = strategy;
    }
    fn forward(
        &mut self,
        hidden_states: &Tensor,      // 【输入】上一层的输出，shape: [batch, seq_len, hidden_size]
        position_ids: &Tensor,       // 【位置编码】每个 token 在完整序列中的绝对位置，shape: [batch, seq_len]
        kv_cache: Option<&mut dyn crate::model::cache::KvCache>, // 【KV 缓存】用于自回归生成时复用之前的 K/V
        attention_mask: Option<&Tensor>, // 【注意力掩码】因果掩码，防止当前 token 看到未来的 token
    ) -> Result<Tensor, ModelError> {
        // 获取输入的各个维度大小
        // batch: 批量大小（一次处理多少条句子，测试里通常是 1）
        // seq_len: 当前输入的 token 数量（比如 8）
        // hidden_size: 隐藏层维度（比如 32）
        let batch = hidden_states.size()[0];
        let seq_len = hidden_states.size()[1];
        let hidden_size = hidden_states.size()[2];

        // ====== 第一步：线性投影（Linear Projection）======
        // Attention 的核心思想：用三个不同的权重矩阵把输入 hidden_states 映射成 Q、K、V。
        // 
        // Q（Query）：当前 token 的"查询向量"，用来问"我和哪些过去的 token 相关？"
        // K（Key）：每个 token 的"关键词向量"，用来回答查询
        // V（Value）：每个 token 的"价值向量"，最终加权平均的就是 V
        //
        // 数学上：q = hidden_states @ W_q^T
        // matmul 是矩阵乘法，transpose(0, 1) 把权重矩阵转置（因为 PyTorch/HF 格式是 [out, in]）
        let mut q = hidden_states.matmul(&self.q_proj.transpose(0, 1));
        if let Some(ref bias) = self.q_bias { q += bias; }  // 如果有偏置，加上去
        let mut k = hidden_states.matmul(&self.k_proj.transpose(0, 1));
        if let Some(ref bias) = self.k_bias { k += bias; }
        let mut v = hidden_states.matmul(&self.v_proj.transpose(0, 1));
        if let Some(ref bias) = self.v_bias { v += bias; }

        // ====== 第二步：reshape 成多头格式 ======
        // 原始 shape: [batch, seq_len, num_heads * head_dim]
        // 先 view 成: [batch, seq_len, num_heads, head_dim]
        // 再 transpose(1, 2) 交换第 1、2 维，变成: [batch, num_heads, seq_len, head_dim]
        // 
        // 这样每个 head 独立处理一部分维度，可以并行计算。
        let q = q.view([batch, seq_len, self.num_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let k = k.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let v = v.view([batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);

        // ====== 第三步：RoPE（旋转位置编码）======
        // RoPE 给 Q 和 K 注入位置信息，让模型知道 token 的先后顺序。
        // 它根据 position_ids（全局位置）对 Q/K 的每一对维度做旋转变换。
        // 旋转角度取决于位置：位置越远，旋转角度越大。
        let (q, k) = self.rope.apply(&q, &k, Some(position_ids));

        // ====== 第四步：更新 KV Cache（可选）======
        // 在自回归生成时，每次只输入一个新 token，但需要看到之前所有 token 的 K/V。
        // KV Cache 把之前的 K/V 存起来，新 token 的 K/V append 到后面，避免重复计算。
        // 
        // 在 prefill 阶段（第一次处理完整 prompt），kv_cache 为 None，直接返回当前 K/V。
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?
        } else {
            (k.shallow_clone(), v.shallow_clone())
        };

        // 记录 prefill 阶段的 KV 长度。decode 阶段发送 peer KV 时，
        // 只发送 prefill 分区，不包含 decode 阶段 append 的新 token。
        if !self.is_prefill_done {
            self.prefill_kv_len = k.size()[2] as usize;
            self.is_prefill_done = true;
        }

        // ====== 第五步：GQA 头重复 ======
        // GQA（Group Query Attention）是一种节省显存的技术：
        // - 标准 MHA：num_kv_heads == num_heads（每个 query head 对应一个 key/value head）
        // - GQA：num_kv_heads < num_heads（多个 query head 共享同一个 key/value head）
        //
        // 例如 num_heads=4, num_kv_heads=1：
        // - 4 个 query head，但只有一个 key head 和一个 value head
        // - 为了让矩阵乘法维度匹配，需要把 K/V 在 head 维度上重复 4 次
        // - repeat(&[1, 4, 1, 1]) 表示：batch 不重复，head 重复 4 次，seq_len 不重复，head_dim 不重复
        let num_rep = self.num_heads / self.num_kv_heads;
        let k = if num_rep > 1 {
            let shape = k.size();
            k.unsqueeze(2)
                .expand([shape[0], shape[1], num_rep as i64, shape[2], shape[3]], false)
                .reshape([shape[0], shape[1] * num_rep as i64, shape[2], shape[3]])
        } else {
            k
        };
        let v = if num_rep > 1 {
            let shape = v.size();
            v.unsqueeze(2)
                .expand([shape[0], shape[1], num_rep as i64, shape[2], shape[3]], false)
                .reshape([shape[0], shape[1] * num_rep as i64, shape[2], shape[3]])
        } else {
            v
        };

        // ====== 第六步：计算全局序列起始位置 ======
        // 在分布式场景下，domain1 处理的是完整序列的后半段（比如 [8, 16)）。
        // 但 causal mask 必须使用"全局位置"才能正确判断哪些 K 对当前 Q 可见。
        // 
        // 例如 domain1 的 position_ids = [8, 9, ..., 15]，min = 8。
        // global_seq_start = 8 意味着本地索引 0 对应全局位置 8。
        // 如果没有传输层（单进程本地模式），global_seq_start = 0。
        // 使用固定的 seq_offset 作为全局序列起始位置。
        // 在 prefill 阶段，seq_offset = domain_id * chunk_size（如 domain0=0, domain1=8）。
        // 在 decode 阶段，seq_offset 保持不变（domain0 始终负责全局位置 0..8 的 KV）。
        // 不能用 position_ids.min()，因为 decode 阶段 position_ids 是当前 token 的全局位置，
        // 与该 domain 负责的历史 KV 的起始位置不同。
        let global_seq_start = self.seq_offset;

        // ====== 第七步：Ring Attention（核心分布式注意力计算）======
        // 把 Q 切成多个 chunk，逐个处理本地 KV 和 peer KV，用 online softmax 合并结果。
        let attn_output = self.ring_attention(&q, &k, &v, attention_mask, global_seq_start)?;

        // ====== 第八步：输出投影（O-projection）======
        // attn_output 的 shape: [batch, num_heads, seq_len, head_dim]
        // transpose(1, 2) → [batch, seq_len, num_heads, head_dim]
        // view → [batch, seq_len, hidden_size]（把多头的结果拼接起来）
        // 最后再乘一个 o_proj 权重矩阵，映射回 hidden_size 维度
        let attn_output = attn_output.transpose(1, 2).contiguous().view([batch, seq_len, hidden_size]);
        Ok(attn_output.matmul(&self.o_proj.transpose(0, 1)))
    }

    #[cfg(feature = "tch-backend")]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::attention::backend::LocalAttentionBackend;
    #[cfg(feature = "tch-backend")]
    use crate::model::cache::KvCache;

    #[cfg(feature = "tch-backend")]
    fn create_test_attention(device: tch::Device) -> crate::model::layers::GqaAttention {
        let hidden_size = 64i64;
        let num_heads = 8usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;

        crate::model::layers::GqaAttention {
            q_proj: Tensor::randn([(num_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            k_proj: Tensor::randn([(num_kv_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            v_proj: Tensor::randn([(num_kv_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            o_proj: Tensor::randn([hidden_size, (num_heads * head_dim) as i64], (Kind::Float, device)),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            num_heads,
            num_kv_heads,
            head_dim,
            rope: crate::model::layers::RotaryEmbedding::new(head_dim, 128, 10000.0, device),
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }

    /// Create a causal mask for local attention testing.
    #[cfg(feature = "tch-backend")]
    fn make_causal_mask(seq_len: i64, device: tch::Device) -> Tensor {
        let mask = Tensor::ones([seq_len, seq_len], (Kind::Float, device))
            .triu(1)
            .to_kind(Kind::Bool);
        Tensor::zeros([seq_len, seq_len], (Kind::Float, device))
            .masked_fill(&mask, f64::NEG_INFINITY)
            .unsqueeze(0)
            .unsqueeze(0)
    }

    /// 【验证 process_kv_block 非因果模式与标准 softmax 等价】
    ///
    /// 这个测试验证：当 apply_causal_mask=false 时，process_kv_block 的 online softmax
    /// 输出与直接 softmax(QK^T/sqrt(d))V 完全一致。
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_chunk_step_vs_softmax_single_block() {
        let device = tch::Device::Cpu;
        let query_len = 4i64;
        let block_len = 4i64;
        let num_heads = 2i64;
        let head_dim = 8i64;

        tch::manual_seed(42);
        let q = Tensor::randn([1, num_heads, query_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, block_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, block_len, head_dim], (Kind::Float, device));

        // 标准 softmax attention（参考值）
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let attn = scores.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        // 用 process_kv_block（非因果模式）计算
        let rope = crate::model::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);
        let backend = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains: 1,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
            disable_overlap: false,
            position_ids: None,
            micro_kv_block_size: 0,
            ..Default::default()
        };

        let mut rm = Tensor::full(
            [1i64, num_heads, query_len],
            f64::NEG_INFINITY,
            (Kind::Float, device),
        );
        let mut rs = Tensor::zeros([1i64, num_heads, query_len], (Kind::Float, device));
        let mut obh = Tensor::zeros(
            [1i64, num_heads, query_len, head_dim],
            (Kind::Float, device),
        );

        // 非因果模式：所有 Q 都能看到所有 K/V
        let q_pos = Tensor::arange_start(0, query_len, (Kind::Int64, Device::Cpu));
        let k_pos = Tensor::arange_start(0, block_len, (Kind::Int64, Device::Cpu));
        backend.process_kv_block(
            &q, &q_pos,
            &k, &v, &k_pos,
            &mut rm, &mut rs, &mut obh,
            false,
        );

        // obh 形状: [1, num_heads, query_len, head_dim]
        let actual = obh;

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Single block diff = {}", diff_val);
        assert!(diff_val < 1e-4, "Single block chunk step differs from softmax: {}", diff_val);
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_local_attention_backend_shape() {
        let device = tch::Device::Cpu;
        let attn = create_test_attention(device);
        let hidden_size = attn.q_proj.size()[1];

        let mut backend = LocalAttentionBackend { attention: attn };
        let batch = 1i64;
        let seq_len = 5i64;
        let hidden = Tensor::randn([batch, seq_len, hidden_size], (Kind::Float, device));
        let pos_ids = Tensor::arange(seq_len, (Kind::Int64, device)).unsqueeze(0);

        let out = backend.forward(&hidden, &pos_ids, None, None).unwrap();
        assert_eq!(out.size(), vec![batch, seq_len, hidden_size]);
    }

    /// Build a ring-attention backend and verify it matches local full attention
    /// (non-causal, all positions attend to all positions).
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_matches_local_full() {
        let device = tch::Device::Cpu;
        let num_heads = 4i64;
        let head_dim = 8i64;
        let seq_len = 16i64;
        let num_domains = 4usize;

        tch::manual_seed(123);
        let q = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let attn = scores.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        // Create a minimal backend (only needs q/k/v/o_proj for local_attention_scores)
        let rope = crate::model::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);
        let mut backend = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
            disable_overlap: false,
            position_ids: None,
            micro_kv_block_size: 0,
                    ..Default::default()
        };

        let actual = backend.ring_attention(&q, &k, &v, None, 0).unwrap();

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Ring vs local full diff = {}", diff_val);
        assert!(diff_val < 1e-5, "Ring attention differs from local full: {}", diff_val);
    }

    /// Verify ring attention with causal mask matches local causal attention.
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_matches_local_causal() {
        let device = tch::Device::Cpu;
        let num_heads = 4i64;
        let head_dim = 8i64;
        let seq_len = 16i64;
        let num_domains = 4usize;

        tch::manual_seed(456);
        let q = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;

        let mask = make_causal_mask(seq_len, device);
        let scores_masked = scores + mask.shallow_clone();
        let attn = scores_masked.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        let rope = crate::model::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);
        let mut backend = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
            disable_overlap: false,
            position_ids: None,
            micro_kv_block_size: 0,
                    ..Default::default()
        };

        let actual = backend.ring_attention(&q, &k, &v, Some(&mask), 0).unwrap();

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Ring vs local causal diff = {}", diff_val);
        assert!(diff_val < 1e-4, "Ring attention differs from local causal: {}", diff_val);
    }

    /// Verify ring attention with seq_len=1 (decode) and long KV cache matches local attention.
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_decode_matches_local() {
        let device = tch::Device::Cpu;
        let num_heads = 4i64;
        let head_dim = 8i64;
        let q_len = 1i64;
        let kv_len = 17i64; // simulate KV cache with 17 tokens
        let num_domains = 2usize;

        tch::manual_seed(999);
        let q = Tensor::randn([1, num_heads, q_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, kv_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, kv_len, head_dim], (Kind::Float, device));

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let attn = scores.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        let rope = crate::model::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);
        let mut backend = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
            disable_overlap: false,
            position_ids: None,
            micro_kv_block_size: 0,
                    ..Default::default()
        };

        // Without transport, ring_attention processes only local KV.
        // Since local KV = full KV (17 tokens), output should match local_attention_scores.
        let actual = backend.ring_attention(&q, &k, &v, None, 0).unwrap();

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Decode ring vs local diff = {}", diff_val);
        assert!(diff_val < 1e-5, "Decode ring attention differs from local: {}", diff_val);
    }

    /// Verify distributed decode (seq_len=1) with peer KV matches local attention.
    /// Simulates real decode KV cache where new token is appended after history.
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_decode_with_peer_kv() {
        use crate::model::transport::MockKvTransport;

        let device = tch::Device::Cpu;
        let num_heads = 4i64;
        let head_dim = 8i64;
        let q_len = 1i64;
        let kv_len = 17i64; // tokens 0..16

        tch::manual_seed(777);
        let q_all = Tensor::randn([1, num_heads, q_len, head_dim], (Kind::Float, device));
        let k_all = Tensor::randn([1, num_heads, kv_len, head_dim], (Kind::Float, device));
        let v_all = Tensor::randn([1, num_heads, kv_len, head_dim], (Kind::Float, device));

        // Reference: local attention on full KV [0..16]
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores_ref = q_all.matmul(&k_all.transpose(2, 3)) * scale;
        let attn_ref = scores_ref.softmax(-1, Kind::Float);
        let expected = attn_ref.matmul(&v_all);

        // Simulate 2-domain distributed decode:
        // domain0 holds history [0..8) + new token 16
        // domain1 holds history [8..16) + new token 16
        let half = (kv_len - 1) / 2; // 8

        // domain0 local KV = [0..7] + [16]
        let k0_local = Tensor::cat(&[k_all.narrow(2, 0, half), k_all.narrow(2, kv_len - 1, 1)], 2);
        let v0_local = Tensor::cat(&[v_all.narrow(2, 0, half), v_all.narrow(2, kv_len - 1, 1)], 2);
        // domain1 local KV = [8..15] + [16]
        let k1_local = Tensor::cat(&[k_all.narrow(2, half, half), k_all.narrow(2, kv_len - 1, 1)], 2);
        let v1_local = Tensor::cat(&[v_all.narrow(2, half, half), v_all.narrow(2, kv_len - 1, 1)], 2);

        // History for sending (exclude new token)
        let k0_hist = k_all.narrow(2, 0, half);
        let v0_hist = v_all.narrow(2, 0, half);
        let k1_hist = k_all.narrow(2, half, half);
        let v1_hist = v_all.narrow(2, half, half);

        let rope = crate::model::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);

        // Worker 0: receives peer KV [8..15] from worker 1
        let mut transport0 = MockKvTransport::new();
        transport0.push(crate::model::transport::KvBlock {
            layer_idx: 0,
            global_seq_start: half as usize,
            global_seq_end: kv_len as usize - 1,
            k: k1_hist.shallow_clone(),
            v: v1_hist.shallow_clone(),
            micro_block_idx: 0,
            total_micro_blocks: 1,
            position_ids: None,
        });

        let mut backend0 = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope: rope.clone(),
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains: 2,
            layer_idx: 0,
            kv_transport: Some(Box::new(transport0)),
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
            disable_overlap: false,
            position_ids: None,
            micro_kv_block_size: 0,
                    ..Default::default()
        };
        let out0 = backend0.ring_attention(&q_all, &k0_local, &v0_local, None, 0).unwrap();

        // Worker 1: receives peer KV [0..7] from worker 0
        let mut transport1 = MockKvTransport::new();
        transport1.push(crate::model::transport::KvBlock {
            layer_idx: 0,
            global_seq_start: 0,
            global_seq_end: half as usize,
            k: k0_hist.shallow_clone(),
            v: v0_hist.shallow_clone(),
            micro_block_idx: 0,
            total_micro_blocks: 1,
            position_ids: None,
        });

        let mut backend1 = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains: 2,
            layer_idx: 0,
            kv_transport: Some(Box::new(transport1)),
            local_domain_id: 1,
            seq_offset: half as usize,
            prefill_kv_len: 0,
            is_prefill_done: false,
            disable_overlap: false,
            position_ids: None,
            micro_kv_block_size: 0,
                    ..Default::default()
        };
        let out1 = backend1.ring_attention(&q_all, &k1_local, &v1_local, None, half as usize).unwrap();

        let diff0 = (&expected - &out0).abs().mean(Kind::Float).double_value(&[]);
        let diff1 = (&expected - &out1).abs().mean(Kind::Float).double_value(&[]);
        let diff01 = (&out0 - &out1).abs().mean(Kind::Float).double_value(&[]);
        println!("Decode with peer KV diff ref-vs-domain0 = {}", diff0);
        println!("Decode with peer KV diff ref-vs-domain1 = {}", diff1);
        println!("Decode with peer KV diff domain0-vs-domain1 = {}", diff01);

        assert!(diff0 < 1e-4, "Decode distributed (domain0) differs from reference: {}", diff0);
        assert!(diff1 < 1e-4, "Decode distributed (domain1) differs from reference: {}", diff1);
        assert!(diff01 < 1e-4, "Decode distributed domain0 differs from domain1: {}", diff01);
    }

    /// Verify HcpRingAttentionBackend::forward matches GqaAttention::forward
    /// when num_domains=1 (no distributed transport) on decode (seq_len=1).
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_backend_matches_gqa_on_decode() {
        let device = tch::Device::Cpu;
        let hidden_size = 32i64;
        let num_heads = 4usize;
        let num_kv_heads = 1usize;
        let head_dim = 8usize;
        let seq_len = 1i64;
        let cache_len = 17i64; // simulate prefill of 16 tokens + 1 decode token

        tch::manual_seed(555);

        // Create GqaAttention
        let gqa = crate::model::layers::GqaAttention {
            q_proj: Tensor::randn([(num_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            k_proj: Tensor::randn([(num_kv_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            v_proj: Tensor::randn([(num_kv_heads * head_dim) as i64, hidden_size], (Kind::Float, device)),
            o_proj: Tensor::randn([hidden_size, (num_heads * head_dim) as i64], (Kind::Float, device)),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            num_heads,
            num_kv_heads,
            head_dim,
            rope: crate::model::layers::RotaryEmbedding::new(head_dim, 128, 10000.0, device),
            scale: 1.0 / (head_dim as f64).sqrt(),
        };

        // Create HcpRingAttentionBackend with same weights
        let rope = gqa.rope.clone();
        let mut ring_backend = HcpRingAttentionBackend {
            q_proj: gqa.q_proj.shallow_clone(),
            k_proj: gqa.k_proj.shallow_clone(),
            v_proj: gqa.v_proj.shallow_clone(),
            o_proj: gqa.o_proj.shallow_clone(),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: gqa.scale,
            num_domains: 1,
            layer_idx: 0,
            kv_transport: None,
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
            disable_overlap: false,
            position_ids: None,
            micro_kv_block_size: 0,
                    ..Default::default()
        };

        // hidden states for decode step
        let hidden_states = Tensor::randn([1, seq_len, hidden_size], (Kind::Float, device));
        let position_ids = Tensor::from_slice(&[16i64]).unsqueeze(0); // position 16

        // Pre-populate KV caches with synthetic history (16 tokens)
        let history_k = Tensor::randn([1, num_kv_heads as i64, cache_len - 1, head_dim as i64], (Kind::Float, device));
        let history_v = Tensor::randn([1, num_kv_heads as i64, cache_len - 1, head_dim as i64], (Kind::Float, device));

        let mut gqa_cache = crate::model::cache::ContiguousKvCache::new();
        let _ = gqa_cache.update(&history_k, &history_v).unwrap();

        let mut ring_cache = crate::model::cache::ContiguousKvCache::new();
        let _ = ring_cache.update(&history_k, &history_v).unwrap();

        // Run both forwards
        let gqa_out = gqa.forward(&hidden_states, &position_ids, Some(&mut gqa_cache), None).unwrap();
        let ring_out = ring_backend.forward(&hidden_states, &position_ids, Some(&mut ring_cache), None).unwrap();

        let diff = (&gqa_out - &ring_out).abs().mean(Kind::Float).double_value(&[]);
        println!("Ring backend vs GqaAttention decode diff = {}", diff);
        assert!(diff < 1e-4, "Ring backend differs from GqaAttention on decode: {}", diff);
    }

    /// Verify distributed ring attention with MockKvTransport matches single-process causal attention.
    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_ring_attention_with_mock_transport() {
        use crate::model::transport::MockKvTransport;

        let device = tch::Device::Cpu;
        let num_heads = 4i64;
        let head_dim = 8i64;
        let seq_len = 16i64;
        let half = seq_len / 2;

        tch::manual_seed(789);
        let q = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));

        // Expected: single-process causal attention
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let mask = make_causal_mask(seq_len, device);
        let scores_masked = scores + mask.shallow_clone();
        let attn = scores_masked.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        // Split into two workers
        let q0 = q.narrow(2, 0, half);
        let k0 = k.narrow(2, 0, half);
        let v0 = v.narrow(2, 0, half);
        let q1 = q.narrow(2, half, half);
        let k1 = k.narrow(2, half, half);
        let v1 = v.narrow(2, half, half);

        let rope = crate::model::layers::RotaryEmbedding::new(head_dim as usize, 128, 10000.0, device);

        // Worker 0: receives peer KV from worker 1
        let mut transport0 = MockKvTransport::new();
        transport0.push(crate::model::transport::KvBlock {
            layer_idx: 0,
            global_seq_start: half as usize,
            global_seq_end: seq_len as usize,
            k: k1.shallow_clone(),
            v: v1.shallow_clone(),
            micro_block_idx: 0,
            total_micro_blocks: 1,
            position_ids: None,
        });

        let mut backend0 = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope: rope.clone(),
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains: 2,
            layer_idx: 0,
            kv_transport: Some(Box::new(transport0)),
            local_domain_id: 0,
            seq_offset: 0,
            prefill_kv_len: 0,
            is_prefill_done: false,
            disable_overlap: false,
            position_ids: None,
            micro_kv_block_size: 0,
                    ..Default::default()
        };
        let out0 = backend0.ring_attention(&q0, &k0, &v0, Some(&mask), 0).unwrap();

        // Worker 1: receives peer KV from worker 0
        let mut transport1 = MockKvTransport::new();
        transport1.push(crate::model::transport::KvBlock {
            layer_idx: 0,
            global_seq_start: 0,
            global_seq_end: half as usize,
            k: k0.shallow_clone(),
            v: v0.shallow_clone(),
            micro_block_idx: 0,
            total_micro_blocks: 1,
            position_ids: None,
        });

        let mut backend1 = HcpRingAttentionBackend {
            q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
            q_bias: None, k_bias: None, v_bias: None,
            rope,
            num_heads: num_heads as usize,
            num_kv_heads: num_heads as usize,
            head_dim: head_dim as usize,
            scale,
            num_domains: 2,
            layer_idx: 0,
            kv_transport: Some(Box::new(transport1)),
            local_domain_id: 1,
            seq_offset: half as usize,
            prefill_kv_len: 0,
            is_prefill_done: false,
            disable_overlap: false,
            position_ids: None,
            micro_kv_block_size: 0,
                    ..Default::default()
        };
        let out1 = backend1.ring_attention(&q1, &k1, &v1, Some(&mask), half as usize).unwrap();

        // Compare each worker against expected slice before concatenation
        let expected0 = expected.narrow(2, 0, half);
        let expected1 = expected.narrow(2, half, half);
        let diff0 = (&expected0 - &out0).abs().mean(Kind::Float).double_value(&[]);
        let diff1 = (&expected1 - &out1).abs().mean(Kind::Float).double_value(&[]);
        println!("worker0 diff={}, worker1 diff={}", diff0, diff1);

        // Concatenate outputs: out0 [1, num_heads, half, head_dim], out1 [1, num_heads, half, head_dim]
        let actual = Tensor::cat(&[out0, out1], 2);

        let diff = (&expected - &actual).abs().mean(Kind::Float);
        let diff_val: f64 = diff.double_value(&[]);
        println!("Distributed ring attention diff = {}", diff_val);
        assert!(diff_val < 1e-4, "Distributed ring attention differs from local causal: {}", diff_val);
    }

    /// 2-domain uneven (3:1) ring attention perf comparison test.
    /// Compares vanilla capacity-aware continuous chunks vs weighted Striped permutation.
    /// Run with: cargo test --features tch-backend test_ring_attention_derivatives_uneven_perf -- --nocapture
    #[cfg(feature = "tch-backend")]
    #[test]
    fn test_ring_attention_derivatives_uneven_perf() {
        let device = tch::Device::Cpu;
        let num_heads = 8i64;
        let head_dim = 128i64;
        let seq_len = 4096i64;
        let chunks = vec![3072usize, 1024usize]; // 3:1

        tch::manual_seed(2024);
        let q = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let k = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));
        let v = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::Float, device));

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)) * scale;
        let mask = make_causal_mask(seq_len, device);
        let scores_masked = scores + mask.shallow_clone();
        let attn = scores_masked.softmax(-1, Kind::Float);
        let expected = attn.matmul(&v);

        for strategy in RingSchedulingStrategy::all() {
            let name = format!("{:?}", strategy).to_lowercase();
            let log_path = format!("/tmp/ring_perf_{}.jsonl", name);
            std::env::set_var("HCP_PERF_LOG", &log_path);
            let _ = std::fs::remove_file(&log_path);

            let (actual, per_worker_diff) = run_uneven_ring_attention(
                &q, &k, &v, &mask, &chunks, *strategy, scale, num_heads, head_dim, device,
            );
            let has_nan = actual.isnan().any().double_value(&[]) != 0.0;
            assert!(!has_nan, "{} output contains NaN", name);
            let diff = (&expected - &actual).abs().mean(Kind::Float).double_value(&[]);
            assert!(diff < 1e-4, "{} uneven ring attention correctness failed: {}", name, diff);
            assert!(per_worker_diff < 1e-4, "{} per-worker diff too large: {}", name, per_worker_diff);

            println!("{} correctness diff = {}", name, diff);
            let rows = parse_perf_log(&log_path);
            println!("\n=== {} ===", name);
            print_perf_summary(&rows);
        }
    }

    /// Run N-domain uneven ring attention with a chosen scheduling strategy.
    /// Returns the output reconstructed in original sequence order and the
    /// max per-position diff against the full local reference.
    #[cfg(feature = "tch-backend")]
    fn run_uneven_ring_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
        chunks: &[usize],
        strategy: RingSchedulingStrategy,
        scale: f64,
        num_heads: i64,
        head_dim: i64,
        device: tch::Device,
    ) -> (Tensor, f64) {
        use crate::model::transport::MockKvTransport;
        use crate::model::attention::strategy::{build_assignment, build_domain_positions, build_inverse_perm, position_ids_tensor};

        let seq_len = q.size()[2] as usize;
        let num_domains = chunks.len();

        let assignment = build_assignment(chunks, strategy);
        let positions = build_domain_positions(&assignment);
        let inverse_perm = build_inverse_perm(&assignment);

        // Gather tensors by domain (local storage order follows the assignment scan).
        let mut q_d: Vec<Tensor> = Vec::new();
        let mut k_d: Vec<Tensor> = Vec::new();
        let mut v_d: Vec<Tensor> = Vec::new();
        let mut pos_d: Vec<Tensor> = Vec::new();
        for domain in 0..num_domains {
            let pos_t = position_ids_tensor(&positions[domain], device);
            q_d.push(q.index_select(2, &pos_t));
            k_d.push(k.index_select(2, &pos_t));
            v_d.push(v.index_select(2, &pos_t));
            pos_d.push(pos_t);
        }

        let rope = crate::model::layers::RotaryEmbedding::new(head_dim as usize, 16384, 10000.0, device);

        // Create transports and backends.
        let mut backends: Vec<HcpRingAttentionBackend> = Vec::new();
        let mut transports: Vec<MockKvTransport> = Vec::new();
        for domain in 0..num_domains {
            let peer = (domain + 1) % num_domains;
            let mut transport = MockKvTransport::new();
            transport.push(crate::model::transport::KvBlock {
                layer_idx: 0,
                global_seq_start: positions[peer].first().copied().unwrap_or(0),
                global_seq_end: positions[peer].last().copied().unwrap_or(0) + 1,
                k: k_d[peer].shallow_clone(),
                v: v_d[peer].shallow_clone(),
                micro_block_idx: 0,
                total_micro_blocks: 1,
                position_ids: Some(pos_d[peer].shallow_clone()),
            });
            transports.push(transport);

            let backend = HcpRingAttentionBackend {
                q_proj: Tensor::randn([1, 1], (Kind::Float, device)),
                k_proj: Tensor::randn([1, 1], (Kind::Float, device)),
                v_proj: Tensor::randn([1, 1], (Kind::Float, device)),
                o_proj: Tensor::randn([1, 1], (Kind::Float, device)),
                q_bias: None, k_bias: None, v_bias: None,
                rope: rope.clone(),
                num_heads: num_heads as usize,
                num_kv_heads: num_heads as usize,
                head_dim: head_dim as usize,
                scale,
                num_domains,
                layer_idx: 0,
                kv_transport: Some(Box::new(transports.pop().unwrap())),
                local_domain_id: domain,
                seq_offset: positions[domain].first().copied().unwrap_or(0),
                prefill_kv_len: 0,
                is_prefill_done: false,
                disable_overlap: false,
                position_ids: Some(pos_d[domain].shallow_clone()),
                micro_kv_block_size: 0,
                strategy,
                kv_base_global_start: positions[domain].first().copied().unwrap_or(0),
                ..Default::default()
            };
            backends.push(backend);
        }

        // Run attention.
        let mut outputs: Vec<Tensor> = Vec::new();
        for domain in 0..num_domains {
            let global_start = positions[domain].first().copied().unwrap_or(0);
            let out = backends[domain]
                .ring_attention(&q_d[domain], &k_d[domain], &v_d[domain], Some(&mask), global_start)
                .unwrap();
            outputs.push(out);
        }

        // Reconstruct original order.
        let inverse_t = Tensor::f_from_slice::<i64>(&inverse_perm).unwrap().to_device(device);
        let concatenated = Tensor::cat(&outputs, 2);
        let actual = concatenated.index_select(2, &inverse_t);

        // Compute per-domain diff against expected slice in original order.
        let expected_full = {
            let scores = q.matmul(&k.transpose(2, 3)) * scale;
            let scores_masked = scores + mask.shallow_clone();
            let attn = scores_masked.softmax(-1, Kind::Float);
            attn.matmul(v)
        };
        let mut max_diff = 0.0f64;
        for domain in 0..num_domains {
            for &orig_pos in &positions[domain] {
                let expected_slice = expected_full.narrow(2, orig_pos as i64, 1);
                let actual_slice = actual.narrow(2, orig_pos as i64, 1);
                let d = (&expected_slice - &actual_slice).abs().mean(Kind::Float).double_value(&[]);
                max_diff = max_diff.max(d);
            }
        }

        (actual, max_diff)
    }

    /// Simple JSONL perf log parser for test summary.
    #[cfg(feature = "tch-backend")]
    fn parse_perf_log(path: &str) -> Vec<std::collections::HashMap<String, String>> {
        let content = std::fs::read_to_string(path).unwrap_or_default();
        content.lines().filter_map(|line| {
            serde_json::from_str::<std::collections::HashMap<String, serde_json::Value>>(line).ok()
                .map(|m| m.into_iter().map(|(k, v)| (k, v.to_string())).collect())
        }).collect()
    }

    #[cfg(feature = "tch-backend")]
    fn print_perf_summary(rows: &[std::collections::HashMap<String, String>]) {
        for row in rows {
            let domain = row.get("domain").cloned().unwrap_or_default();
            let total = row.get("total_ms").cloned().unwrap_or_default();
            let local = row.get("local_compute_ms").cloned().unwrap_or_default();
            let peer = row.get("peer_compute_ms").cloned().unwrap_or_default();
            let recv = row.get("recv_ms").cloned().unwrap_or_default();
            println!("domain {}: total={}ms local_compute={}ms peer_compute={}ms recv={}ms", domain, total, local, peer, recv);
        }
    }
}
