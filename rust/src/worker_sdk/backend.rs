//! 【`WorkerBackend` trait — 分布式 Worker 后端抽象接口】
//!
//! 这是 HCP 的插件化核心：任何深度学习框架只要实现这个 trait，
//! 就能接入 HCP 的分布式推理网络。
//!
//! 【当前实现】
//! - `TchWorkerBackend`: 基于 tch-rs（libtorch）的默认实现
//!
//! 【未来可扩展】
//! - vLLM FFI 后端：复用 vLLM 的 CUDA kernel 和 PagedAttention
//! - TensorRT-LLM 后端：利用 NVIDIA 优化 kernel
//! - MLX 后端：Apple Silicon 原生优化
//!
//! 【设计原则】
//! - `WorkerRuntime` 负责协议、网络、事件循环（与框架无关）
//! - `WorkerBackend` 负责模型加载、forward、KV ring（与框架相关）
//! - 两者通过 trait 接口解耦，互不影响

use crate::model::transport::KvTransport;
use tch::Device;

/// 分布式 Worker 后端抽象。
///
/// 实现者只需关注：
/// 1. 如何加载模型权重
/// 2. 如何执行 prefill / decode forward
/// 3. 如何在 forward 过程中通过已设置的 transports 完成 KV ring 交换
///
/// 协议层（序列化、网络、事件循环）由 `WorkerRuntime` 处理。
pub trait WorkerBackend: Send {
    /// 为每层设置 KV ring 传输通道。
    ///
    /// `WorkerRuntime` 在网络初始化完成后调用此方法，将 per-layer transports
    /// 传给后端。后端负责将这些 transport 绑定到对应的 attention layer。
    ///
    /// 对于 `TchWorkerBackend`，此方法内部调用 `model.setup_distributed_domain()`。
    /// 对于外部框架后端，此方法可以将 transports 存储起来供后续 forward 使用。
    fn setup_kv_transports(&mut self, transports: Vec<Box<dyn KvTransport>>);

    /// 执行 prefill forward。
    ///
    /// 后端 MUST 使用 `setup_kv_transports` 阶段传入的 transports 完成 KV ring 交换
    ///（当 num_domains > 1 时），使输出在数学上等价于全量 attention。
    ///
    /// # Arguments
    /// - `chunk`: token ID 列表，本 domain 负责的 prompt 分片
    /// - `seq_offset`: 本 chunk 在全局序列中的起始位置
    ///
    /// # Returns
    /// - `last_logits`: 最后一个 token 的 logits（`Vec<f32>`，长度 = vocab_size）
    /// - `global_seq_len`: 当前全局序列总长度
    fn prefill(
        &mut self,
        chunk: &[i64],
        seq_offset: usize,
    ) -> Result<(Vec<f32>, usize), String>;

    /// 执行单 token decode forward。
    ///
    /// 同 `prefill`，后端 MUST 使用已设置的 transports 完成 KV ring 交换。
    ///
    /// # Arguments
    /// - `token`: 当前要解码的 token ID
    ///
    /// # Returns
    /// - `logits`: `Vec<f32>`，长度 = vocab_size
    fn decode(&mut self, token: i64) -> Result<Vec<f32>, String>;

    /// 【请求感知的 prefill】支持多请求隔离的 prefill。
    ///
    /// 默认实现调用 `prefill()`（向后兼容单请求后端）。
    /// 多请求后端应 override 此方法来为每个 request_id 创建独立的 KV cache。
    fn prefill_request(
        &mut self,
        _request_id: u64,
        chunk: &[i64],
        seq_offset: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        self.prefill(chunk, seq_offset)
    }

    /// 【请求感知的 decode】支持多请求隔离的 decode。
    ///
    /// 默认实现调用 `decode()`（向后兼容单请求后端）。
    fn decode_request(&mut self, _request_id: u64, token: i64) -> Result<Vec<f32>, String> {
        self.decode(token)
    }

    /// 执行 batch decode forward（多个请求同时解码）。
    ///
    /// 默认实现是逐个调用 `decode_request()`。后端可以 override 此方法来提供
    /// 真正的 kernel-level batching（如 PagedAttention）。
    ///
    /// # Arguments
    /// - `request_tokens`: (request_id, token) 列表
    ///
    /// # Returns
    /// - `request_logits`: (request_id, logits) 列表
    fn decode_batch(&mut self, request_tokens: &[(u64, i64)]) -> Result<Vec<(u64, Vec<f32>)>, String> {
        let mut results = Vec::with_capacity(request_tokens.len());
        for &(request_id, token) in request_tokens {
            let logits = self.decode_request(request_id, token)?;
            results.push((request_id, logits));
        }
        Ok(results)
    }

    /// 【请求感知的 global_seq_len 同步】
    ///
    /// 默认实现调用 `sync_global_seq_len()`。
    fn sync_global_seq_len_for_request(&mut self, _request_id: u64, len: usize) {
        self.sync_global_seq_len(len);
    }

    /// 同步全局序列长度（coordinator 广播）。
    ///
    /// 在 prefill 完成后，coordinator 会取所有 worker 的 `global_seq_len` 最大值，
    /// 然后广播给所有 worker。后端需要更新内部状态以确保 decode 阶段
    /// position_ids 和 causal mask 使用正确的全局位置。
    fn sync_global_seq_len(&mut self, len: usize);

    /// 上报本节点的可用计算资源（显存或内存），单位 MB。
    ///
    /// Coordinator 用此信息做 capacity-aware 分片。
    fn capacity_mb(&self) -> u64;

    /// 模型层数。
    ///
    /// `WorkerRuntime` 用此值创建正确数量的 per-layer QUIC streams。
    fn num_layers(&self) -> usize;

    /// 后端使用的计算设备。
    ///
    /// `WorkerRuntime` 用此值在反序列化 peer KV tensor 时指定目标设备。
    fn device(&self) -> Device;
}
