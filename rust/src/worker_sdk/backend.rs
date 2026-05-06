//! `WorkerBackend` trait — 分布式 Worker 后端抽象接口。
//!
//! 框架适配器（vLLM FFI、TensorRT-LLM、MLX 等）必须实现此 trait。
//! HCP 的 `WorkerRuntime` 负责协议循环和 coordinator 通信，
//! 后端负责模型加载、forward 计算和 KV ring 交换。

use crate::model::kv_transport::KvTransport;
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
