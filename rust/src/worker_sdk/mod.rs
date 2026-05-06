//! HCP Worker SDK — Rust 版
//!
//! 将协议层、传输层、模型计算层解耦，定义 `WorkerBackend` trait，
//! 让外部框架（vLLM FFI、TensorRT-LLM、MLX 等）可以通过 Rust 接口接入
//! HCP 分布式推理网络。
//!
//! 核心结构：
//! - `backend::WorkerBackend`: 后端必须实现的 trait（加载、prefill、decode、capacity 上报）
//! - `runtime::WorkerRuntime`: 通用协议运行时（coordinator 连接、handshake、command loop）
//! - `tch_backend::TchWorkerBackend`: 默认后端，包装现有 `LlamaModel`（tch-rs）
//!
//! 使用示例（默认后端）：
//! ```rust,ignore
//! let backend = TchWorkerBackend::load("/path/to/model", Device::Mps, domain_id, num_domains)?;
//! let mut runtime = WorkerRuntime::new(backend, 0, 2, listen, peer, coord)?;
//! runtime.run()?;
//! ```

#[cfg(feature = "tch-backend")]
pub mod backend;
#[cfg(feature = "tch-backend")]
pub mod runtime;
#[cfg(feature = "tch-backend")]
pub mod tch_backend;

#[cfg(feature = "tch-backend")]
pub use runtime::WorkerRuntime;
#[cfg(feature = "tch-backend")]
pub use tch_backend::TchWorkerBackend;
