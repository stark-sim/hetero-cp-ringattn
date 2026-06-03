//! 【分布式推理模块】
//!
//! HCP 的生产环境分布式推理入口：
//! - `coordinator`: 协调器，负责 prompt 分片、结果汇总、请求调度
//! - `worker`: Worker 进程入口，加载模型并启动 WorkerRuntime
//! - `protocol`: Worker 与 Coordinator 之间的通信协议（QUIC-based）
//! - `transport`: QUIC 网络传输实现（大窗口、keep-alive、NAT 穿透）
//!
//! 架构：1 个 Coordinator + N 个 Worker（N = num_domains）。
//! Coordinator 不执行模型计算，只负责 orchestration。
//! 每个 Worker 持有完整的模型权重，处理自己 domain 的 prompt 分片。

#![allow(dead_code)]

#[cfg(feature = "tch-backend")]
pub mod protocol;
#[cfg(feature = "tch-backend")]
pub mod coordinator;
#[cfg(feature = "tch-backend")]
pub mod worker;
#[cfg(feature = "tch-backend")]
pub mod scheduler;
#[cfg(feature = "tch-backend")]
pub mod transport;
