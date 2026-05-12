//! 【模型层模块】
//!
//! 这个模块包含 HCP 的深度学习模型实现：
//! - `model`: LlamaModel 主结构体和 forward 逻辑
//! - `attention`: Attention 后端（HcpRingAttentionBackend + trait）
//! - `layers`: Transformer 各层实现（Attention、MLP、RMSNorm、RoPE）
//! - `cache`: KV Cache 管理
//! - `config`: 从 HuggingFace config.json 解析模型配置
//! - `weights`: 从 safetensors 加载权重
//! - `generator`: 单节点文本生成器
//! - `distributed_generator`: 单进程分布式生成模拟器
//! - `transport`: KV Block 传输层（Trait + TCP + Mock + QUIC）
//! - `sampling`: Token 采样策略（greedy / temperature / top-p）

pub mod attention;
pub mod cache;
pub mod config;
pub mod sampling;
pub mod generator;
pub mod distributed_generator;
pub mod transport;
pub mod layers;
#[allow(clippy::module_inception)]
pub mod model;
pub mod weights;
pub mod error;
pub use error::ModelError;

#[cfg(feature = "tch-backend")]
pub use transport::KvTransport;

pub use model::LlamaModel;
pub use weights::{ModelWeights, WeightNames};

pub use config::ModelConfig;
