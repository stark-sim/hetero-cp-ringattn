use serde::Serialize;
use thiserror::Error;

/// 【HCP 顶层错误类型】
///
/// 涵盖所有子系统的错误：
/// - InvalidChunkSum: 分布式分片长度之和不等于全局序列长度
/// - InvalidDomain: domain 配置不合法（seq_chunk_len 或 block_size 为 0）
/// - Io: 文件/网络 IO 错误
/// - Json: JSON 解析错误
/// - Protocol: 协议层错误（来自 protocol/message.rs）
/// - InvalidCli: 命令行参数错误
#[derive(Debug, Error)]
pub enum RingError {
    #[error("sum(seq_chunk_len)={actual} does not match global_seq_len={expected}")]
    InvalidChunkSum { actual: usize, expected: usize },
    #[error("domain {domain_id} has invalid seq_chunk_len or block_size")]
    InvalidDomain { domain_id: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("protocol error: {0}")]
    Protocol(#[from] crate::protocol::ProtocolError),
    #[error("invalid cli args: {0}")]
    InvalidCli(String),
}

#[derive(Clone, Copy, Serialize)]
pub struct Tolerance {
    pub max_abs_err: f64,
    pub mean_abs_err: f64,
    pub max_rel_err: f64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub enum ToleranceTier {
    #[default]
    Strict,
    Relaxed,
    EndToEnd,
}

impl std::str::FromStr for ToleranceTier {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "strict" | "s" => Ok(Self::Strict),
            "relaxed" | "r" => Ok(Self::Relaxed),
            "end-to-end" | "e2e" | "e" => Ok(Self::EndToEnd),
            _ => Err(format!("unknown tolerance tier: {s}")),
        }
    }
}

impl ToleranceTier {

    pub fn default_tolerance(self) -> Tolerance {
        match self {
            Self::Strict => Tolerance {
                max_abs_err: 1e-5,
                mean_abs_err: 1e-6,
                max_rel_err: 1e-5,
            },
            Self::Relaxed => Tolerance {
                max_abs_err: 1e-4,
                mean_abs_err: 1e-5,
                max_rel_err: 1e-4,
            },
            Self::EndToEnd => Tolerance {
                max_abs_err: 1e-3,
                mean_abs_err: 1e-4,
                max_rel_err: 1e-3,
            },
        }
    }
}
