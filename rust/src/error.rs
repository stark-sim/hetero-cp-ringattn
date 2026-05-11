use serde::Serialize;
use thiserror::Error;

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

impl ToleranceTier {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "strict" | "s" => Some(Self::Strict),
            "relaxed" | "r" => Some(Self::Relaxed),
            "end-to-end" | "e2e" | "e" => Some(Self::EndToEnd),
            _ => None,
        }
    }

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
