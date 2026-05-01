pub mod backend;
pub mod cache;
pub mod config;
pub mod generate;
pub mod kv_transport;
pub mod layers;
pub mod model;
pub mod weights;

pub use cache::KvCache;
pub use model::LlamaModel;
pub use weights::{ModelWeights, WeightNames};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Safetensors error: {0}")]
    Safetensors(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Shape error: expected {expected:?}, got {got:?}")]
    Shape { expected: Vec<usize>, got: Vec<usize> },
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    #[error("Backend error: {0}")]
    Backend(String),
    #[error("Generation error: {0}")]
    Generation(String),
}

pub use config::ModelConfig;
