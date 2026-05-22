//! OpenAI-compatible API types for `/v1/completions`.

use serde::{Deserialize, Serialize};

/// Request body for `POST /v1/completions`
/// Reference: https://platform.openai.com/docs/api-reference/completions/create
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

fn default_max_tokens() -> usize { 20 }
fn default_temperature() -> f64 { 0.0 }
fn default_top_p() -> f64 { 1.0 }

/// Response body for `POST /v1/completions`
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Health check response
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub workers_connected: usize,
    pub num_domains: usize,
}

/// Simple metrics response
#[derive(Debug, Clone, Serialize)]
pub struct MetricsResponse {
    pub total_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub queued_requests: u64,
    pub active_requests: u64,
}

/// Internal job submitted from HTTP handler to the coordinator loop.
pub struct InferenceJob {
    pub request_id: u64,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    /// Channel to send back the result.
    pub tx: tokio::sync::oneshot::Sender<InferenceResult>,
}

/// Result of an inference job.
pub struct InferenceResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: Option<String>,
}
