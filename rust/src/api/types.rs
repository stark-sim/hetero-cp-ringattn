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
    #[allow(dead_code)]
    pub stop: Option<Vec<String>>,
    /// Continuation session identifier. Required when `keep_kv` or `append`
    /// is set; absent keeps the request fully stateless (default behavior).
    #[serde(default)]
    pub session_id: Option<String>,
    /// Keep this request's KV resident on the workers after completion so a
    /// later `append` request with the same `session_id` can continue on it.
    #[serde(default)]
    pub keep_kv: bool,
    /// This request is a continuation segment appended to an existing
    /// session's frozen KV (stationary continuation, no prefix recompute).
    #[serde(default)]
    pub append: bool,
}

impl CompletionRequest {
    /// Fail-closed session field validation: `append` and `keep_kv` both
    /// require a `session_id`. Session existence is checked later by the
    /// coordinator loop, not here.
    pub fn validate_session_fields(&self) -> Result<(), String> {
        if self.append && self.session_id.is_none() {
            return Err("append=true requires session_id".to_string());
        }
        if self.keep_kv && self.session_id.is_none() {
            return Err("keep_kv=true requires session_id".to_string());
        }
        Ok(())
    }
}

fn default_max_tokens() -> usize {
    20
}
fn default_temperature() -> f64 {
    0.0
}
fn default_top_p() -> f64 {
    1.0
}

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

/// A chunk of streaming inference output.
pub struct StreamChunk {
    /// Text delta for this chunk (only newly generated text).
    pub delta: String,
    /// The token ID.
    #[allow(dead_code)]
    pub token_id: u32,
    /// Finish reason if this is the final chunk.
    pub finish_reason: Option<String>,
}

/// Internal job submitted from HTTP handler to the coordinator loop.
pub struct InferenceJob {
    pub request_id: u64,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    /// Continuation session fields, mirrored from `CompletionRequest`.
    pub session_id: Option<String>,
    pub keep_kv: bool,
    pub append: bool,
    /// For non-streaming: channel to send back the final result.
    pub tx: tokio::sync::oneshot::Sender<InferenceResult>,
    /// For streaming: channel to send per-token chunks.
    pub stream_tx: Option<tokio::sync::mpsc::UnboundedSender<StreamChunk>>,
}

/// Result of an inference job.
pub struct InferenceResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: Option<String>,
}

/// SSE response for streaming completions (OpenAI-compatible).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CompletionStreamChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_without_session_fields_deserializes_with_stateless_defaults() {
        // Backward compatibility: a pre-session request body must behave
        // exactly as before (no session, no keep_kv, no append).
        let req: CompletionRequest =
            serde_json::from_str(r#"{"prompt":"hello","max_tokens":3}"#).unwrap();
        assert_eq!(req.prompt, "hello");
        assert_eq!(req.max_tokens, 3);
        assert!(req.session_id.is_none());
        assert!(!req.keep_kv);
        assert!(!req.append);
        assert!(req.validate_session_fields().is_ok());
    }

    #[test]
    fn request_with_session_fields_deserializes() {
        let req: CompletionRequest = serde_json::from_str(
            r#"{"prompt":"seg","session_id":"s1","keep_kv":true,"append":true}"#,
        )
        .unwrap();
        assert_eq!(req.session_id.as_deref(), Some("s1"));
        assert!(req.keep_kv);
        assert!(req.append);
        assert!(req.validate_session_fields().is_ok());
    }

    #[test]
    fn append_without_session_id_is_rejected() {
        let req: CompletionRequest =
            serde_json::from_str(r#"{"prompt":"seg","append":true}"#).unwrap();
        let err = req.validate_session_fields().unwrap_err();
        assert!(err.contains("append"));
        assert!(err.contains("session_id"));
    }

    #[test]
    fn keep_kv_without_session_id_is_rejected() {
        let req: CompletionRequest =
            serde_json::from_str(r#"{"prompt":"p","keep_kv":true}"#).unwrap();
        let err = req.validate_session_fields().unwrap_err();
        assert!(err.contains("keep_kv"));
        assert!(err.contains("session_id"));
    }

    #[test]
    fn session_id_without_flags_is_accepted_and_stateless() {
        // A bare session_id with neither flag is harmless: the request is
        // processed as a normal stateless request.
        let req: CompletionRequest =
            serde_json::from_str(r#"{"prompt":"p","session_id":"s1"}"#).unwrap();
        assert!(req.validate_session_fields().is_ok());
    }
}
