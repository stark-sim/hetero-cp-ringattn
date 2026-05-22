//! HTTP API service layer for the distributed inference coordinator.
//!
//! Provides OpenAI-compatible `/v1/completions`, `/health`, and `/metrics` endpoints.
//! Requests are queued and processed by the coordinator's main loop.

pub mod server;
pub mod types;

pub use server::{build_router, ApiState};
pub use types::{CompletionRequest, CompletionResponse, HealthResponse, InferenceJob, InferenceResult, MetricsResponse};
