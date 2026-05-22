//! Axum HTTP server for OpenAI-compatible `/v1/completions`.

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;
use tower_http::cors::CorsLayer;

use crate::api::types::{
    CompletionChoice, CompletionRequest, CompletionResponse, HealthResponse,
    InferenceJob, InferenceResult, MetricsResponse, Usage,
};

/// Shared state between HTTP handlers and the coordinator.
#[derive(Clone)]
pub struct ApiState {
    pub job_tx: UnboundedSender<InferenceJob>,
    pub request_counter: Arc<AtomicU64>,
    pub completed_counter: Arc<AtomicU64>,
    pub failed_counter: Arc<AtomicU64>,
    pub workers_connected: Arc<AtomicU64>,
    pub num_domains: usize,
    pub model_name: String,
}

/// Build the axum router.
pub fn build_router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/completions", post(completions_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// `POST /v1/completions`
async fn completions_handler(
    State(state): State<ApiState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    if req.stream {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "Streaming is not yet supported".to_string(),
        ));
    }

    let request_id = state.request_counter.fetch_add(1, Ordering::SeqCst) + 1;

    let (tx, rx) = oneshot::channel();
    let job = InferenceJob {
        request_id,
        prompt: req.prompt.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        tx,
    };

    if state.job_tx.send(job).is_err() {
        state.failed_counter.fetch_add(1, Ordering::SeqCst);
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Coordinator queue is closed".to_string(),
        ));
    }

    let result = match rx.await {
        Ok(r) => r,
        Err(_) => {
            state.failed_counter.fetch_add(1, Ordering::SeqCst);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Coordinator dropped the job".to_string(),
            ));
        }
    };

    state.completed_counter.fetch_add(1, Ordering::SeqCst);

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let response = CompletionResponse {
        id: format!("hcp-completion-{request_id}"),
        object: "text_completion".to_string(),
        created,
        model: req.model.unwrap_or_else(|| state.model_name.clone()),
        choices: vec![CompletionChoice {
            text: result.text,
            index: 0,
            logprobs: None,
            finish_reason: result.finish_reason,
        }],
        usage: Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.prompt_tokens + result.completion_tokens,
        },
    };

    Ok(Json(response))
}

/// `GET /health`
async fn health_handler(State(state): State<ApiState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        workers_connected: state.workers_connected.load(Ordering::SeqCst) as usize,
        num_domains: state.num_domains,
    })
}

/// `GET /metrics`
async fn metrics_handler(State(state): State<ApiState>) -> Json<MetricsResponse> {
    Json(MetricsResponse {
        total_requests: state.request_counter.load(Ordering::SeqCst),
        completed_requests: state.completed_counter.load(Ordering::SeqCst),
        failed_requests: state.failed_counter.load(Ordering::SeqCst),
    })
}
