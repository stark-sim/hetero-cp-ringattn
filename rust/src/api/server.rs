//! Axum HTTP server for OpenAI-compatible `/v1/completions`.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Sse},
    routing::{get, post},
    Router,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;

use crate::api::types::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChoice,
    CompletionStreamResponse, HealthResponse, InferenceJob,
    MetricsResponse, StreamChunk, Usage,
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
    pub queued_counter: Arc<AtomicU64>,
    pub active_counter: Arc<AtomicU64>,
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
) -> Result<axum::response::Response, (StatusCode, String)> {
    let request_id = state.request_counter.fetch_add(1, Ordering::SeqCst) + 1;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model_name = req.model.unwrap_or_else(|| state.model_name.clone());

    if req.stream {
        // Streaming mode: use mpsc channel for per-token chunks.
        let (chunk_tx, chunk_rx) = tokio::sync::mpsc::unbounded_channel::<StreamChunk>();
        let job = InferenceJob {
            request_id,
            prompt: req.prompt.clone(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            tx: oneshot::channel().0, // dummy oneshot for type compatibility
            stream_tx: Some(chunk_tx),
        };

        if state.job_tx.send(job).is_err() {
            state.failed_counter.fetch_add(1, Ordering::SeqCst);
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "Coordinator queue is closed".to_string(),
            ));
        }
        state.queued_counter.fetch_add(1, Ordering::SeqCst);

        let stream = UnboundedReceiverStream::new(chunk_rx)
            .map(move |chunk| {
                let resp = CompletionStreamResponse {
                    id: format!("hcp-completion-{request_id}"),
                    object: "text_completion".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![CompletionStreamChoice {
                        text: chunk.delta,
                        index: 0,
                        finish_reason: chunk.finish_reason.clone(),
                    }],
                };
                let data = serde_json::to_string(&resp).unwrap_or_default();
                Ok::<_, std::convert::Infallible>(
                    axum::response::sse::Event::default().data(data),
                )
            })
            .chain(tokio_stream::once(Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default().data("[DONE]"),
            )));

        let sse = Sse::new(stream);
        return Ok(axum::response::IntoResponse::into_response(sse));
    }

    // Non-streaming mode: use oneshot channel for final result.
    let (tx, rx) = oneshot::channel();
    let job = InferenceJob {
        request_id,
        prompt: req.prompt.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        tx,
        stream_tx: None,
    };

    if state.job_tx.send(job).is_err() {
        state.failed_counter.fetch_add(1, Ordering::SeqCst);
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Coordinator queue is closed".to_string(),
        ));
    }
    state.queued_counter.fetch_add(1, Ordering::SeqCst);

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

    let response = CompletionResponse {
        id: format!("hcp-completion-{request_id}"),
        object: "text_completion".to_string(),
        created,
        model: model_name,
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

    Ok(Json(response).into_response())
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
        queued_requests: state.queued_counter.load(Ordering::SeqCst),
        active_requests: state.active_counter.load(Ordering::SeqCst),
    })
}
