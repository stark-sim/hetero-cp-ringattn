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
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;

use crate::api::types::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChoice,
    CompletionStreamResponse, HealthResponse, InferenceJob, MetricsResponse, StreamChunk, Usage,
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
                Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(data))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::InferenceResult;
    use axum::body::{to_bytes, Body};
    use axum::http::{header::CONTENT_TYPE, Request};
    use serde_json::Value;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tower::ServiceExt;

    fn test_state() -> (ApiState, UnboundedReceiver<InferenceJob>) {
        let (job_tx, job_rx) = tokio::sync::mpsc::unbounded_channel();
        (
            ApiState {
                job_tx,
                request_counter: Arc::new(AtomicU64::new(0)),
                completed_counter: Arc::new(AtomicU64::new(0)),
                failed_counter: Arc::new(AtomicU64::new(0)),
                workers_connected: Arc::new(AtomicU64::new(2)),
                num_domains: 2,
                model_name: "qwen-test".to_string(),
                queued_counter: Arc::new(AtomicU64::new(0)),
                active_counter: Arc::new(AtomicU64::new(0)),
            },
            job_rx,
        )
    }

    fn completion_request(body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn non_streaming_completion_preserves_request_and_response_contract() {
        let (state, mut jobs) = test_state();
        let app = build_router(state.clone());
        let coordinator = tokio::spawn(async move {
            let job = jobs.recv().await.expect("handler must enqueue one job");
            assert_eq!(job.request_id, 1);
            assert_eq!(job.prompt, "contract prompt");
            assert_eq!(job.max_tokens, 3);
            assert_eq!(job.temperature, 0.25);
            assert_eq!(job.top_p, 0.9);
            assert!(job.stream_tx.is_none());
            assert!(job
                .tx
                .send(InferenceResult {
                    text: " result".to_string(),
                    prompt_tokens: 2,
                    completion_tokens: 1,
                    finish_reason: Some("length".to_string()),
                })
                .is_ok());
        });

        let response = app
            .oneshot(completion_request(
                r#"{"model":"served-qwen","prompt":"contract prompt","max_tokens":3,"temperature":0.25,"top_p":0.9}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["id"], "hcp-completion-1");
        assert_eq!(json["object"], "text_completion");
        assert_eq!(json["model"], "served-qwen");
        assert_eq!(json["choices"][0]["text"], " result");
        assert_eq!(json["choices"][0]["index"], 0);
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert!(json["choices"][0]["logprobs"].is_null());
        assert_eq!(json["usage"]["prompt_tokens"], 2);
        assert_eq!(json["usage"]["completion_tokens"], 1);
        assert_eq!(json["usage"]["total_tokens"], 3);
        assert!(json["created"].as_u64().is_some());
        coordinator.await.unwrap();
        assert_eq!(state.request_counter.load(Ordering::SeqCst), 1);
        assert_eq!(state.completed_counter.load(Ordering::SeqCst), 1);
        assert_eq!(state.failed_counter.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn streaming_completion_emits_json_events_finish_reason_and_done() {
        let (state, mut jobs) = test_state();
        let app = build_router(state.clone());
        let coordinator = tokio::spawn(async move {
            let mut job = jobs.recv().await.expect("handler must enqueue one job");
            assert_eq!(job.request_id, 1);
            assert_eq!(job.prompt, "stream prompt");
            let chunks = job.stream_tx.take().expect("streaming job needs chunk tx");
            chunks
                .send(StreamChunk {
                    delta: " first".to_string(),
                    token_id: 11,
                    finish_reason: None,
                })
                .unwrap();
            chunks
                .send(StreamChunk {
                    delta: " second".to_string(),
                    token_id: 12,
                    finish_reason: Some("length".to_string()),
                })
                .unwrap();
        });

        let response = app
            .oneshot(completion_request(
                r#"{"prompt":"stream prompt","max_tokens":2,"stream":true}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers()[CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        let events = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect::<Vec<_>>();
        assert_eq!(events.len(), 3);
        let first: Value = serde_json::from_str(events[0]).unwrap();
        let second: Value = serde_json::from_str(events[1]).unwrap();
        assert_eq!(first["id"], "hcp-completion-1");
        assert_eq!(first["object"], "text_completion");
        assert_eq!(first["model"], "qwen-test");
        assert_eq!(first["choices"][0]["text"], " first");
        assert!(first["choices"][0]["finish_reason"].is_null());
        assert_eq!(second["id"], first["id"]);
        assert_eq!(second["choices"][0]["text"], " second");
        assert_eq!(second["choices"][0]["finish_reason"], "length");
        assert_eq!(events[2], "[DONE]");
        coordinator.await.unwrap();
        assert_eq!(state.request_counter.load(Ordering::SeqCst), 1);
        assert_eq!(state.failed_counter.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn closed_coordinator_queue_returns_service_unavailable() {
        let (state, jobs) = test_state();
        drop(jobs);
        let response = build_router(state.clone())
            .oneshot(completion_request(
                r#"{"prompt":"queue closed","max_tokens":1}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(&body[..], b"Coordinator queue is closed");
        assert_eq!(state.request_counter.load(Ordering::SeqCst), 1);
        assert_eq!(state.failed_counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn malformed_completion_is_rejected_before_enqueue() {
        let (state, mut jobs) = test_state();
        let response = build_router(state.clone())
            .oneshot(completion_request(r#"{"max_tokens":1}"#))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(jobs.try_recv().is_err());
        assert_eq!(state.request_counter.load(Ordering::SeqCst), 0);
    }
}
