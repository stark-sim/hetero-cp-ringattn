//! 【vLLM Worker 后端原型】
//!
//! 通过子进程 + JSON pipe 与 Python vLLM worker 通信。
//!
//! 设计要点：
//! - Rust 侧不加载任何模型权重，所有计算委托给 Python 子进程
//! - 通信协议：line-delimited JSON over stdin/stdout
//! - 单节点模式：不设置 KV ring transport（setup_kv_transports 是 noop）
//!
//! 当前限制：
//! - `device()` 返回 `Device::Cuda(0)`（trait 要求），但实际设备由 Python 侧控制
//! - 仅支持单节点（num_domains == 1），不支持分布式 KV ring

use crate::model::transport::KvTransport;
use crate::worker_sdk::backend::WorkerBackend;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use tch::Device;

/// JSON command sent to Python vLLM worker process.
#[derive(Serialize)]
struct VllmCommand {
    cmd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seq_offset: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_tokens: Option<Vec<(u64, i64)>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    len: Option<usize>,
}

/// JSON response from Python vLLM worker process.
#[derive(Deserialize)]
struct VllmResponse {
    status: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    logits: Vec<f32>,
    #[serde(default)]
    global_seq_len: usize,
    #[serde(default)]
    request_logits: Vec<(u64, Vec<f32>)>,
}

/// vLLM subprocess backend.
///
/// Spawns a Python process that loads vLLM and handles prefill/decode via
/// line-delimited JSON over stdin/stdout.
pub struct VllmWorkerBackend {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    domain_id: usize,
    num_layers: usize,
    capacity_mb: u64,
}

impl VllmWorkerBackend {
    /// Spawn a Python vLLM worker subprocess.
    ///
    /// # Arguments
    /// - `model_dir`: HuggingFace model directory path
    /// - `domain_id`: domain ID for logging
    /// - `python_path`: path to the Python vLLM worker script
    pub fn new(
        model_dir: &str,
        domain_id: usize,
        python_path: &str,
    ) -> Result<Self, String> {
        println!("[VllmWorkerBackend {domain_id}] spawning Python worker: {python_path}");

        let mut child = std::process::Command::new("python3")
            .arg(python_path)
            .arg("--model-dir")
            .arg(model_dir)
            .arg("--backend")
            .arg("vllm")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawn python worker failed: {e}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "failed to capture child stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "failed to capture child stdout".to_string())?;

        let mut backend = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            domain_id,
            num_layers: 0,     // populated by handshake
            capacity_mb: 8192, // placeholder until Python reports real value
        };

        // Handshake: query Python worker for model metadata.
        let resp = backend.send_cmd(&VllmCommand {
            cmd: "handshake".to_string(),
            request_id: None,
            tokens: None,
            seq_offset: None,
            request_tokens: None,
            len: None,
        })?;

        if resp.status != "ok" {
            return Err(format!("handshake failed: {}", resp.message));
        }

        // Parse metadata from handshake response.
        // We encode num_layers and capacity_mb in the message field as JSON.
        let meta: serde_json::Value = serde_json::from_str(&resp.message)
            .map_err(|e| format!("handshake metadata parse failed: {e}"))?;
        backend.num_layers = meta["num_layers"].as_u64().unwrap_or(24) as usize;
        backend.capacity_mb = meta["capacity_mb"].as_u64().unwrap_or(8192);

        println!(
            "[VllmWorkerBackend {domain_id}] handshake ok, num_layers={}, capacity_mb={}",
            backend.num_layers, backend.capacity_mb
        );

        Ok(backend)
    }

    /// Send a command and wait for response.
    fn send_cmd(&mut self, cmd: &VllmCommand) -> Result<VllmResponse, String> {
        let line = serde_json::to_string(cmd)
            .map_err(|e| format!("serialize command failed: {e}"))?;

        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|e| format!("write to child stdin failed: {e}"))?;

        let mut response_line = String::new();
        self.stdout
            .read_line(&mut response_line)
            .map_err(|e| format!("read from child stdout failed: {e}"))?;

        let resp: VllmResponse = serde_json::from_str(&response_line)
            .map_err(|e| format!("parse response failed: {e} (raw: {response_line})"))?;

        if resp.status != "ok" {
            return Err(format!("Python worker error: {}", resp.message));
        }

        Ok(resp)
    }
}

impl WorkerBackend for VllmWorkerBackend {
    fn setup_kv_transports(&mut self, _transports: Vec<Box<dyn KvTransport>>) {
        // vLLM single-node backend does not use KV ring transport.
        println!("[VllmWorkerBackend {}] setup_kv_transports: noop (single-node)", self.domain_id);
    }

    fn prefill(
        &mut self,
        chunk: &[i64],
        seq_offset: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        let resp = self.send_cmd(&VllmCommand {
            cmd: "prefill".to_string(),
            request_id: None,
            tokens: Some(chunk.to_vec()),
            seq_offset: Some(seq_offset),
            request_tokens: None,
            len: None,
        })?;
        Ok((resp.logits, resp.global_seq_len))
    }

    fn decode(&mut self, token: i64) -> Result<Vec<f32>, String> {
        let resp = self.send_cmd(&VllmCommand {
            cmd: "decode".to_string(),
            request_id: None,
            tokens: Some(vec![token]),
            seq_offset: None,
            request_tokens: None,
            len: None,
        })?;
        Ok(resp.logits)
    }

    fn prefill_request(
        &mut self,
        request_id: u64,
        chunk: &[i64],
        seq_offset: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        let resp = self.send_cmd(&VllmCommand {
            cmd: "prefill_request".to_string(),
            request_id: Some(request_id),
            tokens: Some(chunk.to_vec()),
            seq_offset: Some(seq_offset),
            request_tokens: None,
            len: None,
        })?;
        Ok((resp.logits, resp.global_seq_len))
    }

    fn decode_request(&mut self, request_id: u64, token: i64) -> Result<Vec<f32>, String> {
        let resp = self.send_cmd(&VllmCommand {
            cmd: "decode_request".to_string(),
            request_id: Some(request_id),
            tokens: Some(vec![token]),
            seq_offset: None,
            request_tokens: None,
            len: None,
        })?;
        Ok(resp.logits)
    }

    fn decode_batch(&mut self, request_tokens: &[(u64, i64)]) -> Result<Vec<(u64, Vec<f32>)>, String> {
        let resp = self.send_cmd(&VllmCommand {
            cmd: "decode_batch".to_string(),
            request_id: None,
            tokens: None,
            seq_offset: None,
            request_tokens: Some(request_tokens.to_vec()),
            len: None,
        })?;
        Ok(resp.request_logits)
    }

    fn sync_global_seq_len_for_request(&mut self, _request_id: u64, _len: usize) {
        // vLLM manages sequence length internally.
    }

    fn sync_global_seq_len(&mut self, _len: usize) {
        // No-op for vLLM backend.
    }

    fn capacity_mb(&self) -> u64 {
        self.capacity_mb
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn device(&self) -> Device {
        // Trait requires returning a tch::Device, but vLLM manages its own device.
        // We return Cuda(0) as the canonical GPU device; the actual device is
        // controlled by the Python subprocess.
        Device::Cuda(0)
    }
}

impl Drop for VllmWorkerBackend {
    fn drop(&mut self) {
        // Send shutdown command to Python process.
        let _ = self.send_cmd(&VllmCommand {
            cmd: "shutdown".to_string(),
            request_id: None,
            tokens: None,
            seq_offset: None,
            request_tokens: None,
            len: None,
        });
        let _ = self.child.wait();
    }
}
