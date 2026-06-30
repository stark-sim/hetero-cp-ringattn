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
use std::io::{self, BufRead, BufReader, Write};
use std::time::{Duration, Instant};
use tch::Device;

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

/// Poll a file descriptor for readability with a timeout (Unix only).
#[cfg(unix)]
fn poll_readable(fd: i32, timeout: Duration) -> io::Result<bool> {
    let mut pollfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let timeout_ms = timeout.as_millis().min(i32::MAX as u128) as i32;
    let ret = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(ret > 0)
}

/// Stub for non-Unix platforms (always returns true, falls back to blocking read).
#[cfg(not(unix))]
fn poll_readable(_fd: i32, _timeout: Duration) -> io::Result<bool> {
    Ok(true)
}

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
        let python_exe = std::env::var("HCP_PYTHON_PATH").unwrap_or_else(|_| "python3".to_string());
        println!("[VllmWorkerBackend {domain_id}] spawning Python worker: {python_path} (interpreter: {python_exe})");

        let python_backend = std::env::var("HCP_PYTHON_BACKEND_TYPE")
            .unwrap_or_else(|_| "vllm".to_string());
        let mut child = std::process::Command::new(&python_exe)
            .arg(python_path)
            .arg("--model-dir")
            .arg(model_dir)
            .arg("--backend")
            .arg(&python_backend)
            .env_remove("DYLD_LIBRARY_PATH")
            .env_remove("LD_LIBRARY_PATH")
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

    /// Default timeout for waiting on Python worker response.
    /// Covers model load + inference; vllm-metal first init can take 60-90s.
    const DEFAULT_CMD_TIMEOUT: Duration = Duration::from_secs(120);

    /// Send a command and wait for response.
    /// Skips any non-JSON lines on stdout (e.g., spurious child process logs).
    /// Guards against infinite loops with a max skip limit (safety bound).
    /// Uses `poll` with timeout to avoid blocking forever on a hung child.
    fn send_cmd(&mut self, cmd: &VllmCommand) -> Result<VllmResponse, String> {
        const MAX_SKIP_LINES: usize = 1024;
        let deadline = Instant::now() + Self::DEFAULT_CMD_TIMEOUT;

        let line = serde_json::to_string(cmd)
            .map_err(|e| format!("serialize command failed: {e}"))?;

        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|e| format!("write to child stdin failed: {e}"))?;

        for skip_count in 0..MAX_SKIP_LINES {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                // Timeout — check if child exited
                match self.child.try_wait() {
                    Ok(Some(status)) => {
                        return Err(format!(
                            "cmd timeout: child exited with status {status} after {} skips",
                            skip_count
                        ));
                    }
                    Ok(None) => {
                        return Err(format!(
                            "cmd timeout: child still alive but unresponsive after {} skips (timeout={:?})",
                            skip_count, Self::DEFAULT_CMD_TIMEOUT
                        ));
                    }
                    Err(e) => {
                        return Err(format!(
                            "cmd timeout: cannot check child status: {e}"
                        ));
                    }
                }
            }

            // Wait for data on stdout with remaining timeout.
            // IMPORTANT: Check BufReader's internal buffer first — if it already
            // has data from a previous read, we must NOT poll (which would wait
            // for new FD data and ignore the buffered content).
            #[cfg(unix)]
            {
                let has_buffered = !self.stdout.buffer().is_empty();
                if !has_buffered {
                    let fd = self.stdout.get_ref().as_raw_fd();
                    match poll_readable(fd, remaining) {
                        Ok(true) => {} // new data available on FD
                        Ok(false) => {
                            // poll timed out — loop back to check deadline / child status
                            continue;
                        }
                        Err(e) => {
                            return Err(format!("poll on child stdout failed: {e}"));
                        }
                    }
                }
            }

            let mut response_line = String::new();
            let n = self.stdout
                .read_line(&mut response_line)
                .map_err(|e| format!("read from child stdout failed: {e}"))?;

            if n == 0 {
                return Err("child stdout closed without valid JSON response".to_string());
            }

            let trimmed = response_line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match serde_json::from_str::<VllmResponse>(trimmed) {
                Ok(resp) => {
                    if resp.status != "ok" {
                        return Err(format!("Python worker error: {}", resp.message));
                    }
                    return Ok(resp);
                }
                Err(_) => {
                    eprintln!(
                        "[VllmWorkerBackend {}] skipping non-JSON stdout: {}",
                        self.domain_id, trimmed
                    );
                    continue;
                }
            }
        }

        Err(format!(
            "exceeded max {MAX_SKIP_LINES} non-JSON lines on stdout; child process may be misbehaving"
        ))
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
        _position_ids: Option<&[i64]>,
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
        _position_ids: Option<&[i64]>,
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

    fn release_request(&mut self, request_id: u64) {
        // Best-effort release; ignore errors since the request may already be gone.
        let _ = self.send_cmd(&VllmCommand {
            cmd: "release_request".to_string(),
            request_id: Some(request_id),
            tokens: None,
            seq_offset: None,
            request_tokens: None,
            len: None,
        });
    }

    fn capacity_mb(&self) -> u64 {
        self.capacity_mb
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn device(&self) -> Device {
        // Trait requires returning a tch::Device, but vLLM manages its own device.
        // Allow override via HCP_VLLM_DEVICE env var for accurate reporting.
        if let Ok(name) = std::env::var("HCP_VLLM_DEVICE") {
            match name.as_str() {
                "cpu" => Device::Cpu,
                "mps" => Device::Mps,
                "cuda" => Device::Cuda(0),
                "hip" => Device::Cuda(0),
                _ => {
                    if let Some(idx) = name.strip_prefix("cuda:") {
                        if let Ok(i) = idx.parse::<usize>() {
                            Device::Cuda(i)
                        } else {
                            Device::Cuda(0)
                        }
                    } else {
                        Device::Cuda(0)
                    }
                }
            }
        } else {
            Device::Cuda(0)
        }
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vllm_backend_mock_handshake_prefill_decode() {
        // Use mock Python backend so no actual model or GPU is needed.
        std::env::set_var("HCP_PYTHON_BACKEND_TYPE", "mock");
        let repo_root = std::env::var("CARGO_MANIFEST_DIR")
            .map(|d| std::path::Path::new(&d).parent().unwrap().to_path_buf())
            .unwrap_or_else(|_| std::env::current_dir().unwrap());
        let python_path = repo_root.join("python/hcp_worker_process.py");
        let model_dir = repo_root.join("models/Qwen2-0.5B");

        let mut backend = VllmWorkerBackend::new(
            model_dir.to_str().unwrap(),
            0,
            python_path.to_str().unwrap(),
        )
        .expect("create vllm backend failed");

        // Handshake already happened in new(); verify metadata.
        assert_eq!(backend.num_layers(), 2);
        assert_eq!(backend.capacity_mb(), 4096);
        assert!(backend.device() == Device::Cuda(0));

        // Test prefill.
        let (logits, global_seq_len) = backend.prefill(&[1, 2, 3], 0, None).unwrap();
        assert_eq!(logits.len(), 100); // mock vocab_size = 100
        assert_eq!(global_seq_len, 3);

        // Test decode.
        let logits = backend.decode(4).unwrap();
        assert_eq!(logits.len(), 100);

        // Test decode_batch.
        let results = backend.decode_batch(&[(42, 5), (43, 6)]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 42);
        assert_eq!(results[0].1.len(), 100);
        assert_eq!(results[1].0, 43);
        assert_eq!(results[1].1.len(), 100);

        // Test request-aware methods.
        let (logits, global_seq_len) = backend.prefill_request(99, &[10, 11], 0, None).unwrap();
        assert_eq!(logits.len(), 100);
        assert_eq!(global_seq_len, 2);

        let logits = backend.decode_request(99, 12).unwrap();
        assert_eq!(logits.len(), 100);

        // Shutdown happens via Drop.
        println!("test_vllm_backend_mock_handshake_prefill_decode passed");
    }
}
