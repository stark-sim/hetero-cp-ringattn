//! Distributed inference worker process.
//!
//! Supports both single-domain and multi-domain modes.
//! In multi-domain mode (--local-domain-ids), a single process hosts multiple
//! logical workers that share model weights (via shallow_clone) while each
//! maintains its own KV cache and coordinator connection.
//!
//! 本模块已重构为使用 `worker_sdk`：
//! - `TchWorkerBackend` 负责模型加载和 forward 计算
//! - `WorkerRuntime` 负责协议循环和网络传输
//! - 两者通过 `WorkerBackend` trait 解耦

use crate::model::model::LlamaModel;
use crate::model::{ModelConfig, ModelWeights};
use crate::worker_sdk::{TchWorkerBackend, WorkerRuntime};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tch::Device;

/// 【Domain 配置】在 ring 拓扑中，每个 domain 需要：
/// - listen_addr: 本 domain 的监听地址（prev peer 会主动连这里）
/// - next_peer_addr: 本 domain 的 next peer 地址（本 domain 会主动 dial 这里）
///
/// Ring 拓扑由所有 domain 的 listen_addr + next_peer_addr 共同定义：
/// domain(i).next_peer_addr == domain(i+1).listen_addr
/// domain(i).prev 不需要地址，因为 prev 会主动 dial domain(i).listen_addr
#[derive(Debug)]
struct DomainConfig {
    domain_id: usize,
    listen_addr: String,
    next_peer_addr: String,
}

/// A reusable barrier for synchronizing domains within the same process.
/// All domains must arrive before any domain proceeds, then the barrier resets
/// automatically for the next synchronization point.
struct ResetBarrier {
    arrived: AtomicUsize,
    generation: AtomicUsize,
    target: usize,
}

impl ResetBarrier {
    fn new(target: usize) -> Self {
        Self {
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            target,
        }
    }

    fn wait(&self) {
        let my_gen = self.generation.load(Ordering::Relaxed);
        let count = self.arrived.fetch_add(1, Ordering::SeqCst) + 1;
        if count >= self.target {
            self.arrived.store(0, Ordering::SeqCst);
            self.generation.fetch_add(1, Ordering::SeqCst);
        } else {
            while self.generation.load(Ordering::Relaxed) == my_gen {
                std::thread::yield_now();
            }
        }
    }
}

#[derive(Debug)]
struct WorkerArgs {
    domain_configs: Vec<DomainConfig>,
    model_dir: String,
    coordinator_addr: String,
    num_domains: usize,
}

fn parse_args() -> WorkerArgs {
    let mut domain_id = 0usize;
    let mut _seq_offset = 0i64;
    let mut model_dir = String::new();
    let mut listen_addr = String::new();
    let mut next_peer_addr = String::new();
    let mut coordinator_addr = String::new();
    let mut num_domains = 2usize;

    let mut local_domain_ids: Option<Vec<usize>> = None;
    let mut listen_addrs: Option<Vec<String>> = None;
    let mut next_peer_addrs: Option<Vec<String>> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--distributed-role" => { let _ = args.next(); }
            "--domain-id" => domain_id = args.next().unwrap().parse().unwrap(),
            "--seq-offset" => _seq_offset = args.next().unwrap().parse().unwrap(),
            "--model-dir" => model_dir = args.next().unwrap(),
            "--listen-addr" => listen_addr = args.next().unwrap(),
            "--next-peer-addr" => next_peer_addr = args.next().unwrap(),
            "--coordinator-addr" => coordinator_addr = args.next().unwrap(),
            "--num-domains" => num_domains = args.next().unwrap().parse().unwrap(),
            "--local-domain-ids" => {
                local_domain_ids = Some(args.next().unwrap().split(',').map(|s| s.parse().unwrap()).collect());
            }
            "--listen-addrs" => {
                listen_addrs = Some(args.next().unwrap().split(',').map(|s| s.to_string()).collect());
            }
            "--next-peer-addrs" => {
                next_peer_addrs = Some(args.next().unwrap().split(',').map(|s| s.to_string()).collect());
            }
            _ => eprintln!("[worker] unknown arg: {arg}"),
        }
    }

    let domain_configs = if let Some(ids) = local_domain_ids {
        let listens = listen_addrs.expect("--listen-addrs required when --local-domain-ids is set");
        let peers = next_peer_addrs.expect("--next-peer-addrs required when --local-domain-ids is set");
        assert_eq!(ids.len(), listens.len(), "--local-domain-ids and --listen-addrs must have same length");
        assert_eq!(ids.len(), peers.len(), "--local-domain-ids and --next-peer-addrs must have same length");
        ids.into_iter()
            .zip(listens)
            .zip(peers)
            .map(|((id, l), p)| DomainConfig { domain_id: id, listen_addr: l, next_peer_addr: p })
            .collect()
    } else {
        vec![DomainConfig {
            domain_id,
            listen_addr,
            next_peer_addr,
        }]
    };

    WorkerArgs {
        domain_configs,
        model_dir,
        coordinator_addr,
        num_domains,
    }
}

fn select_device() -> Device {
    if let Ok(name) = std::env::var("HCP_TORCH_DEVICE").or_else(|_| std::env::var("HCP_TCH_DEVICE")) {
        match name.as_str() {
            "cpu" => Device::Cpu,
            "mps" => Device::Mps,
            "cuda" => Device::Cuda(0),
            _ => {
                if let Some(idx) = name.strip_prefix("cuda:") {
                    if let Ok(i) = idx.parse::<usize>() {
                        Device::Cuda(i)
                    } else {
                        Device::Cpu
                    }
                } else {
                    Device::Cpu
                }
            }
        }
    } else if cfg!(target_os = "macos") && tch::utils::has_mps() {
        Device::Mps
    } else if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

/// Worker 进程主入口。
///
/// 整体流程：
/// 1. 解析参数（支持单 domain 和 `--local-domain-ids` 多 domain 模式）
/// 2. 选择计算设备（env > MPS > CUDA > CPU）
/// 3. 加载 ModelConfig 和 ModelWeights（只加载一次）
/// 4. 为每个 domain 创建一个线程：
///    a. shallow_clone 权重（共享参数，不复制内存）
///    b. 创建独立的 LlamaModel（每个 domain 有自己的 KV cache）
///    c. 创建 WorkerRuntime（网络初始化 + handshake）
///    d. ResetBarrier 同步所有 domain
///    e. 进入 runtime.run() 事件循环
/// 5. 等待所有线程结束
pub fn run() {
    // Install rustls ring crypto provider once per process (required by rustls 0.23)
    let _ = rustls::crypto::ring::default_provider().install_default();

    let args = parse_args();
    let device = select_device();

    // Load model config and weights once per process.
    // All domains share the weights via shallow_clone.
    let config_path = Path::new(&args.model_dir).join("config.json");
    let config = ModelConfig::from_file(&config_path).expect("load config failed");
    let weights = ModelWeights::from_dir(&args.model_dir, device).expect("load weights failed");
    println!("[multi-worker] loaded model weights once for {} domain(s)", args.domain_configs.len());

    // Barrier to synchronize all domains before command loop.
    // 所有 domain 必须在 runtime.run() 之前到达 barrier，
    // 确保 peer 连接在任何一个 domain 开始发送 KV 之前全部建立。
    let barrier = Arc::new(ResetBarrier::new(args.domain_configs.len()));

    let num_domains = args.num_domains;
    let coordinator_addr = args.coordinator_addr;
    let model_dir = args.model_dir;

    let mut handles = Vec::new();
    for domain_config in args.domain_configs {
        let b = barrier.clone();
        let cfg = config.clone();
        let coord = coordinator_addr.clone();
        let _dir = model_dir.clone();
        // shallow_clone 权重：复制 Tensor 的 metadata 指针，不复制底层数据。
        // 这样多个 domain 共享同一份模型参数，显著降低显存占用。
        let w = ModelWeights {
            #[cfg(feature = "tch-backend")]
            tensors: weights.tensors.iter().map(|(k, v)| (k.clone(), v.shallow_clone())).collect(),
            #[cfg(not(feature = "tch-backend"))]
            tensors: weights.tensors.clone(),
        };
        let handle = std::thread::spawn(move || {
            // 每个 domain 需要独立的 LlamaModel（因为 KV cache 是 domain-specific 的），
            // 但权重共享（shallow_clone）。
            let backend = TchWorkerBackend::from_model(
                LlamaModel::from_weights(cfg, &w, device, num_domains)
                    .expect("build model failed"),
                device,
                domain_config.domain_id,
            );

            let listen_addr: SocketAddr = domain_config.listen_addr.parse()
                .expect("invalid listen_addr");
            let next_peer_addr: SocketAddr = domain_config.next_peer_addr.parse()
                .expect("invalid next_peer_addr");
            let coordinator_addr: SocketAddr = coord.parse()
                .expect("invalid coordinator_addr");

            let mut runtime = WorkerRuntime::new(
                Box::new(backend),
                domain_config.domain_id,
                num_domains,
                listen_addr,
                next_peer_addr,
                coordinator_addr,
            )
            .expect("create runtime failed");

            // 同步点：所有 domain 的网络初始化完成后才能开始处理命令。
            b.wait();
            runtime.run().expect("runtime failed");
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("domain worker thread panicked");
    }
    println!("[multi-worker] all domains finished");
}
