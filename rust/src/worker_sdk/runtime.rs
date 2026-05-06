//! HCP Worker 协议运行时。
//!
//! 负责：
//! 1. 连接 coordinator（QUIC）
//! 2. 发送 handshake（domain_id + capacity_mb）
//! 3. 建立 peer KV ring 传输（QUIC）
//! 4. 循环接收 WorkerCommand，调用 Backend 处理，返回 WorkerResponse
//! 5. 优雅处理 Shutdown
//!
//! 模型计算层通过 `WorkerBackend` trait 解耦，默认实现为 `TchWorkerBackend`。

use crate::distributed_protocol::{
    recv_command_quic, send_response_quic, write_handshake_quic, WorkerCommand, WorkerResponse,
    WorkerHandshake,
};
use crate::model::kv_transport::KvTransport;
use crate::worker_sdk::backend::WorkerBackend;
use quinn::{RecvStream, SendStream};
use std::net::SocketAddr;
use tokio::runtime::Runtime;

/// HCP Worker 协议运行时。
///
/// 泛型参数 `B` 是后端实现（如 `TchWorkerBackend`）。
/// 运行时负责所有网络协议，后端只负责模型计算。
pub struct WorkerRuntime<B: WorkerBackend> {
    backend: B,
    domain_id: usize,
    coord_send: SendStream,
    coord_recv: RecvStream,
    rt: Runtime,
}

impl<B: WorkerBackend> WorkerRuntime<B> {
    /// 创建并初始化 WorkerRuntime。
    ///
    /// 此函数会：
    /// 1. 创建 tokio runtime
    /// 2. 创建 QUIC endpoint 并绑定到 `listen_addr`
    /// 3. 连接到 next peer（`next_peer_addr`）并接受 prev peer 的连接
    /// 4. 为每层创建 bidirectional QUIC stream（KV ring 传输）
    /// 5. 连接到 coordinator（`coordinator_addr`）
    /// 6. 调用 `backend.setup_kv_transports()` 将 per-layer transports 传给后端
    /// 7. 发送 handshake（domain_id + capacity_mb）
    ///
    /// # Arguments
    /// - `backend`: 已加载的后端实例
    /// - `domain_id`: 本 domain 的 ID
    /// - `num_domains`: 总 domain 数
    /// - `listen_addr`: 本 worker 监听 peer 连接的地址
    /// - `next_peer_addr`: 下一个 peer 的地址（ring 中的下游）
    /// - `coordinator_addr`: coordinator 的地址
    pub fn new(
        mut backend: B,
        domain_id: usize,
        num_domains: usize,
        listen_addr: SocketAddr,
        next_peer_addr: SocketAddr,
        coordinator_addr: SocketAddr,
    ) -> Result<Self, String> {
        println!(
            "[worker {domain_id}] starting, listen={listen_addr}, next_peer={next_peer_addr}, coordinator={coordinator_addr}"
        );

        let rt = Runtime::new().map_err(|e| format!("tokio runtime failed: {e}"))?;

        let (kv_transports, mut coord_send, coord_recv) = rt.block_on(async {
            Self::setup_network(
                domain_id,
                num_domains,
                backend.num_layers(),
                backend.device(),
                listen_addr,
                next_peer_addr,
                coordinator_addr,
            )
            .await
        })?;

        // 将 per-layer transports 传给后端
        backend.setup_kv_transports(kv_transports);
        println!("[worker {domain_id}] distributed domain setup complete");

        // 发送 handshake
        let capacity_mb = backend.capacity_mb();
        println!("[worker {domain_id}] capacity: {capacity_mb} MB");
        let handshake = WorkerHandshake {
            domain_id: domain_id as u64,
            capacity_mb,
        };
        write_handshake_quic(&mut coord_send, &handshake, rt.handle())
            .map_err(|e| format!("handshake write failed: {e}"))?;
        println!("[worker {domain_id}] handshake sent to coordinator");

        Ok(Self {
            backend,
            domain_id,
            coord_send,
            coord_recv,
            rt,
        })
    }

    /// 启动 command loop，阻塞直到收到 Shutdown。
    pub fn run(&mut self) -> Result<(), String> {
        let domain_id = self.domain_id;
        let rt_handle = self.rt.handle().clone();

        loop {
            println!("[worker {domain_id}] waiting for command...");
            let cmd = recv_command_quic(&mut self.coord_recv, &rt_handle)
                .map_err(|e| format!("recv_command failed: {e}"))?;
            println!("[worker {domain_id}] received command: {cmd:?}");

            match cmd {
                WorkerCommand::Prefill { chunk, seq_offset } => {
                    let seq_offset = seq_offset as usize;
                    let (logits_vec, global_seq_len) = self
                        .backend
                        .prefill(&chunk, seq_offset)
                        .map_err(|e| format!("prefill failed: {e}"))?;

                    let logits_bytes: Vec<u8> =
                        logits_vec.iter().flat_map(|&v| v.to_le_bytes()).collect();

                    let resp = WorkerResponse::PrefillDone {
                        last_logits_bytes: logits_bytes,
                        global_seq_len,
                    };
                    send_response_quic(&mut self.coord_send, &resp, &rt_handle)
                        .map_err(|e| format!("send PrefillDone failed: {e}"))?;
                }
                WorkerCommand::Decode(token) => {
                    let logits_vec = self
                        .backend
                        .decode(token)
                        .map_err(|e| format!("decode failed: {e}"))?;

                    let logits_bytes: Vec<u8> =
                        logits_vec.iter().flat_map(|&v| v.to_le_bytes()).collect();

                    let resp = WorkerResponse::DecodeDone { logits_bytes };
                    send_response_quic(&mut self.coord_send, &resp, &rt_handle)
                        .map_err(|e| format!("send DecodeDone failed: {e}"))?;
                }
                WorkerCommand::SyncGlobalSeqLen(len) => {
                    self.backend.sync_global_seq_len(len);
                    println!("[worker {domain_id}] synced global_seq_len = {len}");
                }
                WorkerCommand::Shutdown => {
                    println!("[worker {domain_id}] shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// 网络初始化：创建 QUIC endpoint、peer 连接、coordinator 连接、per-layer streams。
    async fn setup_network(
        domain_id: usize,
        num_domains: usize,
        num_layers: usize,
        device: tch::Device,
        listen_addr: SocketAddr,
        next_peer_addr: SocketAddr,
        coordinator_addr: SocketAddr,
    ) -> Result<(Vec<Box<dyn KvTransport>>, SendStream, RecvStream), String> {
        let endpoint = crate::quic_transport::create_endpoint(listen_addr)
            .map_err(|e| format!("QUIC endpoint bind failed: {e}"))?;
        let endpoint_for_accept = endpoint.clone();
        println!("[worker {domain_id}] QUIC endpoint bound to {listen_addr}");

        let dial_fut = endpoint
            .connect(next_peer_addr, "localhost")
            .map_err(|e| format!("connect to next peer failed: {e}"))?;

        let accept_fut = async move {
            loop {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    endpoint_for_accept.accept(),
                )
                .await
                {
                    Ok(Some(incoming)) => break incoming.await.expect("prev peer connection failed"),
                    Ok(None) => panic!("endpoint closed"),
                    Err(_) => {
                        println!("[worker {domain_id}] accept timeout, retrying...");
                        continue;
                    }
                }
            }
        };

        let (conn, prev_conn) = if num_domains == 2 {
            if domain_id == 0 {
                let c = dial_fut
                    .await
                    .map_err(|e| format!("connect to next peer failed: {e}"))?;
                println!("[worker {domain_id}] QUIC connection to next peer established");
                (c.clone(), c)
            } else {
                let c = accept_fut.await;
                println!("[worker {domain_id}] QUIC connection from prev peer established");
                (c.clone(), c)
            }
        } else {
            let dial_handle = tokio::spawn(dial_fut);
            let accept_handle = tokio::spawn(accept_fut);
            let c = dial_handle
                .await
                .map_err(|e| format!("dial task panicked: {e}"))?
                .map_err(|e| format!("connect to next peer failed: {e}"))?;
            println!("[worker {domain_id}] QUIC connection to next peer established");
            let p = accept_handle
                .await
                .map_err(|e| format!("accept task panicked: {e}"))?;
            println!("[worker {domain_id}] QUIC connection from prev peer established");
            (c, p)
        };

        // 为每层创建 bidirectional stream
        let mut outbound = Vec::with_capacity(num_layers);
        let mut inbound = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let (mut send, recv) = conn
                .open_bi()
                .await
                .map_err(|e| format!("layer {layer_idx} open_bi failed: {e}"))?;
            send.write_all(b"\x00")
                .await
                .map_err(|e| format!("layer {layer_idx} dummy write failed: {e}"))?;
            let (peer_send, peer_recv) = prev_conn
                .accept_bi()
                .await
                .map_err(|e| format!("layer {layer_idx} accept_bi failed: {e}"))?;
            outbound.push((send, recv));
            inbound.push((peer_send, peer_recv));
            if layer_idx % 4 == 0 || layer_idx == num_layers - 1 {
                println!("[worker {domain_id}] opened outbound & accepted inbound streams {layer_idx}/{num_layers}");
            }
        }

        // 连接到 coordinator
        println!("[worker {domain_id}] connecting to coordinator {coordinator_addr}");
        let coord_conn = endpoint
            .connect(coordinator_addr, "localhost")
            .map_err(|e| format!("connect to coordinator failed: {e}"))?
            .await
            .map_err(|e| format!("coordinator connection failed: {e}"))?;
        println!("[worker {domain_id}] QUIC connection to coordinator established");
        let (coord_send, coord_recv) = coord_conn
            .open_bi()
            .await
            .map_err(|e| format!("open_bi to coordinator failed: {e}"))?;

        // 构建 per-layer transports
        let kv_transports: Vec<Box<dyn KvTransport>> = outbound
            .into_iter()
            .zip(inbound.into_iter())
            .map(|((send, _recv), (_peer_send, peer_recv))| {
                let transport = crate::quic_transport::QuicKvTransport::new(
                    send,
                    peer_recv,
                    tokio::runtime::Handle::current(),
                    device,
                );
                Box::new(transport) as Box<dyn KvTransport>
            })
            .collect();

        Ok((kv_transports, coord_send, coord_recv))
    }
}
