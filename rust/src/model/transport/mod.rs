//! 【KV 传输层模块】
//!
//! 定义分布式 worker 之间交换 KV block 的接口和实现：
//! - `trait`: KvTransport trait（send/recv/exchange）
//! - `block`: KvBlock 数据结构
//! - `tcp`: TCP 传输实现（测试用）
//! - `mock`: 内存传输实现（单元测试用）
//! - `quic`: QUIC 传输实现（生产环境，在 distributed/transport/quic.rs）

#![allow(dead_code)]

#[cfg(feature = "tch-backend")]
pub mod block;
#[cfg(feature = "tch-backend")]
pub mod mock;
#[cfg(feature = "tch-backend")]
pub mod tcp;
#[cfg(feature = "tch-backend")]
pub mod r#trait;

#[cfg(feature = "tch-backend")]
pub use block::{KvBlock, RingPacket, SelfDrivingPacket};
#[cfg(feature = "tch-backend")]
#[allow(unused_imports)]
pub use mock::{LinkedMockKvTransport, MockKvTransport};
#[cfg(feature = "tch-backend")]
pub use r#trait::{KvTransport, RingMessage};
#[cfg(feature = "tch-backend")]
#[allow(unused_imports)]
pub use tcp::TcpKvTransport;

#[cfg(test)]
#[cfg(feature = "tch-backend")]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use tch::{Device, Kind, Tensor};

    /// 验证 TcpKvTransport 通过本地 loopback 发送和接收 KV block 后，
    /// tensor 值与原始值完全一致（float32 原始字节传输，无精度损失）。
    #[test]
    fn test_tcp_kv_transport_roundtrip() {
        let device = Device::Cpu;
        let k = Tensor::randn([1, 4, 8, 16], (Kind::Float, device));
        let v = Tensor::randn([1, 4, 8, 16], (Kind::Float, device));

        let block = KvBlock {
            layer_idx: 2,
            global_seq_start: 16,
            global_seq_end: 24,
            k: k.shallow_clone(),
            v: v.shallow_clone(),
            micro_block_idx: 0,
            total_micro_blocks: 1,
            position_ids: None,
        };

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        // Server: receive the block and return it
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut transport = TcpKvTransport::new(stream, device).unwrap();
            transport.recv_kv_block().unwrap().unwrap()
        });

        // Client: send the block
        let client = thread::spawn(move || {
            let stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
            let mut transport = TcpKvTransport::new(stream, device).unwrap();
            transport.send_kv_block(&block).unwrap();
        });

        client.join().unwrap();
        let received = server.join().unwrap();

        // Verify metadata
        assert_eq!(received.layer_idx, 2);
        assert_eq!(received.global_seq_start, 16);
        assert_eq!(received.global_seq_end, 24);
        assert_eq!(received.k.size(), vec![1, 4, 8, 16]);
        assert_eq!(received.v.size(), vec![1, 4, 8, 16]);

        // Verify tensor values: float32 raw bytes round-trip should be exact
        let k_diff = (&k - &received.k).abs().max().double_value(&[]);
        let v_diff = (&v - &received.v).abs().max().double_value(&[]);
        println!(
            "TcpKvTransport roundtrip k_diff={} v_diff={}",
            k_diff, v_diff
        );
        assert_eq!(k_diff, 0.0, "K tensor changed after TCP roundtrip");
        assert_eq!(v_diff, 0.0, "V tensor changed after TCP roundtrip");
    }

    #[test]
    fn test_tcp_self_driving_packet_roundtrip() {
        let device = Device::Cpu;
        let packet = SelfDrivingPacket {
            layer_idx: 3,
            residual: Tensor::arange(8, (Kind::Float, device)).reshape([1, 1, 8]),
            normalized: Tensor::arange(8, (Kind::Float, device)).reshape([1, 1, 8]) * 0.5,
            position_ids: Tensor::from_slice(&[16_777_217_i64]).reshape([1, 1]),
            q: Tensor::arange(32, (Kind::Float, device)).reshape([1, 4, 1, 8]),
            attention_output: Tensor::arange(32, (Kind::Float, device)).reshape([1, 4, 1, 8])
                * 0.25,
            lse: Tensor::arange(4, (Kind::Float, device)).reshape([1, 4, 1]),
            assignee: 2,
            current_domain: 2,
            domains: 3,
            visited_domains: 1,
        };
        let expected = packet.clone();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut transport = TcpKvTransport::new(stream, device).unwrap();
            transport.recv_self_driving_packet().unwrap().unwrap()
        });
        let client = thread::spawn(move || {
            let stream = TcpStream::connect(address).unwrap();
            let mut transport = TcpKvTransport::new(stream, device).unwrap();
            transport.send_self_driving_packet(&packet).unwrap()
        });

        let sent_bytes = client.join().unwrap();
        let received = server.join().unwrap();
        assert!(sent_bytes > 0);
        assert_eq!(received.layer_idx, expected.layer_idx);
        assert_eq!(received.assignee, expected.assignee);
        assert_eq!(received.current_domain, expected.current_domain);
        assert_eq!(received.domains, expected.domains);
        assert_eq!(received.visited_domains, expected.visited_domains);
        assert_eq!(received.position_ids.kind(), Kind::Int64);
        for (name, actual, wanted) in [
            ("residual", &received.residual, &expected.residual),
            ("normalized", &received.normalized, &expected.normalized),
            (
                "position_ids",
                &received.position_ids,
                &expected.position_ids,
            ),
            ("q", &received.q, &expected.q),
            (
                "attention_output",
                &received.attention_output,
                &expected.attention_output,
            ),
            ("lse", &received.lse, &expected.lse),
        ] {
            let diff = (actual - wanted).abs().max().double_value(&[]);
            assert_eq!(diff, 0.0, "{name} changed after TCP roundtrip");
        }
    }
}
