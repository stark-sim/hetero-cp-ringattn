//! Minimal NIXL block transport probe (form-B S2 remote smoke).
//!
//! Feature-gated: only builds with nixl-backend on a host with libnixl_capi.so.
//! Proves the FFI links and the agent lifecycle works (create agent, register a
//! block, fetch local metadata) before the ring wiring (S3).

#![allow(dead_code)]

#[cfg(feature = "nixl-backend")]
fn main() {
    use hcp_ringattn_rust::{KvBlockTransport, NixlBlockTransport};
    use tch::{Device, Kind, Tensor};

    let device = Device::cuda_if_available();
    let mut transport = NixlBlockTransport::new("hcp-probe-agent").expect("create agent");
    println!("[nixl-probe] agent created: {}", transport.agent_name());

    // Register a small device block (1x2x3x4 f32 = 96 bytes).
    let tensor = Tensor::arange(24, (Kind::Float, device)).reshape([1, 2, 3, 4]);
    let handle = transport.register_block(&tensor).expect("register block");
    println!(
        "[nixl-probe] registered block id={} len={} addr={}",
        handle.id, handle.desc.len, handle.desc.addr
    );

    let md = transport.local_metadata().expect("local metadata");
    println!("[nixl-probe] local metadata bytes={}", md.len());
    println!("[nixl-probe] OK");
}

#[cfg(not(feature = "nixl-backend"))]
fn main() {
    eprintln!("nixl-probe requires --features nixl-backend");
    std::process::exit(2);
}
