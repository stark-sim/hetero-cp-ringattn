//! NIXL cross-machine block transfer probe (form-B S3a).
//!
//! Feature-gated: only builds with nixl-backend on a host with libnixl_capi.so.
//! Proves the FULL cross-machine lifecycle that S2 did not cover:
//!   register -> exchange metadata+desc (via files, temporary probe channel) ->
//!   load_remote_metadata -> submit_transfer -> poll_transfers -> dump + verify.
//!
//! Each node runs the SAME symmetric flow (white and pearl each register a src
//! and a dest block and each write the peer's dest block), so a single run
//! proves BOTH directions: CUDA->ROCm and ROCm->CUDA.
//!
//! The host-side script moves the metadata/desc/done files between the two
//! hosts. That file exchange is a throwaway probe channel, NOT the HCP
//! side-channel architecture (that is S3b: reuse the coordinator control plane).

#![allow(dead_code)]

use std::path::Path;
use std::time::Duration;

#[cfg(feature = "nixl-backend")]
mod probe {
    use super::*;
    use hcp_ringattn_rust::{KvBlockTransport, NixlBlockTransport, RemoteBlockDesc};
    use std::fs;
    use tch::{Device, Kind, Tensor};

    const SHAPE: [i64; 4] = [1, 2, 3, 4];
    const NUMEL: i64 = 24;

    fn get_arg(args: &[String], name: &str) -> Option<String> {
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if a == name {
                return it.next().cloned();
            }
        }
        None
    }

    fn wait_for_file(path: &str, timeout_secs: u64) -> Result<(), String> {
        let start = std::time::Instant::now();
        loop {
            // Wait for a NON-EMPTY file: the host script writes the peer files
            // with a tmp+rename so an existing-but-empty file (scp mid-write)
            // must not be read as a complete descriptor.
            let non_empty = Path::new(path).exists()
                && std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false);
            if non_empty {
                return Ok(());
            }
            if start.elapsed().as_secs() > timeout_secs {
                return Err(format!("timed out waiting for {path}"));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn dump_tensor(t: &Tensor, path: &str) -> Result<usize, String> {
        let flat = t
            .contiguous()
            .view(-1)
            .to_kind(Kind::Float)
            .to_device(Device::Cpu);
        let values: Vec<f32> =
            Vec::try_from(&flat).map_err(|e| format!("tensor to vec failed: {e}"))?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(path, &bytes).map_err(|e| format!("write {path}: {e}"))?;
        Ok(bytes.len())
    }

    fn run(
        agent: &str,
        seed: f32,
        device: Device,
        md_out: &str,
        md_in: &str,
        desc_out: &str,
        desc_in: &str,
        src_dump_out: &str,
        dest_dump_out: &str,
        done_out: &str,
        done_in: &str,
    ) -> Result<(), String> {
        // S3a validates the cross-vendor (CUDA<->ROCm) path, which must stage
        // through host DRAM (GPU-direct VRAM put has no cross-vendor UCX
        // protocol), so the probe defaults to --device cpu.
        let mut transport =
            NixlBlockTransport::new(agent).map_err(|e| format!("create agent: {e}"))?;
        println!("[xfer-probe] agent created: {}", transport.agent_name());

        // src = arange + seed (unique per node so each direction is verifiable);
        // dest = zeros, to be overwritten by the peer's transfer.
        let src = Tensor::arange(NUMEL, (Kind::Float, device))
            .f_add_scalar(seed as f64)
            .map_err(|e| format!("add scalar: {e:?}"))?
            .reshape(SHAPE);
        let dest = Tensor::zeros(SHAPE, (Kind::Float, device));

        let src_handle = transport
            .register_block(&src)
            .map_err(|e| format!("register src: {e}"))?;
        let dest_handle = transport
            .register_block(&dest)
            .map_err(|e| format!("register dest: {e}"))?;
        println!(
            "[xfer-probe] src block id={} len={} addr={}",
            src_handle.id, src_handle.desc.len, src_handle.desc.addr
        );
        println!(
            "[xfer-probe] dest block id={} len={} addr={}",
            dest_handle.id, dest_handle.desc.len, dest_handle.desc.addr
        );

        // Export local MD + dest desc; dump src for byte-for-byte comparison.
        let md = transport
            .local_metadata()
            .map_err(|e| format!("local md: {e}"))?;
        fs::write(md_out, &md).map_err(|e| format!("write md: {e}"))?;
        println!("[xfer-probe] wrote local md {} bytes", md.len());

        let remote_desc = RemoteBlockDesc {
            agent: agent.to_string(),
            block_id: dest_handle.id,
            desc: dest_handle.desc.clone(),
        };
        let desc_json =
            serde_json::to_string(&remote_desc).map_err(|e| format!("serialize desc: {e}"))?;
        fs::write(desc_out, desc_json).map_err(|e| format!("write desc: {e}"))?;
        println!("[xfer-probe] wrote dest desc");

        let n = dump_tensor(&src, src_dump_out).map_err(|e| format!("dump src: {e}"))?;
        println!("[xfer-probe] dumped src {} bytes", n);

        // Load the peer's MD + dest desc (moved by the host script).
        wait_for_file(md_in, 180)?;
        wait_for_file(desc_in, 180)?;
        let peer_md = fs::read(md_in).map_err(|e| format!("read md_in: {e}"))?;
        let peer_agent = transport
            .load_remote_metadata(&peer_md)
            .map_err(|e| format!("load remote md: {e}"))?;
        println!("[xfer-probe] loaded remote md agent={}", peer_agent);

        let desc_json = fs::read_to_string(desc_in).map_err(|e| format!("read desc_in: {e}"))?;
        let remote: RemoteBlockDesc =
            serde_json::from_str(&desc_json).map_err(|e| format!("parse desc: {e}"))?;
        println!(
            "[xfer-probe] remote dest desc agent={} block_id={} len={}",
            remote.agent, remote.block_id, remote.desc.len
        );

        transport
            .submit_transfer(&src_handle, &remote)
            .map_err(|e| format!("submit transfer: {e}"))?;
        println!("[xfer-probe] transfer submitted (Write -> peer dest)");

        let mut done = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(180);
        while done.is_empty() {
            done = transport
                .poll_transfers()
                .map_err(|e| format!("poll: {e}"))?;
            if std::time::Instant::now() > deadline {
                return Err("timed out polling transfer".to_string());
            }
            if done.is_empty() {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        for c in &done {
            println!(
                "[xfer-probe] transfer complete block_id={} telemetry_bytes={}",
                c.block_id, c.bytes
            );
        }
        println!(
            "[xfer-probe] wire_bytes_sent={} wire_bytes_recv={}",
            transport.wire_bytes_sent(),
            transport.wire_bytes_recv()
        );

        fs::write(done_out, b"done").map_err(|e| format!("write done: {e}"))?;
        println!("[xfer-probe] wrote done signal");

        // Wait for the peer to finish its transfer into OUR dest, then dump it.
        wait_for_file(done_in, 180)?;
        println!("[xfer-probe] peer done signal observed");

        let n = dump_tensor(&dest, dest_dump_out).map_err(|e| format!("dump dest: {e}"))?;
        println!("[xfer-probe] dumped dest {} bytes", n);
        println!("[xfer-probe] OK");
        Ok(())
    }

    pub fn main() -> Result<(), String> {
        let args: Vec<String> = std::env::args().collect();
        let agent = get_arg(&args, "--agent").ok_or("missing --agent")?;
        let seed: f32 = get_arg(&args, "--seed")
            .unwrap_or_else(|| "0".to_string())
            .parse()
            .map_err(|_| "bad --seed")?;
        let device = match get_arg(&args, "--device").as_deref() {
            Some("cuda") => Device::cuda_if_available(),
            _ => Device::Cpu,
        };
        let md_out = get_arg(&args, "--md-out").ok_or("missing --md-out")?;
        let md_in = get_arg(&args, "--md-in").ok_or("missing --md-in")?;
        let desc_out = get_arg(&args, "--desc-out").ok_or("missing --desc-out")?;
        let desc_in = get_arg(&args, "--desc-in").ok_or("missing --desc-in")?;
        let src_dump_out = get_arg(&args, "--src-dump-out").ok_or("missing --src-dump-out")?;
        let dest_dump_out = get_arg(&args, "--dest-dump-out").ok_or("missing --dest-dump-out")?;
        let done_out = get_arg(&args, "--done-out").ok_or("missing --done-out")?;
        let done_in = get_arg(&args, "--done-in").ok_or("missing --done-in")?;

        run(
            &agent,
            seed,
            device,
            &md_out,
            &md_in,
            &desc_out,
            &desc_in,
            &src_dump_out,
            &dest_dump_out,
            &done_out,
            &done_in,
        )
    }
}

#[cfg(feature = "nixl-backend")]
fn main() {
    if let Err(e) = probe::main() {
        eprintln!("[xfer-probe] FAILED: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "nixl-backend"))]
fn main() {
    eprintln!("nixl-xfer-probe requires --features nixl-backend");
    std::process::exit(2);
}
