//! NIXL ring probe (S4-2a): hop-by-hop KV circulation on an N-node ring.
//!
//! Each node registers a "current" block (initially its own KV pattern) and a
//! "recv" buffer. Every round it transfers the current block into its
//! successor's recv buffer (block-direct Write), waits for its own recv buffer
//! to be filled by its predecessor, then swaps (recv into current) so the
//! received KV is what it forwards next round. After N-1 rounds every node has
//! seen every peer's KV.
//!
//! The host script drives desc exchange + per-round done sync via files (the
//! throwaway probe channel; production uses the coordinator control plane).
//! This proves the double-buffering + forward + sync mechanism that S4-2b wires
//! into ring_attention.

#![allow(dead_code)]

use std::path::Path;
use std::time::Duration;

#[cfg(feature = "nixl-backend")]
mod probe {
    use super::*;
    use hcp_ringattn_rust::{KvBlockTransport, NixlBlockTransport, RemoteBlockDesc};
    use std::fs;
    use tch::{Device, Kind, Tensor};

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

    pub fn main() -> Result<(), String> {
        let args: Vec<String> = std::env::args().collect();
        let agent = get_arg(&args, "--agent").ok_or("missing --agent")?;
        let seed: f32 = get_arg(&args, "--seed")
            .unwrap_or_else(|| "0".to_string())
            .parse()
            .map_err(|_| "bad --seed")?;
        let seq: i64 = get_arg(&args, "--seq")
            .unwrap_or_else(|| "64".to_string())
            .parse()
            .map_err(|_| "bad --seq")?;
        let rounds: usize = get_arg(&args, "--rounds")
            .unwrap_or_else(|| "1".to_string())
            .parse()
            .map_err(|_| "bad --rounds")?;
        let md_out = get_arg(&args, "--md-out").ok_or("missing --md-out")?;
        let md_in = get_arg(&args, "--md-in").ok_or("missing --md-in")?;
        // Optional second metadata blob (the predecessor's) so both ring
        // neighbors are loaded — NIXL's UCX link may need both peers to have
        // loaded each other before a Write lands on the right registered block.
        let md_in2 = get_arg(&args, "--md-in2");
        let desc_out = get_arg(&args, "--desc-out").ok_or("missing --desc-out")?;
        let desc_in = get_arg(&args, "--desc-in").ok_or("missing --desc-in")?;
        let done_out = get_arg(&args, "--done-out").ok_or("missing --done-out")?;
        let done_in = get_arg(&args, "--done-in").ok_or("missing --done-in")?;
        let dump_out = get_arg(&args, "--dump-out").ok_or("missing --dump-out")?;

        let device = Device::Cpu; // cross-vendor ring stages through host DRAM (S3a finding)
        let shape: [i64; 4] = [1, 2, seq, 64];
        let numel: i64 = shape.iter().product();

        let mut transport =
            NixlBlockTransport::new(&agent).map_err(|e| format!("create agent: {e}"))?;
        println!("[ring-probe] agent created: {}", transport.agent_name());

        // current = local KV pattern (seed-distinct per node); recv = zeros.
        let mut current = Tensor::arange(numel, (Kind::Float, device))
            .f_add_scalar(seed as f64)
            .map_err(|e| format!("add scalar: {e:?}"))?
            .reshape(shape)
            .to_kind(Kind::BFloat16);
        let recv = Tensor::zeros(shape, (Kind::BFloat16, device));

        let cur0: Vec<f32> = Vec::try_from(
            &current.to_kind(Kind::Float).to_device(Device::Cpu).view(-1),
        )
        .unwrap_or_default();
        println!(
            "[ring-probe] initial current[:4]={:?} (seed={})",
            &cur0[..cur0.len().min(4)],
            seed
        );
        let current_handle = transport
            .register_block(&current)
            .map_err(|e| format!("register current: {e}"))?;
        let recv_handle = transport
            .register_block(&recv)
            .map_err(|e| format!("register recv: {e}"))?;
        let cur_after_reg: Vec<f32> = Vec::try_from(
            &current.to_kind(Kind::Float).to_device(Device::Cpu).view(-1),
        )
        .unwrap_or_default();
        println!(
            "[ring-probe] current AFTER register[:4]={:?}",
            &cur_after_reg[..cur_after_reg.len().min(4)]
        );
        println!(
            "[ring-probe] current id={} len={}, recv id={} len={}",
            current_handle.id, current_handle.desc.len, recv_handle.id, recv_handle.desc.len
        );

        // Export local md + recv desc (the predecessor transfers INTO our recv,
        // so it needs our recv desc).
        let md = transport.local_metadata().map_err(|e| format!("local md: {e}"))?;
        fs::write(&md_out, &md).map_err(|e| format!("write md: {e}"))?;
        let recv_desc = RemoteBlockDesc {
            agent: agent.clone(),
            block_id: recv_handle.id,
            desc: recv_handle.desc.clone(),
        };
        let desc_json =
            serde_json::to_string(&recv_desc).map_err(|e| format!("serialize desc: {e}"))?;
        fs::write(&desc_out, desc_json).map_err(|e| format!("write desc: {e}"))?;
        println!("[ring-probe] wrote local md + recv desc");

        // Load the SUCCESSOR's md + recv desc: we transfer our current block
        // INTO the successor's recv buffer.
        wait_for_file(&md_in, 180)?;
        wait_for_file(&desc_in, 180)?;
        let peer_md = fs::read(&md_in).map_err(|e| format!("read md_in: {e}"))?;
        let peer_agent = transport
            .load_remote_metadata(&peer_md)
            .map_err(|e| format!("load remote md: {e}"))?;
        println!("[ring-probe] loaded remote md agent={}", peer_agent);
        if let Some(md2) = &md_in2 {
            wait_for_file(md2, 180)?;
            let pred_md = fs::read(md2).map_err(|e| format!("read md_in2: {e}"))?;
            let pred_agent = transport
                .load_remote_metadata(&pred_md)
                .map_err(|e| format!("load pred md: {e}"))?;
            println!("[ring-probe] loaded pred md agent={}", pred_agent);
        }
        let desc_str = fs::read_to_string(&desc_in).map_err(|e| format!("read desc: {e}"))?;
        let succ_desc: RemoteBlockDesc =
            serde_json::from_str(&desc_str).map_err(|e| format!("parse desc: {e}"))?;
        println!(
            "[ring-probe] successor recv desc agent={} block_id={} len={} addr={}",
            succ_desc.agent, succ_desc.block_id, succ_desc.desc.len, succ_desc.desc.addr
        );
        println!(
            "[ring-probe] own recv desc addr={} current addr={}",
            recv_handle.desc.addr, current_handle.desc.addr
        );

        // Hop-by-hop circulation.
        for round in 0..rounds {
            transport
                .submit_transfer(&current_handle, &succ_desc)
                .map_err(|e| format!("round {round} submit: {e}"))?;
            let mut done = Vec::new();
            let deadline = std::time::Instant::now() + Duration::from_secs(180);
            while done.is_empty() {
                done = transport
                    .poll_transfers()
                    .map_err(|e| format!("round {round} poll: {e}"))?;
                if std::time::Instant::now() > deadline {
                    return Err(format!("round {round} poll timeout"));
                }
                if done.is_empty() {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            println!(
                "[ring-probe] round {round} transfer complete ({} bytes)",
                done[0].bytes
            );
            // Per-round done files so the host script never races a stale file
            // across rounds (round 0/1 use distinct names).
            let done_out_r = format!("{done_out}.{round}");
            let done_in_r = format!("{done_in}.{round}");
            fs::write(&done_out_r, b"done").map_err(|e| format!("write done: {e}"))?;

            // Wait for the predecessor to finish ITS transfer into our recv.
            wait_for_file(&done_in_r, 180)?;
            let recv_dbg: Vec<f32> = Vec::try_from(
                &recv.to_kind(Kind::Float).to_device(Device::Cpu).view(-1),
            )
            .unwrap_or_default();
            println!(
                "[ring-probe] round {round} BEFORE swap recv[:4]={:?}",
                &recv_dbg[..recv_dbg.len().min(4)]
            );
            // Swap: forward what we just received next round (copy recv into
            // current, preserving the registered current buffer address).
            current.copy_(&recv);
            let cur_dbg: Vec<f32> = Vec::try_from(
                &current.to_kind(Kind::Float).to_device(Device::Cpu).view(-1),
            )
            .unwrap_or_default();
            println!(
                "[ring-probe] round {round} AFTER swap current[:4]={:?}",
                &cur_dbg[..cur_dbg.len().min(4)]
            );
        }

        // Dump recv: after the last round it holds the last predecessor KV.
        let n = dump_tensor(&recv, &dump_out).map_err(|e| format!("dump recv: {e}"))?;
        println!("[ring-probe] dumped recv {} bytes", n);
        println!("[ring-probe] OK");
        Ok(())
    }
}

#[cfg(feature = "nixl-backend")]
fn main() {
    if let Err(e) = probe::main() {
        eprintln!("[ring-probe] FAILED: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "nixl-backend"))]
fn main() {
    eprintln!("nixl-ring-probe requires --features nixl-backend");
    std::process::exit(2);
}
