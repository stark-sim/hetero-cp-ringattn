use crate::report::{TorchBridgeReport, TorchPayloadBlockReport, TorchQueryOutputReport, TorchQueryOutputGroup};
use crate::protocol;
use crate::tch_backend;
use std::env;

pub fn tch_attention_bridge_report() -> TorchBridgeReport {
    if !cfg!(feature = "tch-backend") {
        return TorchBridgeReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            status_code: 0,
            note: "tch-rs attention smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
        };
    }
    match tch_backend::backend::run_attention_block_updates(1) {
        Ok((code, message, _max_rel_err)) => {
            let device_name = env::var("HCP_TCH_DEVICE")
                .or_else(|_| env::var("HCP_TORCH_DEVICE"))
                .unwrap_or_else(|_| "cpu".to_string());
            let (note, status) = match code {
                1 => ("tch-rs attention smoke executed on CPU".to_string(), "pass"),
                2 => ("tch-rs attention smoke executed on MPS".to_string(), "pass"),
                3 => ("tch-rs attention smoke executed on CUDA".to_string(), "pass"),
                _ => ("tch-rs attention smoke returned unexpected status".to_string(), "fail"),
            };
            TorchBridgeReport {
                status,
                compiled: true,
                requested_device: device_name,
                status_code: code,
                note,
                message,
            }
        }
        Err(e) => {
            let device_name = env::var("HCP_TCH_DEVICE")
                .or_else(|_| env::var("HCP_TORCH_DEVICE"))
                .unwrap_or_else(|_| "cpu".to_string());
            TorchBridgeReport {
                status: "fail",
                compiled: true,
                requested_device: device_name,
                status_code: -1,
                note: "tch-rs attention smoke failed".to_string(),
                message: e,
            }
        }
    }
}
pub fn tch_device_name() -> String {
    env::var("HCP_TCH_DEVICE")
        .or_else(|_| env::var("HCP_TORCH_DEVICE"))
        .unwrap_or_else(|_| "cpu".to_string())
}
pub fn tch_status_note_from_code(code: i32, base_note: &str) -> (&'static str, String) {
    match code {
        1 => ("pass", format!("{base_note} executed on CPU")),
        2 => ("pass", format!("{base_note} executed on MPS")),
        3 => ("pass", format!("{base_note} executed on CUDA")),
        _ => ("fail", format!("{base_note} returned unexpected status")),
    }
}
pub fn tch_payload_block_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(feature = "tch-backend") {
        return TorchPayloadBlockReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            requested_blocks,
            processed_blocks: 0,
            status_code: 0,
            note: "tch-rs payload block smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
        };
    }
    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = if blocks.is_empty() {
        "no CP payload blocks captured".to_string()
    } else {
        String::new()
    };
    for block in blocks {
        let block_len = i32::try_from(block.block_len()).unwrap_or(i32::MAX);
        let num_heads = i32::try_from(block.num_heads()).unwrap_or(i32::MAX);
        let head_dim = i32::try_from(block.head_dim()).unwrap_or(i32::MAX);
        match tch_backend::backend::run_payload_block_smoke(block.payload(), block_len, num_heads, head_dim) {
            Ok((c, msg, _max_rel_err)) => {
                code = c;
                message = msg;
            }
            Err(e) => {
                code = -1;
                message = e;
            }
        }
        let (status, _) = tch_status_note_from_code(code, "tch-rs payload block smoke");
        if status == "fail" {
            message = format!("sequence_id={} {message}", block.sequence_id());
            break;
        }
        processed_blocks += 1;
    }
    let (status, note) = tch_status_note_from_code(code, "tch-rs payload block smoke");
    TorchPayloadBlockReport {
        status,
        compiled: true,
        requested_device: tch_device_name(),
        requested_blocks,
        processed_blocks,
        status_code: code,
        note,
        message,
    }
}
pub fn tch_payload_online_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(feature = "tch-backend") {
        return TorchPayloadBlockReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            requested_blocks,
            processed_blocks: 0,
            status_code: 0,
            note: "tch-rs payload online smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs payload online smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    if blocks.iter().any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim) {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs payload online smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    let mut payload = Vec::new();
    let mut block_lens = Vec::with_capacity(blocks.len());
    for block in blocks {
        payload.extend_from_slice(block.payload());
        block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
    }
    let num_heads_i32 = i32::try_from(num_heads).unwrap_or(i32::MAX);
    let head_dim_i32 = i32::try_from(head_dim).unwrap_or(i32::MAX);
    let (code, message) = match tch_backend::backend::run_payload_online_smoke(&payload, &block_lens, num_heads_i32, head_dim_i32) {
        Ok((c, msg, _max_rel_err)) => (c, msg),
        Err(e) => (-1, e),
    };
    let (status, note) = tch_status_note_from_code(code, "tch-rs payload online smoke");
    TorchPayloadBlockReport {
        status,
        compiled: true,
        requested_device: tch_device_name(),
        requested_blocks,
        processed_blocks: if status == "pass" { blocks.len() } else { 0 },
        status_code: code,
        note,
        message,
    }
}
pub fn tch_payload_chunk_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(feature = "tch-backend") {
        return TorchPayloadBlockReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            requested_blocks,
            processed_blocks: 0,
            status_code: 0,
            note: "tch-rs payload chunk smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs payload chunk smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    if blocks.iter().any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim) {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs payload chunk smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    let mut payload = Vec::new();
    let mut block_lens = Vec::with_capacity(blocks.len());
    for block in blocks {
        payload.extend_from_slice(block.payload());
        block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
    }
    let num_heads_i32 = i32::try_from(num_heads).unwrap_or(i32::MAX);
    let head_dim_i32 = i32::try_from(head_dim).unwrap_or(i32::MAX);
    let query_len = i32::try_from(first_block.query_len()).unwrap_or(i32::MAX);
    let (code, message) = match tch_backend::backend::run_payload_chunk_smoke(&payload, &block_lens, query_len, num_heads_i32, head_dim_i32) {
        Ok((c, msg, _max_rel_err)) => (c, msg),
        Err(e) => (-1, e),
    };
    let (status, note) = tch_status_note_from_code(code, "tch-rs payload chunk smoke");
    TorchPayloadBlockReport {
        status,
        compiled: true,
        requested_device: tch_device_name(),
        requested_blocks,
        processed_blocks: if status == "pass" { blocks.len() } else { 0 },
        status_code: code,
        note,
        message,
    }
}
pub fn tch_query_chunk_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(feature = "tch-backend") {
        return TorchPayloadBlockReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            requested_blocks,
            processed_blocks: 0,
            status_code: 0,
            note: "tch-rs query chunk smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs query chunk smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    let query_len = first_block.query_len();
    if blocks.iter().any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim || block.query_len() != query_len) {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs query chunk smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent query_len/num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    let mut groups: std::collections::BTreeMap<&str, Vec<&protocol::CpPayloadBlock>> = std::collections::BTreeMap::new();
    for block in blocks {
        groups.entry(block.compute_domain()).or_default().push(block);
    }
    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = String::new();
    for (compute_domain, group_blocks) in groups {
        let Some(group_first) = group_blocks.first() else { continue; };
        if group_blocks.iter().any(|block| block.query_payload() != group_first.query_payload()) {
            code = -6;
            message = format!("inconsistent query payload for compute_domain={compute_domain}");
            break;
        }
        let mut kv_payload = Vec::new();
        let mut block_lens = Vec::with_capacity(group_blocks.len());
        for block in &group_blocks {
            kv_payload.extend_from_slice(block.payload());
            block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
        }
        let query_len_i32 = i32::try_from(query_len).unwrap_or(i32::MAX);
        let num_heads_i32 = i32::try_from(num_heads).unwrap_or(i32::MAX);
        let head_dim_i32 = i32::try_from(head_dim).unwrap_or(i32::MAX);
        match tch_backend::backend::run_query_chunk_smoke(
            group_first.query_payload(),
            &kv_payload,
            &block_lens,
            query_len_i32,
            num_heads_i32,
            head_dim_i32,
        ) {
            Ok((c, msg, ..)) => {
                code = c;
                message = msg;
            }
            Err(e) => {
                code = -1;
                message = e;
            }
        }
        let (status, _) = tch_status_note_from_code(code, "tch-rs query chunk smoke");
        if status == "fail" {
            message = format!("compute_domain={compute_domain} {message}");
            break;
        }
        processed_blocks += group_blocks.len();
    }
    let (status, note) = tch_status_note_from_code(code, "tch-rs query chunk smoke");
    TorchPayloadBlockReport {
        status,
        compiled: true,
        requested_device: tch_device_name(),
        requested_blocks,
        processed_blocks,
        status_code: code,
        note,
        message,
    }
}
pub fn tch_query_output_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchQueryOutputReport {
    let requested_blocks = blocks.len();
    if !cfg!(feature = "tch-backend") {
        return TorchQueryOutputReport {
            status: "disabled",
            compiled: false,
            requested_device: "none".to_string(),
            requested_blocks,
            processed_blocks: 0,
            status_code: 0,
            note: "tch-rs query output smoke is disabled; build with --features tch-backend".to_string(),
            message: "tch-backend feature not enabled".to_string(),
            output_groups: Vec::new(),
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchQueryOutputReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs query output smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
            output_groups: Vec::new(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    let query_len = first_block.query_len();
    if blocks.iter().any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim || block.query_len() != query_len) {
        return TorchQueryOutputReport {
            status: "fail",
            compiled: true,
            requested_device: tch_device_name(),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "tch-rs query output smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent query_len/num_heads/head_dim across CP payload blocks".to_string(),
            output_groups: Vec::new(),
        };
    }
    let mut groups: std::collections::BTreeMap<&str, Vec<&protocol::CpPayloadBlock>> = std::collections::BTreeMap::new();
    for block in blocks {
        groups.entry(block.compute_domain()).or_default().push(block);
    }
    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = String::new();
    let mut output_groups = Vec::new();
    for (compute_domain, group_blocks) in groups {
        let Some(group_first) = group_blocks.first() else { continue; };
        if group_blocks.iter().any(|block| block.query_payload() != group_first.query_payload()) {
            code = -6;
            message = format!("inconsistent query payload for compute_domain={compute_domain}");
            break;
        }
        let mut kv_payload = Vec::new();
        let mut block_lens = Vec::with_capacity(group_blocks.len());
        for block in &group_blocks {
            kv_payload.extend_from_slice(block.payload());
            block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
        }
        let query_len_i32 = i32::try_from(query_len).unwrap_or(i32::MAX);
        let num_heads_i32 = i32::try_from(num_heads).unwrap_or(i32::MAX);
        let head_dim_i32 = i32::try_from(head_dim).unwrap_or(i32::MAX);
        match tch_backend::backend::run_query_chunk_smoke(
            group_first.query_payload(),
            &kv_payload,
            &block_lens,
            query_len_i32,
            num_heads_i32,
            head_dim_i32,
        ) {
            Ok((c, msg, output_checksum, max_abs_err, _max_rel_err, output_values)) => {
                code = c;
                message = msg;
                let expected_output_values = query_len * num_heads * head_dim;
                if output_values != expected_output_values {
                    code = -6;
                    message = format!(
                        "compute_domain={compute_domain} output values mismatch expected={expected_output_values} actual={output_values}"
                    );
                    break;
                }
                output_groups.push(TorchQueryOutputGroup {
                    compute_domain: compute_domain.to_string(),
                    layer_index: group_first.layer_index(),
                    output_seq_offset: group_first.output_seq_offset(),
                    query_len: group_first.query_len(),
                    output_slot_values: group_first.output_slot_values(),
                    blocks: group_blocks.len(),
                    output_values,
                    output_checksum,
                    max_abs_err,
                });
            }
            Err(e) => {
                code = -1;
                message = e;
            }
        }
        let (status, _) = tch_status_note_from_code(code, "tch-rs query output smoke");
        if status == "fail" {
            message = format!("compute_domain={compute_domain} {message}");
            break;
        }
        processed_blocks += group_blocks.len();
    }
    let (status, note) = tch_status_note_from_code(code, "tch-rs query output smoke");
    TorchQueryOutputReport {
        status,
        compiled: true,
        requested_device: tch_device_name(),
        requested_blocks,
        processed_blocks,
        status_code: code,
        note,
        message,
        output_groups,
    }
}
