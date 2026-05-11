use crate::report::{TorchBridgeReport, TorchBlockUpdateReport, TorchPayloadBlockReport, TorchQueryOutputReport, TorchQueryOutputGroup};
use crate::protocol;
use crate::smoke::reference_algo::PAYLOAD_CHUNK_QUERY_LEN;
use crate::{torch_bridge_enabled_by_env, torch_device_success_code};
use std::collections::BTreeMap;
use std::env;

extern "C" {
    pub fn hcp_ringattn_cxx_smoke_domain_count() -> i32;
    pub fn hcp_ringattn_torch_smoke() -> i32;
    pub fn hcp_ringattn_torch_smoke_message() -> *const std::os::raw::c_char;
    pub fn hcp_ringattn_torch_attention_smoke() -> i32;
    pub fn hcp_ringattn_torch_attention_smoke_message() -> *const std::os::raw::c_char;
    pub fn hcp_ringattn_torch_block_update_smoke(block_updates: i32) -> i32;
    pub fn hcp_ringattn_torch_block_update_smoke_message() -> *const std::os::raw::c_char;
    pub fn hcp_ringattn_torch_payload_block_smoke(
        payload: *const std::os::raw::c_uchar,
        payload_len: usize,
        block_len: i32,
        num_heads: i32,
        head_dim: i32,
    ) -> i32;
    pub fn hcp_ringattn_torch_payload_block_smoke_message() -> *const std::os::raw::c_char;
    pub fn hcp_ringattn_torch_payload_online_smoke(
        payload: *const std::os::raw::c_uchar,
        payload_len: usize,
        block_lens: *const i32,
        block_count: usize,
        num_heads: i32,
        head_dim: i32,
    ) -> i32;
    pub fn hcp_ringattn_torch_payload_online_smoke_message() -> *const std::os::raw::c_char;
    pub fn hcp_ringattn_torch_payload_chunk_smoke(
        payload: *const std::os::raw::c_uchar,
        payload_len: usize,
        block_lens: *const i32,
        block_count: usize,
        query_len: i32,
        num_heads: i32,
        head_dim: i32,
    ) -> i32;
    pub fn hcp_ringattn_torch_payload_chunk_smoke_message() -> *const std::os::raw::c_char;
    pub fn hcp_ringattn_torch_query_chunk_smoke(
        q_payload: *const std::os::raw::c_uchar,
        q_payload_len: usize,
        kv_payload: *const std::os::raw::c_uchar,
        kv_payload_len: usize,
        block_lens: *const i32,
        block_count: usize,
        query_len: i32,
        num_heads: i32,
        head_dim: i32,
    ) -> i32;
    pub fn hcp_ringattn_torch_query_chunk_smoke_message() -> *const std::os::raw::c_char;
    pub fn hcp_ringattn_torch_query_chunk_output_smoke(
        q_payload: *const std::os::raw::c_uchar,
        q_payload_len: usize,
        kv_payload: *const std::os::raw::c_uchar,
        kv_payload_len: usize,
        block_lens: *const i32,
        block_count: usize,
        query_len: i32,
        num_heads: i32,
        head_dim: i32,
        output_checksum: *mut f64,
        max_abs_err: *mut f64,
        output_values: *mut usize,
    ) -> i32;
    pub fn hcp_ringattn_torch_query_chunk_output_smoke_message() -> *const std::os::raw::c_char;
}
pub fn c_string_from_ptr(ptr: *const std::os::raw::c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}
pub fn torch_bridge_report() -> TorchBridgeReport {
    let code = unsafe { hcp_ringattn_torch_smoke() };
    let message = c_string_from_ptr(unsafe { hcp_ringattn_torch_smoke_message() });
    torch_report_from_code(
        code,
        message,
        "C++ libtorch bridge executed on CPU",
        "C++ libtorch bridge executed on MPS",
        "C++ libtorch bridge executed on CUDA",
        "C++ libtorch bridge is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch bridge failed or returned an unexpected status",
    )
}
pub fn torch_attention_bridge_report() -> TorchBridgeReport {
    let code = unsafe { hcp_ringattn_torch_attention_smoke() };
    let message = c_string_from_ptr(unsafe { hcp_ringattn_torch_attention_smoke_message() });
    torch_report_from_code(
        code,
        message,
        "C++ libtorch attention smoke executed on CPU",
        "C++ libtorch attention smoke executed on MPS",
        "C++ libtorch attention smoke executed on CUDA",
        "C++ libtorch attention smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch attention smoke failed or returned an unexpected status",
    )
}
pub fn torch_block_update_bridge_report(requested_updates: usize) -> TorchBlockUpdateReport {
    let block_updates = i32::try_from(requested_updates).unwrap_or(i32::MAX);
    let code = unsafe { hcp_ringattn_torch_block_update_smoke(block_updates) };
    let message = c_string_from_ptr(unsafe { hcp_ringattn_torch_block_update_smoke_message() });
    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch CP block update smoke executed on CPU",
        "C++ libtorch CP block update smoke executed on MPS",
        "C++ libtorch CP block update smoke executed on CUDA",
        "C++ libtorch CP block update smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch CP block update smoke failed or returned an unexpected status",
    );
    TorchBlockUpdateReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_updates,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
    }
}
pub fn torch_payload_block_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(hcp_torch_enabled) {
        let report = torch_report_from_code(
            0,
            "HCP_ENABLE_TORCH is not enabled".to_string(),
            "C++ libtorch payload block smoke executed on CPU",
            "C++ libtorch payload block smoke executed on MPS",
            "C++ libtorch payload block smoke executed on CUDA",
            "C++ libtorch payload block smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch payload block smoke failed or returned an unexpected status",
        );
        return TorchPayloadBlockReport {
            status: report.status,
            compiled: report.compiled,
            requested_device: report.requested_device,
            requested_blocks,
            processed_blocks: 0,
            status_code: report.status_code,
            note: report.note,
            message: report.message,
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
        code = unsafe {
            hcp_ringattn_torch_payload_block_smoke(
                block.payload().as_ptr(),
                block.payload().len(),
                block_len,
                num_heads,
                head_dim,
            )
        };
        message = c_string_from_ptr(unsafe { hcp_ringattn_torch_payload_block_smoke_message() });
        let block_report = torch_report_from_code(
            code,
            message.clone(),
            "C++ libtorch payload block smoke executed on CPU",
            "C++ libtorch payload block smoke executed on MPS",
            "C++ libtorch payload block smoke executed on CUDA",
            "C++ libtorch payload block smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch payload block smoke failed or returned an unexpected status",
        );
        if block_report.status == "fail" {
            message = format!("sequence_id={} {}", block.sequence_id(), message);
            break;
        }
        processed_blocks += 1;
    }

    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch payload block smoke executed on CPU",
        "C++ libtorch payload block smoke executed on MPS",
        "C++ libtorch payload block smoke executed on CUDA",
        "C++ libtorch payload block smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch payload block smoke failed or returned an unexpected status",
    );
    TorchPayloadBlockReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_blocks,
        processed_blocks,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
    }
}
pub fn torch_payload_online_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(hcp_torch_enabled) {
        let report = torch_report_from_code(
            0,
            "HCP_ENABLE_TORCH is not enabled".to_string(),
            "C++ libtorch payload online smoke executed on CPU",
            "C++ libtorch payload online smoke executed on MPS",
            "C++ libtorch payload online smoke executed on CUDA",
            "C++ libtorch payload online smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch payload online smoke failed or returned an unexpected status",
        );
        return TorchPayloadBlockReport {
            status: report.status,
            compiled: report.compiled,
            requested_device: report.requested_device,
            requested_blocks,
            processed_blocks: 0,
            status_code: report.status_code,
            note: report.note,
            message: report.message,
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch payload online smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    if blocks
        .iter()
        .any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim)
    {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch payload online smoke received inconsistent tensor shapes"
                .to_string(),
            message: "inconsistent num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    let mut payload = Vec::new();
    let mut block_lens = Vec::with_capacity(blocks.len());
    for block in blocks {
        payload.extend_from_slice(block.payload());
        block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
    }
    let code = unsafe {
        hcp_ringattn_torch_payload_online_smoke(
            payload.as_ptr(),
            payload.len(),
            block_lens.as_ptr(),
            block_lens.len(),
            i32::try_from(num_heads).unwrap_or(i32::MAX),
            i32::try_from(head_dim).unwrap_or(i32::MAX),
        )
    };
    let message = c_string_from_ptr(unsafe { hcp_ringattn_torch_payload_online_smoke_message() });
    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch payload online smoke executed on CPU",
        "C++ libtorch payload online smoke executed on MPS",
        "C++ libtorch payload online smoke executed on CUDA",
        "C++ libtorch payload online smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch payload online smoke failed or returned an unexpected status",
    );
    let processed_blocks = if report.status == "pass" {
        requested_blocks
    } else {
        0
    };
    TorchPayloadBlockReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_blocks,
        processed_blocks,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
    }
}
pub fn torch_payload_chunk_bridge_report(
    blocks: &[protocol::CpPayloadBlock],
) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(hcp_torch_enabled) {
        let report = torch_report_from_code(
            0,
            "HCP_ENABLE_TORCH is not enabled".to_string(),
            "C++ libtorch payload chunk smoke executed on CPU",
            "C++ libtorch payload chunk smoke executed on MPS",
            "C++ libtorch payload chunk smoke executed on CUDA",
            "C++ libtorch payload chunk smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch payload chunk smoke failed or returned an unexpected status",
        );
        return TorchPayloadBlockReport {
            status: report.status,
            compiled: report.compiled,
            requested_device: report.requested_device,
            requested_blocks,
            processed_blocks: 0,
            status_code: report.status_code,
            note: report.note,
            message: report.message,
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch payload chunk smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    if blocks
        .iter()
        .any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim)
    {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch payload chunk smoke received inconsistent tensor shapes"
                .to_string(),
            message: "inconsistent num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    let mut payload = Vec::new();
    let mut block_lens = Vec::with_capacity(blocks.len());
    for block in blocks {
        payload.extend_from_slice(block.payload());
        block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
    }
    let code = unsafe {
        hcp_ringattn_torch_payload_chunk_smoke(
            payload.as_ptr(),
            payload.len(),
            block_lens.as_ptr(),
            block_lens.len(),
            PAYLOAD_CHUNK_QUERY_LEN,
            i32::try_from(num_heads).unwrap_or(i32::MAX),
            i32::try_from(head_dim).unwrap_or(i32::MAX),
        )
    };
    let message = c_string_from_ptr(unsafe { hcp_ringattn_torch_payload_chunk_smoke_message() });
    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch payload chunk smoke executed on CPU",
        "C++ libtorch payload chunk smoke executed on MPS",
        "C++ libtorch payload chunk smoke executed on CUDA",
        "C++ libtorch payload chunk smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch payload chunk smoke failed or returned an unexpected status",
    );
    let processed_blocks = if report.status == "pass" {
        requested_blocks
    } else {
        0
    };
    TorchPayloadBlockReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_blocks,
        processed_blocks,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
    }
}
pub fn torch_query_chunk_bridge_report(blocks: &[protocol::CpPayloadBlock]) -> TorchPayloadBlockReport {
    let requested_blocks = blocks.len();
    if !cfg!(hcp_torch_enabled) {
        let report = torch_report_from_code(
            0,
            "HCP_ENABLE_TORCH is not enabled".to_string(),
            "C++ libtorch query chunk smoke executed on CPU",
            "C++ libtorch query chunk smoke executed on MPS",
            "C++ libtorch query chunk smoke executed on CUDA",
            "C++ libtorch query chunk smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch query chunk smoke failed or returned an unexpected status",
        );
        return TorchPayloadBlockReport {
            status: report.status,
            compiled: report.compiled,
            requested_device: report.requested_device,
            requested_blocks,
            processed_blocks: 0,
            status_code: report.status_code,
            note: report.note,
            message: report.message,
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch query chunk smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    let query_len = first_block.query_len();
    if blocks
        .iter()
        .any(|block| block.num_heads() != num_heads || block.head_dim() != head_dim)
    {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch query chunk smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent num_heads/head_dim across CP payload blocks".to_string(),
        };
    }
    if blocks.iter().any(|block| block.query_len() != query_len) {
        return TorchPayloadBlockReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch query chunk smoke received inconsistent query shapes".to_string(),
            message: "inconsistent query_len across CP payload blocks".to_string(),
        };
    }

    let mut groups: BTreeMap<&str, Vec<&protocol::CpPayloadBlock>> = BTreeMap::new();
    for block in blocks {
        groups
            .entry(block.compute_domain())
            .or_default()
            .push(block);
    }

    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = String::new();
    for (compute_domain, group_blocks) in groups {
        let Some(group_first) = group_blocks.first() else {
            continue;
        };
        if group_blocks
            .iter()
            .any(|block| block.query_payload() != group_first.query_payload())
        {
            code = -6;
            message = format!("inconsistent query payload for compute_domain={compute_domain}");
            break;
        }
        if group_blocks.iter().any(|block| {
            block.layer_index() != group_first.layer_index()
                || block.output_seq_offset() != group_first.output_seq_offset()
                || block.output_slot_values() != group_first.output_slot_values()
        }) {
            code = -6;
            message = format!("inconsistent output slot for compute_domain={compute_domain}");
            break;
        }

        let mut kv_payload = Vec::new();
        let mut block_lens = Vec::with_capacity(group_blocks.len());
        for block in &group_blocks {
            kv_payload.extend_from_slice(block.payload());
            block_lens.push(i32::try_from(block.block_len()).unwrap_or(i32::MAX));
        }
        code = unsafe {
            hcp_ringattn_torch_query_chunk_smoke(
                group_first.query_payload().as_ptr(),
                group_first.query_payload().len(),
                kv_payload.as_ptr(),
                kv_payload.len(),
                block_lens.as_ptr(),
                block_lens.len(),
                i32::try_from(group_first.query_len()).unwrap_or(i32::MAX),
                i32::try_from(num_heads).unwrap_or(i32::MAX),
                i32::try_from(head_dim).unwrap_or(i32::MAX),
            )
        };
        message = c_string_from_ptr(unsafe { hcp_ringattn_torch_query_chunk_smoke_message() });
        let group_report = torch_report_from_code(
            code,
            message.clone(),
            "C++ libtorch query chunk smoke executed on CPU",
            "C++ libtorch query chunk smoke executed on MPS",
            "C++ libtorch query chunk smoke executed on CUDA",
            "C++ libtorch query chunk smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch query chunk smoke failed or returned an unexpected status",
        );
        if group_report.status == "fail" {
            message = format!("compute_domain={compute_domain} {message}");
            break;
        }
        processed_blocks += group_blocks.len();
    }

    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch query chunk smoke executed on CPU",
        "C++ libtorch query chunk smoke executed on MPS",
        "C++ libtorch query chunk smoke executed on CUDA",
        "C++ libtorch query chunk smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch query chunk smoke failed or returned an unexpected status",
    );
    TorchPayloadBlockReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_blocks,
        processed_blocks,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
    }
}
pub fn torch_query_output_bridge_report(blocks: &[protocol::CpPayloadBlock]) -> TorchQueryOutputReport {
    let requested_blocks = blocks.len();
    if !cfg!(hcp_torch_enabled) {
        let report = torch_report_from_code(
            0,
            "HCP_ENABLE_TORCH is not enabled".to_string(),
            "C++ libtorch query output smoke executed on CPU",
            "C++ libtorch query output smoke executed on MPS",
            "C++ libtorch query output smoke executed on CUDA",
            "C++ libtorch query output smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch query output smoke failed or returned an unexpected status",
        );
        return TorchQueryOutputReport {
            status: report.status,
            compiled: report.compiled,
            requested_device: report.requested_device,
            requested_blocks,
            processed_blocks: 0,
            status_code: report.status_code,
            note: report.note,
            message: report.message,
            output_groups: Vec::new(),
        };
    }
    let Some(first_block) = blocks.first() else {
        return TorchQueryOutputReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch query output smoke has no payload blocks".to_string(),
            message: "no CP payload blocks captured".to_string(),
            output_groups: Vec::new(),
        };
    };
    let num_heads = first_block.num_heads();
    let head_dim = first_block.head_dim();
    let query_len = first_block.query_len();
    if blocks.iter().any(|block| {
        block.num_heads() != num_heads
            || block.head_dim() != head_dim
            || block.query_len() != query_len
    }) {
        return TorchQueryOutputReport {
            status: "fail",
            compiled: cfg!(hcp_torch_enabled),
            requested_device: env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
            requested_blocks,
            processed_blocks: 0,
            status_code: -6,
            note: "C++ libtorch query output smoke received inconsistent tensor shapes".to_string(),
            message: "inconsistent query_len/num_heads/head_dim across CP payload blocks"
                .to_string(),
            output_groups: Vec::new(),
        };
    }

    let mut groups: BTreeMap<&str, Vec<&protocol::CpPayloadBlock>> = BTreeMap::new();
    for block in blocks {
        groups
            .entry(block.compute_domain())
            .or_default()
            .push(block);
    }

    let mut processed_blocks = 0_usize;
    let mut code = -6;
    let mut message = String::new();
    let mut output_groups = Vec::new();
    for (compute_domain, group_blocks) in groups {
        let Some(group_first) = group_blocks.first() else {
            continue;
        };
        if group_blocks
            .iter()
            .any(|block| block.query_payload() != group_first.query_payload())
        {
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

        let mut output_checksum = 0.0_f64;
        let mut max_abs_err = 0.0_f64;
        let mut output_values = 0_usize;
        code = unsafe {
            hcp_ringattn_torch_query_chunk_output_smoke(
                group_first.query_payload().as_ptr(),
                group_first.query_payload().len(),
                kv_payload.as_ptr(),
                kv_payload.len(),
                block_lens.as_ptr(),
                block_lens.len(),
                i32::try_from(group_first.query_len()).unwrap_or(i32::MAX),
                i32::try_from(num_heads).unwrap_or(i32::MAX),
                i32::try_from(head_dim).unwrap_or(i32::MAX),
                &mut output_checksum,
                &mut max_abs_err,
                &mut output_values,
            )
        };
        message =
            c_string_from_ptr(unsafe { hcp_ringattn_torch_query_chunk_output_smoke_message() });
        let group_report = torch_report_from_code(
            code,
            message.clone(),
            "C++ libtorch query output smoke executed on CPU",
            "C++ libtorch query output smoke executed on MPS",
            "C++ libtorch query output smoke executed on CUDA",
            "C++ libtorch query output smoke is disabled; build with HCP_ENABLE_TORCH=1",
            "C++ libtorch query output smoke failed or returned an unexpected status",
        );
        if group_report.status == "fail" {
            message = format!("compute_domain={compute_domain} {message}");
            break;
        }
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
        processed_blocks += group_blocks.len();
    }

    let report = torch_report_from_code(
        code,
        message,
        "C++ libtorch query output smoke executed on CPU",
        "C++ libtorch query output smoke executed on MPS",
        "C++ libtorch query output smoke executed on CUDA",
        "C++ libtorch query output smoke is disabled; build with HCP_ENABLE_TORCH=1",
        "C++ libtorch query output smoke failed or returned an unexpected status",
    );
    TorchQueryOutputReport {
        status: report.status,
        compiled: report.compiled,
        requested_device: report.requested_device,
        requested_blocks,
        processed_blocks,
        status_code: report.status_code,
        note: report.note,
        message: report.message,
        output_groups,
    }
}
pub fn torch_report_from_code(
    code: i32,
    message: String,
    cpu_note: &'static str,
    mps_note: &'static str,
    cuda_note: &'static str,
    disabled_note: &'static str,
    generic_fail_note: &'static str,
) -> TorchBridgeReport {
    let requested_device = env::var("HCP_TORCH_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let expected_code = torch_device_success_code(&requested_device);
    let requested_is_cuda = expected_code == Some(3);
    let status = match (
        cfg!(hcp_torch_enabled),
        torch_bridge_enabled_by_env(),
        expected_code,
    ) {
        (false, false, _) => "skipped",
        (false, true, _) => "fail",
        (true, _, Some(expected)) if code == expected => "pass",
        (true, _, _) => "fail",
    };
    let note = match (
        cfg!(hcp_torch_enabled),
        requested_device.as_str(),
        requested_is_cuda,
        code,
    ) {
        (false, _, _, _) => disabled_note.to_string(),
        (true, "cpu", _, 1) => cpu_note.to_string(),
        (true, "mps", _, 2) => mps_note.to_string(),
        (true, _, true, 3) => cuda_note.to_string(),
        (true, _, _, -4) => {
            "Unsupported HCP_TORCH_DEVICE; expected cpu, mps, cuda, or cuda:N".to_string()
        }
        (true, _, true, -5) => {
            "CUDA backend is unavailable in the current libtorch process".to_string()
        }
        (true, _, _, _) => generic_fail_note.to_string(),
    };
    TorchBridgeReport {
        status,
        compiled: cfg!(hcp_torch_enabled),
        requested_device,
        status_code: code,
        note,
        message,
    }
}
