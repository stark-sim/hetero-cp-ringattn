mod cli;
mod compute_runtime;
#[cfg(feature = "tch-backend")]
mod capacity;
mod distributed;
mod error;
mod infer;
mod model;
mod protocol;
mod remote;
mod report;
mod smoke;
mod tch_backend;
#[cfg(feature = "tch-backend")]
mod worker_sdk;

pub use cli::{CliArgs, parse_cli_args, next_cli_value};
pub use error::{RingError, Tolerance, ToleranceTier};
pub use report::*;
pub use smoke::*;
pub use smoke::reference_algo::*;
pub use smoke::correctness::*;
pub use smoke::bridges::*;
pub use remote::*;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::Path;

fn torch_device_success_code(requested_device: &str) -> Option<i32> {
    match requested_device {
        "cpu" => Some(1),
        "mps" => Some(2),
        "cuda" => Some(3),
        _ => requested_device
            .strip_prefix("cuda:")
            .filter(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
            .map(|_| 3),
    }
}

fn torch_bridge_enabled_by_env() -> bool {
    env::var("HCP_ENABLE_TORCH").ok().as_deref() == Some("1")
}

fn compact_message(message: &str, max_chars: usize) -> String {
    let one_line = message.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= max_chars {
        return one_line;
    }
    let mut compact = one_line.chars().take(max_chars).collect::<String>();
    compact.push_str("...");
    compact
}

fn run(stress_test: bool, tolerance_tier: ToleranceTier) -> Result<Report, RingError> {
    let tol = tolerance_tier.default_tolerance();
    let stress_seeds: Vec<u64> = (0..5).map(|i| 42 + i as u64).collect();
    let cases: Vec<CaseReport> = default_cases()
        .into_iter()
        .enumerate()
        .map(|(index, (name, config))| {
            let seeds = if stress_test && config.global_seq_len <= 256 {
                stress_seeds.clone()
            } else {
                vec![42 + index as u64]
            };
            run_case(name, config, &seeds, tolerance_tier, tol)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let passed = cases.iter().filter(|case| case.status == "pass").count();
    let failed = cases.len() - passed;
    let protocol_smoke = protocol::run_protocol_smoke()?;
    let cp_ring_smoke = protocol::run_cp_ring_node_smoke()?;
    let tch_compute_output_checksum = cp_ring_smoke.compute_output_checksum();
    let torch_bridge = torch_bridge_report();
    let torch_attention_bridge = torch_attention_bridge_report();
    let tch_attention_bridge = tch_attention_bridge_report();
    let torch_block_update_bridge =
        torch_block_update_bridge_report(cp_ring_smoke.compute_updates());
    let torch_payload_block_bridge =
        torch_payload_block_bridge_report(cp_ring_smoke.payload_blocks());
    let torch_payload_online_bridge =
        torch_payload_online_bridge_report(cp_ring_smoke.payload_blocks());
    let torch_payload_chunk_bridge =
        torch_payload_chunk_bridge_report(cp_ring_smoke.payload_blocks());
    let torch_query_chunk_bridge = torch_query_chunk_bridge_report(cp_ring_smoke.payload_blocks());
    let torch_query_output_bridge =
        torch_query_output_bridge_report(cp_ring_smoke.payload_blocks());
    let tch_payload_block_bridge =
        tch_payload_block_bridge_report(cp_ring_smoke.payload_blocks());
    let tch_payload_online_bridge =
        tch_payload_online_bridge_report(cp_ring_smoke.payload_blocks());
    let tch_payload_chunk_bridge =
        tch_payload_chunk_bridge_report(cp_ring_smoke.payload_blocks());
    let tch_query_chunk_bridge =
        tch_query_chunk_bridge_report(cp_ring_smoke.payload_blocks());
    let tch_query_output_bridge =
        tch_query_output_bridge_report(cp_ring_smoke.payload_blocks());
    let status = if failed == 0
        && protocol_smoke.status == "pass"
        && cp_ring_smoke.status == "pass"
        && torch_bridge.status != "fail"
        && torch_attention_bridge.status != "fail"
        && tch_attention_bridge.status != "fail"
        && torch_block_update_bridge.status != "fail"
        && torch_payload_block_bridge.status != "fail"
        && torch_payload_online_bridge.status != "fail"
        && torch_payload_chunk_bridge.status != "fail"
        && torch_query_chunk_bridge.status != "fail"
        && torch_query_output_bridge.status != "fail"
        && tch_payload_block_bridge.status != "fail"
        && tch_payload_online_bridge.status != "fail"
        && tch_payload_chunk_bridge.status != "fail"
        && tch_query_chunk_bridge.status != "fail"
        && tch_query_output_bridge.status != "fail"
    {
        "pass"
    } else {
        "fail"
    };
    Ok(Report {
        status,
        summary: Summary {
            cases: cases.len(),
            passed,
            failed,
        },
        protocol_smoke,
        cp_ring_smoke,
        tch_compute_output_checksum,
        cxx_bridge: CxxBridgeReport {
            status: "ok",
            smoke_domains: unsafe { hcp_ringattn_cxx_smoke_domain_count() },
        },
        torch_bridge,
        torch_attention_bridge,
        tch_attention_bridge,
        torch_block_update_bridge,
        torch_payload_block_bridge,
        torch_payload_online_bridge,
        torch_payload_chunk_bridge,
        torch_query_chunk_bridge,
        torch_query_output_bridge,
        tch_payload_block_bridge,
        tch_payload_online_bridge,
        tch_payload_chunk_bridge,
        tch_query_chunk_bridge,
        tch_query_output_bridge,
        cases,
    })
}

fn write_json_report<T: Serialize>(report_path: &str, report: &T) -> Result<(), RingError> {
    if let Some(parent) = Path::new(report_path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(report_path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

pub fn run_cli() -> Result<(), RingError> {
    let args = parse_cli_args()?;

    // Inference mode: load real model and generate text
    if let Some(ref model_dir) = args.infer_model_dir {
        let prompt = if let Some(ref path) = args.infer_prompt_file {
            std::fs::read_to_string(path)
                .map_err(|e| RingError::InvalidCli(format!("cannot read --infer-prompt-file {path}: {e}")))?
        } else {
            args.infer_prompt.unwrap_or_else(|| "Hello, how are you?".to_string())
        };
        println!("[infer] prompt length: {} chars", prompt.len());
        match infer::run_inference(model_dir, &prompt, args.infer_max_tokens, args.infer_temperature, args.infer_top_p, args.infer_num_domains) {
            Ok(text) => println!("{}", text),
            Err(e) => eprintln!("Inference failed: {}", e),
        }
        return Ok(());
    }

    #[cfg(feature = "tch-backend")]
    if let Some(ref role) = args.distributed_role {
        if role == "worker" {
            distributed::worker::run();
            return Ok(());
        } else if role == "coordinator" {
            distributed::coordinator::run();
            return Ok(());
        } else {
            return Err(RingError::InvalidCli(format!(
                "invalid --distributed-role: {role}; expected worker|coordinator"
            )));
        }
    }

    if args.remote_p2p_role.as_deref() == Some("cp-node") {
        let cp_node = run_remote_cp_node(&args)?;
        let torch_payload_block_bridge =
            torch_payload_block_bridge_report(cp_node.payload_blocks());
        let torch_payload_online_bridge =
            torch_payload_online_bridge_report(cp_node.payload_blocks());
        let torch_payload_chunk_bridge =
            torch_payload_chunk_bridge_report(cp_node.payload_blocks());
        let torch_query_chunk_bridge = torch_query_chunk_bridge_report(cp_node.payload_blocks());
        let torch_query_output_bridge = torch_query_output_bridge_report(cp_node.payload_blocks());
        let tch_payload_block_bridge =
            tch_payload_block_bridge_report(cp_node.payload_blocks());
        let tch_payload_online_bridge =
            tch_payload_online_bridge_report(cp_node.payload_blocks());
        let tch_payload_chunk_bridge =
            tch_payload_chunk_bridge_report(cp_node.payload_blocks());
        let tch_query_chunk_bridge =
            tch_query_chunk_bridge_report(cp_node.payload_blocks());
        let tch_query_output_bridge =
            tch_query_output_bridge_report(cp_node.payload_blocks());
        let tch_compute_output_checksum = cp_node.compute_output_checksum();
        let status = if cp_node.status == "pass"
            && torch_payload_block_bridge.status != "fail"
            && torch_payload_online_bridge.status != "fail"
            && torch_payload_chunk_bridge.status != "fail"
            && torch_query_chunk_bridge.status != "fail"
            && torch_query_output_bridge.status != "fail"
            && tch_payload_block_bridge.status != "fail"
            && tch_payload_online_bridge.status != "fail"
            && tch_payload_chunk_bridge.status != "fail"
            && tch_query_chunk_bridge.status != "fail"
            && tch_query_output_bridge.status != "fail"
        {
            "pass"
        } else {
            "fail"
        };
        let report = RemoteCpNodeRunReport {
            status,
            cp_node,
            torch_payload_block_bridge,
            torch_payload_online_bridge,
            torch_payload_chunk_bridge,
            torch_query_chunk_bridge,
            torch_query_output_bridge,
            tch_payload_block_bridge,
            tch_payload_online_bridge,
            tch_payload_chunk_bridge,
            tch_query_chunk_bridge,
            tch_query_output_bridge,
            tch_compute_output_checksum,
        };
        write_json_report(&args.report_path, &report)?;
        println!(
            "[rust-remote-cp-node] status={} role={} transport={} sent={} received={} compute_updates={} torch_payload_block_status={} torch_payload_block_code={} torch_payload_blocks={}/{} torch_payload_online_status={} torch_payload_online_code={} torch_payload_online_blocks={}/{} torch_payload_chunk_status={} torch_payload_chunk_code={} torch_payload_chunk_blocks={}/{} torch_query_chunk_status={} torch_query_chunk_code={} torch_query_chunk_blocks={}/{} torch_query_output_status={} torch_query_output_code={} torch_query_output_groups={} torch_query_output_blocks={}/{} tch_payload_block_status={} tch_payload_block_code={} tch_payload_block_blocks={}/{} tch_payload_online_status={} tch_payload_online_code={} tch_payload_online_blocks={}/{} tch_payload_chunk_status={} tch_payload_chunk_code={} tch_payload_chunk_blocks={}/{} tch_query_chunk_status={} tch_query_chunk_code={} tch_query_chunk_blocks={}/{} tch_query_output_status={} tch_query_output_code={} tch_query_output_groups={} tch_query_output_blocks={}/{} tch_compute_output_checksum={} report={}",
            report.status,
            report.cp_node.role(),
            report.cp_node.transport(),
            report.cp_node.messages_sent(),
            report.cp_node.messages_received(),
            report.cp_node.compute_updates(),
            report.torch_payload_block_bridge.status,
            report.torch_payload_block_bridge.status_code,
            report.torch_payload_block_bridge.processed_blocks,
            report.torch_payload_block_bridge.requested_blocks,
            report.torch_payload_online_bridge.status,
            report.torch_payload_online_bridge.status_code,
            report.torch_payload_online_bridge.processed_blocks,
            report.torch_payload_online_bridge.requested_blocks,
            report.torch_payload_chunk_bridge.status,
            report.torch_payload_chunk_bridge.status_code,
            report.torch_payload_chunk_bridge.processed_blocks,
            report.torch_payload_chunk_bridge.requested_blocks,
            report.torch_query_chunk_bridge.status,
            report.torch_query_chunk_bridge.status_code,
            report.torch_query_chunk_bridge.processed_blocks,
            report.torch_query_chunk_bridge.requested_blocks,
            report.torch_query_output_bridge.status,
            report.torch_query_output_bridge.status_code,
            report.torch_query_output_bridge.output_groups.len(),
            report.torch_query_output_bridge.processed_blocks,
            report.torch_query_output_bridge.requested_blocks,
            report.tch_payload_block_bridge.status,
            report.tch_payload_block_bridge.status_code,
            report.tch_payload_block_bridge.processed_blocks,
            report.tch_payload_block_bridge.requested_blocks,
            report.tch_payload_online_bridge.status,
            report.tch_payload_online_bridge.status_code,
            report.tch_payload_online_bridge.processed_blocks,
            report.tch_payload_online_bridge.requested_blocks,
            report.tch_payload_chunk_bridge.status,
            report.tch_payload_chunk_bridge.status_code,
            report.tch_payload_chunk_bridge.processed_blocks,
            report.tch_payload_chunk_bridge.requested_blocks,
            report.tch_query_chunk_bridge.status,
            report.tch_query_chunk_bridge.status_code,
            report.tch_query_chunk_bridge.processed_blocks,
            report.tch_query_chunk_bridge.requested_blocks,
            report.tch_query_output_bridge.status,
            report.tch_query_output_bridge.status_code,
            report.tch_query_output_bridge.output_groups.len(),
            report.tch_query_output_bridge.processed_blocks,
            report.tch_query_output_bridge.requested_blocks,
            report.tch_compute_output_checksum,
            args.report_path
        );
        if report.torch_payload_block_bridge.status == "fail"
            && !report.torch_payload_block_bridge.message.is_empty()
        {
            println!(
                "[rust-remote-cp-node] torch_payload_block_message={}",
                compact_message(&report.torch_payload_block_bridge.message, 360)
            );
        }
        if report.torch_payload_online_bridge.status == "fail"
            && !report.torch_payload_online_bridge.message.is_empty()
        {
            println!(
                "[rust-remote-cp-node] torch_payload_online_message={}",
                compact_message(&report.torch_payload_online_bridge.message, 360)
            );
        }
        if report.torch_payload_chunk_bridge.status == "fail"
            && !report.torch_payload_chunk_bridge.message.is_empty()
        {
            println!(
                "[rust-remote-cp-node] torch_payload_chunk_message={}",
                compact_message(&report.torch_payload_chunk_bridge.message, 360)
            );
        }
        if report.torch_query_chunk_bridge.status == "fail"
            && !report.torch_query_chunk_bridge.message.is_empty()
        {
            println!(
                "[rust-remote-cp-node] torch_query_chunk_message={}",
                compact_message(&report.torch_query_chunk_bridge.message, 360)
            );
        }
        if report.torch_query_output_bridge.status == "fail"
            && !report.torch_query_output_bridge.message.is_empty()
        {
            println!(
                "[rust-remote-cp-node] torch_query_output_message={}",
                compact_message(&report.torch_query_output_bridge.message, 360)
            );
        }
        if report.status == "pass" {
            return Ok(());
        }
        std::process::exit(1);
    }
    if args.remote_p2p_role.is_some() {
        let report = run_remote_p2p(&args)?;
        write_json_report(&args.report_path, &report)?;
        println!(
            "[rust-remote-p2p] status={} role={} transport={} sent={} received={} report={}",
            report.status,
            report.role(),
            report.transport(),
            report.messages_sent(),
            report.messages_received(),
            args.report_path
        );
        return Ok(());
    }

    let report = run(args.stress_test, args.tolerance_tier)?;
    write_json_report(&args.report_path, &report)?;
    println!(
        "[rust-ringattn] status={} passed={}/{} protocol_status={} protocol_messages={} cp_ring_status={} cp_ring_messages={} cp_ring_compute_updates={} cxx_domains={} torch_status={} torch_device={} torch_code={} torch_attention_status={} torch_attention_code={} tch_attention_status={} tch_attention_code={} torch_block_update_status={} torch_block_update_code={} torch_block_updates={} torch_payload_block_status={} torch_payload_block_code={} torch_payload_blocks={}/{} torch_payload_online_status={} torch_payload_online_code={} torch_payload_online_blocks={}/{} torch_payload_chunk_status={} torch_payload_chunk_code={} torch_payload_chunk_blocks={}/{} torch_query_chunk_status={} torch_query_chunk_code={} torch_query_chunk_blocks={}/{} torch_query_output_status={} torch_query_output_code={} torch_query_output_groups={} torch_query_output_blocks={}/{} tch_payload_block_status={} tch_payload_block_code={} tch_payload_block_blocks={}/{} tch_payload_online_status={} tch_payload_online_code={} tch_payload_online_blocks={}/{} tch_payload_chunk_status={} tch_payload_chunk_code={} tch_payload_chunk_blocks={}/{} tch_query_chunk_status={} tch_query_chunk_code={} tch_query_chunk_blocks={}/{} tch_query_output_status={} tch_query_output_code={} tch_query_output_groups={} tch_query_output_blocks={}/{} tch_compute_output_checksum={} torch_compiled={} report={}",
        report.status,
        report.summary.passed,
        report.summary.cases,
        report.protocol_smoke.status,
        report.protocol_smoke.messages_sent(),
        report.cp_ring_smoke.status,
        report.cp_ring_smoke.messages_sent(),
        report.cp_ring_smoke.compute_updates(),
        report.cxx_bridge.smoke_domains,
        report.torch_bridge.status,
        report.torch_bridge.requested_device,
        report.torch_bridge.status_code,
        report.torch_attention_bridge.status,
        report.torch_attention_bridge.status_code,
        report.tch_attention_bridge.status,
        report.tch_attention_bridge.status_code,
        report.torch_block_update_bridge.status,
        report.torch_block_update_bridge.status_code,
        report.torch_block_update_bridge.requested_updates,
        report.torch_payload_block_bridge.status,
        report.torch_payload_block_bridge.status_code,
        report.torch_payload_block_bridge.processed_blocks,
        report.torch_payload_block_bridge.requested_blocks,
        report.torch_payload_online_bridge.status,
        report.torch_payload_online_bridge.status_code,
        report.torch_payload_online_bridge.processed_blocks,
        report.torch_payload_online_bridge.requested_blocks,
        report.torch_payload_chunk_bridge.status,
        report.torch_payload_chunk_bridge.status_code,
        report.torch_payload_chunk_bridge.processed_blocks,
        report.torch_payload_chunk_bridge.requested_blocks,
        report.torch_query_chunk_bridge.status,
        report.torch_query_chunk_bridge.status_code,
        report.torch_query_chunk_bridge.processed_blocks,
        report.torch_query_chunk_bridge.requested_blocks,
        report.torch_query_output_bridge.status,
        report.torch_query_output_bridge.status_code,
        report.torch_query_output_bridge.output_groups.len(),
        report.torch_query_output_bridge.processed_blocks,
        report.torch_query_output_bridge.requested_blocks,
        report.tch_payload_block_bridge.status,
        report.tch_payload_block_bridge.status_code,
        report.tch_payload_block_bridge.processed_blocks,
        report.tch_payload_block_bridge.requested_blocks,
        report.tch_payload_online_bridge.status,
        report.tch_payload_online_bridge.status_code,
        report.tch_payload_online_bridge.processed_blocks,
        report.tch_payload_online_bridge.requested_blocks,
        report.tch_payload_chunk_bridge.status,
        report.tch_payload_chunk_bridge.status_code,
        report.tch_payload_chunk_bridge.processed_blocks,
        report.tch_payload_chunk_bridge.requested_blocks,
        report.tch_query_chunk_bridge.status,
        report.tch_query_chunk_bridge.status_code,
        report.tch_query_chunk_bridge.processed_blocks,
        report.tch_query_chunk_bridge.requested_blocks,
        report.tch_query_output_bridge.status,
        report.tch_query_output_bridge.status_code,
        report.tch_query_output_bridge.output_groups.len(),
        report.tch_query_output_bridge.processed_blocks,
        report.tch_query_output_bridge.requested_blocks,
        report.tch_compute_output_checksum,
        report.torch_bridge.compiled,
        args.report_path
    );
    if report.torch_bridge.status == "fail" && !report.torch_bridge.message.is_empty() {
        println!(
            "[rust-ringattn] torch_message={}",
            compact_message(&report.torch_bridge.message, 360)
        );
    }
    if report.torch_attention_bridge.status == "fail"
        && !report.torch_attention_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_attention_message={}",
            compact_message(&report.torch_attention_bridge.message, 360)
        );
    }
    if report.tch_attention_bridge.status == "fail"
        && !report.tch_attention_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_attention_message={}",
            compact_message(&report.tch_attention_bridge.message, 360)
        );
    }
    if report.torch_block_update_bridge.status == "fail"
        && !report.torch_block_update_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_block_update_message={}",
            compact_message(&report.torch_block_update_bridge.message, 360)
        );
    }
    if report.torch_payload_block_bridge.status == "fail"
        && !report.torch_payload_block_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_payload_block_message={}",
            compact_message(&report.torch_payload_block_bridge.message, 360)
        );
    }
    if report.torch_payload_online_bridge.status == "fail"
        && !report.torch_payload_online_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_payload_online_message={}",
            compact_message(&report.torch_payload_online_bridge.message, 360)
        );
    }
    if report.torch_payload_chunk_bridge.status == "fail"
        && !report.torch_payload_chunk_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_payload_chunk_message={}",
            compact_message(&report.torch_payload_chunk_bridge.message, 360)
        );
    }
    if report.torch_query_chunk_bridge.status == "fail"
        && !report.torch_query_chunk_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_query_chunk_message={}",
            compact_message(&report.torch_query_chunk_bridge.message, 360)
        );
    }
    if report.torch_query_output_bridge.status == "fail"
        && !report.torch_query_output_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] torch_query_output_message={}",
            compact_message(&report.torch_query_output_bridge.message, 360)
        );
    }
    if report.tch_payload_block_bridge.status == "fail"
        && !report.tch_payload_block_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_payload_block_message={}",
            compact_message(&report.tch_payload_block_bridge.message, 360)
        );
    }
    if report.tch_payload_online_bridge.status == "fail"
        && !report.tch_payload_online_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_payload_online_message={}",
            compact_message(&report.tch_payload_online_bridge.message, 360)
        );
    }
    if report.tch_payload_chunk_bridge.status == "fail"
        && !report.tch_payload_chunk_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_payload_chunk_message={}",
            compact_message(&report.tch_payload_chunk_bridge.message, 360)
        );
    }
    if report.tch_query_chunk_bridge.status == "fail"
        && !report.tch_query_chunk_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_query_chunk_message={}",
            compact_message(&report.tch_query_chunk_bridge.message, 360)
        );
    }
    if report.tch_query_output_bridge.status == "fail"
        && !report.tch_query_output_bridge.message.is_empty()
    {
        println!(
            "[rust-ringattn] tch_query_output_message={}",
            compact_message(&report.tch_query_output_bridge.message, 360)
        );
    }
    if report.status == "pass" {
        Ok(())
    } else {
        std::process::exit(1);
    }
}
