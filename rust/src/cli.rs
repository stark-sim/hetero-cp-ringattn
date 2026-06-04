use crate::error::RingError;
use crate::error::ToleranceTier;
use std::env;

/// 【命令行参数结构体】
///
/// HCP 的二进制支持多种运行模式：
/// - 单节点推理：`--infer-model-dir` + `--infer-prompt`
/// - 分布式 coordinator：`--distributed-role coordinator`
/// - 分布式 worker：`--distributed-role worker`
/// - 协议 smoke test：`--remote-p2p-role server/client`
/// - correctness test：默认模式，生成 JSON 报告
#[derive(Debug)]
pub struct CliArgs {
    pub report_path: String,
    pub remote_p2p_role: Option<String>,
    pub node_index: Option<usize>,
    pub bind_addr: Option<String>,
    pub connect_addr: Option<String>,
    pub stress_test: bool,
    pub tolerance_tier: ToleranceTier,
    pub infer_model_dir: Option<String>,
    pub infer_prompt: Option<String>,
    pub infer_prompt_file: Option<String>,
    pub infer_max_tokens: usize,
    pub infer_temperature: f64,
    pub infer_top_p: f64,
    pub infer_num_domains: usize,
    pub export_logits_dir: Option<String>,
    pub export_hidden_states_dir: Option<String>,
    pub distributed_role: Option<String>,
}

pub fn parse_cli_args() -> Result<CliArgs, RingError> {
    let mut args = env::args().skip(1);
    let mut report_path = String::from("reports/rust_ringattn_correctness.json");
    let mut remote_p2p_role = None;
    let mut node_index = None;
    let mut bind_addr = None;
    let mut connect_addr = None;
    let mut stress_test = false;
    let mut tolerance_tier = ToleranceTier::default();
    let mut infer_model_dir = None;
    let mut infer_prompt = None;
    let mut infer_prompt_file = None;
    let mut infer_max_tokens = 50;
    let mut infer_temperature = 0.7;
    let mut infer_top_p = 0.9;
    let mut infer_num_domains = 1usize;
    let mut export_logits_dir = None;
    let mut export_hidden_states_dir = None;
    let mut distributed_role = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--report-path" => {
                report_path = next_cli_value(&mut args, "--report-path")?;
            }
            "--remote-p2p-role" => {
                remote_p2p_role = Some(next_cli_value(&mut args, "--remote-p2p-role")?);
            }
            "--node-index" => {
                let value = next_cli_value(&mut args, "--node-index")?;
                node_index = Some(value.parse::<usize>().map_err(|error| {
                    RingError::InvalidCli(format!("invalid --node-index: {error}"))
                })?);
            }
            "--bind" => {
                bind_addr = Some(next_cli_value(&mut args, "--bind")?);
            }
            "--connect" => {
                connect_addr = Some(next_cli_value(&mut args, "--connect")?);
            }
            "--stress-test" => {
                stress_test = true;
            }
            "--tolerance-tier" => {
                let value = next_cli_value(&mut args, "--tolerance-tier")?;
                tolerance_tier = value.parse().map_err(|_| {
                    RingError::InvalidCli(format!(
                        "invalid --tolerance-tier: {value}; expected strict|relaxed|end-to-end"
                    ))
                })?;
            }
            "--infer-model-dir" => {
                infer_model_dir = Some(next_cli_value(&mut args, "--infer-model-dir")?);
            }
            "--infer-prompt" => {
                infer_prompt = Some(next_cli_value(&mut args, "--infer-prompt")?);
            }
            "--infer-prompt-file" => {
                infer_prompt_file = Some(next_cli_value(&mut args, "--infer-prompt-file")?);
            }
            "--infer-max-tokens" => {
                let value = next_cli_value(&mut args, "--infer-max-tokens")?;
                infer_max_tokens = value.parse().map_err(|e| {
                    RingError::InvalidCli(format!("invalid --infer-max-tokens: {e}"))
                })?;
            }
            "--infer-temperature" => {
                let value = next_cli_value(&mut args, "--infer-temperature")?;
                infer_temperature = value.parse().map_err(|e| {
                    RingError::InvalidCli(format!("invalid --infer-temperature: {e}"))
                })?;
            }
            "--infer-top-p" => {
                let value = next_cli_value(&mut args, "--infer-top-p")?;
                infer_top_p = value.parse().map_err(|e| {
                    RingError::InvalidCli(format!("invalid --infer-top-p: {e}"))
                })?;
            }
            "--infer-num-domains" => {
                let value = next_cli_value(&mut args, "--infer-num-domains")?;
                infer_num_domains = value.parse().map_err(|e| {
                    RingError::InvalidCli(format!("invalid --infer-num-domains: {e}"))
                })?;
            }
            "--export-logits" => {
                export_logits_dir = Some(next_cli_value(&mut args, "--export-logits")?);
            }
            "--export-hidden-states" => {
                export_hidden_states_dir = Some(next_cli_value(&mut args, "--export-hidden-states")?);
            }
            "--distributed-role" => {
                distributed_role = Some(next_cli_value(&mut args, "--distributed-role")?);
                // Worker / coordinator parse remaining args themselves;
                // stop parsing here so we don't reject their private flags.
                break;
            }
            _ => {
                return Err(RingError::InvalidCli(format!("unknown argument {arg}")));
            }
        }
    }
    Ok(CliArgs {
        report_path,
        remote_p2p_role,
        node_index,
        bind_addr,
        connect_addr,
        stress_test,
        tolerance_tier,
        infer_model_dir,
        infer_prompt,
        infer_prompt_file,
        infer_max_tokens,
        infer_temperature,
        infer_top_p,
        infer_num_domains,
        export_logits_dir,
        export_hidden_states_dir,
        distributed_role,
    })
}

pub fn next_cli_value(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<String, RingError> {
    args.next()
        .filter(|value| !value.starts_with("--"))
        .ok_or_else(|| RingError::InvalidCli(format!("missing value for {flag}")))
}
