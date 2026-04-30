#[cfg(feature = "tch-backend")]
mod tch_smoke {
    use serde::Serialize;
    use std::env;
    use std::fs;
    use std::path::Path;

    #[derive(Serialize)]
    struct TchSmokeReport {
        status: &'static str,
        requested_device: String,
        status_code: i32,
        note: String,
        message: String,
        ops: Vec<OpReport>,
    }

    #[derive(Serialize)]
    struct OpReport {
        op: &'static str,
        shape_in: Vec<i64>,
        shape_out: Vec<i64>,
        device: String,
        dtype: String,
        max_abs_err: f64,
        mean_abs_err: f64,
        status: &'static str,
    }

    fn device_from_env() -> (tch::Device, String) {
        let raw = env::var("HCP_TCH_DEVICE").unwrap_or_else(|_| "cpu".to_string());
        let device = match raw.as_str() {
            "cpu" => tch::Device::Cpu,
            "mps" => tch::Device::Mps,
            "cuda" => tch::Device::Cuda(0),
            _ => {
                if let Some(index) = raw.strip_prefix("cuda:") {
                    if let Ok(idx) = index.parse::<usize>() {
                        tch::Device::Cuda(idx)
                    } else {
                        eprintln!("Warning: invalid HCP_TCH_DEVICE={raw}, falling back to cpu");
                        tch::Device::Cpu
                    }
                } else {
                    eprintln!("Warning: unknown HCP_TCH_DEVICE={raw}, falling back to cpu");
                    tch::Device::Cpu
                }
            }
        };
        let canonical = match device {
            tch::Device::Cpu => "cpu".to_string(),
            tch::Device::Mps => "mps".to_string(),
            tch::Device::Cuda(idx) => format!("cuda:{idx}"),
            _ => "unknown".to_string(),
        };
        (device, canonical)
    }

    fn success_code_for_device(device: &str) -> i32 {
        match device {
            "cpu" => 1,
            "mps" => 2,
            s if s.starts_with("cuda") => 3,
            _ => 0,
        }
    }

    fn run_smoke(device: tch::Device, device_name: &str) -> Result<TchSmokeReport, String> {
        let mut ops = Vec::new();

        // matmul smoke: A @ B
        let a = tch::Tensor::randn([4, 8], (tch::Kind::Float, device));
        let b = tch::Tensor::randn([8, 6], (tch::Kind::Float, device));
        let c = a.matmul(&b);

        let a_cpu = a.to(tch::Device::Cpu);
        let b_cpu = b.to(tch::Device::Cpu);
        let c_ref = a_cpu.matmul(&b_cpu);
        let c_cpu = c.to(tch::Device::Cpu);

        let diff = (&c_cpu - &c_ref).abs();
        let max_err = diff.max().double_value(&[]);
        let mean_err = diff.mean(tch::Kind::Float).double_value(&[]);
        let matmul_pass = max_err < 1e-4;

        ops.push(OpReport {
            op: "matmul",
            shape_in: vec![4, 8, 8, 6],
            shape_out: c.size(),
            device: device_name.to_string(),
            dtype: "float32".to_string(),
            max_abs_err: max_err,
            mean_abs_err: mean_err,
            status: if matmul_pass { "pass" } else { "fail" },
        });

        // softmax smoke: softmax over last dim
        let x = tch::Tensor::randn([2, 4, 8], (tch::Kind::Float, device));
        let y = x.softmax(-1, tch::Kind::Float);

        let x_cpu = x.to(tch::Device::Cpu);
        let y_ref = x_cpu.softmax(-1, tch::Kind::Float);
        let y_cpu = y.to(tch::Device::Cpu);

        let diff_y = (&y_cpu - &y_ref).abs();
        let max_err_y = diff_y.max().double_value(&[]);
        let mean_err_y = diff_y.mean(tch::Kind::Float).double_value(&[]);
        let softmax_pass = max_err_y < 1e-4;

        ops.push(OpReport {
            op: "softmax",
            shape_in: x.size(),
            shape_out: y.size(),
            device: device_name.to_string(),
            dtype: "float32".to_string(),
            max_abs_err: max_err_y,
            mean_abs_err: mean_err_y,
            status: if softmax_pass { "pass" } else { "fail" },
        });

        // attention-like smoke: Q @ K^T -> softmax -> @ V
        let q = tch::Tensor::randn([2, 4, 8], (tch::Kind::Float, device));
        let k = tch::Tensor::randn([2, 6, 8], (tch::Kind::Float, device));
        let v = tch::Tensor::randn([2, 6, 8], (tch::Kind::Float, device));
        let scores = q.matmul(&k.transpose(1, 2)) / 8.0_f64.sqrt();
        let probs = scores.softmax(-1, tch::Kind::Float);
        let out = probs.matmul(&v);

        let q_ref = q.to(tch::Device::Cpu);
        let k_ref = k.to(tch::Device::Cpu);
        let v_ref = v.to(tch::Device::Cpu);
        let scores_ref = q_ref.matmul(&k_ref.transpose(1, 2)) / 8.0_f64.sqrt();
        let probs_ref = scores_ref.softmax(-1, tch::Kind::Float);
        let out_ref = probs_ref.matmul(&v_ref);
        let out_cpu = out.to(tch::Device::Cpu);

        let diff_o = (&out_cpu - &out_ref).abs();
        let max_err_o = diff_o.max().double_value(&[]);
        let mean_err_o = diff_o.mean(tch::Kind::Float).double_value(&[]);
        let attention_pass = max_err_o < 1e-4;

        ops.push(OpReport {
            op: "attention_like",
            shape_in: vec![2, 4, 8, 2, 6, 8],
            shape_out: out.size(),
            device: device_name.to_string(),
            dtype: "float32".to_string(),
            max_abs_err: max_err_o,
            mean_abs_err: mean_err_o,
            status: if attention_pass { "pass" } else { "fail" },
        });

        let all_pass = matmul_pass && softmax_pass && attention_pass;
        let status_code = if all_pass {
            success_code_for_device(device_name)
        } else {
            -1
        };

        Ok(TchSmokeReport {
            status: if all_pass { "pass" } else { "fail" },
            requested_device: device_name.to_string(),
            status_code,
            note: format!("tch-rs smoke on {device_name}: matmul + softmax + attention_like"),
            message: if all_pass {
                "ok".to_string()
            } else {
                "one or more ops exceeded tolerance".to_string()
            },
            ops,
        })
    }

    pub fn main() {
        let run_id = env::var("RUN_ID").unwrap_or_else(|_| "hcp-ringattn-tch-smoke-local".to_string());
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("rust crate must live under repo_root/rust")
            .to_path_buf();
        let report_dir = repo_root.join("reports").join(&run_id);
        fs::create_dir_all(&report_dir).expect("create report dir");
        let report_path = report_dir.join("tch_smoke.json");

        let (device, device_name) = device_from_env();

        println!("tch-rs smoke starting: requested_device={device_name}");

        let report = match run_smoke(device, &device_name) {
            Ok(r) => r,
            Err(e) => TchSmokeReport {
                status: "fail",
                requested_device: device_name.clone(),
                status_code: -1,
                note: "tch-rs smoke encountered an exception".to_string(),
                message: e,
                ops: Vec::new(),
            },
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize report");
        fs::write(&report_path, json).expect("write report");

        let summary = format!(
            "tch_status={} tch_device={} tch_code={} ops={}/{}",
            report.status,
            report.requested_device,
            report.status_code,
            report.ops.iter().filter(|o| o.status == "pass").count(),
            report.ops.len()
        );
        println!("{summary}");
        println!("report written to {}", report_path.display());

        if report.status == "fail" {
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "tch-backend")]
fn main() {
    tch_smoke::main();
}

#[cfg(not(feature = "tch-backend"))]
fn main() {
    eprintln!("Error: tch-backend feature is not enabled.");
    eprintln!("Build with: cargo run --features tch-backend --bin tch_smoke");
    std::process::exit(1);
}
