#[cfg(feature = "tch-backend")]
pub mod backend {
    use std::env;

    #[derive(Debug, Clone)]
    pub struct TchDeviceSelection {
        pub device: tch::Device,
        pub success_code: i32,
    }

    pub fn select_device() -> Result<TchDeviceSelection, String> {
        let name = env::var("HCP_TCH_DEVICE")
            .or_else(|_| env::var("HCP_TORCH_DEVICE"))
            .unwrap_or_else(|_| "cpu".to_string());

        let (device, success_code) = match name.as_str() {
            "cpu" => (tch::Device::Cpu, 1),
            "mps" => (tch::Device::Mps, 2),
            "cuda" => (tch::Device::Cuda(0), 3),
            _ => {
                if let Some(idx) = name.strip_prefix("cuda:") {
                    if let Ok(i) = idx.parse::<usize>() {
                        (tch::Device::Cuda(i), 3)
                    } else {
                        return Err(format!(
                            "unsupported HCP_TCH_DEVICE={name}; expected cpu, mps, cuda, or cuda:N"
                        ));
                    }
                } else {
                    return Err(format!(
                        "unsupported HCP_TCH_DEVICE={name}; expected cpu, mps, cuda, or cuda:N"
                    ));
                }
            }
        };

        Ok(TchDeviceSelection {
            device,
            success_code,
        })
    }

    pub fn run_attention_block_updates(block_updates: i32) -> Result<(i32, String), String> {
        if block_updates <= 0 {
            return Err("block_updates must be positive".to_string());
        }

        let selection = select_device()?;
        let device = selection.device;

        let cpu = tch::Device::Cpu;
        let float = tch::Kind::Float;

        // Deterministic synthetic data: arange-like via linspace to match C++ bridge spirit
        let q_cpu = tch::Tensor::arange(32, (float, cpu))
            .reshape([4, 8])
            .divide_scalar(17.0);
        let k_cpu = tch::Tensor::arange(48, (float, cpu))
            .reshape([6, 8])
            .divide_scalar(19.0);
        let v_cpu = tch::Tensor::arange(48, (float, cpu))
            .reshape([6, 8])
            .divide_scalar(23.0);

        let scale = 1.0 / (8.0_f64).sqrt();
        let mut max_abs_err = 0.0_f64;
        let mut checksum = 0.0_f64;

        for update in 0..block_updates {
            let shift = update as f64 / 101.0;
            let q_update_cpu = &q_cpu + shift;
            let k_update_cpu = &k_cpu + shift / 3.0;
            let v_update_cpu = &v_cpu + shift / 5.0;

            // CPU reference
            let scores_ref = q_update_cpu.matmul(&k_update_cpu.transpose(0, 1)) * scale;
            let probs_ref = scores_ref.softmax(-1, float);
            let reference = probs_ref.matmul(&v_update_cpu);

            // Device compute
            let q = q_update_cpu.to(device);
            let k = k_update_cpu.to(device);
            let v = v_update_cpu.to(device);
            let scores = q.matmul(&k.transpose(0, 1)) * scale;
            let probs = scores.softmax(-1, float);
            let output = probs.matmul(&v);

            if !device_matches(&output, device) {
                return Err("attention output landed on unexpected device".to_string());
            }

            let output_cpu = output.to(cpu);
            let diff = (&output_cpu - &reference).abs();
            let err = diff.max().double_value(&[]);
            if err > max_abs_err {
                max_abs_err = err;
            }
            checksum += output_cpu.sum(float).double_value(&[]);

            let sizes = output_cpu.size();
            if sizes != vec![4, 8] {
                return Err("unexpected attention output shape".to_string());
            }
        }

        if max_abs_err <= 1.0e-4 {
            let msg = format!(
                "ok updates={block_updates} max_abs_err={max_abs_err} checksum={checksum}"
            );
            Ok((selection.success_code, msg))
        } else {
            Err(format!(
                "attention mismatch updates={block_updates} max_abs_err={max_abs_err}"
            ))
        }
    }

    fn device_matches(tensor: &tch::Tensor, expected: tch::Device) -> bool {
        let actual = tensor.device();
        match expected {
            tch::Device::Cpu => actual == tch::Device::Cpu,
            tch::Device::Mps => actual == tch::Device::Mps,
            tch::Device::Cuda(_) => matches!(actual, tch::Device::Cuda(_)),
            _ => false,
        }
    }
}

#[cfg(not(feature = "tch-backend"))]
pub mod backend {
    pub fn run_attention_block_updates(_block_updates: i32) -> Result<(i32, String), String> {
        Err("tch-backend feature is not enabled".to_string())
    }
}
