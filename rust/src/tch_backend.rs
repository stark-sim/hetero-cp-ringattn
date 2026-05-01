#![allow(dead_code)]
#![allow(clippy::needless_borrows_for_generic_args)]

#[cfg(feature = "tch-backend")]
pub mod backend {
    use std::env;
    use tch::{Kind, Tensor};

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

    fn device_matches(tensor: &Tensor, expected: tch::Device) -> bool {
        let actual = tensor.device();
        match expected {
            tch::Device::Cpu => actual == tch::Device::Cpu,
            tch::Device::Mps => actual == tch::Device::Mps,
            tch::Device::Cuda(_) => matches!(actual, tch::Device::Cuda(_)),
            _ => false,
        }
    }

    // Q: [num_heads, head_dim]
    // K: [block_len, num_heads, head_dim]
    // V: [block_len, num_heads, head_dim]
    // output: [num_heads, head_dim]
    fn payload_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        let scale = 1.0 / (q.size()[1] as f64).sqrt();
        let k_by_head = k.permute(&[1, 0, 2]);
        let v_by_head = v.permute(&[1, 0, 2]);
        let scores = k_by_head.matmul(&q.unsqueeze(2i64)).squeeze_dim(2) * scale;
        let weights = scores.softmax(-1, Kind::Float);
        weights.unsqueeze(1i64).matmul(&v_by_head).squeeze_dim(1)
    }

    // Q: [query_len, num_heads, head_dim]
    // K: [block_len, num_heads, head_dim]
    // V: [block_len, num_heads, head_dim]
    // output: [query_len, num_heads, head_dim]
    fn payload_chunk_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        let scale = 1.0 / (q.size()[2] as f64).sqrt();
        let q_by_head = q.permute(&[1, 0, 2]);
        let k_by_head = k.permute(&[1, 0, 2]);
        let v_by_head = v.permute(&[1, 0, 2]);
        let scores = q_by_head.matmul(&k_by_head.transpose(1, 2)) * scale;
        let weights = scores.softmax(-1, Kind::Float);
        weights.matmul(&v_by_head).permute(&[1, 0, 2])
    }

    pub fn run_attention_block_updates(block_updates: i32) -> Result<(i32, String, f64), String> {
        if block_updates <= 0 {
            return Err("block_updates must be positive".to_string());
        }

        let selection = select_device()?;
        let device = selection.device;

        let cpu = tch::Device::Cpu;
        let float = Kind::Float;

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
        let mut max_rel_err = 0.0_f64;
        let mut checksum = 0.0_f64;

        for update in 0..block_updates {
            let shift = update as f64 / 101.0;
            let q_update_cpu = &q_cpu + shift;
            let k_update_cpu = &k_cpu + shift / 3.0;
            let v_update_cpu = &v_cpu + shift / 5.0;

            let scores_ref = q_update_cpu.matmul(&k_update_cpu.transpose(0, 1)) * scale;
            let probs_ref = scores_ref.softmax(-1, float);
            let reference = probs_ref.matmul(&v_update_cpu);

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
            let rel_diff = diff / (reference.abs() + 1e-12);
            let rel_err = rel_diff.max().double_value(&[]);
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
            checksum += output_cpu.sum(float).double_value(&[]);

            let sizes = output_cpu.size();
            if sizes != vec![4, 8] {
                return Err("unexpected attention output shape".to_string());
            }
        }

        if max_abs_err <= 1.0e-4 && max_rel_err <= 1.0e-3 {
            let msg = format!(
                "ok updates={block_updates} max_abs_err={max_abs_err} max_rel_err={max_rel_err} checksum={checksum}"
            );
            Ok((selection.success_code, msg, max_rel_err))
        } else {
            Err(format!(
                "attention mismatch updates={block_updates} max_abs_err={max_abs_err} max_rel_err={max_rel_err}"
            ))
        }
    }

    pub fn run_payload_block_smoke(
        payload: &[u8],
        block_len: i32,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<(i32, String, f64), String> {
        if block_len <= 0 || num_heads <= 0 || head_dim <= 0 {
            return Err("block_len, num_heads, and head_dim must be positive".to_string());
        }
        let block = block_len as usize;
        let heads = num_heads as usize;
        let dim = head_dim as usize;
        let values_per_tensor = block * heads * dim;
        let expected_values = values_per_tensor * 2;
        let expected_bytes = expected_values * std::mem::size_of::<f32>();
        if payload.len() != expected_bytes {
            return Err(format!(
                "payload byte size mismatch expected={expected_bytes} actual={}",
                payload.len()
            ));
        }

        let selection = select_device()?;
        let device = selection.device;

        let values: Vec<f32> = payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let kv_cpu = Tensor::from_slice(&values)
            .reshape([2, block_len as i64, num_heads as i64, head_dim as i64]);
        let k_cpu = kv_cpu.get(0);
        let v_cpu = kv_cpu.get(1);

        let q_cpu = (Tensor::arange(num_heads as i64 * head_dim as i64, (Kind::Float, tch::Device::Cpu))
            .reshape([num_heads as i64, head_dim as i64])
            / 31.0)
            + 0.125;

        let reference = payload_attention(&q_cpu, &k_cpu, &v_cpu);

        let q = q_cpu.to(device);
        let k = k_cpu.to(device);
        let v = v_cpu.to(device);
        let output = payload_attention(&q, &k, &v);

        if !device_matches(&output, device) {
            return Err("payload attention output landed on unexpected device".to_string());
        }
        if output.size() != vec![num_heads as i64, head_dim as i64] {
            return Err("unexpected payload attention output shape".to_string());
        }

        let output_cpu = output.to(tch::Device::Cpu);
        let diff = (&output_cpu - &reference).abs();
        let max_abs_err = diff.max().double_value(&[]);
        let rel_diff = &diff / (reference.abs() + 1e-12);
        let max_rel_err = rel_diff.max().double_value(&[]);
        let checksum = output_cpu.sum(Kind::Float).double_value(&[]);

        if max_abs_err <= 1.0e-4 && max_rel_err <= 1.0e-3 {
            let msg = format!(
                "ok block_len={block_len} num_heads={num_heads} head_dim={head_dim} max_abs_err={max_abs_err} max_rel_err={max_rel_err} checksum={checksum}"
            );
            Ok((selection.success_code, msg, max_rel_err))
        } else {
            Err(format!(
                "payload attention mismatch block_len={block_len} max_abs_err={max_abs_err} max_rel_err={max_rel_err}"
            ))
        }
    }

    pub fn run_payload_online_smoke(
        payload: &[u8],
        block_lens: &[i32],
        num_heads: i32,
        head_dim: i32,
    ) -> Result<(i32, String, f64), String> {
        if block_lens.is_empty() || num_heads <= 0 || head_dim <= 0 {
            return Err("block_count, num_heads, and head_dim must be positive".to_string());
        }
        let mut token_count = 0usize;
        for &bl in block_lens {
            if bl <= 0 {
                return Err("all block_lens entries must be positive".to_string());
            }
            token_count += bl as usize;
        }
        let expected_values = token_count * (num_heads as usize) * (head_dim as usize) * 2;
        let expected_bytes = expected_values * std::mem::size_of::<f32>();
        if payload.len() != expected_bytes {
            return Err(format!(
                "online payload byte size mismatch expected={expected_bytes} actual={}",
                payload.len()
            ));
        }

        let selection = select_device()?;
        let device = selection.device;
        let cpu = tch::Device::Cpu;
        let float = Kind::Float;

        let values: Vec<f32> = payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let q_cpu = (Tensor::arange(num_heads as i64 * head_dim as i64, (float, cpu))
            .reshape([num_heads as i64, head_dim as i64])
            / 31.0)
            + 0.125;

        let q = q_cpu.to(device);
        let mut running_max =
            Tensor::full(&[num_heads as i64], f64::NEG_INFINITY, (float, device));
        let mut running_sum = Tensor::zeros(&[num_heads as i64], (float, device));
        let mut output = Tensor::zeros(&[num_heads as i64, head_dim as i64], (float, device));

        let mut k_refs: Vec<Tensor> = Vec::new();
        let mut v_refs: Vec<Tensor> = Vec::new();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut offset = 0usize;

        for &block_len in block_lens {
            let block_values =
                (block_len as usize) * (num_heads as usize) * (head_dim as usize) * 2;
            let kv_cpu = Tensor::from_slice(&values[offset..offset + block_values])
                .reshape([2, block_len as i64, num_heads as i64, head_dim as i64]);
            offset += block_values;

            let k_cpu = kv_cpu.get(0);
            let v_cpu = kv_cpu.get(1);
            k_refs.push(k_cpu.shallow_clone());
            v_refs.push(v_cpu.shallow_clone());

            let k = k_cpu.to(device);
            let v = v_cpu.to(device);
            let k_by_head = k.permute(&[1, 0, 2]);
            let v_by_head = v.permute(&[1, 0, 2]);
            let scores = k_by_head.matmul(&q.unsqueeze(2i64)).squeeze_dim(2) * scale;
            let (local_max, _) = scores.max_dim(1, false);
            let weights = (&scores - local_max.unsqueeze(1i64)).exp();
            let local_sum = weights.sum_dim_intlist(&[1i64][..], false, float);
            let local_pv = weights.unsqueeze(1i64).matmul(&v_by_head).squeeze_dim(1);

            let new_max = running_max.max_other(&local_max);
            let exp_prev = (&running_max - &new_max).exp();
            let exp_local = (&local_max - &new_max).exp();
            let new_sum = &exp_prev * &running_sum + &exp_local * &local_sum;
            output = (&exp_prev.unsqueeze(1i64) * &running_sum.unsqueeze(1i64) * &output
                + &exp_local.unsqueeze(1i64) * &local_pv)
                / &new_sum.unsqueeze(1i64);
            running_max = new_max;
            running_sum = new_sum;
        }

        if !device_matches(&output, device) {
            return Err("online output landed on unexpected device".to_string());
        }

        let k_ref = Tensor::cat(&k_refs, 0);
        let v_ref = Tensor::cat(&v_refs, 0);
        let reference = payload_attention(&q_cpu, &k_ref, &v_ref);
        let output_cpu = output.to(cpu);
        let diff = (&output_cpu - &reference).abs();
        let max_abs_err = diff.max().double_value(&[]);
        let rel_diff = &diff / (reference.abs() + 1e-12);
        let max_rel_err = rel_diff.max().double_value(&[]);
        let checksum = output_cpu.sum(float).double_value(&[]);

        if max_abs_err <= 1.0e-4 && max_rel_err <= 1.0e-3 {
            let msg = format!(
                "ok blocks={} tokens={token_count} max_abs_err={max_abs_err} max_rel_err={max_rel_err} checksum={checksum}",
                block_lens.len()
            );
            Ok((selection.success_code, msg, max_rel_err))
        } else {
            Err(format!(
                "online payload mismatch blocks={} tokens={token_count} max_abs_err={max_abs_err} max_rel_err={max_rel_err}",
                block_lens.len()
            ))
        }
    }

    pub fn run_payload_chunk_smoke(
        payload: &[u8],
        block_lens: &[i32],
        query_len: i32,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<(i32, String, f64), String> {
        if query_len <= 0 || num_heads <= 0 || head_dim <= 0 {
            return Err("query_len, num_heads, and head_dim must be positive".to_string());
        }
        if block_lens.is_empty() {
            return Err("block_count must be positive".to_string());
        }
        let mut token_count = 0usize;
        for &bl in block_lens {
            if bl <= 0 {
                return Err("all block_lens entries must be positive".to_string());
            }
            token_count += bl as usize;
        }
        let expected_values = token_count * (num_heads as usize) * (head_dim as usize) * 2;
        let expected_bytes = expected_values * std::mem::size_of::<f32>();
        if payload.len() != expected_bytes {
            return Err(format!(
                "chunk payload byte size mismatch expected={expected_bytes} actual={}",
                payload.len()
            ));
        }

        let selection = select_device()?;
        let device = selection.device;
        let cpu = tch::Device::Cpu;
        let float = Kind::Float;

        let values: Vec<f32> = payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let q_cpu = (Tensor::arange(query_len as i64 * num_heads as i64 * head_dim as i64, (float, cpu))
            .reshape([query_len as i64, num_heads as i64, head_dim as i64])
            / 37.0)
            + 0.0625;

        let q = q_cpu.to(device);
        let q_by_head = q.permute(&[1, 0, 2]);
        let mut running_max =
            Tensor::full(&[num_heads as i64, query_len as i64], f64::NEG_INFINITY, (float, device));
        let mut running_sum = Tensor::zeros(&[num_heads as i64, query_len as i64], (float, device));
        let mut output_by_head =
            Tensor::zeros(&[num_heads as i64, query_len as i64, head_dim as i64], (float, device));

        let mut k_refs: Vec<Tensor> = Vec::new();
        let mut v_refs: Vec<Tensor> = Vec::new();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut offset = 0usize;

        for &block_len in block_lens {
            let block_values =
                (block_len as usize) * (num_heads as usize) * (head_dim as usize) * 2;
            let kv_cpu = Tensor::from_slice(&values[offset..offset + block_values])
                .reshape([2, block_len as i64, num_heads as i64, head_dim as i64]);
            offset += block_values;

            let k_cpu = kv_cpu.get(0);
            let v_cpu = kv_cpu.get(1);
            k_refs.push(k_cpu.shallow_clone());
            v_refs.push(v_cpu.shallow_clone());

            let k = k_cpu.to(device);
            let v = v_cpu.to(device);
            let k_by_head = k.permute(&[1, 0, 2]);
            let v_by_head = v.permute(&[1, 0, 2]);
            let scores = q_by_head.matmul(&k_by_head.transpose(1, 2)) * scale;
            let (local_max, _) = scores.max_dim(2, false);
            let weights = (&scores - local_max.unsqueeze(2i64)).exp();
            let local_sum = weights.sum_dim_intlist(&[2i64][..], false, float);
            let local_pv = weights.matmul(&v_by_head);

            let new_max = running_max.max_other(&local_max);
            let exp_prev = (&running_max - &new_max).exp();
            let exp_local = (&local_max - &new_max).exp();
            let new_sum = &exp_prev * &running_sum + &exp_local * &local_sum;
            output_by_head = (&exp_prev.unsqueeze(2i64) * &running_sum.unsqueeze(2i64) * &output_by_head
                + &exp_local.unsqueeze(2i64) * &local_pv)
                / &new_sum.unsqueeze(2i64);
            running_max = new_max;
            running_sum = new_sum;
        }

        let output = output_by_head.permute(&[1, 0, 2]);
        if !device_matches(&output, device) {
            return Err("chunk output landed on unexpected device".to_string());
        }

        let k_ref = Tensor::cat(&k_refs, 0);
        let v_ref = Tensor::cat(&v_refs, 0);
        let reference = payload_chunk_attention(&q_cpu, &k_ref, &v_ref);
        let output_cpu = output.to(cpu);
        let diff = (&output_cpu - &reference).abs();
        let max_abs_err = diff.max().double_value(&[]);
        let rel_diff = &diff / (reference.abs() + 1e-12);
        let max_rel_err = rel_diff.max().double_value(&[]);
        let checksum = tensor_weighted_checksum(&output_cpu);

        if max_abs_err <= 1.0e-4 && max_rel_err <= 1.0e-3 {
            let msg = format!(
                "ok blocks={} query_len={query_len} tokens={token_count} max_abs_err={max_abs_err} max_rel_err={max_rel_err} checksum={checksum}",
                block_lens.len()
            );
            Ok((selection.success_code, msg, max_rel_err))
        } else {
            Err(format!(
                "chunk payload mismatch blocks={} query_len={query_len} tokens={token_count} max_abs_err={max_abs_err} max_rel_err={max_rel_err}",
                block_lens.len()
            ))
        }
    }

    pub fn run_query_chunk_smoke(
        q_payload: &[u8],
        kv_payload: &[u8],
        block_lens: &[i32],
        query_len: i32,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<(i32, String, f64, f64, f64, usize), String> {
        if query_len <= 0 || num_heads <= 0 || head_dim <= 0 {
            return Err("query_len, num_heads, and head_dim must be positive".to_string());
        }
        let expected_q_values = (query_len as usize) * (num_heads as usize) * (head_dim as usize);
        let expected_q_bytes = expected_q_values * std::mem::size_of::<f32>();
        if q_payload.len() != expected_q_bytes {
            return Err(format!(
                "q payload byte size mismatch expected={expected_q_bytes} actual={}",
                q_payload.len()
            ));
        }

        let q_values: Vec<f32> = q_payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let q_cpu = Tensor::from_slice(&q_values)
            .reshape([query_len as i64, num_heads as i64, head_dim as i64]);

        let selection = select_device()?;
        let device = selection.device;
        let cpu = tch::Device::Cpu;
        let float = Kind::Float;

        let mut token_count = 0usize;
        for &bl in block_lens {
            if bl <= 0 {
                return Err("all block_lens entries must be positive".to_string());
            }
            token_count += bl as usize;
        }
        let expected_kv_values = token_count * (num_heads as usize) * (head_dim as usize) * 2;
        let expected_kv_bytes = expected_kv_values * std::mem::size_of::<f32>();
        if kv_payload.len() != expected_kv_bytes {
            return Err(format!(
                "kv payload byte size mismatch expected={expected_kv_bytes} actual={}",
                kv_payload.len()
            ));
        }

        let kv_values: Vec<f32> = kv_payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let q = q_cpu.to(device);
        let q_by_head = q.permute(&[1, 0, 2]);
        let mut running_max =
            Tensor::full(&[num_heads as i64, query_len as i64], f64::NEG_INFINITY, (float, device));
        let mut running_sum = Tensor::zeros(&[num_heads as i64, query_len as i64], (float, device));
        let mut output_by_head =
            Tensor::zeros(&[num_heads as i64, query_len as i64, head_dim as i64], (float, device));

        let mut k_refs: Vec<Tensor> = Vec::new();
        let mut v_refs: Vec<Tensor> = Vec::new();

        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut offset = 0usize;

        for &block_len in block_lens {
            let block_values =
                (block_len as usize) * (num_heads as usize) * (head_dim as usize) * 2;
            let kv_cpu = Tensor::from_slice(&kv_values[offset..offset + block_values])
                .reshape([2, block_len as i64, num_heads as i64, head_dim as i64]);
            offset += block_values;

            let k_cpu = kv_cpu.get(0);
            let v_cpu = kv_cpu.get(1);
            k_refs.push(k_cpu.shallow_clone());
            v_refs.push(v_cpu.shallow_clone());

            let k = k_cpu.to(device);
            let v = v_cpu.to(device);
            let k_by_head = k.permute(&[1, 0, 2]);
            let v_by_head = v.permute(&[1, 0, 2]);
            let scores = q_by_head.matmul(&k_by_head.transpose(1, 2)) * scale;
            let (local_max, _) = scores.max_dim(2, false);
            let weights = (&scores - local_max.unsqueeze(2i64)).exp();
            let local_sum = weights.sum_dim_intlist(&[2i64][..], false, float);
            let local_pv = weights.matmul(&v_by_head);

            let new_max = running_max.max_other(&local_max);
            let exp_prev = (&running_max - &new_max).exp();
            let exp_local = (&local_max - &new_max).exp();
            let new_sum = &exp_prev * &running_sum + &exp_local * &local_sum;
            output_by_head = (&exp_prev.unsqueeze(2i64) * &running_sum.unsqueeze(2i64) * &output_by_head
                + &exp_local.unsqueeze(2i64) * &local_pv)
                / &new_sum.unsqueeze(2i64);
            running_max = new_max;
            running_sum = new_sum;
        }

        let output = output_by_head.permute(&[1, 0, 2]);
        if !device_matches(&output, device) {
            return Err("chunk output landed on unexpected device".to_string());
        }

        let k_ref = Tensor::cat(&k_refs, 0);
        let v_ref = Tensor::cat(&v_refs, 0);
        let reference = payload_chunk_attention(&q_cpu, &k_ref, &v_ref);
        let output_cpu = output.to(cpu);
        let diff = (&output_cpu - &reference).abs();
        let max_abs_err = diff.max().double_value(&[]);
        let rel_diff = &diff / (reference.abs() + 1e-12);
        let max_rel_err = rel_diff.max().double_value(&[]);
        let checksum = tensor_weighted_checksum(&output_cpu);
        let output_values = output_cpu.numel() as usize;

        if max_abs_err <= 1.0e-4 && max_rel_err <= 1.0e-3 {
            let msg = format!(
                "ok blocks={} query_len={query_len} tokens={token_count} max_abs_err={max_abs_err} max_rel_err={max_rel_err} checksum={checksum}",
                block_lens.len()
            );
            Ok((selection.success_code, msg, checksum, max_abs_err, max_rel_err, output_values))
        } else {
            Err(format!(
                "chunk payload mismatch blocks={} query_len={query_len} tokens={token_count} max_abs_err={max_abs_err} max_rel_err={max_rel_err}",
                block_lens.len()
            ))
        }
    }

    pub fn compute_chunk_attention_step(
        q_payload: &[u8],
        kv_payload: &[u8],
        block_len: i32,
        query_len: i32,
        num_heads: i32,
        head_dim: i32,
        running_max: &mut [f32],
        running_sum: &mut [f32],
        output_acc: &mut [f32],
    ) -> Result<(), String> {
        if block_len <= 0 || query_len <= 0 || num_heads <= 0 || head_dim <= 0 {
            return Err("block_len, query_len, num_heads, and head_dim must be positive".to_string());
        }
        let expected_q_bytes = (query_len as usize) * (num_heads as usize) * (head_dim as usize) * 4;
        if q_payload.len() != expected_q_bytes {
            return Err(format!(
                "q payload size mismatch expected={expected_q_bytes} actual={}",
                q_payload.len()
            ));
        }
        let expected_kv_bytes = (block_len as usize) * (num_heads as usize) * (head_dim as usize) * 2 * 4;
        if kv_payload.len() != expected_kv_bytes {
            return Err(format!(
                "kv payload size mismatch expected={expected_kv_bytes} actual={}",
                kv_payload.len()
            ));
        }
        let expected_max_len = (num_heads as usize) * (query_len as usize);
        if running_max.len() != expected_max_len {
            return Err(format!(
                "running_max len mismatch expected={expected_max_len} actual={}",
                running_max.len()
            ));
        }
        if running_sum.len() != expected_max_len {
            return Err(format!(
                "running_sum len mismatch expected={expected_max_len} actual={}",
                running_sum.len()
            ));
        }
        let expected_out_len = expected_max_len * (head_dim as usize);
        if output_acc.len() != expected_out_len {
            return Err(format!(
                "output_acc len mismatch expected={expected_out_len} actual={}",
                output_acc.len()
            ));
        }

        let selection = select_device()?;
        let device = selection.device;
        let float = Kind::Float;

        let q_values: Vec<f32> = q_payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let q = Tensor::from_slice(&q_values)
            .reshape([query_len as i64, num_heads as i64, head_dim as i64])
            .to(device);
        let q_by_head = q.permute(&[1, 0, 2]);

        let kv_values: Vec<f32> = kv_payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let kv = Tensor::from_slice(&kv_values)
            .reshape([2, block_len as i64, num_heads as i64, head_dim as i64]);
        let k = kv.get(0).to(device);
        let v = kv.get(1).to(device);
        let k_by_head = k.permute(&[1, 0, 2]);
        let v_by_head = v.permute(&[1, 0, 2]);

        let mut rm = Tensor::from_slice(running_max)
            .reshape([num_heads as i64, query_len as i64])
            .to(device);
        let mut rs = Tensor::from_slice(running_sum)
            .reshape([num_heads as i64, query_len as i64])
            .to(device);
        let mut obh = Tensor::from_slice(output_acc)
            .reshape([num_heads as i64, query_len as i64, head_dim as i64])
            .to(device);

        let scale = 1.0 / (head_dim as f64).sqrt();
        println!("DEBUG tch_backend: q_by_head shape={:?}, k_by_head shape={:?}", q_by_head.size(), k_by_head.size());
        let scores = q_by_head.matmul(&k_by_head.transpose(1, 2)) * scale;
        println!("DEBUG tch_backend: scores shape={:?}", scores.size());
        let (local_max, _) = scores.max_dim(2, false);
        println!("DEBUG tch_backend: local_max shape={:?}", local_max.size());
        let local_max_unsqueezed = local_max.unsqueeze(2i64);
        println!("DEBUG tch_backend: local_max_unsqueezed shape={:?}", local_max_unsqueezed.size());
        let weights = (&scores - local_max_unsqueezed).exp();
        let local_sum = weights.sum_dim_intlist(&[2i64][..], false, float);
        let local_pv = weights.matmul(&v_by_head);

        println!("DEBUG tch_backend: before max_other");
        let new_max = rm.max_other(&local_max);
        println!("DEBUG tch_backend: before exp_prev");
        let exp_prev = (&rm - &new_max).exp();
        println!("DEBUG tch_backend: before exp_local");
        let exp_local = (&local_max - &new_max).exp();
        println!("DEBUG tch_backend: before new_sum");
        let new_sum = &exp_prev * &rs + &exp_local * &local_sum;
        println!("DEBUG tch_backend: before obh update");
        obh = (&exp_prev.unsqueeze(2i64) * &rs.unsqueeze(2i64) * &obh
            + &exp_local.unsqueeze(2i64) * &local_pv)
            / &new_sum.unsqueeze(2i64);
        println!("DEBUG tch_backend: before rm= new_max");
        rm = new_max;
        println!("DEBUG tch_backend: before rs= new_sum");
        rs = new_sum;

        let rm_cpu = rm.to(tch::Device::Cpu);
        let rs_cpu = rs.to(tch::Device::Cpu);
        let obh_cpu = obh.to(tch::Device::Cpu);

        let rm_flat = rm_cpu.contiguous().view([-1]);
        let rs_flat = rs_cpu.contiguous().view([-1]);
        let obh_flat = obh_cpu.contiguous().view([-1]);

        let rm_vec: Vec<f32> = Vec::try_from(&rm_flat)
            .map_err(|e| format!("failed to copy running_max: {e}"))?;
        let rs_vec: Vec<f32> = Vec::try_from(&rs_flat)
            .map_err(|e| format!("failed to copy running_sum: {e}"))?;
        let obh_vec: Vec<f32> = Vec::try_from(&obh_flat)
            .map_err(|e| format!("failed to copy output_acc: {e}"))?;

        running_max.copy_from_slice(&rm_vec);
        running_sum.copy_from_slice(&rs_vec);
        output_acc.copy_from_slice(&obh_vec);

        Ok(())
    }

    fn tensor_weighted_checksum(tensor: &Tensor) -> f64 {
        let flat = tensor.contiguous().view([-1]);
        let values: Vec<f32> = Vec::try_from(&flat).unwrap_or_default();
        let mut checksum = 0.0;
        for (i, &v) in values.iter().enumerate() {
            checksum += (v as f64) * ((i % 997) + 1) as f64;
        }
        checksum
    }
}

#[cfg(not(feature = "tch-backend"))]
pub mod backend {
    pub fn run_attention_block_updates(_block_updates: i32) -> Result<(i32, String, f64), String> {
        Err("tch-backend feature is not enabled".to_string())
    }
    pub fn run_payload_block_smoke(
        _payload: &[u8],
        _block_len: i32,
        _num_heads: i32,
        _head_dim: i32,
    ) -> Result<(i32, String, f64), String> {
        Err("tch-backend feature is not enabled".to_string())
    }
    pub fn run_payload_online_smoke(
        _payload: &[u8],
        _block_lens: &[i32],
        _num_heads: i32,
        _head_dim: i32,
    ) -> Result<(i32, String, f64), String> {
        Err("tch-backend feature is not enabled".to_string())
    }
    pub fn run_payload_chunk_smoke(
        _payload: &[u8],
        _block_lens: &[i32],
        _query_len: i32,
        _num_heads: i32,
        _head_dim: i32,
    ) -> Result<(i32, String, f64), String> {
        Err("tch-backend feature is not enabled".to_string())
    }
    pub fn run_query_chunk_smoke(
        _q_payload: &[u8],
        _kv_payload: &[u8],
        _block_lens: &[i32],
        _query_len: i32,
        _num_heads: i32,
        _head_dim: i32,
    ) -> Result<(i32, String, f64, f64, f64, usize), String> {
        Err("tch-backend feature is not enabled".to_string())
    }
    pub fn compute_chunk_attention_step(
        _q_payload: &[u8],
        _kv_payload: &[u8],
        _block_len: i32,
        _query_len: i32,
        _num_heads: i32,
        _head_dim: i32,
        _running_max: &mut [f32],
        _running_sum: &mut [f32],
        _output_acc: &mut [f32],
    ) -> Result<(), String> {
        Err("tch-backend feature is not enabled".to_string())
    }
}
