use crate::protocol::{DomainModelState, OnlineSoftmaxAccumulator, RingAttnMessage};

#[cfg(feature = "tch-backend")]
use tch::Tensor;

pub trait ComputeRuntime {
    type Error: std::fmt::Display;

    fn compute_kv_block(
        &mut self,
        model_state: &DomainModelState,
        message: &RingAttnMessage,
        accumulator: &mut OnlineSoftmaxAccumulator,
    ) -> Result<(), Self::Error>;

    fn finalize_output(
        &mut self,
        model_state: &mut DomainModelState,
        accumulator: &OnlineSoftmaxAccumulator,
    ) -> f64;
}

#[cfg(not(feature = "tch-backend"))]
pub struct NoOpComputeRuntime;

#[cfg(not(feature = "tch-backend"))]
impl NoOpComputeRuntime {
    pub fn compute_kv_block(
        &mut self,
        model_state: &DomainModelState,
        message: &RingAttnMessage,
        accumulator: &mut OnlineSoftmaxAccumulator,
    ) -> Result<(), String> {
        <Self as ComputeRuntime>::compute_kv_block(self, model_state, message, accumulator)
    }

    pub fn finalize_output(
        &mut self,
        model_state: &mut DomainModelState,
        accumulator: &OnlineSoftmaxAccumulator,
    ) -> f64 {
        <Self as ComputeRuntime>::finalize_output(self, model_state, accumulator)
    }
}

#[cfg(not(feature = "tch-backend"))]
impl ComputeRuntime for NoOpComputeRuntime {
    type Error = String;

    fn compute_kv_block(
        &mut self,
        _model_state: &DomainModelState,
        _message: &RingAttnMessage,
        _accumulator: &mut OnlineSoftmaxAccumulator,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn finalize_output(
        &mut self,
        _model_state: &mut DomainModelState,
        _accumulator: &OnlineSoftmaxAccumulator,
    ) -> f64 {
        0.0
    }
}

#[cfg(feature = "tch-backend")]
pub struct TchComputeRuntime;

#[cfg(feature = "tch-backend")]
impl TchComputeRuntime {
    pub fn compute_kv_block(
        &mut self,
        model_state: &DomainModelState,
        message: &RingAttnMessage,
        accumulator: &mut OnlineSoftmaxAccumulator,
    ) -> Result<(), String> {
        <Self as ComputeRuntime>::compute_kv_block(self, model_state, message, accumulator)
    }

    pub fn finalize_output(
        &mut self,
        model_state: &mut DomainModelState,
        accumulator: &OnlineSoftmaxAccumulator,
    ) -> f64 {
        <Self as ComputeRuntime>::finalize_output(self, model_state, accumulator)
    }
}

#[cfg(feature = "tch-backend")]
impl ComputeRuntime for TchComputeRuntime {
    type Error = String;

    fn compute_kv_block(
        &mut self,
        model_state: &DomainModelState,
        message: &RingAttnMessage,
        accumulator: &mut OnlineSoftmaxAccumulator,
    ) -> Result<(), Self::Error> {
        use crate::protocol::RingAttnMessageKind;
        use tch::Kind;

        if message.message_kind != RingAttnMessageKind::KvBlock {
            return Ok(());
        }
        let (Some(block), Some(tensor)) = (&message.block, &message.tensor) else {
            return Ok(());
        };

        // ====== 从 bytes 重建 tch Tensor（inline 替代 compute_chunk_attention_step）======
        let block_len = block.block_len as i64;
        let query_len = model_state.query_len() as i64;
        let num_heads = tensor.num_heads as i64;
        let head_dim = tensor.head_dim as i64;
        let device = tch::Device::Cpu;  // protocol 层当前使用 CPU tensor

        // Q: 从 query_payload bytes 解析 f32，shape [query_len, num_heads, head_dim]
        let q_values: Vec<f32> = model_state.query_payload()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let q = Tensor::from_slice(&q_values)
            .reshape([query_len, num_heads, head_dim])
            .to(device)
            .permute([1, 0, 2]);  // → [num_heads, query_len, head_dim]

        // K/V: 从 message.payload bytes 解析，前一半是 K，后一半是 V
        let kv_values: Vec<f32> = message.payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let kv = Tensor::from_slice(&kv_values)
            .reshape([2, block_len, num_heads, head_dim]);
        let k = kv.get(0).to(device).permute([1, 0, 2]);  // → [num_heads, block_len, head_dim]
        let v = kv.get(1).to(device).permute([1, 0, 2]);  // → [num_heads, block_len, head_dim]

        // Accumulator: 从 Vec<f32> 重建 tensor
        let mut rm = Tensor::from_slice(&accumulator.running_max)
            .reshape([num_heads, query_len])
            .to(device);
        let mut rs = Tensor::from_slice(&accumulator.running_sum)
            .reshape([num_heads, query_len])
            .to(device);
        let mut obh = Tensor::from_slice(&accumulator.output_acc)
            .reshape([num_heads, query_len, head_dim])
            .to(device);

        // Online softmax 计算（与 process_kv_block 数学等价）
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)) * scale;
        let (local_max, _) = scores.max_dim(2, false);
        let weights = (&scores - local_max.unsqueeze(2)).exp();
        let local_sum = weights.sum_dim_intlist(&[2i64][..], false, Kind::Float);
        let local_pv = weights.matmul(&v);

        let new_max = rm.max_other(&local_max);
        let exp_prev = (&rm - &new_max).exp();
        let exp_local = (&local_max - &new_max).exp();
        let new_sum = &exp_prev * &rs + &exp_local * &local_sum;

        obh = (&exp_prev.unsqueeze(2) * &rs.unsqueeze(2) * &obh
            + &exp_local.unsqueeze(2) * &local_pv)
            / &new_sum.unsqueeze(2);
        rm = new_max;
        rs = new_sum;

        // 把结果写回 accumulator（D2H 拷贝）
        let rm_vec: Vec<f32> = Vec::try_from(&rm.to(tch::Device::Cpu).contiguous().view(-1))
            .map_err(|e| format!("failed to copy running_max: {e}"))?;
        let rs_vec: Vec<f32> = Vec::try_from(&rs.to(tch::Device::Cpu).contiguous().view(-1))
            .map_err(|e| format!("failed to copy running_sum: {e}"))?;
        let obh_vec: Vec<f32> = Vec::try_from(&obh.to(tch::Device::Cpu).contiguous().view(-1))
            .map_err(|e| format!("failed to copy output_acc: {e}"))?;

        accumulator.running_max.copy_from_slice(&rm_vec);
        accumulator.running_sum.copy_from_slice(&rs_vec);
        accumulator.output_acc.copy_from_slice(&obh_vec);

        Ok(())
    }

    fn finalize_output(
        &mut self,
        model_state: &mut DomainModelState,
        accumulator: &OnlineSoftmaxAccumulator,
    ) -> f64 {
        use crate::protocol::FLOAT32_BYTES;

        let query_len = model_state.query_len();
        let num_heads = model_state.num_heads();
        let head_dim = model_state.head_dim();
        let hidden_dim = model_state.activation.hidden_dim;
        let mut output_slot = vec![0_u8; query_len * hidden_dim * FLOAT32_BYTES];
        for q in 0..query_len {
            let mut attn_out = Vec::with_capacity(num_heads * head_dim);
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src_idx = (h * query_len + q) * head_dim + d;
                    attn_out.push(accumulator.output_acc[src_idx]);
                }
            }
            let o_proj_out = model_state.weights.project_output(&attn_out);
            let residual_start = q * hidden_dim;
            for (d, &val) in o_proj_out.iter().enumerate() {
                let residual = model_state.activation.residual_input[residual_start + d] + val;
                let offset = residual_start + d;
                output_slot[offset * FLOAT32_BYTES..(offset + 1) * FLOAT32_BYTES]
                    .copy_from_slice(&residual.to_le_bytes());
            }
        }
        model_state.activation.output_slot = output_slot;
        let values: Vec<f32> = model_state
            .activation
            .output_slot
            .chunks_exact(FLOAT32_BYTES)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut checksum = 0.0_f64;
        for (i, &v) in values.iter().enumerate() {
            checksum += (v as f64) * ((i % 997) + 1) as f64;
        }
        checksum
    }
}
