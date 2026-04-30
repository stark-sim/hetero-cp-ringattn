use crate::protocol::{DomainModelState, OnlineSoftmaxAccumulator, RingAttnMessage};

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

pub struct NoOpComputeRuntime;

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

        if message.message_kind != RingAttnMessageKind::KvBlock {
            return Ok(());
        }
        let (Some(block), Some(tensor)) = (&message.block, &message.tensor) else {
            return Ok(());
        };
        crate::tch_backend::backend::compute_chunk_attention_step(
            model_state.query_payload(),
            &message.payload,
            i32::try_from(block.block_len).unwrap_or(i32::MAX),
            i32::try_from(model_state.query_len()).unwrap_or(i32::MAX),
            i32::try_from(tensor.num_heads).unwrap_or(i32::MAX),
            i32::try_from(tensor.head_dim).unwrap_or(i32::MAX),
            &mut accumulator.running_max,
            &mut accumulator.running_sum,
            &mut accumulator.output_acc,
        )
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
            for d in 0..hidden_dim {
                let residual = model_state.activation.residual_input[residual_start + d]
                    + o_proj_out[d];
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
