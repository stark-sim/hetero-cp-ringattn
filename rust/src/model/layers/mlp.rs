use crate::model::{ModelConfig, ModelError, ModelWeights, WeightNames};

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

// ==================== MLP (SwiGLU) ====================

/// 【MLP：前馈神经网络】
///
/// 传统 Transformer 使用 ReLU 或 GELU 作为 FFN 的激活函数。
/// Llama/Qwen2 使用 SwiGLU，它是一种门控结构：
///
///   gate = x @ W_gate^T
///   up   = x @ W_up^T
///   output = silu(gate) * up @ W_down^T
///
/// 其中 silu(x) = x * sigmoid(x)，也叫 Swish 激活函数。
///
/// SwiGLU 的效果：
/// - gate 控制信息流通的"门"，决定哪些信息通过
/// - up 提供升维后的特征表示
/// - silu(gate) * up 是逐元素乘法（Hadamard 积），实现门控效果
/// - down 把维度降回 hidden_size
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct Mlp {
    pub gate_proj: Tensor,  // [intermediate_size, hidden_size]
    pub up_proj: Tensor,    // [intermediate_size, hidden_size]
    pub down_proj: Tensor,  // [hidden_size, intermediate_size]
}

#[cfg(feature = "tch-backend")]
impl Mlp {
    pub fn from_weights(weights: &ModelWeights, layer: usize) -> Result<Self, ModelError> {
        Ok(Self {
            gate_proj: weights.get(&WeightNames::gate_proj_weight(layer))?.shallow_clone(),
            up_proj: weights.get(&WeightNames::up_proj_weight(layer))?.shallow_clone(),
            down_proj: weights.get(&WeightNames::down_proj_weight(layer))?.shallow_clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        const MLP_CHUNK_SIZE: i64 = 8192;
        let seq_len = x.size()[1];

        if seq_len > MLP_CHUNK_SIZE {
            let mut all_outputs = Vec::new();
            for start in (0..seq_len).step_by(MLP_CHUNK_SIZE as usize) {
                let chunk_len = (start + MLP_CHUNK_SIZE).min(seq_len) - start;
                let chunk = x.narrow(1, start, chunk_len);
                let gate = chunk.matmul(&self.gate_proj.transpose(0, 1));
                let up = chunk.matmul(&self.up_proj.transpose(0, 1));
                let activated = gate.silu() * up;
                all_outputs.push(activated.matmul(&self.down_proj.transpose(0, 1)));
            }
            Tensor::cat(&all_outputs, 1)
        } else {
            // gate: 生成门控信号，shape [batch, seq_len, intermediate_size]
            let gate = x.matmul(&self.gate_proj.transpose(0, 1));
            // up: 升维特征，shape [batch, seq_len, intermediate_size]
            let up = x.matmul(&self.up_proj.transpose(0, 1));
            // silu(gate) * up: 门控后的激活，shape [batch, seq_len, intermediate_size]
            let activated = gate.silu() * up;
            // down: 降维回 hidden_size，shape [batch, seq_len, hidden_size]
            activated.matmul(&self.down_proj.transpose(0, 1))
        }
    }
}
