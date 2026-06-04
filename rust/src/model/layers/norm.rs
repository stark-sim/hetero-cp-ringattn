use crate::model::{ModelError, ModelWeights};

#[cfg(feature = "tch-backend")]
use tch::{Kind, Tensor};

pub struct RmsNorm {
    pub weight: Tensor,  // 可学习的缩放参数 gamma，shape [hidden_size]
    pub eps: f64,        // 防止除零的小常数，默认 1e-6
}

#[cfg(feature = "tch-backend")]
impl RmsNorm {
    /// 【从权重加载】
    pub fn from_weights(weights: &ModelWeights, name: &str, eps: f64) -> Result<Self, ModelError> {
        let weight = weights.get(name)?.shallow_clone();
        Ok(Self { weight, eps })
    }

    /// 【前向传播】
    ///
    /// 数学公式：
    ///   variance = mean(x^2, dim=-1, keepdim=True)
    ///   rms = sqrt(variance + eps)
    ///   output = x / rms * weight
    ///
    /// 代码解释：
    /// - pow_tensor_scalar(2): 每个元素平方
    /// - mean_dim(&[-1][..], true, ...): 在最后一个维度（特征维）上求均值，keepdim=true 保留维度以便广播
    /// - rsqrt(): 平方根后取倒数（1/sqrt）
    /// - 最后乘 weight（gamma）做缩放
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let variance = x.pow_tensor_scalar(2i64).mean_dim(&[-1i64][..], true, x.kind());
        let result = x * (variance + self.eps).rsqrt() * &self.weight;
        eprintln!("[rmsnorm] input={:?} weight={:?} output={:?}", x.kind(), self.weight.kind(), result.kind());
        result
    }
}
