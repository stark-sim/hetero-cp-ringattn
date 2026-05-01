use crate::model::{ModelConfig, ModelError};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "tch-backend")]
use tch::{Device, Tensor};

/// 【模型权重集合】从 safetensors 文件加载的全部权重。
///
/// safetensors 是 HuggingFace 推出的一种安全、高效的权重存储格式，
/// 相比传统的 pickle（.bin 文件）不会执行任意代码，更安全。
///
/// 结构体内部用 HashMap<String, Tensor> 存储：
/// - key: HuggingFace 的权重名称（例如 "model.layers.0.self_attn.q_proj.weight"）
/// - value: 对应的 tch::Tensor
#[derive(Debug)]
pub struct ModelWeights {
    /// 权重名到 tensor 的映射。
    /// #[cfg(feature = "tch-backend")] 表示只在启用 tch-backend feature 时编译这段代码。
    #[cfg(feature = "tch-backend")]
    pub tensors: HashMap<String, Tensor>,
    /// 未启用 tch-backend 时的 fallback：存储原始字节（目前未实际使用）。
    #[cfg(not(feature = "tch-backend"))]
    pub tensors: HashMap<String, Vec<u8>>,
}

/// 【权重名称生成器】
/// HuggingFace 的权重有一套固定的命名规则，例如：
/// - Embedding: model.embed_tokens.weight
/// - Layer 0 的 Q 投影权重: model.layers.0.self_attn.q_proj.weight
/// - Layer 0 的 MLP gate 权重: model.layers.0.mlp.gate_proj.weight
///
/// 这个结构体提供静态方法，根据 layer 索引生成对应的权重名称字符串。
pub struct WeightNames;

impl WeightNames {
    /// 【词嵌入权重名称】
    pub fn embedding() -> &'static str {
        "model.embed_tokens.weight"
    }

    /// 【最终层归一化权重名称】
    pub fn layer_norm() -> &'static str {
        "model.norm.weight"
    }

    /// 【语言模型头权重名称】
    /// 如果 tie_word_embeddings=true，这个权重可能与 embedding 共享。
    pub fn lm_head() -> &'static str {
        "lm_head.weight"
    }

    /// 【输入层归一化权重】
    /// 每个 layer 在 attention 之前有一个 RMSNorm。
    pub fn rms_norm_weight(layer: usize) -> String {
        format!("model.layers.{layer}.input_layernorm.weight")
    }

    /// 【Attention 后的层归一化权重】
    /// 每个 layer 在 attention 之后、MLP 之前也有一个 RMSNorm。
    pub fn post_attn_norm_weight(layer: usize) -> String {
        format!("model.layers.{layer}.post_attention_layernorm.weight")
    }

    /// 【Q 投影权重】
    /// 把 hidden states 映射到 Query 向量空间。
    pub fn q_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.q_proj.weight")
    }

    /// 【K 投影权重】
    /// 把 hidden states 映射到 Key 向量空间。
    pub fn k_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.k_proj.weight")
    }

    /// 【V 投影权重】
    /// 把 hidden states 映射到 Value 向量空间。
    pub fn v_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.v_proj.weight")
    }

    /// 【Q 投影偏置】有些模型（如 Qwen）在 Q/K/V 投影中使用偏置。
    pub fn q_proj_bias(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.q_proj.bias")
    }

    /// 【K 投影偏置】
    pub fn k_proj_bias(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.k_proj.bias")
    }

    /// 【V 投影偏置】
    pub fn v_proj_bias(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.v_proj.bias")
    }

    /// 【O 投影权重】
    /// 把 attention 输出（多 head 拼接后）映射回 hidden_size 维度。
    pub fn o_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.o_proj.weight")
    }

    /// 【MLP Gate 投影权重】
    /// SwiGLU 结构的一部分：用 gate_proj 生成门控信号。
    pub fn gate_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.mlp.gate_proj.weight")
    }

    /// 【MLP Up 投影权重】
    /// SwiGLU 结构的另一部分：用 up_proj 升维。
    pub fn up_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.mlp.up_proj.weight")
    }

    /// 【MLP Down 投影权重】
    /// 把 MLP 中间层输出降维回 hidden_size。
    pub fn down_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.mlp.down_proj.weight")
    }
}

impl ModelWeights {
    /// 【从目录加载所有 safetensors 文件】
    ///
    /// 扫描指定目录下的所有 `.safetensors` 文件，按文件名排序后依次加载。
    /// 大模型（如 70B）的权重通常被拆分成多个文件（如 model-00001-of-00004.safetensors），
    /// 所以需要遍历所有文件并合并它们的权重。
    ///
    /// 参数：
    /// - dir: 模型目录路径（例如 ~/models/qwen2-0.5b/）
    /// - device: 目标设备（CPU / MPS / CUDA），加载后的 tensor 会被移到该设备上
    #[cfg(feature = "tch-backend")]
    pub fn from_dir<P: AsRef<Path>>(dir: P, device: Device) -> Result<Self, ModelError> {
        let dir = dir.as_ref();

        // 读取目录中的所有文件，筛选出扩展名为 .safetensors 的文件。
        let mut entries: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())   // 忽略读取失败的条目
            .filter(|e| {
                e.path().extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();
        entries.sort();  // 按文件名排序，确保加载顺序一致

        if entries.is_empty() {
            return Err(ModelError::Safetensors(
                format!("No .safetensors files found in {}", dir.display())
            ));
        }

        // 逐个文件加载权重。
        let mut tensors = HashMap::new();
        for path in entries {
            let file_data = std::fs::read(&path)?;

            // safetensors::SafeTensors::deserialize 解析二进制格式，返回一个视图对象。
            // 这个视图不包含实际数据拷贝，只是对文件内存的映射。
            let st = safetensors::SafeTensors::deserialize(&file_data)
                .map_err(|e| ModelError::Safetensors(e.to_string()))?;

            // 遍历这个文件中的所有 tensor。
            for name in st.names() {
                let view = st.tensor(&name)
                    .map_err(|e| ModelError::Safetensors(e.to_string()))?;
                // 把 safetensors 的 view 转换为 tch::Tensor。
                let tensor = tensor_from_view(&view, device)?;
                tensors.insert(name.to_string(), tensor);
            }
        }
        Ok(Self { tensors })
    }

    /// 【按名称获取权重 tensor】
    /// 如果找不到，返回 MissingWeight 错误。
    #[cfg(feature = "tch-backend")]
    pub fn get(&self, name: &str) -> Result<&Tensor, ModelError> {
        self.tensors.get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))
    }

    /// 【获取 lm_head 权重】
    /// 如果 config 中 tie_word_embeddings=true，lm_head 可能与 embedding 共享权重。
    /// 这种情况下 lm_head 的权重名称可能不存在，需要 fallback 到 embedding 的权重。
    #[cfg(feature = "tch-backend")]
    pub fn get_lm_head(&self, config: &ModelConfig) -> Result<&Tensor, ModelError> {
        // 先尝试直接获取 lm_head 权重。
        if let Ok(t) = self.get(WeightNames::lm_head()) {
            return Ok(t);
        }
        // 如果找不到且启用了权重共享，就返回 embedding 权重。
        if config.tie_word_embeddings {
            return self.get(WeightNames::embedding());
        }
        Err(ModelError::MissingWeight(WeightNames::lm_head().to_string()))
    }
}

/// 【将 safetensors view 转换为 tch::Tensor】
///
/// safetensors 支持多种数据类型（F32、F16、BF16），但 tch-rs 通常需要 F32 tensor。
/// 这个函数负责类型转换：
/// - F32: 直接读取，无需转换
/// - F16: 每个元素 2 字节，需要手动解码为 f32
/// - BF16: 每个元素 2 字节，也需要手动解码为 f32
///
/// 注意：目前不支持 INT8/INT4 量化格式，遇到会报错。
#[cfg(feature = "tch-backend")]
fn tensor_from_view(view: &safetensors::tensor::TensorView, device: Device) -> Result<Tensor, ModelError> {
    use safetensors::tensor::Dtype;

    // safetensors 的 shape 是 &[usize]，tch-rs 需要 Vec<i64>。
    let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();

    let tensor = match view.dtype() {
        // ====== F32: 最简单，直接读取 ======
        Dtype::F32 => {
            // bytemuck::cast_slice 把 &[u8] 安全地 reinterpret 为 &[f32]。
            // 前提是字节对齐和长度匹配（safetensors 保证这一点）。
            let data: &[f32] = bytemuck::cast_slice(view.data());
            Tensor::from_slice(data).to_device(device).view(shape.as_slice())
        }

        // ====== F16 (Half Precision): 需要手动解码 ======
        // F16 每个数占 2 字节（16 bit），格式：1 位符号 + 5 位指数 + 10 位尾数。
        // 大多数 GPU（NVIDIA A100/H100、Apple M 系列）原生支持 F16 计算，可以节省显存和带宽。
        Dtype::F16 => {
            let bytes: &[u8] = view.data();
            let f32_vec: Vec<f32> = bytes
                .chunks_exact(2)  // 每 2 字节一个 F16 数
                .map(|chunk| {
                    // 小端序解码：先读 2 字节成 u16，再转成 f16，最后转成 f32。
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();
            Tensor::from_slice(&f32_vec).to_device(device).view(shape.as_slice())
        }

        // ====== BF16 (Brain Float 16): 也需要手动解码 ======
        // BF16 也是 2 字节，但格式不同：1 位符号 + 8 位指数 + 7 位尾数。
        // 相比 F16，BF16 的指数范围和 F32 相同（都是 8 位），所以数值稳定性更好，
        // 但精度略低（尾数只有 7 位 vs F16 的 10 位）。
        // Google TPU 和 NVIDIA Ampere+ GPU 广泛支持 BF16。
        Dtype::BF16 => {
            let bytes: &[u8] = view.data();
            let f32_vec: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect();
            Tensor::from_slice(&f32_vec).to_device(device).view(shape.as_slice())
        }

        // 其他类型（如 INT8、INT4）暂不支持。
        other => {
            return Err(ModelError::Safetensors(
                format!("Unsupported safetensors dtype: {:?}", other)
            ));
        }
    };
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_names() {
        assert_eq!(WeightNames::q_proj_weight(3), "model.layers.3.self_attn.q_proj.weight");
        assert_eq!(WeightNames::gate_proj_weight(0), "model.layers.0.mlp.gate_proj.weight");
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_load_safetensors() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let weights = ModelWeights::from_dir(&dir, Device::Cpu).expect("load test weights");

        // Verify F32 tensor
        let f32_t = weights.get("test_f32").expect("get test_f32");
        let f32_shape: Vec<i64> = f32_t.size();
        assert_eq!(f32_shape, vec![2, 3]);
        let f32_data: Vec<f32> = f32_t.view(-1).try_into().expect("f32 to vec");
        assert!((f32_data[0] - 1.0).abs() < 1e-6);
        assert!((f32_data[5] - 6.0).abs() < 1e-6);

        // Verify F16 tensor (loaded and converted to F32)
        let f16_t = weights.get("test_f16").expect("get test_f16");
        let f16_shape: Vec<i64> = f16_t.size();
        assert_eq!(f16_shape, vec![2, 3]);
        let f16_data: Vec<f32> = f16_t.view(-1).try_into().expect("f16->f32 to vec");
        assert!((f16_data[0] - 0.5).abs() < 1e-3);
        assert!((f16_data[5] - 5.5).abs() < 1e-3);
    }
}
