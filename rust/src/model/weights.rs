use crate::model::{ModelConfig, ModelError};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

/// A collection of model weights loaded from safetensors files.
#[derive(Debug)]
pub struct ModelWeights {
    /// Map from HuggingFace weight name to tch Tensor.
    #[cfg(feature = "tch-backend")]
    pub tensors: HashMap<String, Tensor>,
    #[cfg(not(feature = "tch-backend"))]
    pub tensors: HashMap<String, Vec<u8>>,
}

/// Standard HuggingFace weight name patterns for Llama-family models.
pub struct WeightNames;

impl WeightNames {
    pub fn embedding() -> &'static str {
        "model.embed_tokens.weight"
    }

    pub fn layer_norm() -> &'static str {
        "model.norm.weight"
    }

    pub fn lm_head() -> &'static str {
        "lm_head.weight"
    }

    pub fn rms_norm_weight(layer: usize) -> String {
        format!("model.layers.{layer}.input_layernorm.weight")
    }

    pub fn post_attn_norm_weight(layer: usize) -> String {
        format!("model.layers.{layer}.post_attention_layernorm.weight")
    }

    pub fn q_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.q_proj.weight")
    }

    pub fn k_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.k_proj.weight")
    }

    pub fn v_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.v_proj.weight")
    }

    pub fn o_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.self_attn.o_proj.weight")
    }

    pub fn gate_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.mlp.gate_proj.weight")
    }

    pub fn up_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.mlp.up_proj.weight")
    }

    pub fn down_proj_weight(layer: usize) -> String {
        format!("model.layers.{layer}.mlp.down_proj.weight")
    }
}

impl ModelWeights {
    /// Load all `.safetensors` files from a model directory.
    #[cfg(feature = "tch-backend")]
    pub fn from_dir<P: AsRef<Path>>(dir: P, device: Device) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let mut entries: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();
        entries.sort();

        if entries.is_empty() {
            return Err(ModelError::Safetensors(
                format!("No .safetensors files found in {}", dir.display())
            ));
        }

        let mut tensors = HashMap::new();
        for path in entries {
            let file_data = std::fs::read(&path)?;
            let st = safetensors::SafeTensors::deserialize(&file_data)
                .map_err(|e| ModelError::Safetensors(e.to_string()))?;
            for name in st.names() {
                let view = st.tensor(&name)
                    .map_err(|e| ModelError::Safetensors(e.to_string()))?;
                let tensor = tensor_from_view(&view, device)?;
                tensors.insert(name.to_string(), tensor);
            }
        }
        Ok(Self { tensors })
    }

    /// Get a tensor by its HF weight name.
    #[cfg(feature = "tch-backend")]
    pub fn get(&self, name: &str) -> Result<&Tensor, ModelError> {
        self.tensors.get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))
    }

    /// Get a tensor, or if missing and tie_word_embeddings is true, fallback to embedding weight.
    #[cfg(feature = "tch-backend")]
    pub fn get_lm_head(&self, config: &ModelConfig) -> Result<&Tensor, ModelError> {
        if let Ok(t) = self.get(WeightNames::lm_head()) {
            return Ok(t);
        }
        if config.tie_word_embeddings {
            return self.get(WeightNames::embedding());
        }
        Err(ModelError::MissingWeight(WeightNames::lm_head().to_string()))
    }
}

/// Convert a safetensors tensor view to a tch::Tensor.
#[cfg(feature = "tch-backend")]
fn tensor_from_view(view: &safetensors::tensor::TensorView, device: Device) -> Result<Tensor, ModelError> {
    use safetensors::tensor::Dtype;
    let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
    let tensor = match view.dtype() {
        Dtype::F32 => {
            let data: &[f32] = bytemuck::cast_slice(view.data());
            Tensor::from_slice(data).to_device(device).view(shape.as_slice())
        }
        Dtype::F16 => {
            let bytes: &[u8] = view.data();
            let f32_vec: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();
            Tensor::from_slice(&f32_vec).to_device(device).view(shape.as_slice())
        }
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
