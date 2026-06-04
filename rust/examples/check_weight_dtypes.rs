use tch::Tensor;

fn main() {
    let model_dir = "/Users/stark_sim/VSCodeProjects/hetero-cp-ringattn/models/Qwen2-0.5B";
    let safetensors = Tensor::read_safetensors(&format!("{}/model.safetensors", model_dir)).unwrap();

    println!("Loaded {} tensors", safetensors.len());

    // Check attention-related weights
    for (name, tensor) in &safetensors {
        if name.contains("proj") || name.contains("embed") || name.contains("norm") || name.contains("lm_head") {
            println!("{}: kind={:?}, shape={:?}", name, tensor.kind(), tensor.size());
        }
    }
}
