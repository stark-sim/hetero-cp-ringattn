use tch::{Device, Kind, Tensor};

fn main() {
    // Test 1: Create BF16 tensor from raw bytes using f_from_data_size
    let data: Vec<u8> = vec![0x80, 0x3F, 0x00, 0x40, 0x40, 0x40]; // 1.0, 2.0, 3.0 in BF16 LE
    let tensor = Tensor::f_from_data_size(&data, &[3], Kind::BFloat16).unwrap();
    println!("BF16 tensor from raw bytes:");
    println!("  kind: {:?}", tensor.kind());
    println!("  shape: {:?}", tensor.size());
    
    let f32_vals: Vec<f32> = Vec::try_from(&tensor.to_kind(Kind::Float)).unwrap();
    println!("  values as f32: {:?}", f32_vals);
    
    // Test 2: Load a real safetensors file with BF16
    let model_dir = "/Users/stark_sim/VSCodeProjects/hetero-cp-ringattn/models/Qwen2-0.5B";
    let safetensors = Tensor::read_safetensors(&format!("{}/model.safetensors", model_dir)).unwrap();
    
    println!("\nLoaded {} tensors from safetensors", safetensors.len());
    
    // Find embedding tensor
    for (name, tensor) in &safetensors {
        if name == "model.embed_tokens.weight" {
            println!("\nEmbedding tensor:");
            println!("  name: {}", name);
            println!("  kind: {:?}", tensor.kind());
            println!("  shape: {:?}", tensor.size());
            
            // Convert to f32 for reading first few values
            let f32_tensor = tensor.to_kind(Kind::Float);
            let first_row: Vec<f32> = f32_tensor.get(0).view(-1).try_into().unwrap();
            println!("  first row [0..3]: {:?}", &first_row[..3]);
            break;
        }
    }
    
    // Test 3: BF16 matmul
    let a = Tensor::randn([4, 8], (Kind::BFloat16, Device::Cpu));
    let b = Tensor::randn([8, 6], (Kind::BFloat16, Device::Cpu));
    let c = a.matmul(&b);
    println!("\nBF16 matmul:");
    println!("  result kind: {:?}", c.kind());
    println!("  result shape: {:?}", c.size());
}
