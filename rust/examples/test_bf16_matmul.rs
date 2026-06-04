use tch::{Device, Kind, Tensor};

fn main() {
    // Test BF16 matmul
    let a = Tensor::randn([1, 14, 4, 64], (Kind::BFloat16, Device::Cpu));
    let b = Tensor::randn([1, 14, 4, 64], (Kind::BFloat16, Device::Cpu));
    
    println!("a kind: {:?}", a.kind());
    println!("b kind: {:?}", b.kind());
    println!("b.transpose kind: {:?}", b.transpose(2, 3).kind());
    
    let c = a.matmul(&b.transpose(2, 3));
    println!("c = a.matmul(b.T) kind: {:?}", c.kind());
    
    // Test with scale
    let scale: f64 = 0.125;
    let d = c * Tensor::from(scale).to_kind(Kind::BFloat16);
    println!("c * scale kind: {:?}", d.kind());
    
    // Test causal mask addition
    let mask = Tensor::zeros([1, 1, 4, 4], (Kind::BFloat16, Device::Cpu));
    let e = d + mask;
    println!("d + mask kind: {:?}", e.kind());
    
    // Test softmax
    let f = e.softmax(-1, Kind::BFloat16);
    println!("softmax kind: {:?}", f.kind());
    
    // Test matmul with v
    let v = Tensor::randn([1, 14, 4, 64], (Kind::BFloat16, Device::Cpu));
    let g = f.matmul(&v);
    println!("attn @ v kind: {:?}", g.kind());
}
