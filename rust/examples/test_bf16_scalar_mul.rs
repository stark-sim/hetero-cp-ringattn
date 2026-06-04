use tch::{Device, Kind, Tensor};

fn main() {
    let a = Tensor::randn([2, 2], (Kind::BFloat16, Device::Cpu));
    println!("a kind: {:?}", a.kind());
    
    // Test f64 scalar multiplication
    let scale: f64 = 0.125;
    let b = &a * scale;
    println!("a * f64 kind: {:?}", b.kind());
    
    // Test f32 scalar multiplication (need explicit cast)
    let scale_f32: f32 = 0.125;
    let c = &a * scale_f32 as f64;
    println!("a * f32->f64 kind: {:?}", c.kind());
    
    // Test Tensor scalar multiplication
    let d = &a * Tensor::from(scale);
    println!("a * Tensor(f64) kind: {:?}", d.kind());
    
    // Test addition with f64
    let e = &a + 1.0f64;
    println!("a + f64 kind: {:?}", e.kind());
}
