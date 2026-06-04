use tch::{Device, Kind, Tensor};

fn main() {
    // Test creating BF16 tensor with NEG_INFINITY
    let neg_inf = Tensor::from(f64::NEG_INFINITY)
        .to_kind(Kind::BFloat16)
        .to_device(Device::Cpu);
    println!("neg_inf kind: {:?}", neg_inf.kind());
    
    let full = Tensor::full([1, 14, 4], f64::NEG_INFINITY, (Kind::BFloat16, Device::Cpu));
    println!("full kind: {:?}", full.kind());
    
    // Test max_other
    let a = Tensor::full([1, 14, 4], f64::NEG_INFINITY, (Kind::BFloat16, Device::Cpu));
    let b = Tensor::randn([1, 14, 4], (Kind::BFloat16, Device::Cpu));
    let c = a.max_other(&b);
    println!("max_other kind: {:?}", c.kind());
    
    // Test subtraction and exp
    let d = (&a - &c).exp();
    println!("exp kind: {:?}", d.kind());
    
    // Test multiplication
    let e = &d * &b;
    println!("mul kind: {:?}", e.kind());
    
    // Test matmul with 4D tensors
    let q = Tensor::randn([1, 14, 4, 64], (Kind::BFloat16, Device::Cpu));
    let k = Tensor::randn([1, 14, 4, 64], (Kind::BFloat16, Device::Cpu));
    let scores = q.matmul(&k.transpose(2, 3));
    println!("scores kind: {:?}", scores.kind());
    
    // Test with NEG_INFINITY mask
    let mask = Tensor::full([1, 1, 4, 4], f64::NEG_INFINITY, (Kind::BFloat16, Device::Cpu));
    let masked = &scores + &mask;
    println!("masked kind: {:?}", masked.kind());
}
