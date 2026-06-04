use tch::{Device, Kind, Tensor};

fn main() {
    // Simulate what happens in attention forward
    let hidden_size = 896i64;
    let num_heads = 14i64;
    let num_kv_heads = 2i64;
    let head_dim = 64i64;
    let seq_len = 4i64;
    
    // hidden_states is BF16
    let hidden_states = Tensor::randn([1, seq_len, hidden_size], (Kind::BFloat16, Device::Cpu));
    println!("hidden_states kind: {:?}", hidden_states.kind());
    
    // q_proj is BF16
    let q_proj = Tensor::randn([num_heads * head_dim, hidden_size], (Kind::BFloat16, Device::Cpu));
    println!("q_proj kind: {:?}", q_proj.kind());
    println!("q_proj.transpose kind: {:?}", q_proj.transpose(0, 1).kind());
    
    // matmul
    let q = hidden_states.matmul(&q_proj.transpose(0, 1));
    println!("q = hidden @ W_q^T kind: {:?}", q.kind());
    
    // view + transpose
    let q = q.view([1, seq_len, num_heads, head_dim]).transpose(1, 2);
    println!("q reshaped kind: {:?}", q.kind());
    
    // narrow
    let q_chunk = q.narrow(2, 0, 2);
    println!("q_chunk kind: {:?}", q_chunk.kind());
    
    // matmul for scores - use same num_heads for k/v to avoid GQA broadcasting issues in test
    let k = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::BFloat16, Device::Cpu));
    let scores = q_chunk.matmul(&k.transpose(2, 3));
    println!("scores kind: {:?}", scores.kind());
    
    // Test with scale as f64 multiplication - THIS IS THE KEY TEST
    let scale: f64 = 0.125;
    let scores_scaled = &scores * scale;
    println!("scores * f64 kind: {:?}", scores_scaled.kind());
    
    // Test with scale as tensor
    let scores_scaled2 = &scores * Tensor::from(scale).to_kind(Kind::BFloat16);
    println!("scores * BF16 tensor kind: {:?}", scores_scaled2.kind());
    
    // Test causal mask with where_self
    let causal = Tensor::ones([2, seq_len], (Kind::Bool, Device::Cpu));
    let neg_inf = Tensor::from(f64::NEG_INFINITY).to_kind(Kind::BFloat16).to_device(Device::Cpu);
    let zero = Tensor::zeros(1, (Kind::BFloat16, Device::Cpu));
    let mask = neg_inf.where_self(&causal.logical_not(), &zero);
    println!("mask kind: {:?}", mask.kind());
    
    let masked_scores = scores_scaled + mask.unsqueeze(0).unsqueeze(0);
    println!("masked_scores kind: {:?}", masked_scores.kind());
    
    // local_max
    let local_max = masked_scores.amax(&[3i64][..], false);
    println!("local_max kind: {:?}", local_max.kind());
    
    // weights
    let weights = (&masked_scores - local_max.unsqueeze(3)).exp();
    println!("weights kind: {:?}", weights.kind());
    
    // local_sum
    let local_sum = weights.sum_dim_intlist(&[3i64][..], false, Kind::BFloat16);
    println!("local_sum kind: {:?}", local_sum.kind());
    
    // local_pv
    let v = Tensor::randn([1, num_heads, seq_len, head_dim], (Kind::BFloat16, Device::Cpu));
    let local_pv = weights.matmul(&v);
    println!("local_pv kind: {:?}", local_pv.kind());
    
    // running max with NEG_INFINITY
    let rm = Tensor::full([1, num_heads, 2], f64::NEG_INFINITY, (Kind::BFloat16, Device::Cpu));
    println!("rm kind: {:?}", rm.kind());
    let new_max = rm.max_other(&local_max);
    println!("new_max kind: {:?}", new_max.kind());
    
    let exp_prev = (&rm - &new_max).exp();
    println!("exp_prev kind: {:?}", exp_prev.kind());
    
    // running sum
    let rs = Tensor::zeros([1, num_heads, 2], (Kind::BFloat16, Device::Cpu));
    let new_sum = &exp_prev * &rs + &Tensor::from(1.0).to_kind(Kind::BFloat16) * &local_sum;
    println!("new_sum kind: {:?}", new_sum.kind());
    
    // obh update
    let obh = Tensor::zeros([1, num_heads, 2, head_dim], (Kind::BFloat16, Device::Cpu));
    let updated = (&exp_prev.unsqueeze(3) * &rs.unsqueeze(3) * &obh
        + &Tensor::from(1.0).to_kind(Kind::BFloat16).unsqueeze(3) * &local_pv)
        / &new_sum.unsqueeze(3);
    println!("updated obh kind: {:?}", updated.kind());
}
