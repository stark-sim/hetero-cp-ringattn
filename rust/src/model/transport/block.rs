#![allow(dead_code)]

#[cfg(feature = "tch-backend")]
use tch::Tensor;

/// A Key/Value block exchanged between distributed workers during ring attention.
#[cfg(feature = "tch-backend")]
#[derive(Debug)]
pub struct KvBlock {
    pub layer_idx: usize,
    pub global_seq_start: usize,
    pub global_seq_end: usize,
    pub k: Tensor,
    pub v: Tensor,
}

impl Clone for KvBlock {
    fn clone(&self) -> Self {
        Self {
            layer_idx: self.layer_idx,
            global_seq_start: self.global_seq_start,
            global_seq_end: self.global_seq_end,
            k: self.k.shallow_clone(),
            v: self.v.shallow_clone(),
        }
    }
}
