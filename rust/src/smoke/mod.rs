//! 【Smoke Test 模块】
//!
//! HCP 的 correctness 验证基础设施：
//! - `bridges`: C++ / Python / Rust 之间的 FFI 桥接测试
//! - `reference_algo`: 参考 attention 算法实现（用于对比验证）
//! - `correctness`: 端到端 correctness test runner，生成 JSON 报告
//!
//! 这些测试验证 HCP 的 attention 计算在不同设备（CPU/MPS/CUDA）
//! 和不同场景（单 block / online / chunk / query chunk）下的数值正确性。

pub mod bridges;
pub mod reference_algo;
pub mod correctness;
