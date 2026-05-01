# tch-backend 设计文档

## 定位

`tch-backend` 是本仓的 **optional PyTorch tensor backend**，与现有 C++ ATen/libtorch bridge 并行存在，逐步替代部分 device compute 路径。

```text
Rust HCP model / protocol
├── pure-rust backend        # 默认 correctness 路径，离线可跑
├── cxx-aten backend         # C++ ATen/libtorch bridge（现有）
└── tch-backend              # tch-rs Rust binding -> libtorch（新增）
```

## 架构原则

1. **并行验证**：tch-backend 不删除 C++ bridge，而是新增一条等价路径，先 smoke 验证对齐后再逐步迁移。
2. **最小侵入**：主 binary 的 `Report` 新增 `tch_attention_bridge` 等字段；未启用 feature 时显示 `status=disabled`，不影响现有流程。
3. **设备码统一**：tch-backend 复用与 C++ bridge 相同的成功码约定：CPU=1, MPS=2, CUDA=3。
4. **Rust/domain-side 保留 ownership**：HCP 控制逻辑（ring order、block traversal、online softmax state、report schema）保留在 Rust；tch 只承载 heavy tensor ops（`matmul`、`softmax`、`to(device)`）。

## 模块结构

```text
rust/src/
├── main.rs              # 集成 tch bridge reports 到 Report / CLI
├── tch_backend.rs       # tch-backend smoke 桥接函数（条件编译）
│   ├── select_device()          # HCP_TCH_DEVICE / HCP_TORCH_DEVICE 解析
│   ├── run_attention_block_updates()     # 对标 C++ RunTorchAttentionBlockUpdates
│   ├── run_payload_block_smoke()         # 对标 C++ RunTorchPayloadBlockSmoke
│   ├── run_payload_online_smoke()        # 对标 C++ RunTorchPayloadOnlineSmoke
│   ├── run_payload_chunk_smoke()         # 对标 C++ RunTorchPayloadChunkSmoke
│   └── run_query_chunk_smoke()           # 对标 C++ RunTorchQueryChunkSmoke
├── compute_runtime.rs   # ComputeRuntime trait + TchComputeRuntime 实时计算
│   ├── select_tch_device_from_env()      # 环境变量设备解析（共享）
│   └── TchComputeRuntime                 # protocol 实时 compute 路径
└── bin/
    └── tch_smoke.rs     # 独立最小 tensor smoke
```

## 数据流

1. Rust 侧从环境变量解析目标设备（`cpu`/`mps`/`cuda:N`）。
2. 在 **CPU** 上构造 synthetic Q/K/V（`arange` + `reshape` + `scale` + `shift`）。
3. CPU 上计算 reference：`softmax(Q @ K^T / sqrt(d)) @ V`。
4. 将 Q/K/V `to(device)`，在目标设备上执行同样的 attention 公式。
5. 将设备输出 `to(CPU)`，与 reference 比较 `max_abs_err`。
6. 返回成功码 + message；`main.rs` 包装成 `TorchBridgeReport` 写入 JSON。

## 与 C++ bridge 的对应关系

| C++ bridge | tch-backend 等价函数 | 状态 |
|------------|---------------------|------|
| `hcp_ringattn_torch_smoke` | `tch_smoke.rs` 基础 op smoke | ✅ |
| `RunTorchAttentionBlockUpdates` | `run_attention_block_updates()` | ✅ |
| `RunTorchPayloadBlockSmoke` | `run_payload_block_smoke()` | ✅ |
| `RunTorchPayloadOnlineSmoke` | `run_payload_online_smoke()` | ✅ |
| `RunTorchPayloadChunkSmoke` | `run_payload_chunk_smoke()` | ✅ |
| `RunTorchQueryChunkSmoke` | `run_query_chunk_smoke()` | ✅ |
| `RunTorchQueryOutputSmoke` | `run_query_chunk_smoke()` (返回 checksum/max_err/output_values) | ✅ |

## 构建与链接

### macOS
- 只需 `LIBTORCH=/Users/stark_sim/libtorch`。
- 运行时需 `DYLD_LIBRARY_PATH=${LIBTORCH}/lib`。
- 不要设置 `LIBTORCH_INCLUDE`/`LIBTORCH_LIB`，否则 `torch-sys` 会重复追加路径。

### Linux (CUDA)
- `LIBTORCH=/home/stark/libtorch`。
- `torch-sys` build script 会链接 `libtorch_cuda`，但 Linux linker 的 `--as-needed` 会丢弃未被直接引用的 registration library。
- 我们在 `build.rs` 中使用 `cargo:rustc-link-arg-bins` 在 binary 链接阶段前置 `--push-state,--no-as-needed,-ltorch_cuda,-lc10_cuda,--pop-state`，确保 CUDA backend 被注册加载。

## 验证矩阵

| 平台 | 设备 | tch_smoke | tch_attention | tch payload/query bridges | 实时 compute (TchComputeRuntime) | 备注 |
|------|------|-----------|---------------|---------------------------|----------------------------------|------|
| macOS | CPU | ✅ code=1 | ✅ code=1 | ✅ code=1 30/30 | ✅ checksum=1093.59 | |
| macOS | MPS | ✅ code=2 | ✅ code=2 | ✅ code=2 30/30 | ✅ checksum=1093.59 | 非沙箱进程 |
| Linux | CPU | — | — | — | — | 待验证 |
| Linux | CUDA | ✅ code=3 | ✅ code=3 | ✅ code=3 30/30 | ✅ checksum=238.88 | 远端 GPU 100.64.0.93 |
| 3-node remote CP | MPS+CUDA | — | — | ✅ code=2/3 12/12 | ✅ checksum=71.35/238.88/406.41 | MPS node0/2 + CUDA node1 |

## Runtime 设备配置

`tch-backend` 支持运行时切换目标设备，不依赖编译时 feature flag。

### 环境变量

| 变量 | 优先级 | 说明 |
|------|--------|------|
| `HCP_TCH_DEVICE` | 高 | 首选设备名：`cpu`、`mps`、`cuda`、`cuda:N` |
| `HCP_TORCH_DEVICE` | 中 | fallback，与 C++ bridge 兼容 |
| 默认 | 低 | 未设置时 fallback 到 `cpu` |

### ComputeRuntime trait

```rust
pub trait ComputeRuntime {
    fn compute_kv_block(&mut self, ...);
    fn finalize_output(&mut self, ...) -> f64;
}
```

- `TchComputeRuntime::from_env()` — 从环境变量解析设备并创建
- `TchComputeRuntime::new(device)` — 显式传入设备
- `NoOpComputeRuntime` — 无 `tch-backend` feature 时的编译兼容 stub

`TchComputeRuntime` 在 `protocol.rs` 的 `run_cp_ring_node`、`run_remote_cp_node`、`run_cp_ring_node_smoke` 中被使用，执行实时 online softmax 更新和 output finalize。

## 启动方式

```bash
# 本地 CPU/MPS
cd rust && cargo run --bin hcp-ringattn-rust
HCP_TCH_DEVICE=mps cargo run --bin hcp-ringattn-rust

# 远端 CUDA
ssh gpu_host "bash -l -c 'cd hetero-cp-ringattn/rust && HCP_TCH_DEVICE=cuda:0 cargo run --bin hcp-ringattn-rust'"

# 3-node remote CP smoke（统一 launcher）
export GPU_HOST=100.64.0.93 GPU_USER=user MAC_NODE_ADDR=100.64.0.95
bash scripts/run_rust_remote_cp_3node_smoke.sh
```

## 下一步

- [ ] 远端 Linux CPU smoke 验证。
- [ ] 考虑为 `ComputeRuntime` 增加 config 文件支持（JSON/YAML），减少纯环境变量配置。
- [ ] 评估 `TchComputeRuntime` 是否可进一步与 `cuda` feature 解耦，实现编译时可选 CUDA kernel 路径。
