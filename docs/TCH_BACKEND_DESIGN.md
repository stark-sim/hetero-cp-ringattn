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
├── main.rs              # 集成 tch_attention_bridge_report 到 Report / CLI
├── tch_backend.rs       # tch-backend 核心模块（条件编译）
│   ├── select_device()          # HCP_TCH_DEVICE / HCP_TORCH_DEVICE 解析
│   └── run_attention_block_updates()  # 对标 C++ RunTorchAttentionBlockUpdates
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
| `RunTorchPayloadBlockSmoke` | （待实现） | ⏳ |
| `RunTorchPayloadOnlineSmoke` | （待实现） | ⏳ |
| `RunTorchPayloadChunkSmoke` | （待实现） | ⏳ |
| `RunTorchQueryChunkSmoke` | （待实现） | ⏳ |

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

| 平台 | 设备 | tch_smoke | tch_attention_bridge | 备注 |
|------|------|-----------|----------------------|------|
| macOS | CPU | ✅ code=1 | ✅ code=1 | |
| macOS | MPS | ✅ code=2 | ✅ code=2 | 非沙箱进程 |
| Linux | CPU | — | — | 待验证 |
| Linux | CUDA | ✅ code=3 | ✅ code=3 | 远端 GPU 192.168.8.172 |

## 启动方式

```bash
# 本地 CPU/MPS
bash scripts/run_tch_ringattn_smoke.sh
HCP_TCH_DEVICE=mps bash scripts/run_tch_ringattn_smoke.sh

# 主 binary 启用 tch-backend
cd rust && cargo run --features tch-backend --bin hcp-ringattn-rust

# 远端 CUDA
ssh gpu_host "bash -l -c 'cd hetero-cp-ringattn/rust && HCP_TCH_DEVICE=cuda:0 cargo run --features tch-backend --bin hcp-ringattn-rust'"
```

## 下一步

- [ ] 实现 payload block、online softmax、chunk、query chunk 等完整接口。
- [ ] 在 `Cargo.toml` 中将 `tch-backend` 设为 default feature（编译时间允许后）。
- [ ] 远端 Linux CPU smoke 验证。
