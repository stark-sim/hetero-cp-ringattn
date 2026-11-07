# Thor 原生 ARM 验证记录

> 记录日期：2026-11-06；整理与发布：2026-11-07。
> 范围：HCP 在 NVIDIA Thor 上的原生 aarch64 构建、libtorch 数值 smoke 和 Qwen2-0.5B 单域短推理。
> 本记录不包含多域 ring、异构联合计算、PD、长上下文或性能结论。

## 1. 结论

HCP 在不使用 Docker 的原生 ARM 环境中完成以下验证：

1. PyTorch CUDA 初始化、`sm_110` 识别和 CUDA matmul 成功。
2. Rust `tch_smoke` 的 matmul、softmax、attention-like 三项 CUDA/CPU 参考检查全部通过。
3. `hcp-ringattn-rust` 加载 Qwen2-0.5B 配置、权重和 tokenizer，使用 `Cuda(0)` 完成单域、短 prompt、最多 3 token 的 greedy 推理。
4. 两个二进制均为 ELF ARM aarch64；在指定的命令级库路径下没有缺失动态库。

这些结果证明 **Thor 原生平台适配和单设备最小闭环**，不能外推为多域正确性、长上下文容量或性能优势。

## 2. 代码、模型与硬件

| 项目 | 实测值 |
|---|---|
| HCP commit | `5f49bb0cd03ae05fdb1ad578f0a1aee1e41469cd` |
| 节点 | `sd-thor`，SSH `user@100.64.0.179` |
| OS / 架构 | Ubuntu 24.04.5 LTS；Linux 6.8.12-1021-tegra；aarch64 |
| CPU / 内存 | 14 logical CPUs；122 GiB 统一内存 |
| GPU | NVIDIA Thor；compute capability 11.0（`sm_110`） |
| 驱动 | 595.78 |
| CUDA toolkit | `cuda-toolkit-13-2=13.2.2-1`；nvcc 13.2.86 |
| Python / PyTorch | Python 3.12.3；`torch=2.11.0+cu130`；`torch.version.cuda=13.0` |
| PyTorch arch list | `sm_80`、`sm_90`、`sm_100`、`sm_110`、`sm_120` |
| Rust / Cargo | 1.99.1（原生 rustup stable） |
| CMake | 3.28.3 |
| NCCL | `libnccl2=2.32.3-1+cuda13.4`；仅满足动态链接，未测试 collective |
| 模型 | Qwen2-0.5B；24 层、14 attention heads、2 KV heads、BF16 config |
| 模型 SHA-256 | `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad` |

Thor 上的路径：

```text
/home/user/hcp-thor/hetero-cp-ringattn-runtime
/home/user/hcp-thor/models/Qwen2-0.5B
/home/user/hcp-thor/venv
/home/user/hcp-thor/target
```

## 3. 数据来源与完整性

- 运行源码由本地 `git archive HEAD` 生成，仅包含 `rust/`、`python/`、`scripts/`、`src/`、`include/`、`CMakeLists.txt` 和 `README.md`，大小 **644,778 bytes**。不包含 `.git`、未提交修改、模型、reports 或构建缓存。
- 模型由 Thor 直接从 ModelScope 下载；权重 SHA-256 与本地已有权重一致。未通过 Tailscale 搬运模型。
- Cargo 依赖由 Thor 通过 RsProxy sparse 获取；实际构建和运行使用 `--locked --offline`。
- PyTorch 使用官方 `download.pytorch.org/whl/cu130` 原生 aarch64 wheel。`ldd` 发现的实际缺口由 cuDNN、cuSparseLt、NVSHMEM 和系统 NCCL 补齐；没有全局 `LD_PRELOAD`。
- Docker 路线被明确否决。最终 Docker service 为 `inactive`、`disabled`；本记录中的所有结果均来自原生进程。

## 4. 运行方式

以下为实际使用的核心命令；`NVIDIA_LIBS` 由 venv 下的 `nvidia/*/lib` 目录生成，并追加 `/usr/local/cuda/lib64`。库路径仅作用于当前命令。

```bash
cd /home/user/hcp-thor/hetero-cp-ringattn-runtime/rust
export PATH=/home/user/.cargo/bin:/home/user/hcp-thor/venv/bin:$PATH
export HCP_ENABLE_TORCH=1 LIBTORCH_USE_PYTORCH=1
export CARGO_NET_OFFLINE=true CARGO_TARGET_DIR=/home/user/hcp-thor/target
NVIDIA_LIBS=$(/home/user/hcp-thor/venv/bin/python -c 'import glob; print(":".join(glob.glob("/home/user/hcp-thor/venv/lib/python3.12/site-packages/nvidia/*/lib")))')
export LD_LIBRARY_PATH="$NVIDIA_LIBS:/usr/local/cuda/lib64"
```

### 4.1 `tch_smoke`

```bash
HCP_TCH_DEVICE=cuda:0 RUN_ID=thor-native-tch-20261106 \
  /home/user/.cargo/bin/rustup run stable \
  cargo run --locked --offline -q --bin tch_smoke
```

### 4.2 单域 Qwen 推理

```bash
env -u HCP_TORCH_DEVICE HCP_TCH_DEVICE=cuda:0 \
  /home/user/.cargo/bin/rustup run stable \
  cargo run --locked --offline -q --bin hcp-ringattn-rust -- \
  --infer-model-dir /home/user/hcp-thor/models/Qwen2-0.5B \
  --infer-prompt "Hello world" \
  --infer-max-tokens 3 \
  --infer-temperature 0 \
  --infer-num-domains 1
```

## 5. 结果

### 5.1 PyTorch CUDA smoke

- `cuda_available=True`，设备为 `NVIDIA Thor`，capability 为 `(11, 0)`。
- BF16 支持查询为 `True`。
- 一个 float32 matmul 与 CPU 参考的 `max_abs_err=0.0`。

### 5.2 HCP `tch_smoke`

顶层结果：`status=pass`、`requested_device=cuda:0`、`status_code=3`、`ops=3/3`。

| op | dtype | max_abs_err | mean_abs_err | 结果 |
|---|---|---:|---:|---|
| matmul | float32 | 9.5367431640625e-7 | 1.2293457984924316e-7 | pass |
| softmax | float32 | 4.470348358154297e-8 | 4.554749466478825e-9 | pass |
| attention-like | float32 | 1.341104507446289e-7 | 2.5116605684161186e-8 | pass |

Thor 原始报告位置：

```text
/home/user/hcp-thor/hetero-cp-ringattn-runtime/reports/thor-native-tch-20261106/tch_smoke.json
```

报告 SHA-256：

```text
84ce17405520518efcbabb806f41059b522f68510fff14d74e308442fb0eed29
```

按仓库约定，原始生成的 report JSON 不纳入 Git；本文记录其关键数值、路径和哈希。

### 5.3 单域模型输出

实际日志关键行：

```text
[infer] prompt length: 11 chars
[infer] device: Cuda(0)
[infer] building model (24 layers, 14 heads)
[infer] generating (max_tokens=3, temperature=0)...
! I'm
```

未出现 `Inference failed:`。该次调用的总 wall time 为 36.00 秒，包含剩余构建和启动开销，**不是推理性能测量**。

### 5.4 原生二进制

| 二进制 | 格式 | 大小 |
|---|---|---:|
| `hcp-ringattn-rust` | ELF 64-bit LSB PIE，ARM aarch64 | 165,951,280 bytes |
| `tch_smoke` | ELF 64-bit LSB PIE，ARM aarch64 | 29,225,704 bytes |

在命令级 NVIDIA/CUDA 库路径下，`ldd` 未报告 `not found`。二进制不纳入 Git。

## 6. 验证边界与后续实验

本次没有验证：

- 独立 Hugging Face golden 或端到端 logits 对照；
- QUIC 控制面、多域 KV ring、非均分调度或异构联合计算；
- NCCL collective；
- HCP→vLLM PD 注入、KV 搬运、尾部补算或 decode 正确性；
- 长上下文、TTFT、吞吐、峰值内存或可扩展性。

Thor 已实测具有 122 GiB 统一内存，因此后续应优先作为**强单节点基线**。有信息价值的下一步是本地/loopback 的真实 CP 产物→PD 注入 kill test，以及同模型、同输入下的强单机对照；仅把 Thor 加进另一个短文本 ring PASS 不能回答当前产品问题。
