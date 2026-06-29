# Progress Timeline

按时间倒序排列的重要进展、实验和学到的教训。

### [2026-06-19] 1M context 本地异构分布式推理成功

type: `session` · status: `closed` · confidence: 1.0 · importance: 1.0 · source: `memory-bank/progress.md`

white RTX 4090 CUDA + pearl RX 9060 XT HIP，2.5G 有线直连。Qwen2-0.5B-1M，capacity-aware 3:1 分片（white 750K / pearl 250K）。Prefill 24/24 全通，decode 5 tokens 全通，exit=0。总耗时 ~2h 11m，white 显存峰值 23,999 MB。攻克：KV channel buffer 512、QUIC timeout 14400s、max_position_embeddings=1048576 patch、pearl 碎片化 OOM 通过 3:1 分片缓解。

_updated: 2026-06-29 05:34:19_
### [2026-06-17] 昇腾 910B NPU 控制面 E2E 打通

type: `session` · status: `closed` · confidence: 1.0 · importance: 0.75 · source: `memory-bank/progress.md`

单机 1× Ascend 910B4 (32 GB HBM) 上完成 Python vLLM worker ↔ Rust coordinator 控制面 E2E。Rust coordinator 脱离 libtorch feature 可编译运行，纯 Rust 采样替代 tch::Tensor。Coordinator 输出 generated: ! I'm。

_updated: 2026-06-29 05:34:19_
### 证据：1:1/2:1/3:2 split 均导致 pearl OOM

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `memory-bank/progress.md`

在 1M context 尝试中，均分 500K 及 2:1、3:2 split 均在 layer 23/24 因 pearl 16GB 显存分配失败而 OOM。只有 3:1 split 成功。

_updated: 2026-06-29 05:34:19_
### 证据：同构分布式 BF16 也有 ~0.3-0.4 logits 差异

type: `evidence` · status: `held` · confidence: 0.85 · importance: 0.8 · source: `memory-bank/systemPatterns.md`

White CUDA loopback 双 domain 3B max_diff=0.406，0.5B max_diff=0.344，argmax=10/10。跨平台单节点 0.438，异构分布式 0.484。证明跨平台 BLAS 仅贡献 ~0.1 额外差异，不是 logits 差异主导因素。

_updated: 2026-06-29 05:34:19_
