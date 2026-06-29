# Progress Timeline

按时间倒序排列的重要进展、实验和学到的教训。

### CPU mock 只能验证语法和逻辑依赖，不能指导 LLM 服务架构设计

type: `lesson` · status: `held` · confidence: 0.95 · importance: 0.9

在 Striped Attention 原型验证中发现：CPU 上 correctness diff 和 perf 数字对 LLM 服务架构设计的实际作用几乎没有意义。\n\n原因：\n1. CPU 与加速卡（CUDA/HIP/MPS）的算力结构、memory bandwidth、kernel launch 开销完全不同。\n2. CPU mock 无法反映真实 heterogeneous 场景下各 domain 的计算速度差异、显存压力、P2P / 网络传输瓶颈。\n3. Striped 对负载均衡的影响取决于"慢 domain 到底有多慢"以及"peer compute 转移是否能被快 domain 吸收"，这些信息 CPU 无法提供。\n\n结论：代码逻辑层面的正确性可以在 CPU 快速验证；任何关于调度策略、overlap、分片比例、端到端吞吐/延迟的设计决策，必须在真实加速卡硬件上复跑后才能得出结论。

_updated: 2026-06-29 12:35:36_
### [2026-06-29] Striped correctness原型在CPU mock上验证通过

type: `evidence` · status: `held` · confidence: 0.9 · importance: 0.85 · source: `cargo test / rust/src/model/attention/ring.rs`

测试：cargo test --features tch-backend test_ring_attention_uneven_perf -- --nocapture
配置：seq_len=4096, 2 domain, chunk=[3072,1024] (3:1), CPU Float32, mock transport。

Correctness：
- Vanilla diff = 2.8e-8
- Striped diff = 2.6e-8
均 < 1e-4，数值正确。

Perf（单次 layer，CPU mock）：
Vanilla：
- domain 0 total=118.5ms (local=117.0ms, peer=0.02ms)
- domain 1 total=46.3ms (local=15.8ms, peer=30.0ms)
Striped：
- domain 0 total=184.6ms (local=129.6ms, peer=53.3ms)
- domain 1 total=50.8ms (local=15.9ms, peer=34.6ms)

关键发现：在 homogenous CPU 上，Striped 把部分 peer compute 从 domain 1 转移到 domain 0，使原本就是瓶颈的 domain 0 更慢；domain 0/1 总耗时比从约 2.6x 扩大到约 3.6x。

_updated: 2026-06-29 10:46:05_
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
