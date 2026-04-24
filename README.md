# HCP Ring Attention Repo

这个仓库是从 `honolulu` 主仓中抽离出来的独立研究仓，目标只做一件事：

- **HCP（Heterogeneous Context Parallelism）**
- **场景**：超长 context
- **层级**：`intra-layer / low-boundary`
- **通信**：跨异构 domain 只允许 `P2P`

它不再承载 `HLPP` 主线职责，也不再依赖 `phase2_native/` 或 `phase3_layerwise/` 的源码边界。

## 这个仓库要解决什么问题

长上下文需求正在持续增长。过去很多系统把 `200k` 当成高门槛，后来演进到 `1M`，后面还会继续走向更长的 context。问题不只是“模型想不想支持”，而是：

- 单卡显存很快成为硬上限
- 同构高端卡资源昂贵且稀缺
- 现实中团队往往手里是混合算力，而不是整齐划一的同构集群
- 如果这些设备无法合作，context 一长，系统就只能截断、降级，甚至直接放弃任务

HCP 的产品出发点是：

- 不要求单张卡吃下整个长上下文
- 允许不同能力的设备以**不均分**方式合作
- 随着 context 从 `200k -> 1M -> 10M` 继续增长，系统理论上可以通过增加 domain / 增加卡来继续支撑，而不是只能投降

## 为什么它不是 HLPP

`HLPP` 和 `HCP` 是两条不同路线，不是一个粗一个细。

- `HLPP` 处理的是 `layer-wise / high-boundary` 问题
  - 不同 domain 负责不同 `layer range`
  - 跨域传的是 `hidden_states`
- `HCP` 处理的是 `intra-layer / low-boundary` 问题
  - 多个 domain 共同完成同一个 attention layer
  - 跨域持续传的是 `K/V blocks + online-softmax state`

这就是为什么 HCP 需要学习 `Ring Attention`。一旦目标变成“让多个异构设备共同承担同一个 attention 层”，就不能再沿用高边界 layer-wise 叙事，也不能假设 collective 在异构环境里仍然有效。

## 为什么是 Ring Attention / P2P

本仓的核心判断是：

- 跨异构 domain 不适合把 all-gather / reduce-scatter / all-to-all / all-reduce 当成主通信假设
- 异构参与者在算力、显存、延迟、带宽上天然不对称
- 对同一个 attention 层做跨域协作时，更自然的做法是：
  - 每个 domain 维护本地 `Q chunk`
  - `K/V block` 沿 ring 持续流动
  - online softmax state 逐 block 聚合

也就是说，本仓研究的不是“如何让所有卡同步地像一个同构集群那样工作”，而是“如何让不对称设备也能参与同一个长上下文 attention 任务”。

## 当前状态

当前仓库已经具备独立骨架，但还不是完整闭环：

- 已完成：
  - 独立 `Status`
  - 独立 `TensorDType` / `BoundaryTensor`
  - 独立 `RingAttnProtocol` / `RingAttnRuntime`
  - 最小 `NoOp` runtime
  - C++ coordinator smoke
  - Python controller / worker / kernel stub
- 尚未完成：
  - online softmax correctness report
  - 真实 message serialization
  - 最小 P2P send/recv transport
  - heterogeneous remote runtime stub
  - 2-domain remote end-to-end smoke

## 文档地图

- `docs/PRODUCT_THESIS.md`
  - 为什么 HCP 是一个超长 context 产品问题
- `docs/HLPP_VS_HCP.md`
  - HCP 与 HLPP 的边界、layer-wise vs intra-layer、为什么学习 Ring Attention
- `docs/HISTORY_AND_LESSONS.md`
  - 从 `phase1 / phase2 / phase3` 继承哪些经验，拒绝哪些历史包袱
- `docs/VALIDATION_PLAN.md`
  - HCP 应该如何被验证，而不只是“代码能编译”
- `docs/ROADMAP.md`
  - 从 skeleton 到 remote heterogeneous smoke 的阶段路线
- `docs/DESIGN.md`
  - 当前协议和 online softmax 设计骨架

## 当前范围

- `include/hcp_ringattn/core/`
  - 独立公共类型
  - Ring Attention 协议与 runtime 抽象
- `src/`
  - 最小 `NoOp` runtime
  - coordinator smoke
- `python/`
  - controller / worker / kernel stub
- `scripts/`
  - 本仓自举 smoke 入口

## 非目标

- 不承载 `HLPP` 的 layer-wise 编排
- 不承载域内 `TP / collective / kernel` 优化
- 不把 `honolulu` 主仓继续作为源码级前置依赖
- 不在当前阶段追求性能最优

## 本地构建

```bash
cmake -S . -B build
cmake --build build --target ringattn_coordinator_smoke -j4
./build/ringattn_coordinator_smoke
```

## 本地 smoke

```bash
bash scripts/run_local_ringattn_smoke.sh
```

Python 原型依赖 `numpy`。若当前解释器没有 `numpy`，脚本会跳过 Python smoke，只跑 C++ smoke。

如果当前环境不允许本地端口绑定，可以显式跳过 Python worker/controller：

```bash
SKIP_PYTHON_SMOKE=1 bash scripts/run_local_ringattn_smoke.sh
```

## 目录

```text
hcp_ringattn_repo/
├── CMakeLists.txt
├── README.md
├── include/hcp_ringattn/core/
├── src/
├── python/
├── config/
├── scripts/
├── docs/
└── reports/
```
