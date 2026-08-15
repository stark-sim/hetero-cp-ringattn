# NIXL Block Transport — 第三种传输（block-direct 数据面）

> 状态：设计（decision-nixl-as-transport-20260816 裁定走形态 B）
> 目标：把 NIXL 接上 transport 抽象，作为 QUIC/TCP 之外的第三种通信，且 block 级数据面同时是未来对接 vLLM paged KV 的同一抽象。

---

## 1. 背景与动机

K10 已建立 KvTransport 的 wire_bytes_sent/recv 计量接口。现有传输实现是字节流语义：把 tensor 序列化成 frame（JSON meta + 原始字节）再 send/recv。

NIXL 的语义不是字节流，是 block 级 device-memory 零拷贝传输：

1. registerMem(descs) — 注册本地 device 内存块（{addr, len, devId, metaInfo}）。
2. getLocalMD / loadRemoteMD — 经 side channel 交换 agent 元数据 + block 描述符。
3. prepXferDlist → makeXferReq/createXferReq → postXferReq（异步，返回 IN_PROG）。
4. getXferStatus / getNotifs — 轮询完成。
5. getXferTelemetry — 传输字节数（直接填 K10 的 wire_bytes_sent/recv 口径）。

这就是为什么形态 B 需要新的 block 级 trait 面，而非把 NIXL 硬塞进字节流 KvTransport。

## 2. 关键架构判断

### 2.1 数据面分两条路径（不是替换，是并行）

| 路径 | 载荷 | 语义 | 传输 |
|---|---|---|---|
| prefill KV ring | KvBlock 的 K/V tensor（大，几十 MB 级） | block-direct（零拷贝） | 新 KvBlockTransport（NIXL / 序列化 fallback） |
| decode SD packet | residual/normalized/q/attention_output/lse（小，KB 级） | 字节流 | 现有 KvTransport（QUIC/TCP） |

理由：NIXL 的 register+transfer 开销只在搬大 KV block 时划算；decode 的自驱动 packet 是小激活量、无 K/V，不值得注册内存 + 异步 transfer 的生命周期。decode 路径不动。

### 2.2 side channel 复用 HCP 控制面（不新增 side channel）

vLLM PD 用独立 TCP side channel（VLLM_NIXL_SIDE_CHANNEL_HOST/PORT）交换 NIXL agent 元数据。HCP 已有 coordinator↔worker 的 QUIC 控制面（WorkerCommand/WorkerResponse，bincode），coordinator 本来就知道全拓扑。NIXL 元数据 + block 描述符经既有控制面交换，不新增端口/依赖——这更符合网络自由卖点（coordinator 是唯一拓扑知识源）。

### 2.3 block 单元：先对齐现有 KvBlock，paged 化后置

现有 KvBlock { layer_idx, global_seq_start/end, k, v, position_ids } 的 k/v 是 [1, num_kv_heads, seq, head_dim] 连续 tensor。第一步 block-direct 直接注册并传输现有 KvBlock 的 k/v device 内存，元数据（layer_idx/seq range/position_ids）走控制面小 payload。docs/BLOCK_RING_FUSION.md 论证的 ring 交换粒度 = vLLM block_size(16) 是后续 paged-KV 化节点（把 seq 维切成 16-token 物理 block + block_table），本节点只建 block 传输抽象 + NIXL FFI，不重写 ring_attention 的连续 seq 假设。

## 3. Trait 面（KvBlockTransport）

与 KvTransport 并列，定义在 rust/src/model/transport/：

    pub trait KvBlockTransport: Send {
        fn register_block(&mut self, tensor: &Tensor) -> Result<BlockHandle, String>;
        fn deregister_block(&mut self, handle: &BlockHandle) -> Result<(), String>;
        fn local_metadata(&self) -> Result<Vec<u8>, String>;
        fn load_remote_metadata(&mut self, blob: &[u8]) -> Result<String, String>;
        fn submit_transfer(&mut self, local: &BlockHandle, remote: &RemoteBlockDesc) -> Result<(), String>;
        fn poll_transfers(&mut self) -> Result<Vec<TransferCompletion>, String>;
        fn wire_bytes_sent(&self) -> u64;
        fn wire_bytes_recv(&self) -> u64;
    }

BlockHandle 是注册块的稳定引用；RemoteBlockDesc 是经 side channel 传来的 peer 块描述符（含 peer agent 名 + addr/len/devId + metaInfo）。这套 addr+len+devId+metadata 描述符模型与 NIXL 和 vLLM 物理 block 完全同构。

## 4. 实现与 feature gating

- rust/src/model/transport/block_transport.rs：trait + BlockHandle/RemoteBlockDesc/TransferCompletion 类型。
- rust/src/model/transport/serialized_block.rs：fallback 实现（复用现有 frame 序列化，Mac 可测，作为 NIXL 必须对齐的参考基线）。
- rust/src/distributed/transport/nixl.rs：NixlBlockTransport，FFI 绑定 libnixl_capi.so，feature nixl-backend 门控（默认 off）。
- Cargo：nixl-backend = [] feature，NIXL FFI 只在 white/pearl（CUDA/ROCm）编译。

## 5. 分阶段验证

1. S1（Mac 可绿）：KvBlockTransport trait + SerializedKvBlockTransport fallback + round-trip 测试。
2. S2（white/pearl）：NixlBlockTransport FFI，cargo build --features nixl-backend + 双机 register→transfer→poll 探针。
3. S3（white/pearl）：prefill KV ring 走 block 路径（coordinator 控制面做 side channel），N=2 CUDA↔ROCm 数值对照 single-node reference。
4. S4（后续节点）：paged-KV 化（block_size=16 + block_table），对接 vLLM paged KV。

## 6. 边界（本节点不做）

- 不重写 ring_attention 的连续 seq 假设；不引入 paged block_table（S4）。
- 不改 decode SD packet 路径（仍走 KvTransport 字节流）。
- 不新增独立 NIXL side channel（复用 HCP 控制面）。
- Mac 只验证 trait + fallback；NIXL FFI 的真实编译/smoke 在 white/pearl。
