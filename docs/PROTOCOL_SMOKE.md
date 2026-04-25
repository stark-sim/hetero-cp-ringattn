# Ring Attention Protocol Smoke

## 目标

本阶段把 Ring Attention / Context Parallel 中“ring 内流动的东西”从内存内抽象推进成可序列化、可发送、可接收、可检查的协议单元。

它不是性能实现。当前目标是固定低边界协议语义，并用本地 P2P 队列表达 point-to-point send / recv 语义：

- 每个 source domain 按自己的 `block_size` 切出 K/V block。
- K/V block 沿 ring 逐 hop 转发。
- 每个 receiver 可以从消息中恢复 source / sender / receiver / block metadata。
- softmax state 和 terminate 控制消息也走同一套编码路径。

## Context Parallel 数据流

当前 smoke 使用 3 个 domain：

```text
domain-0 -> domain-1 -> domain-2 -> domain-0
```

对任意 source domain 的一个 K/V block：

1. source domain 本地消费该 block。
2. source domain 将 block 发送给 ring next domain。
3. receiver 解码并验证消息。
4. receiver 继续将同一个 source block 转发给下一个 domain。
5. 直到其他 domain 都收到该 source block。

因此 `N` 个 domain 下，每个 source block 会产生 `N - 1` 条跨域 K/V block message。

## Message Schema

Rust 侧当前定义的最小 schema 包括：

- `schema_version`
- `sequence_id`
- `layer_index`
- `ring_step`
- `source_domain`
- `sender_domain`
- `receiver_domain`
- `message_kind`
- `payload_kind`
- `block`
- `tensor`
- `payload`

其中：

- `source_domain` 表示 K/V block 的原始归属 domain。
- `sender_domain` 表示当前 hop 的发送方。
- `receiver_domain` 表示当前 hop 的接收方。
- `block` 记录 `global_offset`、`block_len`、`source_seq_offset`。
- `tensor` 记录 dtype、head shape、payload byte size 和 checksum。

当前 wire format 使用 `serde_json`，因为本阶段优先可读性和可诊断性。后续如果需要性能，可以在 schema 稳定后替换成二进制格式。

## P2P Transport 语义

这里的 P2P 不是特指 IP/TCP。它表示跨 domain 只做点对点消息传递，不把 all-gather / reduce-scatter / all-to-all / all-reduce 作为主通信假设。

当前 smoke 使用 `local_p2p_queue`：

- 每个 domain 有独立 inbox。
- send 必须指定 sender / receiver。
- recv 会校验 expected sender。
- frame 内容仍是完整序列化后的 `RingAttnMessage`。

这一步先固定协议语义，不绑定到底层 IP/TCP。后续可以把同一个 `RingAttnMessage` 映射到 TCP、UCX/RDMA、NCCL send/recv、共享内存或自定义 GPU-direct transport。

远端 NVIDIA GPU 当前在 `192.168.8.172`。代码变更只通过 git 同步：本地提交并 push，远端只执行 `git pull` 和 smoke 命令，不在远端直接编辑源码。

## Smoke Report

运行：

```bash
bash scripts/run_rust_ringattn_smoke.sh
```

本机硬件验证仍使用 MPS：

```bash
HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=mps bash scripts/run_rust_ringattn_smoke.sh
```

通过时 CLI 会显示：

```text
protocol_status=pass protocol_messages=22
```

JSON report 中新增 `protocol_smoke`：

```json
{
  "status": "pass",
  "transport": "local_p2p_queue",
  "schema_version": 1,
  "domains": 3,
  "source_blocks": 10,
  "summary": {
    "kv_block_messages": 20,
    "softmax_state_messages": 1,
    "terminate_messages": 1,
    "messages_sent": 22,
    "messages_received": 22
  }
}
```

`route_preview` 会记录前几条 K/V block 的 source / sender / receiver / block range，用于检查 ring order 是否符合 Context Parallel 预期。

## 后续

下一步应为 `local_p2p_queue` 抽出明确 transport trait，再增加一个可选 remote transport 实现。remote 版本可以先用 TCP 做工程 smoke，但不应把 Ring Attention protocol 本身定义成 TCP。
