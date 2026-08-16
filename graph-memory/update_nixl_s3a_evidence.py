import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@7e475d6"

EVIDENCE = "evidence-nixl-s3a-cross-machine-transfer-20260816"
S3_TASK = "task-nixl-s3-cross-machine-transfer-20260816"
S3_DECISION = "decision-nixl-s3-cross-machine-20260816"

def upsert_node(conn, node_id, node_type, layer, title, content, status, importance, confidence, source):
    conn.execute(
        "INSERT INTO nodes (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now')) ON CONFLICT(id) DO UPDATE SET type=excluded.type,layer=excluded.layer,project=excluded.project,title=excluded.title,content=excluded.content,importance=excluded.importance,confidence=excluded.confidence,status=excluded.status,source=excluded.source,updated_at=datetime('now')",
        (node_id, node_type, layer, PROJECT, title, content, importance, confidence, status, source),
    )

def upsert_edge(conn, source, target, edge_type, note):
    conn.execute(
        "INSERT INTO edges(source,target,type,weight,note) VALUES (?,?,?,1.0,?) ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note",
        (source, target, edge_type, note),
    )

CONTENT = """NIXL block-direct S3a 跨机异构 transfer 全生命周期真实验证通过（commit 7e475d6）。

【验证结果】white(RTX4090,CUDA)↔pearl(RX9060XT,ROCm) 双向 register→metadata→transfer→poll→dump：
- 双向字节级一致：white.dest == pearl.src（ROCm→CUDA）与 pearl.dest == white.src（CUDA→ROCm）均 max|diff|=0.0（96 字节 f32 block）。
- telemetry.total_bytes=96 == block len，wire_bytes_sent=96 wire_bytes_recv=96，对齐 K10 wire-byte 口径（block-direct 无序列化 meta 开销，真实字节 = block len）。
- 探针 scripts/nixl_xfer_probe.sh 本机编排：ssh -f 后台化 + scp tmp+mv 原子交换 metadata/desc/done + 字节级对比。

【重大发现——跨厂商 GPU-direct 受限】跨异构（CUDA↔ROCm）GPU-direct VRAM put 无 UCX 远程协议：white 的 CUDA 构建 UCX 报 "cannot find remote protocol for put(multi) from cuda/GPU0 to rocm"、postXferReq 后 remote agent disconnected。跨厂商没有 GPU-direct 互操作，NIXL 跨异构必须 host DRAM staging + tcp 传输（数据经 host 中转，应用层做 GPU↔host 拷贝）。

【修复清单（均 commit，S3a 驱动发现）】
1. enable_listen_thread=true + AgentConfig（new_configured）：跨机 UCX 需接受 incoming 连接，默认 enable_listen_thread=false 使 agent client-only，load_remote_md 报 "UCX endpoint create failed: Connection refused"。
2. listen_port=0：让 NIXL/OS 分配临时端口，避免固定 8888 的残留占用冲突（"Socket Bind failed: Address already in use"）。
3. UCX_TLS=tcp：强制 TCP、排除 cuda_ipc GPU-direct（跨厂商无协议）；注意 UCX_TLS 若只写 tcp 会连带禁用 CUDA/ROCm 支持导致 VRAM 注册失败（"UCX CUDA support not found"），所以只在 host-staging 路径用。
4. capture_telemetry=true：get_telemetry() 不依赖 NIXL_TELEMETRY_ENABLE env。
5. poll_transfers 的 get_telemetry 在 NoTelemetry 时 fallback 到 desc.len（S2 遗留 bug，S3a 跨机才暴露）。
6. BlockDesc 加 mem_type（BlockMemType::Dram/Vram）：register_block 按 tensor.device() 自动选，submit_transfer 用 desc.mem_type 构造 XferDescList。

【架构含义（对 NIXL 作为 HCP ring 传输的价值评估）】NIXL 零拷贝（GPU-direct VRAM）只服务同构（CUDA↔CUDA / ROCm↔ROCm）；跨异构（HCP 核心卖点=异构支持）必须 host staging，NIXL 的零拷贝价值在异构场景退化为「host 内存 + tcp」，与 QUIC/TCP 字节流同量级（都需序列化/拷贝经 host）。这是「网络自由=手段」卖点下 NIXL 定位的关键约束：NIXL 是同构加速、异构兜底，不是异构场景的零拷贝解。

【证据边界】单 block（96 bytes f32）、N=2 white↔pearl、host DRAM（CPU tensor）+ UCX_TLS=tcp、双向 Write transfer；不覆盖同构 VRAM GPU-direct 跨机、大 KV block、prefill KV ring 接线（S3c）、paged-KV 化（S4）、性能/吞吐。S3b（side-channel 接 coordinator 控制面）与 S3c（prefill KV ring 走 block 路径）仍为后续节点。"""

def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn, EVIDENCE, "evidence", "active",
        "NIXL S3a 跨机异构 transfer 跑通：双向字节级一致 + 跨厂商 GPU-direct 受限（需 host DRAM staging）",
        CONTENT, "verified", 0.95, 1.0, SOURCE)

    edges = [
        (EVIDENCE, S3_TASK, "PART_OF", "S3a cross-machine transfer is the first checkpoint of the S3 task"),
        (EVIDENCE, S3_DECISION, "PART_OF", "S3a evidence for the S3 cross-machine verification decision"),
        (EVIDENCE, "evidence-nixl-sys-white-cuda-verified-20260816", "BUILDS_ON", "builds on the white/pearl single-machine register/metadata smoke"),
        (EVIDENCE, "decision-k10-kv-byte-ledger-20260816", "BUILDS_ON", "telemetry.total_bytes fills the K10 wire-byte caliber"),
        (EVIDENCE, "decision-hcp-first-principles-value-20260815", "FOLLOWS", "cross-vendor host-staging constraint refines NIXL's value under network-freedom selling point"),
    ]
    for e in edges:
        upsert_edge(conn, *e)

    conn.commit()
    conn.close()
    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("evidence-nixl-s3a-cross-machine-transfer-20260816")

if __name__ == "__main__":
    main()
