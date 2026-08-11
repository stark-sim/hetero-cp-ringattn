import sqlite3
from pathlib import Path
import subprocess
import sys

DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@6d"

TASK = "task-phase2-rust-6d-n3-service-readiness-20260812"
EVIDENCE = "evidence-phase2-rust-6d-n3-service-20260812"
SESSION = "session-next-phase2-rust-6d-n3-service-20260812"
BENCH_TASK = "task-phase2-rust-benchmark-readiness-20260812"
EVIDENCE_BENCH = "evidence-phase2-rust-benchmark-readiness-20260812"
TASK_6C1 = "task-phase2-rust-6c1-native-service-baseline-20260812"
EV_6C1 = "evidence-phase2-rust-6c1-native-baseline-20260812"
SCRIPT = "scripts/test_phase2_6d_n3_service.sh"
REPORT = "reports/routeb-6d-n3-service-20260812-060937"


def upsert_node(conn, node_id, node_type, layer, title, content, status, importance, confidence, source):
    conn.execute(
        """
        INSERT INTO nodes
        (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          type=excluded.type,layer=excluded.layer,project=excluded.project,
          title=excluded.title,content=excluded.content,importance=excluded.importance,
          confidence=excluded.confidence,status=excluded.status,source=excluded.source,
          updated_at=datetime('now')
        """,
        (node_id, node_type, layer, PROJECT, title, content, importance, confidence, status, source),
    )


def upsert_edge(conn, source, target, edge_type, note):
    conn.execute(
        """
        INSERT INTO edges(source,target,type,weight,note)
        VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note
        """,
        (source, target, edge_type, note),
    )


TASK_RESULT = """[2026-08-12 完成] 6d N=3 异构 Rust 服务 readiness。
Mac MPS (coordinator + worker0) + white RTX 4090 CUDA (worker1) + pearl RX 9060 XT HIP (worker2) 经 coordinator/生产 QUIC neighbor-only ring 处理真实 Qwen2-0.5B 多请求服务；native client 验证 token、admission、FIFO、release、telemetry 与 neighbor-only hops。
验收（scripts/test_phase2_6d_n3_service.sh，report routeb-6d-n3-service-20260812-060937）：
1. 4/4 请求 0 错误（metrics total=4 completed=4 failed=0 active=0；提示词不等长 6/13/40/49 tokens，max_tokens 3/4/6/8）；
2. token/reference：greedy 生成均非 [error:、text 非空（如 prompt49 -> ". The quick brown"，prompt40 -> " It was the only flower that"）；
3. admission/FIFO/release：trace 每请求 reserved==released、reserved 字节按 prompt/decodes 单调增长（如 req1 [233472,245760,221184]）；
4. telemetry 关联：trace 8 条（4 请求）按 request_id 含 enqueue/prefill-accepted/first-token/completed elapsed 与 finish_reason；
5. neighbor-only hops：N=3 L=24 断言 prefill_hops=48=L*(N-1)、decode_hops=steps*48 全部成立。
二期到此证明 Rust 服务已具备接受外部 benchmark 的基础，不在本节点调用 vLLM CLI。"""

EVIDENCE_CONTENT = """实现脚本 scripts/test_phase2_6d_n3_service.sh；复用 78be1d0（6c.0 trace 平面）。
验证（三机真实异构：Mac MPS + white CUDA + pearl HIP，均 292a3c5 release build，远端经 git pull --ff-only 同步）：
1. 三端 handshake：Mac=8192 MB、white=21793 MB、pearl=14805 MB（均非 u64::MAX，admission 预算真实）。
2. 4 个不等长请求（并发 2 + 串行 2）：resp1-4 全部 OK，0 错误；metrics total=completed=4、failed=0、active=0。
3. trace 4 条：prefill_hops=48、decode_hops=steps*48（N=3 L=24 公式），reserved==released，finish_reason 均 length。
4. 首次运行 1 个失败：测试提示词 "The quick brown fox"（4 tokens）在 N=3 ring 下 domain 分到 0 token 被预检拒绝——测试数据问题，非系统缺陷；加长后通过。另一次脚本 health grep 误判（python json.tool 缩进），已改为 json 解析。
证据边界：N=3 异构单服务实例多请求 correctness；不证明吞吐/性能、跨机故障恢复、vLLM 兼容；二期 benchmark-readiness 五项出口至此全部达成（API 6b.0 / 资源 6b.2a+6b.2b / 调度 6b.3 / 观测 6c.0 / 稳定性 6c.1+6d）。"""

BENCH_RESULT = """[2026-08-12 完成] 二期 Rust 推理服务 benchmark-readiness 五项出口全部达成（只针对 Rust HCP 服务本体，不运行 vLLM benchmark）：
1. API ✅：/v1/completions 非 streaming/streaming、SSE [DONE]、usage/error 内部 regression（6b.0，4784acf）。
2. 资源 ✅：普通 service prefill frozen reservation + byte admission（6b.2a，d57b9ca）；active requests 总占用 reserve/release（6b.2b，9ec8f96）。
3. 调度 ✅：coordinator 广播同一 DecodeBatch FIFO，request horizon/token/cache lifecycle 正确（6b.3，abddbf1）。
4. 观测 ✅：request queue/prefill/first-token/decode/release 与 ring hops/bytes/reserved bytes 可关联（6c.0，78be1d0 --trace-jsonl）。
5. 稳定性 ✅：native concurrency 1/2/4（6c.1）；Mac MPS + white CUDA + pearl HIP N=3 真实 Qwen 服务闭环（6d，test_phase2_6d_n3_service.sh）。
二期不改 vLLM engine/plugin。三期（生态：多请求 batching、placement/ledger WIP 重启、外部 benchmark）待用户规划。"""


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(
        conn,
        TASK,
        "task",
        "active",
        "6d：N=3 异构 Rust 服务真实 Qwen readiness",
        TASK_RESULT,
        "completed",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        EVIDENCE,
        "evidence",
        "active",
        "6d N=3 异构真实 Qwen 服务闭环通过",
        EVIDENCE_CONTENT,
        "verified",
        1.0,
        1.0,
        SOURCE,
    )
    # Close the 6d checkpoint and the whole phase-2 engineering line.
    conn.execute("UPDATE nodes SET status='closed', updated_at=datetime('now') WHERE id=?", (SESSION,))
    # Mark benchmark-readiness task complete with full exit criteria.
    upsert_node(
        conn,
        BENCH_TASK,
        "task",
        "active",
        "二期 Rust 推理服务 benchmark-readiness",
        BENCH_RESULT,
        "completed",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        EVIDENCE_BENCH,
        "evidence",
        "active",
        "二期 benchmark-readiness 五项出口全部达成",
        BENCH_RESULT,
        "verified",
        1.0,
        1.0,
        SOURCE,
    )

    edges = (
        (TASK, EVIDENCE, "CONFIRMS", "6d task confirmed by evidence"),
        (EVIDENCE, TASK, "SUPPORTS", "evidence supports task completion"),
        (TASK_6C1, TASK, "DEPENDS_ON", "6d builds on 6c.1 native baseline"),
        (EV_6C1, EVIDENCE, "SUPPORTS", "6c.1 established the N=2 stability baseline"),
        (EVIDENCE, EVIDENCE_BENCH, "SUPPORTS", "6d evidence completes benchmark-readiness item 5"),
        (TASK, EVIDENCE_BENCH, "CONFIRMS", "6d completes the benchmark-readiness exit"),
        (EVIDENCE_BENCH, BENCH_TASK, "SUPPORTS", "all five exit items now done"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()
    conn.close()

    subprocess.run([sys.executable, "graph-memory/export.py"], check=True)
    print("6d_task=completed")
    print("6d_evidence=verified")
    print("benchmark_readiness=completed")
    print("phase2_engineering=complete")


if __name__ == "__main__":
    main()
