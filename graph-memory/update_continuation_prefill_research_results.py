import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@4b6dc76"

RESEARCH_TASK = "task-continuation-prefill-inference-research-20260807"
OLD_READINESS = "decision-continuation-local-kv-readiness-20260805"
NEW_READINESS = "decision-continuation-local-kv-readiness-revised-20260807"
REVISION = "revision-initial-prefill-kv-ring-necessity-20260807"
EQUIVALENCE = "belief-continuation-kv-stationary-equivalence-20260807"
COST_BELIEF = "belief-continuation-route-cost-model-20260807"
EVIDENCE = "ev-continuation-prefill-inference-research-20260807"
NEXT_TASK = "task-continuation-packet-layer-contract-20260807"
FULL_TASK = "task-continuation-batched-accumulator-ring-20260803"
PREFERENCE = "preference-motivation-analysis-20260721"

RESEARCH_RESULT = """研究已完成并形成 docs/CONTINUATION_PREFILL_INFERENCE_RESEARCH.md。结论边界：历史 KV 是否沿 ring 移动不是 attention 数学必要性；KV-ring 与 KV-stationary accumulator 是同一精确 softmax 的两种数据移动对偶。完整 m-segment activation packet 与 continuation prefill 直接交合，因为它同时解决 owner-local new-KV generation、全 shard attention 与层间 residual/Norm/MLP activation。近期保留 KV-ring baseline 和完整 packet 两种显式实验模式，不引入动态 planner；query-shard N-hop return 与重算 Norm 压缩 packet 仅保留为后续候选。"""

NEW_READINESS_CONTENT = """KV readiness 的修订定义如下。
1. 对任意 prefill/decode/continuation attention，每个节点只需在执行自己的 local partial 前，由当前层 normalized activation 生成本节点负责 position 的新 K/V，并 append 到本地 capacity-weighted positioned shard；不需要全局 new-KV barrier。
2. 对互斥完备的 positioned KV shards，只要每个 query 的 O/LSE accumulator 恰好合并所有可见 shard，历史 KV 数学上不需要移动；绝对位置 p_k<=p_q 给出 segment 内 causal 语义。
3. Initial prefill 继续使用 KV-ring 是当前 HCP 的阶段性路线选择，不是数学必要性。对 GQA Qwen2-0.5B，T=0 时 KV payload 显著小于完整 activation packet，且 query-sharded non-attention 计算可并行，因此 KV-ring 是近期合理基线。
4. Decode m=1 保留现有 KV-stationary self-driving packet。Continuation 同时保留 KV-ring baseline 与 m-segment packet 实验路线，先静态比较，不引入动态 planner。
5. 当前只完成研究与 attention-level batched oracle；m>1 整层 packet、owner-local position subset projection、wire 和 24 层递推仍待验证。
这份修订替代 2026-08-05 决策中“initial prefill 仍需 KV ring”的必要性表述，同时保留其 owner-local readiness 核心。VERDICT: IMPLEMENT THE SMALLEST PACKET CONTRACT EXPERIMENT AFTER USER CONFIRMATION。"""

REVISION_CONTENT = """旧决策正确识别了 decode 的 owner-local KV generation 和无全局 barrier，但把“initial prefill 仍需 KV ring”写成了必要条件。Ring Attention 原论文证明 circulating KV 有效；FlashInfer/SGLang/vLLM DCP/Helix 与可结合 O/LSE 推导共同证明 KV-stationary 也精确成立。修订后的判断是：initial prefill 使用 KV-ring 属于当前 GQA 通信成本和 query-sharded 并行下的路线选择，而非数学必要性。旧决策保留为 superseded 历史，新决策给出完整替代表述。"""

EQUIVALENCE_CONTENT = """对每层互斥完备的 positioned KV 分片 S_i，以及每个 continuation query 的可见集合 V_r={p|p<=T+r}，局部 max/sum/numerator 状态使用 max-shifted online-softmax 合并是结合的；等价地，normalized partial (O_i,LSE_i) 以 LSE-weighted sum 精确合并。因此固定 Q 并环传 KV，与固定 KV 并环传 Q/O/LSE，在实数域中得到相同 attention。历史 KV 移动是执行选择，不是正确性条件。浮点实现仅受合并顺序舍入影响。置信度 0.99：数学推导、vLLM DCP A2A 源码、SGLang extend merge 和现有 Rust batched positioned oracle 相互支持。"""

COST_CONTENT = """每请求每层的理想 byte-hop 模型：C_KV=(N-1)*2(T+m)D_kv*b；当前完整 residual+normalized+Q+O+LSE packet 为 C_packet4=(N-1)*m(4H+h_q)*b；重算 input Norm 的候选为 C_packet3=(N-1)*m(3H+h_q)*b；query-shard 回原节点为 C_Q_return=N*m(2H+h_q)*b。对 Qwen2-0.5B H=896,D_kv=128,h_q=14,g=7,BF16,N=3，完整 packet 对 KV-ring 的纯网络 break-even 是 T>13.0546875m；重算 Norm 是 T>9.5546875m；Q/O/LSE N-1 理想下界是 T>6.0546875m；query-shard N-hop return 是 T>9.58203125m。公式不含 framing、kernel latency、overlap 和异构最慢边，不能直接成为动态 runtime selector。"""

EVIDENCE_CONTENT = """研究报告提交 4b6dc76。两轮来源核验覆盖 Ring Attention 论文 Appendix C、作者 JAX inference kernel、PyTorch CP、Megatron Core CP、vLLM Context Parallel 文档与 DCP A2A 源码、FlashInfer paged KV append、SGLang forward_extend/FlashInfer backend 和 Helix Parallelism。主要事实：原论文选择 circulation KV；serving extend 只 append 新 KV；SGLang 可把 new-segment causal partial 与 historical-prefix non-causal partial 用 O/LSE 合并；vLLM DCP A2A 对 stationary KV shards 的 partial O/LSE 做 exact weighted combination。
新鲜验证：git diff --check -- docs/CONTINUATION_PREFILL_INFERENCE_RESEARCH.md -> exit 0；独立 Python assertions -> report_contract=ok, citations=10, formula_examples=ok；五个主要官方/论文 URL curl 均 HTTP 200；代码核对确认 LayerPacket 字段、m=1 validate_route 限制与 positioned partial/merge API 仍与报告描述一致。
证据边界：这是数学、权威来源和代码映射研究，不是 m>1 整层实现、真实网络性能或异构硬件结果。"""

NEXT_TASK_CONTENT = """下一最小节点只推广 LayerPacket 的整层 shape 合同：允许 m>1，使用一层 synthetic oracle 验证 residual、normalized、position_ids、Q、O/LSE、output projection、residual、post-attention Norm 与 MLP 的 shape 和 dense-reference 数值。该节点不实现 capacity-weighted position assignment、不接 wire/runtime、多请求或动态 planner，也不声称 continuation 闭环。完成后再单独实现 per-position owner-local K/V generation，最后才做 N=3、L=24、tickets=[1,3,2] mixed-history continuation。开始实现前等待用户确认。"""


def upsert_node(
    conn,
    node_id,
    node_type,
    layer,
    title,
    content,
    status,
    importance,
    confidence,
    source,
):
    conn.execute(
        """
        INSERT INTO nodes
        (id, type, layer, project, title, content, importance, confidence,
         status, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          type=excluded.type, layer=excluded.layer, project=excluded.project,
          title=excluded.title, content=excluded.content,
          importance=excluded.importance, confidence=excluded.confidence,
          status=excluded.status, source=excluded.source,
          updated_at=datetime('now')
        """,
        (
            node_id,
            node_type,
            layer,
            PROJECT,
            title,
            content,
            importance,
            confidence,
            status,
            source,
        ),
    )


def upsert_edge(conn, source, target, edge_type, note):
    conn.execute(
        """
        INSERT INTO edges (source, target, type, weight, note)
        VALUES (?, ?, ?, 1.0, ?)
        ON CONFLICT(source, target, type) DO UPDATE SET note=excluded.note
        """,
        (source, target, edge_type, note),
    )


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("BEGIN IMMEDIATE")

    old_task = conn.execute(
        "SELECT content FROM nodes WHERE id=?", (RESEARCH_TASK,)
    ).fetchone()
    assert old_task is not None
    task_content = old_task[0]
    if RESEARCH_RESULT not in task_content:
        task_content += "\n\n[2026-08-07 完成]\n" + RESEARCH_RESULT
    upsert_node(
        conn,
        RESEARCH_TASK,
        "task",
        "progress",
        "研究 continuation prefill 是否需要移动历史 KV",
        task_content,
        "closed",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        NEW_READINESS,
        "decision",
        "active",
        "KV readiness 与阶段路线的研究后修订",
        NEW_READINESS_CONTENT,
        "held",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        REVISION,
        "revision",
        "progress",
        "Initial prefill 的 KV-ring 是路线选择而非数学必要",
        REVISION_CONTENT,
        "held",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        EQUIVALENCE,
        "belief",
        "blueprint",
        "KV-ring 与 KV-stationary accumulator 的 attention 等价性",
        EQUIVALENCE_CONTENT,
        "held",
        1.0,
        0.99,
        SOURCE,
    )
    upsert_node(
        conn,
        COST_BELIEF,
        "belief",
        "blueprint",
        "Continuation 路线的 byte-hop 成本模型",
        COST_CONTENT,
        "held",
        0.95,
        0.98,
        SOURCE,
    )
    upsert_node(
        conn,
        EVIDENCE,
        "evidence",
        "progress",
        "Continuation prefill inference 研究与公式核验完成",
        EVIDENCE_CONTENT,
        "held",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        NEXT_TASK,
        "task",
        "active",
        "先验证 m>1 LayerPacket 的单层完整合同",
        NEXT_TASK_CONTENT,
        "planning",
        1.0,
        0.95,
        SOURCE,
    )

    conn.execute(
        "UPDATE nodes SET layer='progress', status='superseded', replaced_by=?, updated_at=datetime('now') WHERE id=?",
        (NEW_READINESS, OLD_READINESS),
    )
    conn.execute(
        "UPDATE nodes SET status='planning', updated_at=datetime('now') WHERE id=?",
        (FULL_TASK,),
    )

    edges = (
        (EVIDENCE, RESEARCH_TASK, "CONFIRMS", "research deliverable and checks complete"),
        (EVIDENCE, EQUIVALENCE, "CONFIRMS", "math, upstream source, and Rust oracle agree"),
        (EVIDENCE, COST_BELIEF, "SUPPORTS", "formula and Qwen examples independently asserted"),
        (EVIDENCE, NEW_READINESS, "SUPPORTS", "supports revised phase semantics"),
        (REVISION, OLD_READINESS, "REVISION_OF", "revises only the initial-prefill necessity clause"),
        (REVISION, NEW_READINESS, "LEADS_TO", "replacement decision preserves valid local-readiness content"),
        (NEW_READINESS, OLD_READINESS, "SUPERSEDES", "full replacement after inference research"),
        (OLD_READINESS, NEW_READINESS, "REPLACED_BY", "preserve decision history"),
        (PREFERENCE, NEW_READINESS, "GOVERNS", "motivation analysis and evidence boundary"),
        (NEW_READINESS, EQUIVALENCE, "DEPENDS_ON", "stationary route relies on exact shard merge"),
        (NEXT_TASK, RESEARCH_TASK, "DEPENDS_ON", "research chooses the smallest packet experiment"),
        (FULL_TASK, NEXT_TASK, "DEPENDS_ON", "24-layer mixed-history proof follows one-layer contract"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)

    conn.commit()

    expected = {
        RESEARCH_TASK: RESEARCH_RESULT,
        NEW_READINESS: NEW_READINESS_CONTENT,
        REVISION: REVISION_CONTENT,
        EQUIVALENCE: EQUIVALENCE_CONTENT,
        COST_BELIEF: COST_CONTENT,
        EVIDENCE: EVIDENCE_CONTENT,
        NEXT_TASK: NEXT_TASK_CONTENT,
    }
    for node_id, marker in expected.items():
        content = conn.execute(
            "SELECT content FROM nodes WHERE id=?", (node_id,)
        ).fetchone()[0]
        assert marker in content
    old_status, replaced_by = conn.execute(
        "SELECT status,replaced_by FROM nodes WHERE id=?", (OLD_READINESS,)
    ).fetchone()
    assert (old_status, replaced_by) == ("superseded", NEW_READINESS)
    edge_count = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE source IN (?,?,?,?,?) OR target IN (?,?,?,?,?)",
        (
            EVIDENCE,
            NEW_READINESS,
            REVISION,
            NEXT_TASK,
            FULL_TASK,
            EVIDENCE,
            NEW_READINESS,
            REVISION,
            NEXT_TASK,
            FULL_TASK,
        ),
    ).fetchone()[0]
    assert edge_count >= len(edges)
    conn.close()

    print("research_task=closed")
    print("replacement_decision=held")
    print("old_decision=superseded")
    print("content_markers=exact")
    print(f"relevant_edges={edge_count}")


if __name__ == "__main__":
    main()
