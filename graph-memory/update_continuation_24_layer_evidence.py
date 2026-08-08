import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@a7a583d"

TASK = "task-continuation-24-layer-mixed-history-20260808"
DECISION = "decision-continuation-24-layer-mixed-history-20260808"
ROUTE = "hypothesis-continuation-route-full-activation-packet-20260807"
PORTFOLIO = "task-continuation-route-experiment-portfolio-20260807"
EVIDENCE = "evidence-continuation-24-layer-mixed-history-20260808"
LESSON = "lesson-file-scoped-rustfmt-in-route-worktree-20260808"
NEXT_TASK = "task-continuation-after-stationary-decode-20260808"

TASK_RESULT = """[2026-08-08 完成]
路线 B 已完成 N=3、L=24、m=6、tickets=[1,3,2] 的 mixed-history stationary continuation。起点由六位置 initial prefill 和一轮 capacity-scheduled decode 构成；continuation 历史 KV 全程留在 ReservedPositionedKvShard，LayerPacket 只沿 successor ring 合并本地 positioned attention partial。每层新增 KV 精确按 [1,3,2] append，24 层最终 domain KV 总量为 [52,156,104]，仍严格 1:3:2；每层 N-1=2 hops，总计 48 hops。该结论限本机 CPU synthetic correctness，不是性能、真实网络或异构硬件证据。"""

EVIDENCE_CONTENT = """路线分支 codex/route-b-continuation-stationary-packet；动机与计划提交 cc542e0；实现提交 a7a583d（rust: validate 24-layer stationary continuation）。
TDD 与验证：
1. RED：新增 twenty_four_layer_stationary_continuation_uses_mixed_positioned_history 后，focused cargo test 因 run_model_ring_with_reserved_history_for_positions 不存在而以 E0425 失败；既有 primitive 未被误判为 24 层闭环。
2. GREEN：focused test 1 passed、0 failed。runner 在任何 shard mutation 前校验所有 domain offsets 对 0..m 互斥完备；每层只调用 process_layer_packet_with_reserved_history_for_positions，并把 finisher hidden 交给下一层 starter。
3. 数据流：initial prefill positions 0..5 按 [1,3,2] 保存，一轮 decode 写 position 6；stationary continuation 写 positions 7..12。每层实际 shard position union 严格等于 0..13，continuation 增量为 [1,3,2]，storage pointer 不变且 committed_len 等于 reservation；24 层总 domain KV 为 [52,156,104]。
4. 路由与数值：starter/finisher 逐层轮转，visited_domains 逐层等于 successor 顺序；总 hops=24*(3-1)=48，末层 logits producer 回到初始 domain。最终 hidden/logits 与 contiguous dense reference 在既有 1e-3 容差内一致。
5. 完整验证：cargo test --features tch-backend 为 106 passed、0 failed、3 ignored；model::self_driving::tests 为 24 passed、0 failed、1 ignored；cargo clippy --features tch-backend --lib --tests exit 0，仅既存 warnings；file-scoped rustfmt --check、git diff --check 均 exit 0。
证据边界：mac-local-shell + libtorch CPU synthetic；不证明 MPS/CUDA/HIP、TCP/QUIC、runtime、多请求、性能，亦未在 continuation 后再执行 decode。"""

LESSON_CONTENT = """[2026-08-08 verified incident]
症状：在干净的路线 B worktree 对单一 Rust 文件改动后运行 cargo fmt --manifest-path rust/Cargo.toml，产生 55 个文件 diff，其中 54 个与任务无关。
根因：cargo fmt 以整个 crate 为格式化范围，而该分支基线尚未整体采用当前 rustfmt 输出；除空白折行外还会重排 use/mod，造成大面积非业务 diff。
影响：若直接提交会污染路线比较锚点，使功能 diff 难以审查。
已验证恢复：规划提交后工作树原本为空，测试 patch 只改 self_driving.rs；对其余 Rust diff reverse apply 后，git status 只剩目标文件，随后 focused/full tests 与 clippy 全部通过，最终 a7a583d 只包含 self_driving.rs。
预防条件：本仓库的小节点只对实际修改文件运行 rustfmt --edition 2021 <file>，然后执行 git diff --name-only、git diff --check；除非任务本身就是全 crate 格式化，不运行 crate-wide cargo fmt。"""

NEXT_TASK_CONTENT = """下一候选小节点：在当前路线 B CPU synthetic oracle 上，从 prefix + 第一轮 decode + m=6 stationary continuation 的同一 request state 继续一次 m=1 decode。预先把两轮 decode 的 48 个 layer-KV append 一起纳入 [1,3,2] frozen schedule 和 reservation；用 continuation 的末位置 logits 采样 token，由其 finisher domain 启动 position 13 的 decode。验收包括 dense reference hidden/logits/token 对齐、每层 position union=0..14、post-continuation decode 每层只 append 一次、两轮 decode 总 capacity counts=[8,24,16]、storage pointer 不变、阶段边界仍为 N-1 hops/layer。范围仍限 in-process CPU synthetic，不接 wire/runtime、多请求或性能测量；开始前做独立动机剖析。"""


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
        (id,type,layer,project,title,content,importance,confidence,status,source,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          type=excluded.type,layer=excluded.layer,project=excluded.project,
          title=excluded.title,content=excluded.content,importance=excluded.importance,
          confidence=excluded.confidence,status=excluded.status,source=excluded.source,
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
        INSERT INTO edges(source,target,type,weight,note)
        VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note
        """,
        (source, target, edge_type, note),
    )


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    old_task = conn.execute("SELECT title,content FROM nodes WHERE id=?", (TASK,)).fetchone()
    assert old_task is not None
    task_content = old_task[1]
    if TASK_RESULT not in task_content:
        task_content += "\n\n" + TASK_RESULT
    upsert_node(
        conn,
        TASK,
        "task",
        "progress",
        old_task[0],
        task_content,
        "closed",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        EVIDENCE,
        "evidence",
        "progress",
        "24 层 mixed-history stationary continuation 验证完成",
        EVIDENCE_CONTENT,
        "verified",
        1.0,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        LESSON,
        "lesson",
        "progress",
        "路线 worktree 使用 file-scoped rustfmt",
        LESSON_CONTENT,
        "held",
        0.9,
        1.0,
        SOURCE,
    )
    upsert_node(
        conn,
        NEXT_TASK,
        "task",
        "active",
        "候选：stationary continuation 后恢复 decode",
        NEXT_TASK_CONTENT,
        "planning",
        1.0,
        0.95,
        "proposed-after-a7a583d",
    )

    edges = (
        (EVIDENCE, TASK, "CONFIRMS", "24-layer route B task verified on its isolated branch"),
        (EVIDENCE, DECISION, "CONFIRMS", "the minimal runner closes the full-layer recurrence gap"),
        (EVIDENCE, ROUTE, "SUPPORTS", "third correctness checkpoint for stationary-history route"),
        (LESSON, TASK, "LEARNED_FROM", "crate-wide formatting incident occurred during this task"),
        (LESSON, NEXT_TASK, "GOVERNS", "use file-scoped rustfmt on the next route checkpoint"),
        (NEXT_TASK, TASK, "DEPENDS_ON", "post-continuation decode requires the verified 24-layer continuation state"),
        (NEXT_TASK, ROUTE, "PART_OF", "next route B correctness checkpoint"),
        (NEXT_TASK, PORTFOLIO, "PART_OF", "retains route comparison boundaries"),
    )
    for edge in edges:
        upsert_edge(conn, *edge)
    conn.commit()

    task_state = conn.execute(
        "SELECT layer,status,source,content FROM nodes WHERE id=?", (TASK,)
    ).fetchone()
    evidence_state = conn.execute(
        "SELECT status,source,content FROM nodes WHERE id=?", (EVIDENCE,)
    ).fetchone()
    lesson_state = conn.execute(
        "SELECT status,source,content FROM nodes WHERE id=?", (LESSON,)
    ).fetchone()
    next_state = conn.execute(
        "SELECT layer,status,source,content FROM nodes WHERE id=?", (NEXT_TASK,)
    ).fetchone()
    assert task_state[:3] == ("progress", "closed", SOURCE)
    assert TASK_RESULT in task_state[3]
    assert evidence_state == ("verified", SOURCE, EVIDENCE_CONTENT)
    assert lesson_state == ("held", SOURCE, LESSON_CONTENT)
    assert next_state == ("active", "planning", "proposed-after-a7a583d", NEXT_TASK_CONTENT)
    assert conn.execute(
        "SELECT COUNT(*) FROM edges WHERE source=?", (EVIDENCE,)
    ).fetchone()[0] == 3
    conn.close()

    print("continuation_24_layer_task=closed")
    print("evidence=verified")
    print("rustfmt_lesson=held")
    print("next_task=planning")


if __name__ == "__main__":
    main()
