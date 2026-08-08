import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "hetero-cp-ringattn@73cd0e8"

TASK = "task-continuation-position-owner-local-kv-20260807"
DECISION = "decision-continuation-position-owner-local-kv-20260807"
ROUTE = "hypothesis-continuation-route-full-activation-packet-20260807"
BRANCH_DECISION = "decision-continuation-route-branch-discipline-20260807"
PORTFOLIO = "task-continuation-route-experiment-portfolio-20260807"
EVIDENCE = "evidence-continuation-position-owner-local-kv-20260808"
NEXT_TASK = "task-continuation-24-layer-mixed-history-20260808"
OLD_NEXT_TASK = "task-continuation-batched-accumulator-ring-20260803"

TASK_RESULT = """[2026-08-08 完成]
路线 B 分支 codex/route-b-continuation-stationary-packet 已完成单层 position owner-local KV 合同。N=3、m=6、tickets=[1,3,2] 的 frozen request schedule 把新增 positions 精确分成 [1,3,2]；每个 domain 只 index-select、投影并 append 自己的 normalized/absolute-position subset，随后仍用全部 m 个 Q 对本地完整 positioned shard 计算 causal partial。旧单-assignee API 保留为 all-or-empty offsets wrapper；LayerPacket、SelfDrivingPacket、transport 与 runtime 均未增加 owner vector。该节点是 CPU synthetic correctness，不是性能或真实异构部署结果。"""

EVIDENCE_CONTENT = """路线分支 codex/route-b-continuation-stationary-packet；计划提交 99c3084；实现提交 73cd0e8（rust: distribute continuation KV by position owner）。
TDD 与验证：
1. RED：新增 multi_token_packet_generates_new_kv_by_capacity_weighted_position_owner 后，cargo test 因 process_layer_packet_with_reserved_history_for_positions 不存在而以 E0425 失败，证明测试命中缺失合同。
2. GREEN：同一测试通过。N=3、m=6、tickets=[1,3,2]，owner offsets 完备且无重复；三个 ReservedPositionedKvShard 新增数为 [1,3,2]，全部写满但不越过预留容量，storage pointer 不变；packet tensor payload 仍为 m*(4H+h_q+1)，不含历史 KV 或 owner vector；attention max diff <1e-4，整层 hidden max diff <2e-4。
3. 负路径 position_owner_offsets_reject_duplicates_and_out_of_range_values 通过：重复/越界 offset 在任何 append 前返回错误，committed_len 保持 0。
4. cargo test --features tch-backend model::self_driving::tests：23 passed、0 failed、1 ignored；旧 m=1 decode、m>1 uniform assignee、任意 N、wrap-around 与 24 层 cache reuse 回归保持通过。
5. cargo clippy --features tch-backend --lib --tests、目标文件 rustfmt --check、git diff --check 均 exit 0；clippy 仅报告仓内既存 warnings。
审查修订：normalized 与 position_ids 的 index tensors 分别放到各自输入 device，避免隐含同设备假设。
证据边界：本机 CPU synthetic，不证明 MPS/CUDA/HIP、wire/runtime、24 层 continuation 或路线性能。全局 owner 完备性由 frozen plan 生成/校验，packet 本身不携带或重复校验全局 owner vector。"""

NEXT_TASK_CONTENT = """下一候选小节点：在路线 B 分支用 N=3、L=24、tickets=[1,3,2] 验证 mixed-history continuation。起点应包含已经按 positioned shards 保存的历史 prefix 与至少一轮 decode 增量；随后对 m>1 continuation segment 逐层使用 position owner-local KV 和 self-driving activation packet，验证每层 position union 完整无重复、累计 KV 字节不越各 worker reservation、starter/finisher 跨层轮转、最终 logits 对齐 dense reference。仍限 in-process synthetic，不接 wire/runtime、多请求或性能测量。开始前单独做动机剖析，明确是否在同一节点内再追加一个 decode token，避免把两个里程碑混成一项。"""


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
        (node_id,node_type,layer,PROJECT,title,content,importance,confidence,status,source),
    )


def upsert_edge(conn, source, target, edge_type, note):
    conn.execute(
        """
        INSERT INTO edges(source,target,type,weight,note)
        VALUES (?,?,?,1.0,?)
        ON CONFLICT(source,target,type) DO UPDATE SET note=excluded.note
        """,
        (source,target,edge_type,note),
    )


def main():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    old_task = conn.execute("SELECT title,content FROM nodes WHERE id=?",(TASK,)).fetchone()
    assert old_task is not None
    task_content = old_task[1]
    if TASK_RESULT not in task_content:
        task_content += "\n\n" + TASK_RESULT
    upsert_node(conn,TASK,"task","progress",old_task[0],task_content,"closed",1.0,1.0,SOURCE)
    upsert_node(conn,EVIDENCE,"evidence","progress","Position owner-local continuation KV 单层合同验证完成",EVIDENCE_CONTENT,"verified",1.0,1.0,SOURCE)
    upsert_node(conn,NEXT_TASK,"task","active","候选：24 层 mixed-history stationary continuation",NEXT_TASK_CONTENT,"planning",1.0,0.95,"proposed-after-73cd0e8")
    conn.execute(
        "UPDATE nodes SET layer='progress',status='superseded',replaced_by=?,updated_at=datetime('now') WHERE id=?",
        (NEXT_TASK,OLD_NEXT_TASK),
    )

    edges = (
        (EVIDENCE,TASK,"CONFIRMS","position-local KV task verified on route B branch"),
        (EVIDENCE,DECISION,"CONFIRMS","frozen request plan can govern owner-local subset projection"),
        (EVIDENCE,ROUTE,"SUPPORTS","second correctness checkpoint for stationary-history route"),
        (EVIDENCE,BRANCH_DECISION,"SUPPORTS","implementation and evidence are isolated on the recorded route branch"),
        (NEXT_TASK,TASK,"DEPENDS_ON","24-layer proof requires the single-layer position-owner contract"),
        (NEXT_TASK,ROUTE,"PART_OF","next route B correctness checkpoint"),
        (NEXT_TASK,PORTFOLIO,"PART_OF","retains route comparison boundaries"),
        (NEXT_TASK,OLD_NEXT_TASK,"SUPERSEDES","replaces the generic accumulator task with a bounded 24-layer route-B checkpoint"),
        (OLD_NEXT_TASK,NEXT_TASK,"REPLACED_BY","preserves the earlier planning history"),
    )
    for edge in edges:
        upsert_edge(conn,*edge)
    conn.commit()

    task_state = conn.execute("SELECT layer,status,source,content FROM nodes WHERE id=?",(TASK,)).fetchone()
    evidence_state = conn.execute("SELECT status,source,content FROM nodes WHERE id=?",(EVIDENCE,)).fetchone()
    next_state = conn.execute("SELECT layer,status,content FROM nodes WHERE id=?",(NEXT_TASK,)).fetchone()
    assert task_state[:3] == ("progress","closed",SOURCE)
    assert TASK_RESULT in task_state[3]
    assert evidence_state[:2] == ("verified",SOURCE)
    assert evidence_state[2] == EVIDENCE_CONTENT
    assert next_state[:2] == ("active","planning")
    assert next_state[2] == NEXT_TASK_CONTENT
    assert conn.execute(
        "SELECT layer,status,replaced_by FROM nodes WHERE id=?",(OLD_NEXT_TASK,)
    ).fetchone() == ("progress","superseded",NEXT_TASK)
    assert conn.execute("SELECT COUNT(*) FROM edges WHERE source=?",(EVIDENCE,)).fetchone()[0] == 4
    conn.close()

    print("position_owner_task=closed")
    print("evidence=verified")
    print("next_task=planning")
    print("evidence_edges=4")


if __name__ == "__main__":
    main()
