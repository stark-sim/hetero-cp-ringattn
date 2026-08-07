import sqlite3
from pathlib import Path


DB = Path("graph-memory/graph.db")
PROJECT = "hetero-cp-ringattn"
SOURCE = "user-confirmed-2026-08-07"

PORTFOLIO = "task-continuation-route-experiment-portfolio-20260807"
NEXT_TASK = "task-continuation-packet-layer-contract-20260807"
DECISION = "decision-continuation-packet-layer-contract-motivation-20260807"
USER_PREFERENCE = "preference-bandwidth-first-kv-stationary-continuation-20260807"
METHOD_PREFERENCE = "preference-motivation-analysis-20260721"
RESEARCH_TASK = "task-continuation-prefill-inference-research-20260807"
RESEARCH_EVIDENCE = "ev-continuation-prefill-inference-research-20260807"
COST_BELIEF = "belief-continuation-route-cost-model-20260807"
EQUIVALENCE = "belief-continuation-kv-stationary-equivalence-20260807"
BATCHED_CONTRACT = "task-continuation-batched-accumulator-contract-20260805"

ROUTE_KV_RING = "hypothesis-continuation-route-kv-ring-20260807"
ROUTE_PACKET4 = "hypothesis-continuation-route-full-activation-packet-20260807"
ROUTE_PACKET3 = "hypothesis-continuation-route-norm-recompute-packet-20260807"
ROUTE_Q_RETURN = "hypothesis-continuation-route-query-shard-return-20260807"
ROUTE_Q_ROTATE = "hypothesis-continuation-route-query-shard-rotate-20260807"
ROUTE_SPLIT_PARTIAL = "hypothesis-continuation-route-split-prefix-segment-partial-20260807"
ROUTE_HYBRID = "hypothesis-continuation-route-stage-hybrid-20260807"

PORTFOLIO_CONTENT = """长期保存 continuation prefill 的可组合实验路线，不把当前优先路线误写成唯一架构。每条路线单独记录通信对象、hop 合同、永久/临时显存、非 attention 计算位置、成立条件、牺牲项和重访触发条件。当前推进顺序由 HCP 的目标环境决定：异构节点间传输带宽是主要瓶颈，因此先实验历史 KV 原地的完整 activation packet；KV-ring 继续作为 correctness/大 segment 基线；其余路线不实现但保留可回溯关系。路线切换现阶段只允许显式实验模式，不引入动态 planner。"""

PREFERENCE_CONTENT = """用户确认 HCP 面向传输带宽高度受限的非同构集群，continuation/decode 的近期探索应优先避免传输长度为 T 的历史 KV。这个偏好不删除 KV-ring：initial prefill、大 m/T continuation、并行 query shard 或未来高速互联场景仍可能更适合它。永久 KV 始终按设备 capacity weight 分配；任何候选路线都不得让单节点永久承担不符合其 capacity 的历史 KV。"""

DECISION_CONTENT = """【动机剖析六问】
1. 问题：attention-level batched positioned accumulator 已证明 m>1 的 O/LSE 合并，但现有 LayerPacket、Q/KV projection 与整层 finisher 路径仍限制 m=1，不能证明 continuation segment 在不传历史 KV 时能完成一整层。
2. 现状：LayerPacket 已携带 residual、normalized、position_ids、Q、O/LSE；reserved positioned shard 能原地保存 KV；positioned partial kernel 支持 m>1 causal mask。缺口是这些能力尚未通过同一 LayerPacket 状态机组合，普通 tuple history 又没有 position metadata。
3. 目标：单层 synthetic、N>=2、m>1 测试通过。packet 访问每个节点的 ReservedPositionedKvShard，历史 KV 不进入 payload；唯一临时 assignee 可为本小节点追加整段新 K/V；finisher 的 attention output 与 hidden_states 对齐 dense causal reference；m=1 回归保持。
4. 他者：SGLang forward_extend 与 FlashInfer paged prefill 都让新增 query 读取 positioned/paged 历史并只 append 新 KV；vLLM DCP 合并 stationary shards 的 O/LSE。可复用的是 positioned causal merge 与增量 append，不复用 collective、paged allocator 或 scheduler。
5. 本方案：直接把现有 LayerPacket shape 从 m=1 推广到 m>=1；reserved path 使用 packet.position_ids 和 shard.position_tensor 调用 positioned partial/merge；legacy tuple path继续只允许 m=1。先写失败测试，再做最小实现，不改 wire/runtime/capacity schedule。
6. 为什么：另建 continuation packet 会复制现有状态机；只扩 attention oracle 已经完成却不能证明 Norm/MLP；直接扩现有 packet 是最小纵向证明，并保持 decode 为 m=1 特例。
【牺牲四问】
1. 默认 KV-ring 为什么存在：它利用 GQA 较小的 KV width，并让 position-sharded query、Norm 和 MLP 并行，initial prefill 与大 segment 时网络更省。
2. 当前牺牲什么：完整 packet 传 residual+normalized+Q+O/LSE，单层 attention 按节点串行，MLP 集中在 finisher；本节点还把整段新 K/V 暂交一个 assignee，不能证明最终 capacity weighting。
3. 被牺牲能力的作用：KV-ring 的 query 并行和紧凑 payload 降低短历史/大 m 的 TTFT；per-position assignment 才能给可变 segment 提供严格的 capacity-weighted byte placement。
4. 对 HCP 的意义：当前目标环境首先受历史 KV 传输限制，先验证 bytes 与 T 无关值得接受上述实验性牺牲；但这些牺牲禁止被提升为最终性能结论，后续必须单独补 per-position assignment 与 24 层验证。
VERDICT: IMPLEMENT。用户已确认先沿不传历史 KV 路线探索。"""

ROUTES = {
    ROUTE_KV_RING: (
        "Continuation 路线 A：完整 committed KV ring",
        "baseline",
        """问题：让 position-sharded query 覆盖完整 T+m KV。现状/方法：每个永久 KV shard 经 predecessor/successor ring 访问其他 N-1 节点，query activation 留在原节点，Norm/MLP 可按 token 并行。完成条件：positioned causal correctness、每 shard 恰好访问一次、foreign KV 仅为 bounded streaming buffer。业界对应：Ring Attention、Megatron/PyTorch CP 的 KV gather/ring。HCP 价值：协议最成熟，是 initial prefill 与 continuation correctness 基线。代价：C_KV=(N-1)*2(T+m)D_kv*b，历史越长越消耗带宽，并临时持有外来 KV chunk。牺牲：不满足“历史 KV 完全不上环”的近期目标。VERDICT: KEEP AS BASELINE。重访触发：m/T 较大、GQA KV payload 极小、query 并行收益超过历史传输，或高速互联显著降低 KV ring 成本。""",
    ),
    ROUTE_PACKET4: (
        "Continuation 路线 B：完整 activation 自驱动 packet",
        "active-experiment",
        """问题：在历史 KV 原地时完成 attention 与下一层 activation。方法：单 packet 携带 residual、normalized、Q、O/LSE，依次访问 N 个 local positioned KV shards；各节点在本地生成负责的新 KV，finisher 执行 W_o+residual+Norm+MLP。完成条件：m>1 单层合同、per-position K/V assignment、24 层 mixed-history、wire/runtime 依次通过。业界对应：KV-stationary partial O/LSE merge，但 HCP 用 neighbor-only P2P 和 rotating finisher 替代 collective。优势：C_packet4=(N-1)*m(4H+h_q)*b，与 T 无关，永久 KV 不移动。代价：大 m payload 较大，单层 attention 串行，MLP 集中。VERDICT: ACTIVE EXPERIMENT。当前只做第一项单层合同。""",
    ),
    ROUTE_PACKET3: (
        "Continuation 路线 C：重算 Norm 的压缩 packet",
        "deferred",
        """问题：完整 packet 中 normalized activation 额外占 mH 元素。方法：只传 residual、Q、O/LSE，每个节点从 residual 重算 input Norm 后投影本地 K/V。预期成本 C_packet3=(N-1)*m(3H+h_q)*b，与 T 无关。默认传 normalized 的原因是 Norm 只算一次且各节点使用完全相同的数值结果。牺牲：input Norm 从一次变为最多 N 次，增加计算、能耗和跨设备舍入差异；还要求每个后端能够一致重算。其作用是以计算换带宽。对 HCP 的意义：带宽极弱时可能值得，但在 packet4 尚无真实 profiling 前没有证据。VERDICT: DEFER。重访触发：packet4 correctness 完成且测得 normalized 占主导链路时间。""",
    ),
    ROUTE_Q_RETURN: (
        "Continuation 路线 D：Q-shard 完整 N 跳回原节点",
        "deferred",
        """问题：同时保持历史 KV 原地与 tokenwise Norm/MLP 并行。方法：每个节点保留自己的 position activation 并生成本地 Q/K/V；每个 Q shard 携带 O/LSE 绕完整 N 条边回到原节点，再本地完成 W_o、residual、Norm、MLP。成本 C_Q_return=N*m(2H+h_q)*b。优势：不传 residual/normalized，non-attention 计算按 query shard 并行。代价：多个并行 packet、完整 N hops、需要 request/packet 区分与并发流控。业界对应：Helix/vLLM 的 query/partial-result collective 对偶，但 HCP 只能邻接 P2P。VERDICT: DEFER。重访触发：packet4 的 finisher 计算成为瓶颈，且 ring 能稳定承载多 packet。""",
    ),
    ROUTE_Q_ROTATE: (
        "Continuation 路线 E：Q-shard N-1 跳并轮转 activation",
        "deferred",
        """问题：避免 Q-shard return 的最后一跳。方法：Q/O/LSE 加 residual 在 N-1 hops 后停在 predecessor/finisher，由该节点完成本 token shard 的 W_o+MLP，下一层 activation holder 随之旋转。成本约 C_Q_rotate=(N-1)*m(3H+h_q)*b。优势：保持 neighbor-only 线性成本并省回程。代价：多 shard 的 activation ownership 跨层变化，路由、重组和最终 token 顺序更复杂；如果不传 residual 就无法精确完成 residual connection。VERDICT: DEFER。重访触发：Q-return 路线已证明但最后一跳显著，或需要更细粒度分散 finisher 计算。""",
    ),
    ROUTE_SPLIT_PARTIAL: (
        "Continuation 组合技术 F：历史 prefix 与新增 segment 分拆 partial",
        "deferred",
        """问题：continuation 的历史 T 对所有新增 query 非因果可见，而新增 m 内部需要 causal mask。方法：每个节点分别计算 historical-prefix non-causal partial 与 new-segment causal partial，再用 O/LSE 合并；永久 KV 与通信路线不变。业界对应：SGLang FlashInfer ragged extend 的 o1/s1 与 o2/s2 safe merge。优势：可能复用更快的专用 kernel，且语义清楚。代价：两次 kernel/merge、需要区分 local history 与 current positions，当前 positioned whole-shard kernel 已经正确，无性能证据。VERDICT: DEFER AS COMPOSABLE KERNEL OPTION。重访触发：m>1 packet 正确后，profile 显示 whole positioned causal kernel 是瓶颈。""",
    ),
    ROUTE_HYBRID: (
        "Continuation 路线 G：阶段显式混合",
        "candidate",
        """问题：KV-ring 成本随 T+m，activation packet 成本随 m，不存在全区间单一最优路线。方法：保留两种显式执行模式；initial prefill 默认 KV-ring，decode 默认 KV-stationary，continuation 在实验中按固定场景选择，未来才考虑基于 T/m、链路与算力的选择器。优势：保留两个阶段各自强项，可组合 split-partial、packet 压缩或 query-shard 子路线。代价：两套经过验证的路径和一致 cache 合同增加维护面。默认单路线的作用是简化系统；牺牲它会增加测试矩阵。对 HCP 的意义：positioned cache 格式统一后，路线切换不要求迁移永久 KV，组合价值高。VERDICT: KEEP AS CANDIDATE, NO DYNAMIC PLANNER NOW。重访触发：packet4 与 KV-ring 在相同模型/硬件上获得可比较数据。""",
    ),
}


def upsert_node(conn, node_id, node_type, layer, title, content, status, importance, confidence):
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
        (node_id,node_type,layer,PROJECT,title,content,importance,confidence,status,SOURCE),
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
    conn=sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")

    upsert_node(conn,PORTFOLIO,"task","active","Continuation prefill 多路线实验组合空间",PORTFOLIO_CONTENT,"ongoing",1.0,1.0)
    upsert_node(conn,USER_PREFERENCE,"preference","active","带宽受限异构环境优先探索历史 KV 原地",PREFERENCE_CONTENT,"held",1.0,1.0)
    upsert_node(conn,NEXT_TASK,"task","active","先验证 m>1 LayerPacket 的单层完整合同",
                "下一小节点只推广 LayerPacket 的整层 shape 与 positioned causal 合同。历史 KV 保持在 ReservedPositionedKvShard；只验证一层 synthetic correctness 和 payload 不随 T 增长。整段新 K/V 暂由单一 assignee append，因此不声称 capacity-weighted placement 完成；不接 wire/runtime、多请求或动态 planner。", "active",1.0,1.0)
    upsert_node(conn,DECISION,"decision","active","先完成不传历史 KV 的 m>1 单层 packet 合同",DECISION_CONTENT,"held",1.0,1.0)

    for route_id,(title,status,content) in ROUTES.items():
        confidence=0.99 if route_id in (ROUTE_KV_RING,ROUTE_PACKET4,ROUTE_SPLIT_PARTIAL) else 0.9
        upsert_node(conn,route_id,"hypothesis","blueprint",title,content,status,0.95,confidence)

    edges=[
        (DECISION,NEXT_TASK,"PART_OF","approved pre-action motivation analysis"),
        (METHOD_PREFERENCE,DECISION,"GOVERNS","required six-question and sacrifice analysis"),
        (USER_PREFERENCE,DECISION,"GOVERNS","bandwidth-first route priority"),
        (NEXT_TASK,PORTFOLIO,"PART_OF","first experiment in the retained route portfolio"),
        (NEXT_TASK,RESEARCH_TASK,"DEPENDS_ON","research established the route boundary"),
        (NEXT_TASK,BATCHED_CONTRACT,"DEPENDS_ON","reuses verified m>1 positioned accumulator math"),
        (NEXT_TASK,ROUTE_PACKET4,"DEPENDS_ON","tests the first contract of the active route"),
        (RESEARCH_EVIDENCE,PORTFOLIO,"SUPPORTS","official sources and formulas justify retaining alternatives"),
        (ROUTE_PACKET4,EQUIVALENCE,"DEPENDS_ON","stationary packet relies on exact shard merge"),
        (ROUTE_KV_RING,EQUIVALENCE,"DEPENDS_ON","KV movement computes the same exact merge"),
        (ROUTE_PACKET3,ROUTE_PACKET4,"REFINES","removes normalized payload by repeated norm"),
        (ROUTE_Q_RETURN,ROUTE_PACKET4,"ALTERNATIVE_TO","trades full activation packet for multiple returning Q shards"),
        (ROUTE_Q_ROTATE,ROUTE_Q_RETURN,"REFINES","removes return hop by rotating activation ownership"),
        (ROUTE_SPLIT_PARTIAL,ROUTE_PACKET4,"COMBINES_WITH","optional local kernel decomposition"),
        (ROUTE_SPLIT_PARTIAL,ROUTE_Q_RETURN,"COMBINES_WITH","optional local kernel decomposition"),
        (ROUTE_HYBRID,ROUTE_KV_RING,"COMBINES","retains KV ring for large m/T"),
        (ROUTE_HYBRID,ROUTE_PACKET4,"COMBINES","retains stationary packet for large T/m"),
    ]
    for route_id in ROUTES:
        edges.append((route_id,PORTFOLIO,"PART_OF","retained route for future comparison"))
        edges.append((route_id,COST_BELIEF,"DEPENDS_ON","uses the common byte-hop model"))
    for edge in edges:
        upsert_edge(conn,*edge)

    conn.commit()

    for node_id,(_,status,content) in ROUTES.items():
        got=conn.execute("SELECT status,content FROM nodes WHERE id=?",(node_id,)).fetchone()
        assert got==(status,content)
    assert conn.execute("SELECT content FROM nodes WHERE id=?",(DECISION,)).fetchone()[0]==DECISION_CONTENT
    assert conn.execute("SELECT status FROM nodes WHERE id=?",(NEXT_TASK,)).fetchone()[0]=="active"
    edge_count=conn.execute("SELECT COUNT(*) FROM edges WHERE target=?",(PORTFOLIO,)).fetchone()[0]
    assert edge_count>=8
    conn.close()

    print("routes_exact=7")
    print("motivation_exact=1")
    print("next_task=active")
    print(f"portfolio_incoming_edges={edge_count}")


if __name__=="__main__":
    main()
