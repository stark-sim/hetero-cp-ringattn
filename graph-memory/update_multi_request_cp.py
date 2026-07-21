import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
conn = sqlite3.connect(DB)
c = conn.cursor()

def insert_node(id_, type_, layer, title, content, importance=0.7, confidence=0.7, status='open', source='experiment'):
    c.execute('''
        INSERT OR REPLACE INTO nodes
        (id, type, layer, project, title, content, importance, confidence, status, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    ''', (id_, type_, layer, PROJECT, title, content, importance, confidence, status, source))

def insert_edge(src, tgt, type_, weight=1.0, note=None):
    c.execute('''
        INSERT OR REPLACE INTO edges (source, target, type, weight, note)
        VALUES (?, ?, ?, ?, ?)
    ''', (src, tgt, type_, weight, note))

# Evidence: multi-request (continuous-batching) CP path validated
insert_node(
    'ev-ring-multi-request-cp-20260721', 'evidence', 'progress',
    '[2026-07-21] 多请求并发 CP 路径验证通过：staging 按 chunk 键 + 每请求 kv_transfer_params',
    '决策 decision-per-request-staging-20260721 的实现与验证(commit ec8e528..8bb2553)。\n'
    '实现：\n'
    '1. PEER_KV_STAGING 键从 layer 改为 (chunk_key, layer)；新增 PEER_REQ_MAP 以请求首块 id '
    '(生命周期内稳定)绑定请求→chunk，forward 按 batch 行查 peer KV；\n'
    '2. 每请求参数走 vLLM 原生通道 SamplingParams.extra_args.kv_transfer_params.hcp_ring '
    '(chunk_id/prefix_len/peer_url)，全局 extra_config 保留为回退；显式 prefix_len=0 可退出 CP '
    '(存在性覆盖，非真值覆盖)——非 CP 请求与 CP 请求可同引擎混跑；\n'
    '3. staged KV 按 chunk 引用计数，请求结束时释放；清理在携带 finished_req_ids 的那一步 '
    'forward 之前执行(connector metadata 携带)，shutdown() 兜底。\n'
    '验证(pearl, ROCm)：\n'
    '- validate_ring_concurrent.py：2 请求(各 1024 token、不同 prompt、各挂 peer chunk c0/c1)一次 '
    'generate(max_num_seqs=2)，token 与单节点参考全对；STAGING_STATS 显示 2 chunk×24 层并发 staging；'
    'BATCH_STATS.max_reqs=2(CP 路径真进连续批)；chunk-A 槽位本地写入 0；结束后 staging 清空。PASS。\n'
    '- validate_ring_connector.py 单请求回归(2048/1024)：PASS。\n'
    '跨节点复验(ringconc-232830, scripts/run_cross_node_ring_concurrent.sh, HEAD=8bb2553 双机一致)：\n'
    'white(RTX 4090 CUDA) producer 算 2 个 chunk(c0/c1, 各 512 token)并经 HTTP 供取；'
    'pearl(ROCm gfx1200) consumer 一次 generate 提交 2 个全 prompt(1024 token),'
    '每请求经 kv_transfer_params 各挂各的 peer chunk。结果：双请求 token 与 pearl 单节点参考全对；\n'
    '2 chunk×24 层并发 staging(经 HTTP 来自 white, producer 日志 GET 来源 100.111.242.55)；\n'
    'BATCH_STATS.max_reqs=2(跨节点 CP 路径进连续批)；chunk-A 槽位本地写入 0；staging 用后释放。PASS。\n'
    '排障记录(有复用价值)：\n'
    '- 块 id 复用竞态(真修复)：finished 请求清理原在 forward 后(get_finished)，回收首块的新请求可能被绑到 '
    '过期 chunk；改为 metadata 携带 finished ids、start_load_kv 里 forward 前清理；\n'
    '- 768 overlap 误报(非 bug)：验证脚本遗留 HCP_RING_SPLIT_TOKENS=1024 使短非 CP 请求落入 env-split '
    '分支，WRITE_TRACK 把其自身写入误记为 peer 区域污染(32 槽×24 层)；attention 经 n_a 分支始终正确。'
    'connector 验证脚本已显式置 0。',
    importance=0.9, confidence=0.95, status='held', source='experiment'
)
insert_edge('ev-ring-multi-request-cp-20260721', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('ev-ring-multi-request-cp-20260721', 'decision-per-request-staging-20260721', 'CONFIRMS', weight=0.95,
            note='按决策实施并验证通过；顺序 3→2→1 的第 3 步完成')
insert_edge('ev-ring-multi-request-cp-20260721', 'decision-ring-paged-kernel-20260721', 'ENABLES', weight=0.85,
            note='per-request staging 数据结构就位,kernel 化(第 2 步)可在其上按请求取 peer KV')

# Close the per-request staging decision as implemented
c.execute("""
    UPDATE nodes SET status = 'closed', updated_at = datetime('now')
    WHERE id = 'decision-per-request-staging-20260721'
""")

# Lesson: 调试探针(WRITE_TRACK)与环境回退路径(env-split)的交互会产生误报
insert_node(
    'lesson-debug-probe-interaction-20260721', 'lesson', 'progress',
    '[2026-07-21] 调试探针的假设要与所有激活路径核对，尤其遗留回退路径',
    'WRITE_TRACK 假设"peer 区域槽位绝不被本地写入"，该假设只对 connector-staged 路径成立；'
    '遗留的 HCP_RING_SPLIT_TOKENS env-split 路径(单进程 PoC 用)故意从本地 cache 读 peer,'
    '两条路径同时激活时探针把正常写入报成污染(768)。教训：\n'
    '1. 新增验证手段时，穷举它会影响的所有代码路径(含遗留回退)，逐路径核对假设；\n'
    '2. 验证脚本应显式固定行为相关环境变量(如 HCP_RING_SPLIT_TOKENS=0)，不依赖默认值；\n'
    '3. 出现"数值异常但结果正确"时，先怀疑探针假设，再怀疑被测逻辑——token 全对 + overlap 异常 '
    '这个组合本身就是探针误报的特征。',
    importance=0.8, confidence=0.9, status='held', source='reflection'
)
insert_edge('lesson-debug-probe-interaction-20260721', 'ev-ring-multi-request-cp-20260721', 'BASED_ON', weight=0.9)
insert_edge('lesson-debug-probe-interaction-20260721', 'bp-simplicity-principle', 'RELATES_TO', weight=0.5,
            note='遗留回退路径是简洁性债务：多路径并存提升调试成本')

conn.commit()
conn.close()
print('multi-request CP evidence + lesson recorded in graph.db')
