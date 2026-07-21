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

# Evidence: cross-node heterogeneous ring split-CP via KV connector
insert_node(
    'ev-ring-cross-node-split-cp-20260721', 'evidence', 'progress',
    '[2026-07-21] 异构跨节点切分 CP 验证通过：white(CUDA) producer + pearl(ROCm) consumer 经 HcpRingKvConnector',
    'run_id=ringx-210415，驱动 scripts/run_cross_node_ring_cp.sh，HEAD=cce069e（双机一致）。\n'
    '拓扑：white(RTX 4090, vllm-v1) 以 CUSTOM backend(HcpRingAttentionBackend)+HcpRingKvConnector(role=producer) '
    '只算 chunk A(2048-token prompt 的前 1024 token)，24 层 KV 存 safetensors 并经 HTTP(0.0.0.0:8901) 供取；\n'
    'pearl(RX 9060 XT gfx1200, vllm-rocm) 以 CUSTOM backend+HcpRingKvConnector(role=consumer) 跑全 prompt，'
    '调度侧把 chunk A 标 external（全局 RoPE 位置、不重算），worker 侧经 HTTP 把 peer KV 拉进 ring backend 的 '
    'TRANSIENT PEER_KV_STAGING（不写 pearl 常驻 paged pool / block table），online softmax 合并 local(chunk B, causal)+peer(chunk A, transient)。\n'
    '结果：\n'
    '1. greedy 4 token 与 pearl 单节点参考完全一致：ref=[14579,220,22,21] cons=[14579,220,22,21]；\n'
    '2. 末步 logits max|diff|=0.037（阈值 0.1，argmax 处 0.0）；\n'
    '3. 显存切分证据：24/24 层 peer KV 经 HTTP 来自 white（producer 日志 GET 来自 100.111.242.55），'
    '1024 token/层；pearl 本地写 pool 槽位 1027（仅自身 chunk B），chunk-A 区域槽位本地写入 = 0；\n'
    '4. report: reports/ring-cross-ringx-210415/{consumer,producer}.log。\n'
    '意义：三步顺序（flash_attn→decode 充分验证→异构跨节点切分 CP）全部完成；'
    'vLLM worker 对 vLLM worker 组环 + KV connector 瞬时切分路线在真异构跨节点（CUDA↔ROCm）闭环。',
    importance=0.95, confidence=0.95, status='held', source='experiment'
)
insert_edge('ev-ring-cross-node-split-cp-20260721', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('ev-ring-cross-node-split-cp-20260721', 'plan-vllm-line-next-steps-20260721', 'SUPPORTS', weight=0.95,
            note='三步顺序第 3 步（异构跨节点切分 CP）完成，三步全部收官')
insert_edge('ev-ring-cross-node-split-cp-20260721', 'task-vllm-online-softmax-20260717', 'SUPPORTS', weight=0.95,
            note='online softmax 显存切分从单机 2 进程扩展到真跨节点异构')
insert_edge('ev-ring-cross-node-split-cp-20260721', 'ev-hcp-ring-connector-20260721', 'CONFIRMS', weight=0.9,
            note='同一套 connector/backend 从 loopback 2 进程走向 white+pearl 真跨节点')
insert_edge('ev-ring-cross-node-split-cp-20260721', 'hyp-block-kv-vllm', 'SUPPORTS', weight=0.85,
            note='路线 A（插件解耦）可行性进一步增强：KV connector + CUSTOM backend 不改 vLLM 内核')

# Update the 3-step plan node: all steps done
plan_append = (
    '\n[2026-07-21 更新] 三步全部完成：'
    '1) flash_attn 双平台(white vendored FA 含 LSE / pearl TRITON_ATTN+CUSTOM)；'
    '2) decode 充分验证（连续批 6 请求、多步 decode=8/16 全过）；'
    '3) 异构跨节点切分 CP（ringx-210415 PASS，见 ev-ring-cross-node-split-cp-20260721）。'
)
c.execute("""
    UPDATE nodes SET content = content || ?,
        status = 'held', updated_at = datetime('now')
    WHERE id = 'plan-vllm-line-next-steps-20260721'
""", (plan_append,))

# Close the online-softmax memory-split task
c.execute("""
    UPDATE nodes SET status = 'closed', updated_at = datetime('now')
    WHERE id = 'task-vllm-online-softmax-20260717'
""")
insert_edge('task-vllm-online-softmax-20260717', 'ev-ring-cross-node-split-cp-20260721', 'RESOLVED_BY', weight=1.0,
            note='显存切分 online softmax 在跨节点异构环境验证通过，任务收官')

# Next focus: vLLM 生态插件化 + block KV 整合
insert_node(
    'active-vllm-ecosystem-plugin-20260721', 'task', 'active',
    '当前焦点：hetero-cp-ringattn 向 vLLM 生态插件收敛 + PageAttn/block KV 整合',
    '三步顺序完成后，vLLM 线已具备：CUSTOM ring backend(online softmax 显存切分) + '
    'HcpRingKvConnector(切分瞬时 peer KV) + 跨节点异构(CUDA↔ROCm)闭环。\n'
    '下一步（用户给定方向）：\n'
    '1. 把 hetero-cp-ringattn 分布式调度框架整理成标准 vLLM 生态插件（entry points 注册、'
    '配置化、可随 vLLM 官方更新跟进），既有异构长上下文能力又不 fork 内核；\n'
    '2. 整合 PageAttn 与 hetero-cp-ringattn 的 block KV：现在 ring backend 用 plain-PyTorch '
    'fp32 逐请求算 attention，需评估与 vLLM paged attention/flash_attn 内核的融合路径；\n'
    '3. 解除 PoC 限制：PEER_KV_STAGING 按 layer 键限单并发(max_num_seqs=1)，consumer 必须关 prefix caching；'
    '工程化需支持多请求并发 staging（按 request 键）。',
    importance=0.95, confidence=0.85, status='ongoing', source='user-direction'
)
insert_edge('active-vllm-ecosystem-plugin-20260721', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('active-vllm-ecosystem-plugin-20260721', 'ev-ring-cross-node-split-cp-20260721', 'BASED_ON', weight=0.95)
insert_edge('active-vllm-ecosystem-plugin-20260721', 'hyp-block-kv-vllm', 'RELATES_TO', weight=0.85,
            note='插件化收敛 + PageAttn 整合正是路线 A 的工程化落地')

conn.commit()
conn.close()
print('cross-node split-CP evidence + next focus recorded in graph.db')
