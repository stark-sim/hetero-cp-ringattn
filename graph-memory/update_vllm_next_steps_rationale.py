import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
conn = sqlite3.connect(DB)
c = conn.cursor()

def insert_node(id_, type_, layer, title, content, importance=0.7, confidence=0.7, status='open', source='user-input'):
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

PARENT = 'active-vllm-ecosystem-plugin-20260721'

# ---------------------------------------------------------------------------
# Beliefs: 对外部世界（vLLM）的理解，挂证据链，可被 revision 推翻
# ---------------------------------------------------------------------------
insert_node(
    'belief-vllm-cascade-attn-20260721', 'belief', 'active',
    'vLLM cascade attention 与 HCP local+peer LSE merge 数学同构',
    'vLLM 的 cascade attention 在多请求共享前缀时，对共享前缀与各请求私有后缀分别算 attention，'
    '再用各自的 LSE(logsumexp) 合并。HCP ring backend 的 local(chunk B, causal) + peer(chunk A, '
    'non-causal) LSE merge 是同一种数学，只是合并的两段住在不同节点上。'
    '此外 FlashAttention kernel 本身支持输出 LSE(white 上 vendored FA 已验证含 LSE)。'
    '推论：HCP 的 ring merge 可以换成"两个原生 kernel + 一次 cascade 式合并"，不需要自写 attention 数学。',
    importance=0.85, confidence=0.8, status='held', source='code-reading'
)
insert_edge('belief-vllm-cascade-attn-20260721', 'ev-ring-cross-node-split-cp-20260721', 'BASED_ON', weight=0.7,
            note='ring_backend.py 的 _lse_merge 即 cascade 合并形状，已跨节点验证正确')

insert_node(
    'belief-vllm-pd-full-kv-20260721', 'belief', 'active',
    'vLLM 官方长上下文分布路线是 disaggregated prefill(全量 KV 搬移)',
    'vLLM 官方对"长上下文分布式"的答案是 P/D 分离：prefill 节点算完全量 KV，整体搬给 decode 节点。'
    '该路线每个节点都必须容纳全量 KV；HCP 切分 CP 不需要——各节点只持有自己 chunk 的 KV，'
    'peer chunk 仅以瞬时 staging 参与计算。这是 HCP 相对 vLLM 官方路线的差异化价值，'
    '也是三步工程化值得做的原因：把差异化的正确性证明变成差异化的可用能力。',
    importance=0.85, confidence=0.85, status='held', source='code-reading'
)
insert_edge('belief-vllm-pd-full-kv-20260721', 'decision-ring-kv-connector-split-20260717', 'BASED_ON', weight=0.9,
            note='全量搬移 vs 切分瞬时的区分决策')

insert_node(
    'belief-connector-api-experimental-20260721', 'belief', 'active',
    'KVConnectorBase_V1 是 experimental API，插件边界收敛才能跟进 vLLM 升级',
    'vLLM 运行日志明示 "KVConnectorBase_V1. This API is experimental and subject to change"。'
    'HCP 对 vLLM 的依赖面 = attention backend 注册表(CUSTOM) + KV connector 接口两个扩展点。'
    '不收敛成干净插件边界，vLLM 升级可能悄悄破坏兼容性且无人发现；收敛后每次升级跑一遍兼容性验证即可。',
    importance=0.8, confidence=0.9, status='held', source='experiment'
)
insert_edge('belief-connector-api-experimental-20260721', 'ev-ring-cross-node-split-cp-20260721', 'BASED_ON', weight=0.8,
            note='consumer.log 中 vllm 官方 warning 原文')

# ---------------------------------------------------------------------------
# Decisions: 三个下一步的动机剖析（现状 / 为什么做 / vLLM 怎么做 / 做完什么样）
# ---------------------------------------------------------------------------
insert_node(
    'decision-per-request-staging-20260721', 'decision', 'active',
    '步骤3(先做)：peer KV staging 从全局 dict 改为按请求键，解除单并发限制',
    '【现状】PEER_KV_STAGING 是全局 dict，键为 layer 名 => 同一时刻全引擎只能有一份 peer KV，'
    '所有请求共享同一 peer chunk。故 PoC 强制 max_num_seqs=1，consumer 必须关 prefix caching。'
    '这是正确性限制，不是性能限制；第 2 步验证的连续批走的是非 CP 路径，CP 路径无并发能力。\n'
    '【动机】continuous batching 是 vLLM 存在的意义，单并发只是演示品；且这是 N 节点真 ring 的前提——'
    'N 个 chunk 时每请求要挂多个 peer 块，全局 dict 结构上不支持。\n'
    '【vLLM 怎么做】框架本就提供按请求携带状态的通道：build_connector_meta 产出的 connector metadata '
    '按请求组织(RingReqMeta 已在其中)；AttentionMetadata(seq_lens/block_table/query_start_loc) 全是按请求索引的 '
    'batched 结构。PoC 用全局 dict 是抄近路，不是框架缺能力。\n'
    '【目标态】staging 键从 layer 改为 (request_id, layer)，metadata 携带每请求的 peer chunk 列表，'
    'forward 按请求取各自 peer KV。CP 路径进入连续批，max_num_seqs 不限 1，为 N 节点真 ring 铺平数据结构。',
    importance=0.9, confidence=0.85, status='held', source='user-direction'
)
insert_edge('decision-per-request-staging-20260721', PARENT, 'PART_OF')
insert_edge('decision-per-request-staging-20260721', 'ev-hcp-ring-connector-20260721', 'BASED_ON', weight=0.9,
            note='PoC 验证中确立的限制(max_num_seqs=1、关 prefix caching)')

insert_node(
    'decision-ring-paged-kernel-20260721', 'decision', 'active',
    '步骤2(次做)：ring attention 从 plain-PyTorch 换成原生 paged kernel + cascade 式 LSE 合并',
    '【现状】ring_backend._attn_with_lse 为自写 plain PyTorch fp32：每请求每层把 K/V 从 paged cache gather '
    '成连续张量，einsum 物化完整 score 矩阵 [H, Tq, Tk] 再手动 softmax+LSE。2048 token 可跑，但 score 矩阵显存 '
    '随长度平方增长，128K/1M(HCP 卖点)直接爆显存；fp32 无 kernel 融合，速度差原生一个量级。\n'
    '【动机】显存切分省下的显存会被自实现低效吃回去；不长上下文，跨节点能力无实用价值。\n'
    '【vLLM 怎么做】(a) PagedAttention：KV 按 block 分页，kernel 以 block table 为索引直接读分页内存，'
    '不 gather、不物化 score 矩阵，内部本即 online softmax 分块；(b) cascade attention：与 HCP merge 数学同构'
    '(见 belief-vllm-cascade-attn-20260721)；(c) FlashAttention kernel 支持输出 LSE。\n'
    '【目标态】chunk B 走 vLLM 原生 paged kernel(带 LSE)，chunk A 对 staging buffer 跑一次 flash kernel(带 LSE)，'
    '再一次 LSE merge。score 矩阵不再物化，长度天花板消失，速度接近原生。',
    importance=0.9, confidence=0.85, status='held', source='user-direction'
)
insert_edge('decision-ring-paged-kernel-20260721', PARENT, 'PART_OF')
insert_edge('decision-ring-paged-kernel-20260721', 'belief-vllm-cascade-attn-20260721', 'BASED_ON', weight=0.9)
insert_edge('decision-ring-paged-kernel-20260721', 'decision-per-request-staging-20260721', 'DEPENDS_ON', weight=0.85,
            note='kernel 化时需按请求取 staging，先改数据结构再换 kernel 冲突最少')

insert_node(
    'decision-vllm-plugin-packaging-20260721', 'decision', 'active',
    '步骤1(最后做)：从研究脚本收敛为标准 vLLM 生态插件',
    '【现状】插件能工作但形态是研究脚本：pip install -e 本仓库、手写长 kv_transfer_config dict、'
    '环境变量控制行为；验证脚本硬编码模型路径与节点 IP；无版本兼容声明。\n'
    '【动机】KVConnectorBase_V1 是 experimental API(见 belief-connector-api-experimental-20260721)，'
    '不收敛插件边界则 vLLM 升级可能悄悄破坏兼容性；收敛后别人 pip install + 两个参数即可获得异构长上下文能力。\n'
    '【vLLM 怎么做】官方答案就是插件：两条标准扩展面(KV connector 接口 + attention backend 注册表)，'
    'NIXL connector / LMCache / Mooncake 均走同一 KVConnectorBase_V1 接口，无人 fork 内核；'
    '我们的 ring backend 注册在 CUSTOM，与官方后端机制平级。此步非发明新东西，是打磨已在正确接口上的代码。\n'
    '【目标态】entry points 自动注册、配置项收敛为文档化的少数键、声明兼容的 vLLM 版本区间、留最小可跑示例；'
    'vLLM 升级时跑兼容性验证脚本即知坏没坏。\n'
    '【为何最后】插件定义的对外配置面应等 staging(3)与 kernel(2)定型后再冻结，避免刚发布就改配置。',
    importance=0.9, confidence=0.85, status='held', source='user-direction'
)
insert_edge('decision-vllm-plugin-packaging-20260721', PARENT, 'PART_OF')
insert_edge('decision-vllm-plugin-packaging-20260721', 'belief-connector-api-experimental-20260721', 'BASED_ON', weight=0.9)
insert_edge('decision-vllm-plugin-packaging-20260721', 'belief-vllm-pd-full-kv-20260721', 'BASED_ON', weight=0.8,
            note='插件化是差异化能力(P/D 全量 vs HCP 切分)对外交付的形态')
insert_edge('decision-vllm-plugin-packaging-20260721', 'decision-ring-paged-kernel-20260721', 'DEPENDS_ON', weight=0.7,
            note='对外配置面等 kernel/staging 定型后冻结')

# 更新父任务：固化 3→2→1 顺序及理由
parent_append = (
    '\n[2026-07-21 更新] 执行顺序修正为 3→2→1(原记录 1→2→3)：'
    'per-request staging 是地基(数据结构正确性)；paged kernel 化建在其上(按请求取 staging)；'
    '插件化最后(对外配置面等二者定型再冻结)。三者动机剖析已落 decision 节点：'
    'decision-per-request-staging-20260721 / decision-ring-paged-kernel-20260721 / '
    'decision-vllm-plugin-packaging-20260721。'
)
c.execute("""
    UPDATE nodes SET content = content || ?, updated_at = datetime('now')
    WHERE id = ?
""", (parent_append, PARENT))

conn.commit()
conn.close()
print('3 decisions + 3 beliefs + ordering recorded in graph.db')
