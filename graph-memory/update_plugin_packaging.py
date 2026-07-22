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

insert_node(
    'ev-vllm-plugin-packaging-20260722', 'evidence', 'progress',
    '[2026-07-22] 第 1 步完成:vLLM 生态插件 v0.1 包装(entry point 自动注册双平台验证)',
    '决策 decision-vllm-plugin-packaging-20260721 的实施(commit c5e95a5)。\n'
    '交付:\n'
    '1. hcp_vllm_plugin/README.md:组件表、安装、兼容性(vLLM 0.23.1rc1 + torch 2.13,CUDA/ROCm 验证;'
    '依赖接口面明示:KVConnectorBase_V1 experimental / backend 注册表 / merge_attn_states / triton_utils)、'
    '快速开始(producer/consumer 配置 + 每请求 kv_transfer_params)、环境变量表、v0.1 限制清单'
    '(单 peer chunk、consumer 关 prefix caching、fp16/bf16、eager、kernel-hardening backlog 指向)、验证脚本索引;\n'
    '2. hcp_vllm_plugin/compat_check.py:免 engine 冒烟——vllm 版本、KVConnectorBase_V1 方法面、'
    'CUSTOM 注册表、merge_attn_states、KVConnectorFactory、register() 执行、插件模块导入;'
    'pearl 与 white 均 PASS(0 warnings);\n'
    '3. entry point 自动注册实证:探针 engine 不传 kv_connector_module_path,仅凭 '
    'kv_connector="HcpRingKvConnector" 解析成功并生成 token(pearl + white 均过),'
    'vllm.general_plugins 入口真正生效;\n'
    '4. 包 docstring 更新为 ring memory-splitting 语义(原描述停留在全量 context-passing 时代)。\n'
    '至此三步顺序(3 staging→2 kernel→1 packaging)全部完成,vLLM 线具备:'
    '可 pip install 的插件形态 + 双平台 triton kernel + 多请求连续批 CP + 跨节点异构闭环。',
    importance=0.9, confidence=0.95, status='held', source='experiment'
)
insert_edge('ev-vllm-plugin-packaging-20260722', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('ev-vllm-plugin-packaging-20260722', 'decision-vllm-plugin-packaging-20260721', 'CONFIRMS', weight=0.95,
            note='按决策实施;顺序 3→2→1 全部收官')

# Close the packaging decision and the parent focus task
c.execute("""
    UPDATE nodes SET status = 'closed', updated_at = datetime('now')
    WHERE id = 'decision-vllm-plugin-packaging-20260721'
""")
c.execute("""
    UPDATE nodes SET status = 'closed', updated_at = datetime('now')
    WHERE id = 'active-vllm-ecosystem-plugin-20260721'
""")
insert_edge('active-vllm-ecosystem-plugin-20260721', 'ev-vllm-plugin-packaging-20260722', 'RESOLVED_BY', weight=1.0,
            note='三步(staging/kernel/packaging)全部完成,焦点任务收官;后续见 task-kernel-hardening-backlog-20260721')

conn.commit()
conn.close()
print('packaging evidence recorded; step-1 and parent focus closed')
