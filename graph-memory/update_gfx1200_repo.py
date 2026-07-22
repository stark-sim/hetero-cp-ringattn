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
    'ev-gfx1200-repo-20260722', 'evidence', 'progress',
    '[2026-07-22] gfx1200 适配 repo 整理完成:vllm-rocm-gfx1200(private),解耦全部落地',
    'github.com/stark-sim/vllm-rocm-gfx1200(private)。\n'
    '内容:5 个补丁(从 pearl /home/stark/vllm 源码树 git diff 提取,base commit 3f99883d9 '
    'v0.23.1rc1.dev905):\n'
    '0001 spinloop.cpp 改 x86intrin(ROCm Clang 23 编译错误);0002 禁用 GPTQ(HIP 缺 half2 atomicAdd);'
    '0003 ROCm 平台识别 torch.version.hip 兜底(amdsmi 不可用);0004 _get_gcn_arch 走 torch.cuda + '
    'HCP_ROCM_GCN_ARCH 覆盖;0005 pyproject 解除 torch==2.11.0 钉版。'
    '外加构建脚本(clone→checkout→apply→pip install -e)、LD_LIBRARY_PATH 运行 wrapper、README 兼容性矩阵。\n'
    'pearl 迁移:插件 clone /home/stark/hcp-vllm-plugin + pip install -e 重装,compat_check 6/6 PASS;'
    'pearl GitHub 访问走已有用户 SSH key(known_hosts 补齐),跨网段 ssh 不稳时经 white(192.168.8.176)跳转。\n'
    '两个产品 repo 至此全部独立: hcp-vllm-plugin + vllm-rocm-gfx1200;主仓=研究/驱动/知识库。',
    importance=0.85, confidence=0.95, status='held', source='experiment'
)
insert_edge('ev-gfx1200-repo-20260722', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('ev-gfx1200-repo-20260722', 'decision-repo-decoupling-20260722', 'CONFIRMS', weight=0.95,
            note='解耦决策全部落地(两个 repo + 双机迁移)')

c.execute("""
    UPDATE nodes SET status = 'closed', updated_at = datetime('now')
    WHERE id = 'task-gfx1200-repo-extraction'
""")
insert_edge('task-gfx1200-repo-extraction', 'ev-gfx1200-repo-20260722', 'RESOLVED_BY', weight=1.0)

conn.commit()
conn.close()
print('gfx1200 repo + decoupling completion recorded')
