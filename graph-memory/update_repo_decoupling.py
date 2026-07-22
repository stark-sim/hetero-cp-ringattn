import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
PROJECT = 'hetero-cp-ringattn'
conn = sqlite3.connect(DB)
c = conn.cursor()

def insert_node(id_, type_, layer, title, content, importance=0.7, confidence=0.7, status='open', source='user-direction'):
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
    'decision-repo-decoupling-20260722', 'decision', 'active',
    '决策：两个产品级产出解耦为独立 GitHub repo(private),主仓保留研究/驱动/知识库',
    '用户方向(2026-07-22):HCP vLLM 插件与 gfx1200 适配是两个不同生命周期的产出,'
    '独立成 repo 比保留子文件夹更清晰。\n'
    '执行:\n'
    '1. hcp_vllm_plugin/ 经 git subtree split 带全部 23 个 commit 历史切出 => '
    'github.com/stark-sim/hcp-vllm-plugin(private, main);clone 验证(文件/历史/语法)通过;\n'
    '2. 主仓删除 hcp_vllm_plugin/ 避免双源漂移,新增根 README.md 仓库地图;'
    '跨节点驱动脚本改为读 *_PLUGIN_REPO(默认 /home/stark/hcp-vllm-plugin);\n'
    '3. white 已迁移:/home/stark/hcp-vllm-plugin clone + pip install -e 重装,import 验证通过;\n'
    '4. 第二 repo(gfx1200 适配)待 pearl 恢复后从 /home/stark/vllm 源码树整理补丁。\n'
    '主仓定位:Rust/Python 调度核心、transformers 线、跨节点驱动、graph-memory、docs/reports。',
    importance=0.9, confidence=0.95, status='held', source='user-direction'
)
insert_edge('decision-repo-decoupling-20260722', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('decision-repo-decoupling-20260722', 'ev-vllm-plugin-packaging-20260722', 'BASED_ON', weight=0.85,
            note='插件 v0.1 定型后才适合独立成 repo')
insert_edge('decision-repo-decoupling-20260722', 'bp-simplicity-principle', 'RELATES_TO', weight=0.7,
            note='产物边界=生命周期边界,不为"可能复用"而保持单体')

insert_node(
    'task-gfx1200-repo-extraction', 'task', 'active',
    '待办:gfx1200 适配 repo 整理(等 pearl 恢复可达)',
    'pearl(Tailscale 100.111.242.55 / LAN 192.168.8.176)当前不可达(ping 100% 丢包)。'
    '恢复后:\n'
    '1. 盘点 /home/stark/vllm 源码树的本地 patch(git status/diff vs upstream tag)及构建/运行脚本;\n'
    '2. 整理为 github.com/stark-sim/vllm-rocm-gfx1200(private):补丁、构建脚本、LD_LIBRARY_PATH wrapper、'
    '兼容性说明(torch 2.13+rocm7.13 / gfx1200);\n'
    '3. pearl 迁移插件 clone:/home/stark/hcp-vllm-plugin + pip install -e 重装 + import 验证;\n'
    '4. 双机跑 compat_check + 一次跨节点并发验证确认迁移无损。',
    importance=0.8, confidence=0.9, status='blocked', source='user-direction'
)
insert_edge('task-gfx1200-repo-extraction', 'decision-repo-decoupling-20260722', 'PART_OF')

conn.commit()
conn.close()
print('repo decoupling recorded in graph.db')
