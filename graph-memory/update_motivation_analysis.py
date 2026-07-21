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

# ---------------------------------------------------------------------------
# Preference: 行动前动机剖析六问(用户确立的工作方式规则)
# ---------------------------------------------------------------------------
insert_node(
    'preference-motivation-analysis-20260721', 'preference', 'active',
    '工作方式规则：任何工作开始前先做动机剖析六问',
    '用户确立的通用工作方式(2026-07-21，适用于优化工作与普通工作)：开始任何一项工作前，'
    '必须先能回答六个问题，并把答案写进对应 decision/task 节点的 content(或 commit message)：\n'
    '1. 面对什么问题——要解决的问题/缺口是什么；\n'
    '2. 现状是什么——当前代码/系统处于什么状态，为什么不够用；\n'
    '3. 做完能怎样——完成后的目标态与可验证标准；\n'
    '4. 其他人怎么做——生态/同行(特别是 vLLM)遇到同样或类似问题时的解法，能否直接复用；\n'
    '5. 我们怎么做——本项目采用的具体方案；\n'
    '6. 为什么我们要这么做——相对第 4 问的现成方案，我们的方案差异在哪、为什么差异是必要的。\n'
    '扩展规则：若工作属于优化/做减法类(丢弃现有行为换速度/显存/简洁)，在六问之外追加牺牲四问'
    '(为什么默认存在/牺牲了什么/被牺牲者的用途/对本项目的意义)，并给出 implement/defer/reject 结论；'
    'reject 也要记录，避免同一想法被重复提出。\n'
    '全局沉淀：该方法论已融入 graph-memory skill 的 "Pre-Action Motivation Analysis" 一节'
    '(含六问→节点/边的映射：DEPENDS_ON 记顺序、belief+证据记外部做法、GOVERNS 关联规则与应用)。'
    '原 optimization-trade-off skill 已按用户决策退役(移入 _removed)，其牺牲四问作为扩展条款并入；'
    '项目 AGENTS.md 对应章节已同步改为动机剖析六问+牺牲扩展。',
    importance=0.9, confidence=0.95, status='held', source='user-direction'
)
insert_edge('preference-motivation-analysis-20260721', 'proj-hetero-cp-ringattn', 'PART_OF')
insert_edge('preference-motivation-analysis-20260721', 'bp-simplicity-principle', 'RELATES_TO', weight=0.8,
            note='动机剖析是简洁性原则的前置：弄清问题与现状才能拒绝投机性复杂度')

# 三个下一步 decision 是该规则的首次完整应用实例
for dec in ('decision-per-request-staging-20260721',
            'decision-ring-paged-kernel-20260721',
            'decision-vllm-plugin-packaging-20260721'):
    insert_edge('preference-motivation-analysis-20260721', dec, 'GOVERNS', weight=0.9,
                note='该 decision 的 content 即按六问结构(现状/动机/别人怎么做/我们怎么做/为什么/目标态)书写')

# ---------------------------------------------------------------------------
# Belief: 动机剖析方法论有效(挂证据链，可被后续证据修订)
# ---------------------------------------------------------------------------
insert_node(
    'belief-motivation-analysis-value-20260721', 'belief', 'active',
    '动机剖析六问能在行动前暴露顺序错误与现成轮子，值得作为默认动作',
    '首次完整应用(2026-07-21，vLLM 线三个下一步)即产生两类实质收益：\n'
    '(a) 暴露顺序错误——原记录顺序 1→2→3(插件化→kernel→staging)，剖析依赖后修正为 3→2→1'
    '(staging 是数据结构地基，kernel 化需按请求取 staging，插件配置面最后冻结)，避免返工；\n'
    '(b) 暴露现成轮子——"别人怎么做"一问发现 vLLM cascade attention 与 HCP local+peer LSE merge '
    '数学同构、AttentionMetadata/connector metadata 本就按请求组织，两步工作都可直接复用框架机制 '
    '而非自造。\n'
    '代价：每项工作启动前增加约一次剖析的固定开销。对多步骤、跨系统的工作收益大于开销；'
    '对单行修复类琐碎工作可从简。',
    importance=0.85, confidence=0.8, status='held', source='experiment'
)
insert_edge('belief-motivation-analysis-value-20260721', 'preference-motivation-analysis-20260721', 'SUPPORTS', weight=0.9,
            note='支撑该规则值得保留为默认工作方式')
for dec in ('decision-per-request-staging-20260721',
            'decision-ring-paged-kernel-20260721',
            'decision-vllm-plugin-packaging-20260721'):
    insert_edge(dec, 'belief-motivation-analysis-value-20260721', 'SUPPORTS', weight=0.8,
                note='剖析产出：修正执行顺序为 3→2→1 / 发现 cascade attention 与 per-request metadata 可复用')

conn.commit()
conn.close()
print('motivation-analysis methodology recorded in graph.db')
