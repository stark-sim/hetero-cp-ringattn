import sqlite3
from pathlib import Path

DB = Path('graph-memory/graph.db')
OUT = Path('graph-memory')
PROJECT = 'hetero-cp-ringattn'

def query(sql, params=()):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(sql, params)
    rows = c.fetchall()
    conn.close()
    return rows

def fmt_node(n):
    lines = [f"### {n['title']}\n"]
    meta = []
    if n['type']:
        meta.append(f"type: `{n['type']}`")
    if n['status']:
        meta.append(f"status: `{n['status']}`")
    if n['confidence'] is not None:
        meta.append(f"confidence: {n['confidence']}")
    if n['importance'] is not None:
        meta.append(f"importance: {n['importance']}")
    if n['source']:
        meta.append(f"source: `{n['source']}`")
    if meta:
        lines.append(' · '.join(meta) + '\n')
    if n['content']:
        lines.append(n['content'] + '\n')
    lines.append(f"_updated: {n['updated_at']}_\n")
    return '\n'.join(lines)

# active.md
active = query('''
    SELECT * FROM nodes
    WHERE layer = 'active' AND project = ? AND status != 'closed'
    ORDER BY importance DESC, updated_at DESC
''', (PROJECT,))
with open(OUT / 'active.md', 'w') as f:
    f.write('# Active Context\n\n')
    f.write('当前活跃的任务、决策、风险和假设。\n\n')
    for n in active:
        f.write(fmt_node(n))

# progress.md
progress = query('''
    SELECT * FROM nodes
    WHERE layer = 'progress' AND project = ?
    ORDER BY created_at DESC
''', (PROJECT,))
with open(OUT / 'progress.md', 'w') as f:
    f.write('# Progress Timeline\n\n')
    f.write('按时间倒序排列的重要进展、实验和学到的教训。\n\n')
    for n in progress:
        f.write(fmt_node(n))

# systemPatterns.md
sys = query('''
    SELECT * FROM nodes
    WHERE layer = 'blueprint' AND project = ?
      AND type IN ('blueprint', 'decision', 'belief', 'component', 'api', 'assumption')
    ORDER BY type, importance DESC
''', (PROJECT,))
with open(OUT / 'systemPatterns.md', 'w') as f:
    f.write('# System Patterns\n\n')
    f.write('架构概览、关键设计模式与架构决策。\n\n')
    for n in sys:
        f.write(fmt_node(n))

# techContext.md
tech = query('''
    SELECT * FROM nodes
    WHERE layer = 'blueprint' AND project = ?
      AND type IN ('component', 'api', 'dependency', 'claim')
    ORDER BY type, importance DESC
''', (PROJECT,))
with open(OUT / 'techContext.md', 'w') as f:
    f.write('# Tech Context\n\n')
    f.write('技术栈、依赖与关键实现细节。\n\n')
    for n in tech:
        f.write(fmt_node(n))

# productContext.md
prod = query('''
    SELECT * FROM nodes
    WHERE layer = 'blueprint' AND project = ?
      AND type IN ('fact', 'decision', 'belief')
    ORDER BY type, importance DESC
''', (PROJECT,))
with open(OUT / 'productContext.md', 'w') as f:
    f.write('# Product Context\n\n')
    f.write('产品问题、目标用户、成功标准与产品决策。\n\n')
    for n in prod:
        f.write(fmt_node(n))

print('Exported views generated')
