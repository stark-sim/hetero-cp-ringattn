# Graph Memory - 通用 Agent 规则

> 本文件适用于所有参与本项目的 AI coding agent。
> 每次会话开始时，都必须读取本文件、读取 `graph-memory/blueprint.md`，并视需要查询 `graph-memory/graph.db`。

## 文件结构

| 文件 / 路径 | 用途 |
|------------|------|
| `graph-memory/RULES.md` | 本文件，规则文件。创建后不可修改 |
| `graph-memory/graph.db` | SQLite 源数据：节点、边、FTS5 全文搜索 |
| `graph-memory/blueprint.md` | 人类可读的项目蓝图，由人维护、由 agent 读取 |
| `graph-memory/active.md` | `active` 层节点的导出视图 |
| `graph-memory/progress.md` | `progress` 层时间线的导出视图 |
| `graph-memory/systemPatterns.md` | `blueprint` 层架构决策的导出视图 |
| `graph-memory/techContext.md` | `blueprint` 层技术栈的导出视图 |
| `graph-memory/productContext.md` | `blueprint` 层产品上下文的导出视图 |

## 记忆模型

- **信念（belief）不是事实**：每个 `belief`、`hypothesis`、`claim` 必须附带 `confidence` 和证据链。
- **证据必须链接**：不要创建孤立的 `evidence` 节点；必须通过 `SUPPORTS` / `CONTRADICTS` / `CONFIRMS` / `REFUTES` / `QUESTIONS` 边连接到信念或决策。
- **矛盾触发修订**：当强证据与现有信念冲突时，创建 `revision` 节点，并用 `SUPERSEDES` 或 `REPLACED_BY` 标记旧信念，而不是直接覆盖。

## 必须遵守的协议

### 会话开始

1. 检查 `graph-memory/` 目录是否存在。
2. 读取 `graph-memory/RULES.md`。
3. 读取 `graph-memory/blueprint.md`，理解项目范围和当前架构。
4. 读取 `graph-memory/active.md` 和 `graph-memory/progress.md`，理解当前工作和状态。
5. 按需查询 `graph-memory/graph.db` 获取更详细的节点/边上下文。

### 更新触发条件

| 事件 | 需要执行的操作 |
|------|----------------|
| 功能完成 | 在 `graph.db` 插入 `task` / `evidence` / `lesson` 节点，并重新导出 `active.md` / `progress.md` |
| 架构决策发生变化 | 在 `graph.db` 插入/更新 `decision` 或 `belief` 节点，并链接相关证据 |
| 新依赖或工具链变化 | 在 `graph.db` 插入/更新 `component` / `dependency` 节点，并导出 `techContext.md` |
| Bug 修复或问题定位完成 | 在 `graph.db` 插入 `evidence` 或 `lesson` 节点 |
| 用户偏好被确认 | 在 `graph.db` 插入/更新 `preference` 节点 |
| 分支、任务或阶段变化 | 在 `graph.db` 插入/更新 `task` / `session` 节点，并导出 `active.md` |

### 会话结束

1. 总结本次会话改变了什么。
2. 记录产生了哪些决策或证据。
3. 明确下一步。
4. 询问用户是否需要更新 graph memory。

## 更新规则

- **永远不要修改 `graph-memory/RULES.md`**。本文件是 immutable 规则文件。
- **`blueprint.md` 由人类维护**：agent 只应在用户明确批准范围变化时更新它。
- **不要写入敏感信息**，包括 API key、token、密码、私有凭证。
- 时间相关条目使用日期戳：`[YYYY-MM-DD]`。
- `blueprint.md` 保持在 300 行以内；导出视图保持在 500 行以内。
- 不要在多个导出文件中重复同一段详细信息；让 `graph.db` 作为单一来源。

## 命令约定

| 命令 | 行为 |
|------|------|
| `memory bank update` / `graph memory update` | 检查并更新 graph memory，重新生成导出视图 |
| `memory bank status` / `graph memory status` | 展示当前活跃节点和近期进展摘要 |
| `memory bank read` / `graph memory read` | 读取 `blueprint.md`、导出视图并展示关键上下文 |
| `export memory` / `regenerate markdown` | 从 `graph.db` 重新生成 markdown 导出视图 |

## SQLite vs Markdown 使用原则

| 操作 | 使用 SQLite (`graph.db`) | 使用 Markdown |
|------|--------------------------|---------------|
| Agent 写入记忆 | ✅ | ❌（只读导出） |
| 全文搜索 | ✅（FTS5） | ❌ |
| 关系/证据追踪 | ✅（edges） | ❌ |
| 人类阅读项目概览 | ❌ | ✅ `blueprint.md` |
| Git diff 审查 | ❌（二进制） | ✅ 导出视图 |
