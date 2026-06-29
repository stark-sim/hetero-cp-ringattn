# Memory Bank - 通用 Agent 规则

> 本文件适用于所有参与本项目的 AI coding agent。
> 每次会话开始时，都必须读取本文件和 `memory-bank/` 下的上下文文件。

## 文件结构

| 文件 | 用途 |
|------|------|
| `RULES.md` | 本文件，规则文件。创建后不可修改 |
| `projectbrief.md` | 项目范围、愿景、核心需求 |
| `productContext.md` | 产品背景、目标用户、成功标准 |
| `systemPatterns.md` | 架构、设计模式、关键技术决策 |
| `techContext.md` | 技术栈、依赖、开发环境 |
| `activeContext.md` | 当前工作、近期变化、活跃决策 |
| `progress.md` | 已完成、进行中、已知问题、里程碑 |

## 必须遵守的协议

### 会话开始

1. 检查 `memory-bank/` 目录是否存在。
2. 读取 `memory-bank/RULES.md`。
3. 读取 `memory-bank/activeContext.md`，理解当前工作。
4. 读取 `memory-bank/progress.md`，理解当前状态。
5. 按需读取其他 memory bank 文件。

### 更新触发条件

| 事件 | 需要更新的文件 |
|------|----------------|
| 功能完成 | `activeContext.md` + `progress.md` |
| 架构决策发生变化 | `systemPatterns.md` |
| 新依赖或工具链变化 | `techContext.md` |
| Bug 修复或问题定位完成 | `activeContext.md` |
| 用户偏好被确认 | `activeContext.md` |
| 分支、任务或阶段变化 | `activeContext.md` |

### 会话结束

1. 总结本次会话改变了什么。
2. 记录发生了哪些决策。
3. 明确下一步。
4. 询问用户是否需要更新 memory bank。

## 更新规则

- **永远不要修改 `RULES.md`**。本文件是 immutable 规则文件。
- **不要修改 `projectbrief.md`**，除非用户明确要求。
- **不要写入敏感信息**，包括 API key、token、密码、私有凭证。
- 时间相关条目使用日期戳：`[YYYY-MM-DD]`。
- 每个文件保持在 500 行以内。
- 不要在多个文件中重复同一段详细信息。

## 命令约定

| 命令 | 行为 |
|------|------|
| `memory bank update` | 检查并更新全部 memory bank 文件 |
| `memory bank status` | 展示当前状态摘要 |
| `memory bank read` | 读取全部文件并展示完整上下文 |
