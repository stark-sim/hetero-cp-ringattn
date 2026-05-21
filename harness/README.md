# Project-Level Harness

本项目使用 Harness Agent Governance Framework 进行约束。

## 目录说明

```
harness/
├── AGENTS.md          # 项目特定规则（优先级高于全局 ~/.kimi/AGENTS.md）
├── infra.yaml         # 项目基础设施清单
├── pending/           # 待处理队列
│   ├── operation.yaml # 待执行操作（由主 agent 写入，executor 读取）
│   └── review.yaml    # 待审查请求（由主 agent 写入，reviewer 读取）
├── operations/        # 已执行操作记录（executor 写入）
├── reviews/           # 已审查记录（reviewer 写入）
└── memory/            # 项目经验库
    ├── active/        # 已确认经验
    └── pending/       # 待确认经验
```

## 使用流程

1. 主 agent 检测到危险操作
2. 主 agent 写入 `pending/operation.yaml`
3. 主 agent 调用 `Agent` 工具启动 Executor
4. Executor 读取 `pending/operation.yaml` 和 `infra.yaml`
5. Executor 执行操作，写入 `operations/<timestamp>-<id>.yaml`
6. Executor 删除 `pending/operation.yaml`
7. 主 agent 读取操作结果，生成审查请求，写入 `pending/review.yaml`
8. 主 agent 调用 `Agent` 工具启动 Reviewer
9. Reviewer 审查，写入 `reviews/<timestamp>-<id>.yaml`
10. 主 agent 检查 verdict，决定继续或回滚

## 规则优先级

1. 项目级 `harness/AGENTS.md`（最高优先级）
2. 全局 `~/.kimi/AGENTS.md`
3. 全局 `~/.agents/AGENTS.md`
