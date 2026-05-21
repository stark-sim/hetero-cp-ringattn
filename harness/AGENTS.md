# Project-Level Harness Rules for hetero-cp-ringattn

> 优先级：项目级规则 > 全局规则 (`~/.kimi/AGENTS.md`)
> 本文件与项目根目录 `AGENTS.md` (memory-bank 规则) 并行存在，互不覆盖

## 项目基础设施

查询路径：`./harness/infra.yaml`

关键路径（禁止猜测）：
- Mac libtorch: `/Users/stark_sim/libtorch`
- white libtorch: `/home/stark/libtorch`
- Mac models: `/Users/stark_sim/models/qwen2-0.5b`
- white models: `/home/stark/models/Qwen2-0.5B`
- white SSH: `stark@100.64.0.2`
- white Cargo PATH: `PATH=/home/stark/.cargo/bin:$PATH`

## Dangerous Operations Protocol

任何涉及以下操作的任务，必须遵循 Harness Protocol：

1. **SSH 到 white (100.64.0.2)**
2. **写入或修改代码/配置文件**
3. **cargo build / cargo test / 运行 binary**
4. **删除文件或目录**
5. **跨节点分布式测试启动**

### Protocol Steps

#### 1. 写入 pending/operation.yaml
```yaml
timestamp: "2026-05-13T21:00:00+08:00"
type: "dangerous-operation"
stage: "cross-node-smoke-4k"
target:
  mac_local: true
  white_remote: true
command: "Run cross-node 4K A/B test"
risk_level: "medium"
steps:
  - "Build locally"
  - "Build on white via SSH"
  - "Launch coordinator + workers"
  - "Wait for completion"
```

#### 2. 调用 Agent 工具启动 Executor
- Executor 读取 `./harness/pending/operation.yaml`
- Executor 读取 `./harness/infra.yaml`
- Executor 验证 target 已注册
- Executor 执行或拒绝

#### 3. 调用 Agent 工具启动 Reviewer
- Reviewer 零上下文隔离
- Reviewer 输出 YAML verdict

#### 4. 检查 verdict
- APPROVE → 继续
- WARN → 记录警告后继续
- REJECT → 停止并报告用户

## 项目特定约束

### 1. 先查后做
- 操作前查询 `harness/infra.yaml`，禁止猜测路径
- 跨节点测试前查询 `memory-bank/` 中的已知问题（端口占用、超时等）

### 2. SSH 规范
- 非交互式 SSH 不加载 Cargo，必须显式 `PATH=/home/stark/.cargo/bin:$PATH`
- 禁止直接编辑远程源码，必须通过 git pull 同步

### 3. 测试规范
- 本地 smoke 优先验证 correctness（45 tests）
- 跨节点测试前确保端口未被占用（`lsof -i :<port>`）
- 大 scale 跨节点（4K+）需评估网络稳定性

### 4. Git 规范（继承自根 AGENTS.md）
- 分 checkpoint 提交
- 提交前运行相关验证
- 禁止 force-push
- commit 后 push 到 remote
