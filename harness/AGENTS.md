# Project-Level Harness Rules for hetero-cp-ringattn v0.2

> 优先级：项目级规则 > 全局规则 (`~/.kimi/AGENTS.md`)
> 本文件与项目根目录 `AGENTS.md` (memory-bank 规则) 并行存在，互不覆盖

## 核心原则：主 Agent 永远是执行方

- **主 Agent** 直接执行所有具体操作（Shell、WriteFile、SSH 等）
- **Executor Subagent** 只做**安全审查**，不执行任何操作
- **Reviewer Subagent** 做**质量审查**，事后验证结果

## 项目基础设施

查询路径：`./harness/infra.yaml`

关键路径（禁止猜测）：
- Mac libtorch: `/Users/stark_sim/libtorch`
- white libtorch: `/home/stark/libtorch`
- sd-1 libtorch: `/home/user/libtorch`
- sd-2 libtorch: `/home/user/libtorch`
- Mac models: `/Users/stark_sim/models/qwen2-0.5b`
- white models: `/home/stark/models/Qwen2-0.5B`
- sd-1 models: `/home/user/models/qwen2-0.5b`
- sd-2 models: `/home/user/models/qwen2-0.5b`
- white SSH: `stark@100.64.0.2`
- sd-1 SSH: `user@100.64.0.93`
- sd-2 SSH: `user@100.64.0.94`
- white Cargo PATH: `PATH=/home/stark/.cargo/bin:$PATH`
- sd-1/sd-2 Cargo PATH: `PATH=/home/user/.cargo/bin:$PATH`

## 危险操作 Protocol

### 危险操作（必须走完整 Protocol）
- 删除文件或目录（`rm -rf`、truncate）
- 修改系统配置文件
- 安装/卸载包
- 首次 SSH 到未注册主机
- `curl | sh` 类操作

### 豁免操作（主 Agent 可直接执行）
- 只读查询：`ls`、`cat`、`du`、`df`、`cargo --version`
- 读取文件、搜索代码、查看日志
- 本地测试运行：`cargo test --features tch-backend`
- **已注册主机的 SSH 只读查询**（如 `ssh user@sd-1 'cargo --version'`）
- **已注册主机的项目目录写操作**（如 `scp` 模型文件、修改项目代码）

### Protocol Steps

1. **主 Agent 写入 pending/operation.yaml**
2. **调用 Executor Subagent（安全审查）** — Executor 只审查，不执行
3. **收到 AUTHORIZED 后，主 Agent 直接执行**
4. **调用 Reviewer Subagent（质量审查）**
5. **检查 verdict**

## 项目特定约束

### 1. 先查后做
- 操作前查询 `harness/infra.yaml`，禁止猜测路径
- 跨节点测试前查询 `harness/memory/active/` 中的已知问题

### 2. SSH 规范
- 非交互式 SSH 不加载 Cargo，必须显式 `PATH=/home/stark/.cargo/bin:$PATH`（white）或 `PATH=/home/user/.cargo/bin:$PATH`（sd-1/sd-2）
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
