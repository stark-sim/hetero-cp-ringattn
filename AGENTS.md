## Graph Memory Protocol (Required)

This project uses a graph-backed memory system in `graph-memory/` for cross-session context continuity.

- **Source of truth:** `graph-memory/graph.db` (SQLite + FTS5 + graph).
- **Human-readable blueprint:** `graph-memory/blueprint.md` — read this first every session.
- **Exported views:** `graph-memory/active.md`, `graph-memory/progress.md`, `graph-memory/systemPatterns.md`, `graph-memory/techContext.md`, `graph-memory/productContext.md`.

### At Session Start - ALWAYS:
1. Read `graph-memory/RULES.md` - all rules are there
2. Read `graph-memory/blueprint.md` - project scope and current architecture
3. Read `graph-memory/active.md` - current work and active decisions
4. Read `graph-memory/progress.md` - recent status
5. Query `graph-memory/graph.db` when deeper context is needed

### During Work - Update When:
- Feature completed -> insert/update nodes in `graph.db`, then regenerate `active.md` / `progress.md`
- Architecture decision made -> insert/update `decision` / `belief` nodes in `graph.db`, then regenerate `systemPatterns.md`
- New dependency added -> insert/update `component` / `dependency` nodes in `graph.db`, then regenerate `techContext.md`
- User preference learned -> insert/update `preference` nodes in `graph.db`, then regenerate `active.md`
- Evidence contradicts a belief -> run conflict resolution: create `revision` / `evidence` nodes, update confidence, mark supersession with `REPLACED_BY` / `SUPERSEDES` edges

### Git Discipline:
- Split work into task-sized checkpoints.
- After each task checkpoint is implemented and verified, create a dedicated git commit.
- Do not wait until the end of a long session to make one large mixed commit.
- Keep each commit focused on one coherent change: setup, correctness model, Rust bridge, docs, graph-memory update, etc.
- Before committing, run the relevant verification for that checkpoint and mention it in the commit context.
- Raw `reports/**/*.json` files are ignored by git by default. Do not commit generated report JSON unless the user explicitly asks for that exact artifact to be versioned.
- When an experiment documents project progress or a known-good validation point, summarize the result in docs or graph-memory instead of committing raw generated JSON.
- Do not commit build outputs, transient logs, cache directories, or large binary artifacts unless explicitly requested.
- After committing on `main`, push the commit to the configured remote.
- Never use `git push --force`, `git push -f`, or any force-push variant. Force-push is prohibited for all branches and remotes unless the user explicitly changes this rule in writing.

### Sudo / System Changes:
- If a fix requires `sudo`, root-owned paths, `/opt`, `/usr/local`, system linkers, launch services, or other machine-level changes, stop and ask the user to run the command manually.
- Do not try to work around missing sudo by patching third-party binaries, rewriting install names, copying system libraries, or changing vendor artifacts unless the user explicitly approves that exact approach.
- Prefer explaining the root cause and giving the minimal sudo command sequence for the user to run.
- After the user confirms the system-level fix is done, verify normally from the project.
- For standalone libtorch on macOS, if `libtorch_cpu.dylib` requires `/opt/llvm-openmp/lib/libomp.dylib`, ask the user to create the `/opt/llvm-openmp/lib` symlink with sudo instead of modifying libtorch dylibs.

### Local Hardware Smoke Discipline:
- On the local Mac, libtorch hardware smoke should use MPS in a non-sandbox/escalated process: `HCP_ENABLE_TORCH=1 HCP_TORCH_DEVICE=mps bash scripts/run_rust_ringattn_smoke.sh`.
- Do not treat CPU-only local libtorch smoke as a meaningful hardware validation result; CPU smoke is only a fallback for compile/link or no-accelerator checks.
- The normal sandbox hides Metal devices, so sandbox MPS failures are not valid evidence that MPS is unavailable.

### Remote GPU Discipline:
- The NVIDIA GPU host is `192.168.8.172`.
- Do not directly edit source files on the remote GPU host. Make code changes locally, commit them, push to the configured remote, then sync on the GPU host with `git pull`.
- Remote GPU smoke results can be recorded in reports/docs, but source-of-truth code changes must flow through git.
- For dual-machine P2P smoke, do not use `127.0.0.1` as the validation endpoint. Bind the server to `0.0.0.0` or the target subnet address and connect the client to the `192.168.8.x` GPU host address.
- Non-interactive SSH on the GPU host may not load Cargo. Prefer an explicit PATH prefix such as `PATH=/home/stark/.cargo/bin:$PATH` when launching Rust smoke commands remotely; do not edit remote shell startup files just to work around this.

### Cross-Node Heterogeneous Validation Discipline:
- **Every platform in a heterogeneous setup MUST run at least one worker**. The coordinator only handles tokenizer sharding and token broadcasting; it performs NO model computation. A setup where Platform A runs only the coordinator and Platform B runs only the worker does NOT validate heterogeneous compute capability.
- **Correct architecture**: Mac runs `coordinator + worker 0 (MPS)`, GPU host runs `worker 1 (CUDA)`. Both platforms execute model forward passes and participate in the KV ring exchange.
- `distributed_worker.rs` supports `--local-domain-ids 0,1` for multi-domain workers in a single process, but this is for local development convenience. For true cross-node heterogeneous validation, each platform must run its own worker process(es).
- Record cross-node results with explicit worker distribution: which domain runs on which platform and device.

### Special Commands:
- `memory bank update` / `graph memory update` / `memory bank güncelle` -> Review and update graph memory, regenerate markdown views
- `memory bank status` / `graph memory status` / `memory bank durumu` -> Show current active/progress summary
- `memory bank read` / `graph memory read` / `memory bank oku` -> Read blueprint + active + progress and present context
- `export memory` / `regenerate markdown` -> Regenerate markdown views from `graph.db`

### Optimization Trade-off Discipline:
Whenever proposing an optimization, you MUST analyze the trade-off before implementing:
1. **Why the default exists**: What problem does the current (non-optimized) approach solve?
2. **What is sacrificed**: What capability, correctness guarantee, or flexibility does the optimization discard?
3. **What the sacrificed thing does**: What is its intrinsic purpose in the general case?
4. **What it means for this project**: Why does that sacrifice matter (or not matter) specifically here?

Do NOT treat "faster / less memory" as an unqualified win. Record the analysis in the commit message or graph-memory.

**Example — "last token only" LM head optimization:**
- **Why default computes full logits**: `LlamaModel::forward` returns `[batch, seq, vocab]` so that callers can inspect logits at *any* position (e.g., perplexity evaluation, contrastive search, speculative decoding verification, training loss). It is the canonical transformer output contract.
- **What is sacrificed**: The ability to compute per-token loss, do beam-search over intermediate positions, or use the model as a general scoring function (e.g., reward model). The forward signature semantics change from "full sequence logits" to "last position only (unless flagged otherwise)".
- **What the sacrificed thing does**: Per-position logits are needed for (a) training (cross-entropy over all positions), (b) evaluation (perplexity), (c) advanced decoding (contrative search compares multiple position scores), (d) speculative decoding (draft model needs to score all draft tokens).
- **What it means for this project**: HCP is **inference-only** (90% inference, 10% training per product thesis). For greedy/temperature sampling decode, only the last token matters. However, our distributed correctness tests currently compare *all* positions (`ref_logits.narrow(1, i, 1)` vs `dist_logits.narrow(1, i, 1)`). "Last token only" would break those tests unless we add a `return_full_logits: bool` flag. The flag adds API surface complexity. Given that prefill is a one-time cost and decode is the loop, the win is limited to prefill phase only. **Conclusion: skip for now, revisit if prefill becomes the bottleneck.**

### NEVER:
- Modify `graph-memory/RULES.md` (it's immutable)
- Write secrets (API keys, tokens, passwords) to graph-memory files or `graph.db`
- Skip reading graph memory at session start

---
