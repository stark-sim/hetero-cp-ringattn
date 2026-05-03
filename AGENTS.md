## Memory Bank Protocol (Required)

This project uses a Memory Bank system in `memory-bank/` for cross-session context continuity.

### At Session Start - ALWAYS:
1. Read `memory-bank/RULES.md` - all rules are there
2. Read `memory-bank/activeContext.md` - current work and decisions
3. Read `memory-bank/progress.md` - current status
4. Read other files as needed (`systemPatterns.md`, `techContext.md`, `productContext.md`, `projectbrief.md`)

### During Work - Update When:
- Feature completed -> update `activeContext.md` + `progress.md`
- Architecture decision made -> update `systemPatterns.md`
- New dependency added -> update `techContext.md`
- User preference learned -> update `activeContext.md`

### Git Discipline:
- Split work into task-sized checkpoints.
- After each task checkpoint is implemented and verified, create a dedicated git commit.
- Do not wait until the end of a long session to make one large mixed commit.
- Keep each commit focused on one coherent change: setup, correctness model, Rust bridge, docs, memory-bank update, etc.
- Before committing, run the relevant verification for that checkpoint and mention it in the commit context.
- Raw `reports/**/*.json` files are ignored by git by default. Do not commit generated report JSON unless the user explicitly asks for that exact artifact to be versioned.
- When an experiment documents project progress or a known-good validation point, summarize the result in docs or memory-bank instead of committing raw generated JSON.
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

### Special Commands:
- `memory bank update` / `memory bank güncelle` -> Review and update ALL memory bank files
- `memory bank status` / `memory bank durumu` -> Show current status summary
- `memory bank read` / `memory bank oku` -> Read all files and present context

### Optimization Trade-off Discipline:
Whenever proposing an optimization, you MUST analyze the trade-off before implementing:
1. **Why the default exists**: What problem does the current (non-optimized) approach solve?
2. **What is sacrificed**: What capability, correctness guarantee, or flexibility does the optimization discard?
3. **What the sacrificed thing does**: What is its intrinsic purpose in the general case?
4. **What it means for this project**: Why does that sacrifice matter (or not matter) specifically here?

Do NOT treat "faster / less memory" as an unqualified win. Record the analysis in the commit message or memory-bank.

**Example — "last token only" LM head optimization:**
- **Why default computes full logits**: `LlamaModel::forward` returns `[batch, seq, vocab]` so that callers can inspect logits at *any* position (e.g., perplexity evaluation, contrastive search, speculative decoding verification, training loss). It is the canonical transformer output contract.
- **What is sacrificed**: The ability to compute per-token loss, do beam-search over intermediate positions, or use the model as a general scoring function (e.g., reward model). The forward signature semantics change from "full sequence logits" to "last position only (unless flagged otherwise)".
- **What the sacrificed thing does**: Per-position logits are needed for (a) training (cross-entropy over all positions), (b) evaluation (perplexity), (c) advanced decoding (contrastive search compares multiple position scores), (d) speculative decoding (draft model needs to score all draft tokens).
- **What it means for this project**: HCP is **inference-only** (90% inference, 10% training per product thesis). For greedy/temperature sampling decode, only the last token matters. However, our distributed correctness tests currently compare *all* positions (`ref_logits.narrow(1, i, 1)` vs `dist_logits.narrow(1, i, 1)`). "Last token only" would break those tests unless we add a `return_full_logits: bool` flag. The flag adds API surface complexity. Given that prefill is a one-time cost and decode is the loop, the win is limited to prefill phase only. **Conclusion: skip for now, revisit if prefill becomes the bottleneck.**

### NEVER:
- Modify `memory-bank/RULES.md` (it's immutable)
- Write secrets (API keys, tokens, passwords) to memory bank files
- Skip reading memory bank at session start

---
