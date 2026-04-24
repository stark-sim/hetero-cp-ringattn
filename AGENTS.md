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
- Commit structured experiment reports when they document project progress or a known-good validation point.
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

### Special Commands:
- `memory bank update` / `memory bank güncelle` -> Review and update ALL memory bank files
- `memory bank status` / `memory bank durumu` -> Show current status summary
- `memory bank read` / `memory bank oku` -> Read all files and present context

### NEVER:
- Modify `memory-bank/RULES.md` (it's immutable)
- Write secrets (API keys, tokens, passwords) to memory bank files
- Skip reading memory bank at session start

---
