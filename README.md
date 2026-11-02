# tfa — thinking-first assistant

A local, verified, self-improving coding assistant built around small open models: a 9B worker in VRAM, a ~35B-A3B
mixture-of-experts senior in RAM, a verifier cascade (apply → lint → typecheck → tests → build), explicit memory
(Notebook + skill library), novelty-gated routing, idle-time consolidation, and weekly LoRA self-training. Target
machine: one developer's Windows PC with an RTX 3080 12 GB and 32 GB RAM.

**Start here:** `docs/pdf/20-agent-build-prompt.pdf`. It is the master instruction: the document set, the standing
rules, and phases P0–P6 with acceptance tests and session prompts. `docs/pdf/10-interface-contract.pdf` is normative.
`docs/thinking-first-local-ai-thesis.pdf` explains the design. `docs/pdf/A…I-*.pdf` are the research packs; the
Markdown copies under `docs/markdown/` are the ones agents read.

## State: P0 done, waiting for the owner's acceptance test

| P0 task | Where | Contract |
|---|---|---|
| 1 bootstrap, layout, config | `src/config.ts`, `tfa.config.json`, `src/types.ts`, `src/ids.ts`, CI | §2, §3, §10 |
| 2 model I/O | `src/llm/client.ts` (`Reader`, `HttpReader`, `LoggedReader`), `src/llm/prompts/` (`buildPrompt`) | §4, I-6, I-9 |
| 3 verifier + write guard | `src/files.ts` (I-1 guard), `src/think/verify-adapter.ts`, `proposals/P0-3-verifier.patch` | §7, I-1, I-4 |
| 4 trajectory log | `src/log.ts` | §5 |
| 5 CLI | `src/cli.ts`: `doctor`, `verify`, `fix`, `models` | §8 (P0 subset) |
| 6 model commands | `src/models.ts` | §4 |

Edits are applied by `src/llm/edits.ts`, which enforces I-8 (no path escapes, `find` matches exactly once,
CRLF-safe). `src/verify/**` is human-owned. Its contract migration waits in `proposals/P0-3-verifier.patch` (see
ASK-HUMAN Q1); until you apply it, a transitional adapter gives the same guarantees.

## Run the tests (no GPU needed)

```bash
npm ci
npm run typecheck        # src and tests
npm test                 # vitest: fixture repos + mock llama-server
```

CI runs the same three commands on `ubuntu-latest` and `windows-latest`.

## P0 acceptance test (owner, on the PC)

Run this in PowerShell. Replace `C:\code\myrepo` with one of your TypeScript repositories.

```powershell
# 0. tool
git clone https://github.com/C3Studium/tfa; cd tfa
npm ci; npm run typecheck; npm test
# optional, after reviewing it (ASK-HUMAN Q1):  git apply proposals/P0-3-verifier.patch; npm test; git commit -am "P0/3: apply verifier proposal"
npm run build; npm link                     # puts `tfa` on PATH

# 1. papers (once)
powershell -ExecutionPolicy Bypass -File docs/sources/download-papers.ps1

# 2. llama.cpp + worker model
#    llama.cpp: from github.com/ggml-org/llama.cpp/releases take the win-cuda x64 zip and the matching
#    cudart zip; unzip both into one folder and add it to PATH.
pip install -U "huggingface_hub[cli]"
tfa models                                  # confirm the GGUF names, then run the [worker] and [draft] download lines
#    In a second terminal, run the [worker] "serve:" line from `tfa models`, from the tfa folder. Leave it running.
curl.exe http://127.0.0.1:8080/health       # {"status":"ok"}

# 3. pre-flight on your repo
tfa doctor C:\code\myrepo                   # expect "typecheck ... [scripts.typecheck]" or "[tsc]" and "✓ ready for tfa fix"

# 4. the test: a real TypeScript error (or introduce one), then
tfa fix C:\code\myrepo "Fix the type error in src/<file>.ts"
cd C:\code\myrepo; npm run build            # must pass
```

**Pass criterion:** `tfa fix` ends with `✓ verified on attempt k`, and `npm run build` in the repo passes.
`tfa fix` prints attempts, wall-clock time and a JSON record at the end. Save that record as
`C:\code\myrepo\.tfa\bench\runs\p0-first.json` (a human-owned path, so you write it). Then report the attempts and
wall-clock here, with your answers to `ASK-HUMAN.md`. The full trajectory is in `.tfa\trajectories\<taskId>.jsonl`.

If it fails: `tfa verify C:\code\myrepo` shows which stage fails without a model. Exit code 2 from `tfa fix` means
the worker is not reachable. `--attempts N` raises the attempt budget (default 4). `--dry` verifies and then rolls back.

## CLI

```
tfa doctor [repo]                          environment, detected cascade, slot health, human-owned paths
tfa verify [repo] [--up-to <stage>]        run the cascade on the working tree (skipped stages never count as passed)
tfa fix <repo> "<task>" [--dry] [--attempts N]
tfa models                                 download + llama-server commands: worker, draft, senior A/B, embed, rerank
```

Config: `tfa.config.json` (tool) < `<repo>/.tfa/config.json` (project). `TFA_WORKER_URL`, `TFA_SENIOR_URL`,
`TFA_DRAFT_URL`, `TFA_EMBED_URL` and `TFA_RERANK_URL` override slot URLs only (contract §10).

## Conventions

- Human-owned paths: `src/verify/**`, `src/bench/**` (in this repo) and `.tfa/bench/**` (in every repo). The write
  guard in `src/files.ts` refuses them, and `test/guard.test.ts` checks that every write in `src/` goes through it.
  Agents propose changes in `proposals/<id>.patch`.
- Constants from papers carry `// arXiv:<id> §<n>`.
- Questions for the owner go to `ASK-HUMAN.md`.
- Commits: `<phase>/<task>: <what> (<arXiv ids used>)`.
