# tfa — thinking-first assistant

A local, verified, self-improving coding assistant built around small open models: a 9B worker in VRAM, a ~35B-A3B
mixture-of-experts senior in RAM, a verifier cascade (lint → typecheck → tests → build), explicit memory (Notebook +
skill library), novelty-gated routing, idle-time consolidation, and weekly LoRA self-training. Target machine: one
developer's Windows PC with an RTX 3080 12 GB and 32 GB RAM.

**Start here:** `docs/pdf/20-agent-build-prompt.pdf` — the master instruction (document set, standing rules, phases
P0–P6 with acceptance tests and copy-paste session prompts). `docs/pdf/10-interface-contract.pdf` is normative.
`docs/thinking-first-local-ai-thesis.pdf` is the design rationale. `docs/pdf/A…I-*.pdf` are the research packs.

## State

**P0 scaffold** (reader harness + verifier cascade + fix loop + trajectory log + CLI + tests). Not yet migrated to the
interface contract — that migration is P0 task list 1–6 in the build prompt.

## Run

```bash
npm install
npm run typecheck
npm test                 # vitest: fixture repo + mock model server; no GPU needed
npm run build && npm link
tfa models               # download/serve commands for the 3080
tfa doctor <repo>        # detected cascade stages, model reachability
tfa fix <repo> "<task>"  # draft → apply → verify → retry; keeps verified edits
```

## P0 acceptance test (owner runs on the PC)

1. `powershell -ExecutionPolicy Bypass -File docs/sources/download-papers.ps1` (original papers → `docs/papers/`).
2. Install a CUDA build of llama.cpp; run the worker download/serve command from `tfa models`; `curl http://127.0.0.1:8080/health`.
3. `tfa doctor <one of your repos>` — typecheck and build detected.
4. `tfa fix <repo> "Fix the type error in <file>"` on a real or introduced error. Done when it ends with
   `✓ verified on attempt k` and `npm run build` passes. Record attempts and wall-clock.

## Conventions

- Human-owned paths: `src/verify/**`, `src/bench/**`, `.tfa/bench/**`. Agents propose changes in `proposals/`.
- Constants from papers carry `// arXiv:<id> §<n>`.
- Questions for the owner go to `ASK-HUMAN.md`.
- Commits: `<phase>/<task>: <what> (<arXiv ids used>)`.
