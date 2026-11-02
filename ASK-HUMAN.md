# ASK-HUMAN

Questions from the build agents to the owner. Agents append; the owner answers inline and marks `[answered]`.
Format per entry: date · phase/task · question · options · agent's recommendation.

---

### Q1 · 2026-11-02 · P0/3 · Who writes the `src/verify` migration?
The build prompt's P0 task 3 asks the agent to migrate `src/verify/*` to the contract §7 `Verifier`. But `src/verify`
is human-owned from the first commit (I-1), and it was already in the bootstrap commit `4bde90c`.
- **A.** The agent ships the migration as `proposals/P0-3-verifier.patch` (plus its tests). You review it and run
  `git apply proposals/P0-3-verifier.patch`. Until then, a small temporary adapter outside `src/verify`
  (`src/think/verify-adapter.ts`) wraps the existing cascade, so P0 works either way.
- **B.** A one-time waiver: the agent edits `src/verify` directly for P0/3, and it is frozen from then on.

**Recommendation: A** (already done this way). Nothing else in P0 waits on your answer.

### Q2 · 2026-11-02 · P0/3 · Should the write guard itself be human-owned?
The I-1 write guard lives in `src/files.ts`, which an agent could weaken.
- **A.** Add `src/files.ts` to the human-owned list (agents then treat it as read-only and send patches).
- **B.** Move it into `src/verify/guard.ts` with the next verifier proposal.
- **C.** Leave it agent-owned; the CI test (`test/guard.test.ts`) is the only protection.

**Recommendation: A**, effective once P0 is accepted.

### Q3 · 2026-11-02 · P0/3 · Which human-owned paths apply in target repos?
`.tfa/bench/**` is always guarded. `src/verify/**` and `src/bench/**` are tfa's own paths. A Next.js app can have a
legitimate `src/verify/` folder.
- **A.** Guard `src/verify/**` and `src/bench/**` only when the target repo is tfa itself (its `package.json` name is `tfa`).
- **B.** Guard them in every repo, which is stricter but can block legitimate fixes.

**Recommendation: A** (implemented).

### Q4 · 2026-11-02 · P0/4 · Where does the trajectory schema marker go?
Contract §11 says every JSONL store carries `"schema"` in its first line. §5 says every line is an event with `t`.
- **A.** The first event (`t:"task"`) carries an extra `"schema": 1` field.
- **B.** A separate header line `{"t":"schema","schema":1}`.

**Recommendation: A** (implemented; readers accept both). This needs a one-line note in the contract.
