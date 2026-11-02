/**
 * P0 loop (`tfa fix`): context → draft edits → verify (apply first, then the cascade) → feedback → retry.
 *
 * Sequential attempts with the failure fed back each round: verbal feedback from the failed attempt
 * — arXiv:2303.11366 §3 (Reflexion, memory bound 1 for code). Candidates are generated and checked by
 * execution before acceptance, the generate-then-verify pattern of arXiv:2207.10397 §2 (CodeT).
 * P2 replaces this with Thinker.search; nothing here implements later phases.
 *
 * Trajectory (contract §5): task → decompose (one subtask) → retrieve → [prompt → response → candidate →
 * stage… → verify]×attempts → result. Model calls go through LoggedReader (I-6), prompts through
 * buildPrompt (I-9), edits through the guarded applyEdits (I-1, I-8), verdicts through the Verifier (I-4).
 *
 * Safety: the repo must be a git repository; failed candidates are rolled back exactly; verified edits stay
 * in the working tree for review (or are rolled back with keep:false).
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type { Config } from "../config.js";
import { newId } from "../ids.js";
import { HttpReader, LoggedReader, type Reader } from "../llm/client.js";
import { parseEdits, snapshot, type Backup } from "../llm/edits.js";
import { buildPrompt, loadCore } from "../llm/prompts/index.js";
import { FIX_SYSTEM, applyFailedFeedback, parseFailedFeedback, stageFailedFeedback } from "../llm/prompts/fix.js";
import { Trajectory } from "../log.js";
import { buildContext } from "../retrieve/context.js";
import type { Candidate, Costs, Edit, Msg, Stage, StageOutcome, Verifier } from "../types.js";
import { exec, tailLog } from "../verify/exec.js";
import { TransitionalVerifier } from "./verify-adapter.js";

export type FixOptions = {
  repo: string;
  task: string;
  cfg: Config;
  reader?: Reader;
  verifier?: Verifier;
  /** Keep the verified edits in place (default). With false, roll back and only record the diff. */
  keep?: boolean;
  /** Sequential attempts; default budgets.slow.routine.maxCandidates. */
  attempts?: number;
  log?: (line: string) => void;
};

export type FixOutcome = {
  ok: boolean;
  attempts: number;
  taskId: string;
  trajectory: string;
  diff?: string;
  failedAt?: Stage;
  lastLog?: string;
  note?: string;
  costs: Costs;
};

/** Rough chars-per-token for budgeting before a tokenizer is wired in (heuristic, not from a paper). */
export const CHARS_PER_TOKEN = 4;

export function formatOutcome(s: StageOutcome): string {
  const mark = s.skipped ? "–" : s.ok ? "✓" : "✗";
  return `${mark} ${s.stage.padEnd(9)} ${s.skipped ? "skipped" : (s.ms / 1000).toFixed(1) + "s"}`;
}

export async function fix(o: FixOptions): Promise<FixOutcome> {
  const { repo, task, cfg } = o;
  const log = o.log ?? (() => {});
  if (!fs.existsSync(path.join(repo, ".git"))) throw new Error(`${repo} is not a git repository (tfa needs git so every change is revertable)`);
  const maxAttempts = o.attempts ?? cfg.budgets.slow.routine.maxCandidates;

  const traj = Trajectory.create(repo, { text: task, origin: "user" });
  const subtaskId = newId("sub");
  const reader = new LoggedReader(o.reader ?? new HttpReader(cfg.slots), traj, subtaskId);
  const verifier = o.verifier ?? new TransitionalVerifier(cfg, (s) => log(formatOutcome(s)));
  log(`trajectory: ${path.relative(repo, traj.file)}`);

  // Pre-flight: the current type-check output is the best cheap signal of what is wrong.
  const tsc = verifier.stages(repo).find((c) => c.stage === "typecheck");
  let tscLog: string | undefined;
  if (tsc) {
    log(`pre-flight: ${tsc.cmd}`);
    const r = await exec(tsc.cmd, repo, cfg.timeouts.typecheck ?? 300_000);
    tscLog = tailLog(r.out, 8000);
    log(r.code === 0 ? "pre-flight: type-check clean" : `pre-flight: type-check failing (${r.out.split("\n").filter((l) => /error TS/.test(l)).length} errors)`);
  }

  const ctx = buildContext(repo, task, tscLog, cfg.budgets.contextTokens * CHARS_PER_TOKEN);
  traj.append({ t: "decompose", at: Date.now(), subtasks: [{ id: subtaskId, text: task, touches: ctx.files }] });
  // P0 context is type-check driven (P1 replaces it with the Library); one hit per included file, uncited.
  traj.append({
    t: "retrieve",
    at: Date.now(),
    subtaskId,
    query: task,
    hits: ctx.files.map(() => ({ chunkId: newId("chunk"), score: 1, cited: false })),
    tokens: Math.ceil(ctx.text.length / CHARS_PER_TOKEN),
  });

  const core = loadCore(repo);
  const conversation: Msg[] = [];
  let last: { failedAt?: Stage; log?: string; note?: string } = {};

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    log(`\nattempt ${attempt}/${maxAttempts} → worker`);
    // Diversify on retry (+0.2 per attempt is a design choice), capped at CodeT's sampling temperature 0.8 — arXiv:2207.10397 §A
    const temperature = Math.min(0.8, cfg.slots.worker.temperature + (attempt - 1) * 0.2);
    const messages = buildPrompt({ system: FIX_SYSTEM, core, context: ctx.text, task: `${task}\n\nProduce the edits now.`, conversation });
    let text: string;
    try {
      const res = await reader.chat("worker", messages, { temperature, idSlot: 0, attempt });
      text = res.text;
      log(`model answered in ${(res.ms / 1000).toFixed(1)}s (${res.completionTokens ?? "?"} tokens)`);
    } catch (err) {
      const note = `worker call failed: ${(err as Error).message}`;
      traj.result({ ok: false, subtaskId, note });
      log(`✗ ${note}`);
      return { ok: false, attempts: attempt, taskId: traj.id, trajectory: traj.file, note, costs: traj.costs() };
    }

    let edits: Edit[] = [];
    let parseError: string | undefined;
    try {
      edits = parseEdits(text);
    } catch (err) {
      parseError = `could not parse edits: ${(err as Error).message}`;
    }
    const cand: Candidate = { id: newId("cand"), subtaskId, edits, createdBy: "worker" };
    traj.candidate(cand);

    if (parseError) {
      // A parse failure is cascade stage `apply` (contract §4), cost ≈ 0.
      traj.append({ t: "stage", at: Date.now(), candidateId: cand.id, stage: "apply", ok: false, ms: 0, log: parseError });
      traj.append({ t: "verify", at: Date.now(), candidateId: cand.id, ok: false, failedAt: "apply", totalMs: 0 });
      log(`✗ apply     ${parseError}`);
      last = { failedAt: "apply", log: parseError };
      conversation.push({ role: "assistant", content: text }, { role: "user", content: parseFailedFeedback(parseError) });
      continue;
    }

    let dry: Backup | undefined;
    if (o.keep === false) {
      try {
        dry = snapshot(repo, edits);
      } catch {
        /* an unsafe path fails the apply stage below */
      }
    }
    const vr = await verifier.run(cand, repo);
    for (const s of vr.stages) traj.append({ t: "stage", at: Date.now(), candidateId: cand.id, stage: s.stage, ok: s.ok, skipped: s.skipped, ms: s.ms, log: s.log });
    traj.append({ t: "verify", at: Date.now(), candidateId: cand.id, ok: vr.ok, failedAt: vr.failedAt, totalMs: vr.totalMs });

    if (vr.ok) {
      const diff = gitDiff(repo, edits.map((e) => e.path));
      const ev = traj.result({ ok: true, subtaskId, candidateId: cand.id, diff });
      dry?.restore();
      log(`\n✓ verified on attempt ${attempt} in ${(vr.totalMs / 1000).toFixed(1)}s`);
      return { ok: true, attempts: attempt, taskId: traj.id, trajectory: traj.file, diff, costs: ev.costs };
    }

    if (!vr.failedAt) {
      // Every stage was skipped: nothing can verify any candidate, so retrying is pointless (fail closed, I-4).
      const note = "nothing verified: no cascade stage has a command in this repo (see `tfa doctor`)";
      const ev = traj.result({ ok: false, subtaskId, candidateId: cand.id, note });
      log(`\n✗ ${note}`);
      return { ok: false, attempts: attempt, taskId: traj.id, trajectory: traj.file, note, costs: ev.costs };
    }

    const failed = vr.stages.find((s) => s.stage === vr.failedAt)!;
    last = { failedAt: vr.failedAt, log: failed.log };
    log(vr.failedAt === "apply" ? `✗ apply: ${failed.log}` : `rolled back; ${vr.failedAt} failed`);
    conversation.push(
      { role: "assistant", content: text },
      { role: "user", content: vr.failedAt === "apply" ? applyFailedFeedback(failed.log) : stageFailedFeedback(vr.failedAt, failed.log) },
    );
  }

  const note = `no verified solution in ${maxAttempts} attempts (last failure: ${last.failedAt ?? "unknown"})`;
  const ev = traj.result({ ok: false, subtaskId, note });
  log(`\n✗ ${note}`);
  return { ok: false, attempts: maxAttempts, taskId: traj.id, trajectory: traj.file, failedAt: last.failedAt, lastLog: last.log, note, costs: ev.costs };
}

/** Diff of the files a candidate touched (tracked changes plus new files), via git without a shell. */
export function gitDiff(repo: string, paths: string[]): string {
  const rels = [...new Set(paths.map((p) => p.replace(/\\/g, "/")))];
  if (rels.length === 0) return "";
  const git = (args: string[]) => execFileSync("git", args, { cwd: repo, encoding: "utf8", maxBuffer: 50 * 1024 * 1024, windowsHide: true });
  let diff = git(["--no-pager", "diff", "--no-color", "--", ...rels]);
  const untracked = git(["ls-files", "--others", "--exclude-standard", "--", ...rels]).split(/\r?\n/).filter(Boolean);
  for (const f of untracked) {
    const abs = path.join(repo, f);
    if (fs.existsSync(abs) && fs.statSync(abs).size < 200_000) {
      const lines = fs.readFileSync(abs, "utf8").replace(/\r\n/g, "\n").replace(/\n$/, "").split("\n");
      diff += `--- /dev/null\n+++ b/${f}\n@@ -0,0 +1,${lines.length} @@\n` + lines.map((l) => "+" + l).join("\n") + "\n";
    }
  }
  return diff;
}
