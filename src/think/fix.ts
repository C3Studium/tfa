/**
 * P0 loop: understand → draft edits → apply → cascade → (reflect) → retry, up to N attempts.
 *
 * Sequential best-of-N with the failure log fed back each round (a minimal Reflexion).
 * Every prompt, response, apply and stage result is written to the trajectory, so P3
 * (lessons, skills) and P5 (training data) have data from the first run.
 *
 * Safety: edits are applied in place with an exact backup and rolled back on failure;
 * the repo must be a git repository so the user can always inspect or revert.
 */
import fs from "node:fs";
import path from "node:path";
import type { Config } from "../config.js";
import { HttpReader, type Reader } from "../llm/client.js";
import { applyEdits, parseEdits, type Edit } from "../llm/edits.js";
import { buildPrompt, loadCore } from "../llm/prompts/index.js";
import { FIX_SYSTEM, applyFailedFeedback, parseFailedFeedback, stageFailedFeedback } from "../llm/prompts/fix.js";
import type { Msg } from "../types.js";
import { Trajectory } from "../log.js";
import { runCascade, formatStage, type CascadeResult } from "../verify/cascade.js";
import { detectCommands } from "../verify/detect.js";
import { exec, tailLog } from "../verify/exec.js";
import { buildContext } from "../retrieve/context.js";

export type FixOptions = {
  repo: string;
  task: string;
  cfg: Config;
  reader?: Reader;
  /** Keep the successful edits in place (default). With false, roll back and only write the diff. */
  keep?: boolean;
  /** Sequential attempts; default budgets.slow.routine.maxCandidates. */
  attempts?: number;
  log?: (line: string) => void;
};

export type FixOutcome = { ok: boolean; attempts: number; diff?: string; trajectory: string; failedAt?: string; lastLog?: string };

const TRAJECTORY_DIR = ".tfa/trajectories"; // contract §2
/** Rough chars-per-token for budgeting before a tokenizer is wired in (heuristic, not from a paper). */
const CHARS_PER_TOKEN = 4;


export async function fix(o: FixOptions): Promise<FixOutcome> {
  const { repo, task, cfg } = o;
  const log = o.log ?? (() => {});
  const reader = o.reader ?? new HttpReader(cfg.slots);
  const maxAttempts = o.attempts ?? cfg.budgets.slow.routine.maxCandidates;
  if (!fs.existsSync(path.join(repo, ".git"))) throw new Error(`${repo} is not a git repository (tfa needs git so every change is revertable)`);

  const traj = new Trajectory(repo, TRAJECTORY_DIR, task);
  log(`trajectory: ${path.relative(repo, traj.file)}`);

  // Pre-flight: current type-check output is the best cheap signal of what is wrong.
  const cmds = detectCommands(repo, cfg.cascade);
  const tsc = cmds.find((c) => c.stage === "typecheck");
  let tscLog: string | undefined;
  if (tsc) {
    log(`pre-flight: ${tsc.cmd}`);
    const r = await exec(tsc.cmd, repo, cfg.timeouts.typecheck ?? 300_000);
    tscLog = tailLog(r.out, 8000);
    log(r.code === 0 ? "pre-flight: type-check clean" : `pre-flight: type-check failing (${r.out.split("\n").filter((l) => /error TS/.test(l)).length} errors)`);
  }

  const ctx = buildContext(repo, task, tscLog, cfg.budgets.contextTokens * CHARS_PER_TOKEN);
  traj.write({ t: "context", at: Date.now(), chars: ctx.text.length, files: ctx.files });

  const conversation: Msg[] = [];
  const prompt = () => buildPrompt({ system: FIX_SYSTEM, core: loadCore(repo), context: ctx.text, task: `${task}\n\nProduce the edits now.`, conversation });

  let lastCascade: CascadeResult | undefined;
  let lastReason = "";
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    log(`\nattempt ${attempt}/${maxAttempts} → worker`);
    // Diversify on retry (+0.2 per attempt is a design choice), capped at CodeT's sampling temperature 0.8 — arXiv:2207.10397 §A
    const temperature = Math.min(0.8, cfg.slots.worker.temperature + (attempt - 1) * 0.2);
    const messages = prompt();
    traj.write({ t: "prompt", at: Date.now(), attempt, model: "worker", messages });
    const res = await reader.chat("worker", messages, { temperature, idSlot: 0, attempt });
    traj.write({ t: "response", at: Date.now(), attempt, model: "worker", text: res.text, ms: res.ms, promptTokens: res.promptTokens, completionTokens: res.completionTokens });
    log(`model answered in ${(res.ms / 1000).toFixed(1)}s (${res.completionTokens ?? "?"} tokens)`);

    let edits: Edit[];
    try {
      edits = parseEdits(res.text);
    } catch (err) {
      lastReason = `could not parse edits: ${(err as Error).message}`;
      traj.write({ t: "apply", at: Date.now(), attempt, ok: false, reason: lastReason, touched: [] });
      log(`✗ apply: ${lastReason}`);
      conversation.push({ role: "assistant", content: res.text }, { role: "user", content: parseFailedFeedback(lastReason) });
      continue;
    }

    const applied = applyEdits(repo, edits);
    traj.write({ t: "apply", at: Date.now(), attempt, ok: applied.ok, reason: applied.ok ? undefined : applied.reason, touched: applied.touched });
    if (!applied.ok) {
      lastReason = applied.reason;
      log(`✗ apply: ${applied.reason}`);
      conversation.push({ role: "assistant", content: res.text }, { role: "user", content: applyFailedFeedback(applied.reason) });
      continue;
    }
    log(`applied: ${applied.touched.join(", ")}`);

    lastCascade = await runCascade(repo, cfg, {
      commands: cmds,
      onStage: (s) => {
        log(formatStage(s));
        traj.write({ t: "stage", at: Date.now(), attempt, stage: s.stage, ok: s.ok, ms: s.ms, log: s.log });
      },
    });

    if (lastCascade.ok) {
      const diff = await gitDiff(repo);
      traj.write({ t: "result", at: Date.now(), ok: true, attempt, diff });
      if (o.keep === false) applied.backup.restore();
      log(`\n✓ verified on attempt ${attempt} in ${(lastCascade.totalMs / 1000).toFixed(1)}s`);
      return { ok: true, attempts: attempt, diff, trajectory: traj.file };
    }

    const failed = lastCascade.stages.find((s) => !s.ok && !s.skipped)!;
    lastReason = `${failed.stage} failed`;
    applied.backup.restore();
    log(`rolled back; ${failed.stage} failed`);
    conversation.push({ role: "assistant", content: res.text }, { role: "user", content: stageFailedFeedback(failed.stage, failed.log) });
  }

  traj.write({ t: "result", at: Date.now(), ok: false, attempt: maxAttempts, note: lastReason });
  const failedAt = lastCascade?.failedAt ?? "apply";
  log(`\n✗ no verified solution in ${maxAttempts} attempts (last: ${lastReason})`);
  return { ok: false, attempts: maxAttempts, trajectory: traj.file, failedAt, lastLog: lastCascade?.stages.find((s) => !s.ok && !s.skipped)?.log };
}

async function gitDiff(repo: string): Promise<string> {
  const r = await exec("git --no-pager diff", repo, 20_000);
  const untracked = await exec("git ls-files --others --exclude-standard", repo, 20_000);
  let diff = r.out;
  for (const f of untracked.out.split("\n").map((s) => s.trim()).filter(Boolean)) {
    if (f.startsWith(".tfa/")) continue;
    const abs = path.join(repo, f);
    if (fs.existsSync(abs) && fs.statSync(abs).size < 200_000) diff += `\n--- /dev/null\n+++ b/${f}\n` + fs.readFileSync(abs, "utf8").split("\n").map((l) => "+" + l).join("\n") + "\n";
  }
  return diff;
}
