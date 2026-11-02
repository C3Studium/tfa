/**
 * The verifier cascade: stages ordered by cost; a candidate dies at its first failure,
 * so the expensive stages (build, browser) see only survivors.
 */
import type { CascadeStage, Config } from "../config.js";
import { detectCommands, type StageCommand } from "./detect.js";
import { exec, tailLog } from "./exec.js";

export type StageResult = { stage: CascadeStage; ok: boolean; skipped?: boolean; cmd?: string; ms: number; log: string };
export type CascadeResult = { ok: boolean; stages: StageResult[]; failedAt?: CascadeStage; totalMs: number };

export async function runCascade(
  repo: string,
  cfg: Config,
  opts: { stages?: CascadeStage[]; onStage?: (r: StageResult) => void; commands?: StageCommand[] } = {},
): Promise<CascadeResult> {
  const stages = opts.stages ?? cfg.cascade;
  const cmds = opts.commands ?? detectCommands(repo, stages);
  const byStage = new Map(cmds.map((c) => [c.stage, c]));
  const results: StageResult[] = [];
  const t0 = Date.now();
  for (const stage of stages) {
    const c = byStage.get(stage);
    if (!c) {
      const r: StageResult = { stage, ok: true, skipped: true, ms: 0, log: "(no command detected; skipped)" };
      results.push(r);
      opts.onStage?.(r);
      continue;
    }
    const timeout = cfg.timeouts[stage] ?? 300_000;
    const x = await exec(c.cmd, repo, timeout);
    const ok = x.code === 0 && !x.timedOut;
    const r: StageResult = { stage, ok, cmd: c.cmd, ms: x.ms, log: tailLog(x.out) + (x.timedOut ? `\n(timed out after ${timeout} ms)` : "") };
    results.push(r);
    opts.onStage?.(r);
    if (!ok) return { ok: false, stages: results, failedAt: stage, totalMs: Date.now() - t0 };
  }
  return { ok: true, stages: results, totalMs: Date.now() - t0 };
}

/** A human-readable one-liner per stage, for the CLI. */
export function formatStage(r: StageResult): string {
  const mark = r.skipped ? "–" : r.ok ? "✓" : "✗";
  const secs = (r.ms / 1000).toFixed(1);
  return `${mark} ${r.stage.padEnd(9)} ${r.skipped ? "skipped" : secs + "s"}${r.cmd ? "  (" + r.cmd + ")" : ""}`;
}
