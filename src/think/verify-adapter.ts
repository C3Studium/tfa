/**
 * TRANSITIONAL — removed when proposals/P0-3-verifier.patch is applied (ASK-HUMAN Q1).
 *
 * Presents the human-owned cascade (src/verify, read-only to agents) through the contract §7 `Verifier`
 * interface. It never weakens a verdict; it only adds the two contract rules the P0 cascade lacks:
 *  - stage `apply` always first (contract §4 "Edit format"), via the I-8/I-1-guarded applyEdits;
 *  - I-4: a skipped stage is reported `skipped` and `ok:false`, never as passed; and if no stage ran at all
 *    the candidate is not verified (fail closed, contract §1 principle 7).
 * Verified edits stay in the working tree; a failed candidate is rolled back exactly.
 */
import type { Config } from "../config.js";
import { applyEdits } from "../llm/edits.js";
import type { Candidate, Stage, StageCommand, StageOutcome, Verifier, VerifyResult } from "../types.js";
import { runCascade } from "../verify/cascade.js";
import { detectCommands } from "../verify/detect.js";

export function stagesUpTo(cascade: Stage[], upTo?: Stage): Stage[] {
  if (!upTo) return cascade;
  if (upTo === "apply") return [];
  const i = cascade.indexOf(upTo);
  return i < 0 ? cascade : cascade.slice(0, i + 1);
}

/** I-4 normalisation of a stage list: skipped is never ok; ok requires at least one stage to have run. */
export function aggregate(stages: StageOutcome[], requested: number): { ok: boolean; failedAt?: Stage } {
  const failed = stages.find((s) => !s.ok && !s.skipped);
  if (failed) return { ok: false, failedAt: failed.stage };
  const ran = stages.filter((s) => s.stage !== "apply" && !s.skipped).length;
  return { ok: requested === 0 || ran > 0 };
}

export class TransitionalVerifier implements Verifier {
  constructor(
    private readonly cfg: Config,
    private readonly onStage?: (s: StageOutcome) => void,
  ) {}

  stages(repo: string): StageCommand[] {
    return detectCommands(repo, this.cfg.cascade);
  }

  async run(c: Candidate, repo: string, o: { upTo?: Stage } = {}): Promise<VerifyResult> {
    const t0 = Date.now();
    const applied = applyEdits(repo, c.edits);
    const apply: StageOutcome = {
      stage: "apply",
      ok: applied.ok,
      ms: Date.now() - t0,
      log: applied.ok ? `applied ${c.edits.length} edit(s): ${applied.touched.join(", ") || "(none)"}` : applied.reason,
    };
    this.onStage?.(apply);
    if (!applied.ok) return { ok: false, failedAt: "apply", stages: [apply], totalMs: Date.now() - t0 };

    const wanted = stagesUpTo(this.cfg.cascade, o.upTo);
    const r = await runCascade(repo, this.cfg, {
      stages: wanted,
      commands: detectCommands(repo, wanted),
      onStage: (s) => this.onStage?.(toOutcome(s)),
    });
    const stages = [apply, ...r.stages.map(toOutcome)];
    const agg = aggregate(stages, wanted.length);
    if (!agg.ok) applied.backup.restore();
    return { ok: agg.ok, failedAt: agg.failedAt, stages, totalMs: Date.now() - t0 };
  }
}

function toOutcome(s: { stage: Stage; ok: boolean; skipped?: boolean; ms: number; log: string }): StageOutcome {
  return s.skipped ? { stage: s.stage, ok: false, skipped: true, ms: s.ms, log: s.log } : { stage: s.stage, ok: s.ok, ms: s.ms, log: s.log };
}
