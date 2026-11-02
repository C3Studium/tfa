/**
 * Trajectories — contract §5: `<repo>/.tfa/trajectories/<taskId>.jsonl`, one event per line, `t` = type.
 * The system's raw memory and the sole source of training data (P5), written from the first task.
 *
 * Rules enforced here:
 *  - append-only: the file is created with `wx` and only ever appended to; there is no rewrite method;
 *  - `log` fields are tails: ≤ 6 kB, ANSI stripped;
 *  - schema marker: the first line (the `task` event) carries `"schema": 1` (contract §11; ASK-HUMAN Q4);
 *  - costs (§5 `Costs`) are accumulated from the events themselves, so a `result` always carries them.
 * Every write goes through the guarded file layer (I-1).
 */
import fs from "node:fs";
import path from "node:path";
import { appendFile, createFile } from "./files.js";
import { newId } from "./ids.js";
import type { EventSink, PromptEvent, ResponseEvent } from "./llm/client.js";
import { SCHEMA, type Candidate, type Costs, type Edit, type GateFeatures, type Id, type LessonDelta, type Ms, type PlaybookDelta, type Stage } from "./types.js";

export type Origin = "user" | "queue" | "bench" | "sleep";

export type Ev =
  | { t: "task"; id: Id; at: Ms; repo: string; text: string; origin: Origin; schema?: number }
  | { t: "decompose"; at: Ms; subtasks: { id: Id; text: string; touches: string[] }[] }
  | { t: "gate"; at: Ms; subtaskId: Id; decision: "fast" | "slow" | "ask"; features: GateFeatures; score: number; threshold: number }
  | { t: "retrieve"; at: Ms; subtaskId: Id; query: string; hits: { chunkId: Id; score: number; cited: boolean }[]; tokens: number }
  | PromptEvent
  | ResponseEvent
  | { t: "candidate"; at: Ms; subtaskId: Id; candidateId: Id; planId?: Id; parentId?: Id; edits: Edit[] }
  | { t: "stage"; at: Ms; candidateId: Id; stage: Stage; ok: boolean; skipped?: boolean; ms: number; log: string }
  | { t: "verify"; at: Ms; candidateId: Id; ok: boolean; failedAt?: Stage; totalMs: number }
  | { t: "reflect"; at: Ms; subtaskId: Id; lessons: LessonDelta[]; playbook: PlaybookDelta[] }
  | { t: "save"; at: Ms; subtaskId: Id; skillId?: Id; noteOps: Id[]; trainingExample?: Id; preferencePair?: Id }
  | { t: "ask"; at: Ms; subtaskId: Id; question: string; answeredAt?: Ms; answer?: string }
  | { t: "correction"; at: Ms; subtaskId: Id; candidateId: Id; humanDiff: string }
  | { t: "result"; at: Ms; ok: boolean; subtaskId?: Id; candidateId?: Id; diff?: string; note?: string; costs: Costs };

export type EvType = Ev["t"];
export const EV_TYPES: readonly EvType[] = ["task", "decompose", "gate", "retrieve", "prompt", "response", "candidate", "stage", "verify", "reflect", "save", "ask", "correction", "result"];

export const LOG_TAIL_BYTES = 6 * 1024; // contract §5: log fields are tails ≤ 6 kB

// CSI and OSC escape sequences (colours, cursor moves, hyperlinks) and single-char escapes.
// eslint-disable-next-line no-control-regex
const ANSI = /\u001b\[[0-?]*[ -/]*[@-~]|\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)|\u001b[@-Z\\-_]/g;

export function stripAnsi(s: string): string {
  return s.replace(ANSI, "");
}

/** ANSI-stripped, LF-normalised tail of at most `maxBytes` UTF-8 bytes, with a marker when cut. */
export function tail(s: string, maxBytes = LOG_TAIL_BYTES): string {
  const clean = stripAnsi(s).replace(/\r\n/g, "\n");
  if (Buffer.byteLength(clean, "utf8") <= maxBytes) return clean;
  const marker = "…(truncated)…\n";
  let t = clean.slice(-(maxBytes - Buffer.byteLength(marker, "utf8")));
  while (Buffer.byteLength(marker + t, "utf8") > maxBytes) t = t.slice(1);
  return marker + t;
}

export function trajectoryDir(repo: string): string {
  return path.join(repo, ".tfa", "trajectories"); // contract §2
}

export class Trajectory implements EventSink {
  private tokensIn = 0;
  private tokensOut = 0;
  private seniorCalls = 0;
  private stagesRun = 0;

  private constructor(
    readonly repo: string,
    readonly id: Id,
    readonly file: string,
    readonly startedAt: Ms,
  ) {}

  /** Start a task: create `<taskId>.jsonl` (never overwrites) and write the `task` event with the schema marker. */
  static create(repo: string, task: { text: string; origin: Origin; id?: Id }): Trajectory {
    const at = Date.now();
    const id = task.id ?? newId("task", at);
    const file = path.join(trajectoryDir(repo), `${id}.jsonl`);
    const first: Ev = { t: "task", id, at, repo: path.resolve(repo), text: task.text, origin: task.origin, schema: SCHEMA };
    createFile(repo, file, JSON.stringify(first) + "\n");
    return new Trajectory(repo, id, file, at);
  }

  append(ev: Ev): void {
    if (!EV_TYPES.includes(ev.t)) throw new Error(`unknown trajectory event type: ${String((ev as { t: unknown }).t)}`);
    if (ev.t === "task") throw new Error("the task event is written once, by Trajectory.create");
    let out: Ev = ev;
    if (ev.t === "stage") {
      out = { ...ev, log: tail(ev.log) };
      if (!ev.skipped) this.stagesRun++;
    } else if (ev.t === "response") {
      this.tokensIn += ev.promptTokens ?? 0;
      this.tokensOut += ev.completionTokens ?? 0;
      if (ev.slot === "senior") this.seniorCalls++;
    }
    appendFile(this.repo, this.file, JSON.stringify(out) + "\n");
  }

  costs(now: Ms = Date.now()): Costs {
    return { wallMs: now - this.startedAt, promptTokens: this.tokensIn, completionTokens: this.tokensOut, seniorCalls: this.seniorCalls, stagesRun: this.stagesRun };
  }

  candidate(c: Candidate): void {
    this.append({ t: "candidate", at: Date.now(), subtaskId: c.subtaskId, candidateId: c.id, planId: c.planId, parentId: c.parentId, edits: c.edits });
  }

  /** Write the final `result` event, with costs, and return it. */
  result(r: { ok: boolean; subtaskId?: Id; candidateId?: Id; diff?: string; note?: string }): Extract<Ev, { t: "result" }> {
    const at = Date.now();
    const ev = { t: "result" as const, at, ...r, costs: this.costs(at) };
    this.append(ev);
    return ev;
  }
}

/** Read a trajectory: CRLF-safe, blank lines ignored, accepts a `{"t":"schema"}` header line (ASK-HUMAN Q4 option B). */
export function readTrajectory(file: string): Ev[] {
  const out: Ev[] = [];
  const lines = fs.readFileSync(file, "utf8").replace(/^﻿/, "").split(/\r?\n/);
  lines.forEach((line, i) => {
    if (!line.trim()) return;
    let v: { t?: string; schema?: number };
    try {
      v = JSON.parse(line) as { t?: string; schema?: number };
    } catch {
      throw new Error(`${file}:${i + 1}: not JSON`);
    }
    if (v.schema !== undefined && v.schema > SCHEMA) throw new Error(`${file}: schema ${v.schema} is newer than this tfa (schema ${SCHEMA})`);
    if (v.t === "schema") return;
    out.push(v as Ev);
  });
  return out;
}
