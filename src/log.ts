/**
 * Trajectory log: one JSONL file per task, every prompt, candidate and cascade result.
 * This is the raw material for P3 (lessons, skills) and P5 (training data), so it is
 * written from day one, even though P0 only reads it by eye.
 */
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

export type TrajectoryEvent =
  | { t: "task"; id: string; at: number; repo: string; text: string }
  | { t: "context"; at: number; chars: number; files: string[] }
  | { t: "prompt"; at: number; attempt: number; model: string; messages: { role: string; content: string }[] }
  | { t: "response"; at: number; attempt: number; model: string; text: string; ms: number; promptTokens?: number; completionTokens?: number }
  | { t: "apply"; at: number; attempt: number; ok: boolean; reason?: string; touched: string[] }
  | { t: "stage"; at: number; attempt: number; stage: string; ok: boolean; ms: number; log: string }
  | { t: "result"; at: number; ok: boolean; attempt: number; diff?: string; note?: string };

export class Trajectory {
  readonly id: string;
  readonly file: string;
  constructor(repo: string, dir: string, text: string) {
    this.id = new Date().toISOString().replace(/[:.]/g, "-") + "-" + crypto.randomBytes(3).toString("hex");
    const abs = path.resolve(repo, dir);
    fs.mkdirSync(abs, { recursive: true });
    this.file = path.join(abs, `${this.id}.jsonl`);
    this.write({ t: "task", id: this.id, at: Date.now(), repo, text });
  }
  write(ev: TrajectoryEvent) {
    fs.appendFileSync(this.file, JSON.stringify(ev) + "\n");
  }
}

export function newId(): string {
  return crypto.randomBytes(4).toString("hex");
}
