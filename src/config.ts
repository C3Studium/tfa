/**
 * Configuration. One JSON file (tfa.config.json) at the repo root of tfa itself or in the
 * target repo; environment variables override. Every model is a slot behind an
 * OpenAI-compatible HTTP endpoint (llama.cpp `llama-server`), so slots are swappable.
 */
import fs from "node:fs";
import path from "node:path";

export type ModelSlot = {
  /** Base URL of a llama.cpp server, e.g. http://127.0.0.1:8080 */
  url: string;
  /** Model name sent in the request (llama.cpp ignores it, but logs it). */
  name: string;
  /** Max tokens the model may generate per call. */
  maxTokens: number;
  /** Default sampling temperature. */
  temperature: number;
};

export type CascadeStage = "apply" | "lint" | "typecheck" | "test" | "build" | "browser";

export type Config = {
  worker: ModelSlot;
  senior?: ModelSlot;
  /** Stages in order of cost. "apply" is implicit and always first. */
  cascade: CascadeStage[];
  /** Per-stage timeout in ms. */
  timeouts: Partial<Record<CascadeStage, number>>;
  /** Max sequential candidates for `tfa fix` in P0. */
  maxAttempts: number;
  /** Where trajectories (JSONL) are written, relative to the target repo. */
  trajectoryDir: string;
  /** Max characters of repo context sent to the model. */
  contextBudgetChars: number;
};

export const DEFAULT_CONFIG: Config = {
  worker: { url: "http://127.0.0.1:8080", name: "worker", maxTokens: 4096, temperature: 0.2 },
  senior: { url: "http://127.0.0.1:8081", name: "senior", maxTokens: 8192, temperature: 0.3 },
  cascade: ["lint", "typecheck", "test", "build"],
  timeouts: { apply: 10_000, lint: 120_000, typecheck: 300_000, test: 600_000, build: 900_000, browser: 600_000 },
  maxAttempts: 4,
  trajectoryDir: ".tfa/trajectories",
  contextBudgetChars: 24_000,
};

export function loadConfig(repoDir: string): Config {
  const candidates = [path.join(repoDir, "tfa.config.json"), path.join(process.cwd(), "tfa.config.json")];
  let cfg: Config = structuredClone(DEFAULT_CONFIG);
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      const user = JSON.parse(fs.readFileSync(c, "utf8")) as Partial<Config>;
      cfg = { ...cfg, ...user, worker: { ...cfg.worker, ...(user.worker ?? {}) }, senior: user.senior === undefined ? cfg.senior : user.senior ? { ...cfg.senior!, ...user.senior } : undefined, timeouts: { ...cfg.timeouts, ...(user.timeouts ?? {}) } };
      break;
    }
  }
  if (process.env.TFA_WORKER_URL) cfg.worker.url = process.env.TFA_WORKER_URL;
  if (process.env.TFA_SENIOR_URL && cfg.senior) cfg.senior.url = process.env.TFA_SENIOR_URL;
  if (process.env.TFA_MAX_ATTEMPTS) cfg.maxAttempts = Number(process.env.TFA_MAX_ATTEMPTS);
  return cfg;
}
