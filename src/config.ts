/**
 * Configuration — contract §10.
 *
 * Sources, in increasing precedence:
 *   1. DEFAULT_CONFIG below
 *   2. tool config: tfa.config.json at the root of the tfa package
 *   3. project override: <target-repo>/.tfa/config.json
 *   4. environment variables, which override slot URLs only: TFA_WORKER_URL, TFA_SENIOR_URL,
 *      TFA_DRAFT_URL, TFA_EMBED_URL, TFA_RERANK_URL
 * Objects merge deeply; arrays and scalars replace.
 *
 * Unlabelled numbers are design defaults to be tuned by the benchmark (contract, "Scope").
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SLOTS, STAGES, type Budget, type Slot, type SlotConfig, type Stage } from "./types.js";

export type Difficulty = "routine" | "moderate" | "hard";

export type Config = {
  slots: Record<Slot, SlotConfig>;
  cascade: Stage[];
  timeouts: Partial<Record<Stage, number>>;
  budgets: {
    fast: { maxCandidates: 1; maxWallMs: 60_000 };
    slow: Record<Difficulty, Budget>;
    queue: Budget;
    contextTokens: number;
    coreTokens: number;
  };
  gate: { mode: "rule" | "model"; fastThreshold: number; dailySlowBudgetMs: number };
  skills: { matchThreshold: number; dedupeThreshold: number };
  notebook: { recencyDecay: number; retrieveK: number };
  sleep: { idleAfterMs: number; maxMs: number; maxTokens: number; decayHalfLifeDays: number };
  train: { weekday: number; minExamples: number; r: number; alpha: number; lr: number; dpoBeta: number; replayFraction: number; activateDelta: number };
  bench: { devCount: 20; holdoutCount: 10; rotateDay: number; contaminationThreshold: 0.95 };
  decompose: { seniorThreshold: number };
};

/** Legacy alias: src/verify (human-owned) imports `CascadeStage` from here. */
export type CascadeStage = Stage;

const MIN = 60_000;

export const DEFAULT_CONFIG: Config = {
  slots: {
    // Ports match `tfa models`. Worker temperature 0.0 — arXiv:2405.15793 §(SWE-agent; Pack H §2.4 "Sampling temperature (worker)")
    worker: { url: "http://127.0.0.1:8080", maxTokens: 4096, temperature: 0.0, ctx: 32_768, family: "qwen3.5", resident: true },
    senior: { url: "http://127.0.0.1:8081", maxTokens: 8192, temperature: 0.0, ctx: 32_768, family: "qwen3.6", resident: false },
    // The draft is loaded inside the worker server (--model-draft); it has no endpoint of its own.
    draft: { url: "http://127.0.0.1:8080", maxTokens: 0, temperature: 0.0, ctx: 32_768, family: "qwen3.5", resident: true },
    embed: { url: "http://127.0.0.1:8082", maxTokens: 0, temperature: 0, ctx: 8192, family: "qwen3-embedding", resident: true },
    rerank: { url: "http://127.0.0.1:8083", maxTokens: 0, temperature: 0, ctx: 8192, family: "reranker", resident: true },
  },
  // "apply" is implicit and always first (contract §4 "Edit format"); stages ordered cheapest-first.
  cascade: ["lint", "typecheck", "test", "build"],
  timeouts: { apply: 10_000, lint: 2 * MIN, typecheck: 5 * MIN, test: 10 * MIN, build: 15 * MIN, browser: 10 * MIN, citation: 1 * MIN },
  budgets: {
    fast: { maxCandidates: 1, maxWallMs: 60_000 },
    // Pack C §2.4 "Suggested local schedule" (derived there, not a paper result). routine = 1 plan + ≤3 revisions,
    // matching Reflexion's retry stop after 3 consecutive failures — arXiv:2303.11366 §4.2
    slow: {
      routine: { maxCandidates: 4, maxDepth: 1, maxWallMs: 10 * MIN, allowSenior: false, stopOnAgreement: true },
      moderate: { maxCandidates: 12, maxDepth: 3, maxWallMs: 30 * MIN, allowSenior: true, stopOnAgreement: true },
      hard: { maxCandidates: 40, maxDepth: 5, maxWallMs: 60 * MIN, allowSenior: true, stopOnAgreement: true },
    },
    queue: { maxCandidates: 40, maxDepth: 5, maxWallMs: 120 * MIN, allowSenior: true, stopOnAgreement: true },
    // 13k-token retrieved-context tier — arXiv:2310.06770 §Table 3 / Fig. 5 (resolve rate falls with more context);
    // also ≈ 40% of a 32k window (Pack G §2.4 "Hard context budget", derivation).
    contextTokens: 13_000,
    coreTokens: 2_000, // contract §6.1
  },
  // Cold-start rule: skillCovered || simTask ≥ 0.85 → fast (contract §6.4).
  gate: { mode: "rule", fastThreshold: 0.85, dailySlowBudgetMs: 180 * MIN },
  skills: { matchThreshold: 0.8, dedupeThreshold: 0.92 }, // contract §6.2
  notebook: {
    recencyDecay: 0.995, // arXiv:2304.03442 §4.1 (Generative Agents: decay 0.995 per hour)
    retrieveK: 10, // arXiv:2502.12110 §4 (A-MEM: k=10, sweep 10–50 plateaus; Table 8)
  },
  sleep: { idleAfterMs: 15 * MIN, maxMs: 120 * MIN, maxTokens: 2_000_000, decayHalfLifeDays: 30 },
  train: {
    weekday: 0,
    minExamples: 50,
    r: 16, // arXiv:2305.14314 §A.1 (QLoRA: r unrelated to performance when all layers adapted)
    alpha: 16, // arXiv:2305.14314 §B.2 (fix alpha, tune LR)
    lr: 2e-4, // arXiv:2305.14314 §Table 9 (7B/13B, constant)
    dpoBeta: 0.1, // arXiv:2305.18290 §B (DPO appendix B)
    replayFraction: 0.2, // contract §6.7
    activateDelta: 0.02, // contract §6.6
  },
  // contaminationThreshold — arXiv:2406.18665 §(contamination check; section number not in Pack H) UNVERIFIED
  bench: { devCount: 20, holdoutCount: 10, rotateDay: 1, contaminationThreshold: 0.95 },
  decompose: { seniorThreshold: 200 },
};

/** Root of the tfa package (works from src/ via tsx and from dist/ after build). */
export const TFA_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export type ConfigSources = { tool?: string; project?: string; env: string[] };
export type LoadedConfig = Config & { sources: ConfigSources };

type Plain = Record<string, unknown>;
const isPlain = (v: unknown): v is Plain => !!v && typeof v === "object" && !Array.isArray(v);

/** Deep merge: objects merge, arrays and scalars replace. Never mutates its inputs. */
export function deepMerge<T>(base: T, over: unknown): T {
  if (!isPlain(base) || !isPlain(over)) return structuredClone(over === undefined ? base : over) as T;
  const out: Plain = structuredClone(base);
  for (const [k, v] of Object.entries(over)) {
    if (v === undefined) continue;
    out[k] = isPlain(out[k]) && isPlain(v) ? deepMerge(out[k], v) : structuredClone(v);
  }
  return out as T;
}

function readJson(file: string): Plain {
  // Strip a UTF-8 BOM (Notepad on Windows writes one).
  const raw = fs.readFileSync(file, "utf8").replace(/^﻿/, "");
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch (err) {
    throw new Error(`invalid JSON in ${file}: ${(err as Error).message}`);
  }
  if (!isPlain(v)) throw new Error(`${file} must contain a JSON object`);
  const { schema: _schema, $comment: _comment, ...rest } = v;
  return rest;
}

export function validateConfig(cfg: Config): void {
  for (const s of SLOTS) {
    const sc = cfg.slots[s];
    if (!sc || typeof sc.url !== "string" || !/^https?:\/\//.test(sc.url)) throw new Error(`config: slots.${s}.url must be an http(s) URL`);
    for (const k of ["maxTokens", "temperature", "ctx"] as const) if (typeof sc[k] !== "number") throw new Error(`config: slots.${s}.${k} must be a number`);
  }
  for (const st of cfg.cascade) if (!STAGES.includes(st)) throw new Error(`config: unknown cascade stage "${st}"`);
  if (cfg.cascade.includes("apply")) throw new Error(`config: "apply" is implicit and always first; remove it from cascade`);
}

export function loadConfig(repoDir: string, o: { toolRoot?: string; env?: NodeJS.ProcessEnv } = {}): LoadedConfig {
  const env = o.env ?? process.env;
  const sources: ConfigSources = { env: [] };
  let cfg: Config = structuredClone(DEFAULT_CONFIG);

  const tool = path.join(o.toolRoot ?? TFA_ROOT, "tfa.config.json");
  if (fs.existsSync(tool)) {
    cfg = deepMerge(cfg, readJson(tool));
    sources.tool = tool;
  }
  const project = path.join(repoDir, ".tfa", "config.json");
  if (fs.existsSync(project)) {
    cfg = deepMerge(cfg, readJson(project));
    sources.project = project;
  }
  for (const s of SLOTS) {
    const name = `TFA_${s.toUpperCase()}_URL`;
    const v = env[name];
    if (v) {
      cfg.slots[s].url = v.replace(/\/+$/, "");
      sources.env.push(name);
    }
  }
  validateConfig(cfg);
  return Object.assign(cfg, { sources });
}
