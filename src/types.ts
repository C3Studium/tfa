/**
 * Shared types from the interface contract (docs/markdown/10-interface-contract.md, schema 1).
 * §3 identifiers, §4 slots, §7 shared types and module interfaces implemented in P0.
 * Changing a type here requires a schema bump and a migration (contract §11).
 */

// ---- §3 identifiers, time, provenance ----
export type Id = string; // `${kind}_${base32(time)}${rand4}`, see src/ids.ts
export type Ms = number; // Unix epoch milliseconds, UTC
export type Hash = string; // sha256 hex of file content or canonical JSON
export type Provenance = { taskId: Id; trajectory: string; attempt?: number; candidateId?: Id; at: Ms };

export const SCHEMA = 1;

// ---- §4 model server contract ----
export type Slot = "worker" | "senior" | "draft" | "embed" | "rerank";
export const SLOTS: readonly Slot[] = ["worker", "senior", "draft", "embed", "rerank"];
export type SlotConfig = { url: string; maxTokens: number; temperature: number; ctx: number; family: string; resident: boolean };

export type Msg = { role: "system" | "user" | "assistant"; content: string };

/** Edit format (§4): `find` must match exactly once; paths are repo-relative and never escape the repo (I-8). */
export type Edit = { path: string; find: string; replace: string } | { path: string; content: string };

// ---- §7 shared types ----
export type Stage = "apply" | "lint" | "typecheck" | "test" | "build" | "browser" | "citation";
export const STAGES: readonly Stage[] = ["apply", "lint", "typecheck", "test", "build", "browser", "citation"];

export type Candidate = { id: Id; subtaskId: Id; planId?: Id; parentId?: Id; edits: Edit[]; createdBy: Slot | "skill" };

export type StageOutcome = { stage: Stage; ok: boolean; skipped?: boolean; ms: number; log: string };
export type VerifyResult = { ok: boolean; failedAt?: Stage; stages: StageOutcome[]; totalMs: number };

export type Budget = { maxCandidates: number; maxDepth: number; maxWallMs: number; allowSenior: boolean; stopOnAgreement: boolean };

export type Costs = { wallMs: number; promptTokens: number; completionTokens: number; seniorCalls: number; stagesRun: number; wh?: number };

/** A verifier command detected for one stage of a target repo. */
export type StageCommand = { stage: Stage; cmd: string; source: string };

/** §7 Verifier. The implementation lives in src/verify (HUMAN-OWNED, I-1). */
export interface Verifier {
  stages(repo: string): StageCommand[];
  run(c: Candidate, repo: string, o?: { upTo?: Stage }): Promise<VerifyResult>;
}

// ---- §6.4 gate features (logged in Ev.gate; the gate itself is P2) ----
export type GateFeatures = {
  simTask: number;
  simSkill: number;
  skillCovered: boolean;
  footprintKnown: number;
  selfEstimate: number;
  size: number;
};

/** Defined by the reflector in P3 (Pack A, ACE bullet deltas); opaque until then. */
export type LessonDelta = Record<string, unknown>;
export type PlaybookDelta = Record<string, unknown>;
