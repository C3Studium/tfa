/**
 * The file layer — every write the tool makes goes through here (enforced by test/guard.test.ts).
 *
 * I-1: human-owned paths are never written by any module; changes go to proposals/<id>.patch.
 *   - always:            .tfa/bench/**   (benchmark tasks, holdout, runs)
 *   - in tfa itself:     src/verify/**, src/bench/**   (the verifier and the bench runner)
 *   - lesson-tests dir:  added in P3 when its path is fixed
 * Matching is case-insensitive and done on both the lexical and the real (link-resolved) path,
 * so `SRC/Verify/x.ts`, `src/../src/verify/x.ts` and a junction into src/verify are all refused.
 */
import fs from "node:fs";
import path from "node:path";

export const HUMAN_OWNED_ALWAYS = [".tfa/bench/"] as const;
export const HUMAN_OWNED_IN_TFA = ["src/verify/", "src/bench/"] as const;

export class HumanOwnedPathError extends Error {
  constructor(
    readonly rel: string,
    readonly pattern: string,
  ) {
    super(`refused: ${rel} is human-owned (${pattern}**, invariant I-1); write a proposal to proposals/<id>.patch instead`);
  }
}

const WIN = process.platform === "win32";

export function within(root: string, abs: string): boolean {
  const r = root.endsWith(path.sep) ? root : root + path.sep;
  return WIN ? abs.toLowerCase().startsWith(r.toLowerCase()) : abs.startsWith(r);
}

/** Realpath of the nearest existing ancestor (the file itself may not exist yet). */
export function realAncestor(abs: string): string {
  let p = abs;
  const rest: string[] = [];
  while (!fs.existsSync(p)) {
    const parent = path.dirname(p);
    if (parent === p) break;
    rest.unshift(path.basename(p));
    p = parent;
  }
  return path.join(fs.realpathSync.native(p), ...rest);
}

/** True if `root` is the tfa repository itself (its package.json is named "tfa"). */
export function isTfaRepo(root: string): boolean {
  try {
    const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8").replace(/^﻿/, "")) as { name?: string };
    return pkg.name === "tfa";
  } catch {
    return false;
  }
}

function relPosix(root: string, abs: string): string {
  return path.relative(root, abs).split(path.sep).join("/").toLowerCase();
}

/** The human-owned pattern that covers `abs` inside `root`, or null. */
export function humanOwned(root: string, abs: string): string | null {
  const patterns: string[] = [...HUMAN_OWNED_ALWAYS, ...(isTfaRepo(root) ? HUMAN_OWNED_IN_TFA : [])];
  const rootAbs = path.resolve(root);
  const target = path.resolve(rootAbs, abs);
  const candidates = [relPosix(rootAbs, target)];
  if (fs.existsSync(rootAbs)) candidates.push(relPosix(fs.realpathSync.native(rootAbs), realAncestor(target)));
  for (const rel of candidates) {
    for (const p of patterns) if (rel === p.slice(0, -1) || rel.startsWith(p)) return p;
  }
  return null;
}

export function assertWritable(root: string, abs: string): void {
  const p = humanOwned(root, abs);
  if (p) throw new HumanOwnedPathError(path.relative(root, path.resolve(root, abs)).split(path.sep).join("/"), p);
}

// ---- guarded writes: the only fs mutations allowed in src/ outside human-owned dirs ----

export function writeFile(root: string, abs: string, data: string | Uint8Array): void {
  assertWritable(root, abs);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, data);
}

/** Create a new file; fails if it exists (`wx`). */
export function createFile(root: string, abs: string, data: string): void {
  assertWritable(root, abs);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, data, { flag: "wx" });
}

export function appendFile(root: string, abs: string, data: string): void {
  assertWritable(root, abs);
  fs.appendFileSync(abs, data, { flag: "a" });
}

export function ensureDir(root: string, abs: string): void {
  assertWritable(root, abs);
  fs.mkdirSync(abs, { recursive: true });
}

export function removeFile(root: string, abs: string): void {
  assertWritable(root, abs);
  if (fs.existsSync(abs)) fs.rmSync(abs);
}
