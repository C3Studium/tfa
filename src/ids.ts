/**
 * Contract §3: ids are `${kind}_${base32(time)}${rand4}` — globally unique and sortable by time.
 * Time is 48-bit epoch ms in Crockford base32 (10 chars, fixed width, so lexical order = time order).
 */
import crypto from "node:crypto";
import type { Hash, Id } from "./types.js";

const B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"; // Crockford

export function base32Time(ms: number): string {
  let n = Math.floor(ms);
  let s = "";
  for (let i = 0; i < 10; i++) {
    s = B32[n % 32] + s;
    n = Math.floor(n / 32);
  }
  return s;
}

function rand4(): string {
  const b = crypto.randomBytes(4);
  let s = "";
  for (let i = 0; i < 4; i++) s += B32[b[i] % 32];
  return s;
}

export type IdKind = "task" | "sub" | "note" | "skill" | "cand" | "plan" | "lora" | "run" | "chunk" | "doc" | "gate" | "ex" | "pair" | "prop";

export function newId(kind: IdKind, at: number = Date.now()): Id {
  return `${kind}_${base32Time(at)}${rand4()}`;
}

export function sha256(s: string | Buffer): Hash {
  return crypto.createHash("sha256").update(s).digest("hex");
}

/** Canonical JSON: object keys sorted recursively, so equal values hash equally. */
export function canonicalJson(v: unknown): string {
  if (Array.isArray(v)) return "[" + v.map(canonicalJson).join(",") + "]";
  if (v && typeof v === "object") {
    const o = v as Record<string, unknown>;
    return "{" + Object.keys(o).filter((k) => o[k] !== undefined).sort().map((k) => JSON.stringify(k) + ":" + canonicalJson(o[k])).join(",") + "}";
  }
  return JSON.stringify(v);
}

export function hashJson(v: unknown): Hash {
  return sha256(canonicalJson(v));
}
