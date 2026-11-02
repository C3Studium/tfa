/**
 * The edit format the model must produce (contract §4 "Edit format") and how it is applied.
 *
 * Small models are unreliable at unified diffs. A JSON list of exact find/replace edits
 * (or whole-file writes for new/small files) is far more robust, and a failed `find`
 * is a cheap, early rejection — cascade stage `apply`, always first, cost ≈ 0.
 *
 * I-1: edits to human-owned paths are refused by the file layer (src/files.ts).
 * I-8: edits never escape the target repo, never touch `.git`, and `find` must match exactly once.
 * Applying is in place with a backup, so a failed candidate is rolled back exactly.
 */
import fs from "node:fs";
import path from "node:path";
import { assertWritable, realAncestor, removeFile, within, writeFile } from "../files.js";
import type { Edit } from "../types.js";

export type { Edit };

export type ApplyResult = { ok: true; touched: string[] } | { ok: false; reason: string; touched: string[] };

export const EDIT_FORMAT_INSTRUCTIONS = `Respond with ONLY a JSON array of edits, no prose, no markdown fence. Each edit is either
{"path": "<relative file path>", "find": "<exact existing text, including whitespace>", "replace": "<new text>"}
or, for a new file or a complete rewrite of a small file,
{"path": "<relative file path>", "content": "<entire file content>"}.
Rules: "find" must match the file text exactly once. Keep edits minimal. Do not touch files that are not needed.`;

/** Tolerant parse: strips code fences and leading prose, finds the first JSON array. */
export function parseEdits(text: string): Edit[] {
  let s = text.trim();
  s = s.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "");
  const start = s.indexOf("[");
  const end = s.lastIndexOf("]");
  if (start < 0 || end < start) throw new Error("no JSON array found in model output");
  const arr = JSON.parse(s.slice(start, end + 1)) as unknown;
  if (!Array.isArray(arr)) throw new Error("edits is not an array");
  const edits: Edit[] = [];
  for (const e of arr) {
    if (!e || typeof e !== "object" || typeof (e as Edit).path !== "string") throw new Error("edit without path");
    const o = e as Record<string, unknown>;
    if (typeof o.content === "string") edits.push({ path: o.path as string, content: o.content });
    else if (typeof o.find === "string" && typeof o.replace === "string") edits.push({ path: o.path as string, find: o.find, replace: o.replace });
    else throw new Error(`edit for ${o.path} has neither content nor find/replace`);
  }
  if (edits.length === 0) throw new Error("empty edit list");
  return edits;
}

/**
 * Resolve a model-supplied path inside `root`, or throw. Rejects absolute paths (POSIX, drive letters, UNC),
 * `..` escapes, `.git`, NUL bytes, and escapes through symlinks or directory junctions.
 */
export function safeJoin(root: string, rel: string): string {
  if (typeof rel !== "string" || rel.trim() === "") throw new Error("edit path is empty");
  if (rel.includes("\0")) throw new Error(`edit path contains a NUL byte: ${JSON.stringify(rel)}`);
  if (path.isAbsolute(rel) || path.win32.isAbsolute(rel) || /^[a-zA-Z]:/.test(rel)) throw new Error(`edit path escapes repo (absolute): ${rel}`);
  const rootAbs = path.resolve(root);
  const abs = path.resolve(rootAbs, rel);
  if (!within(rootAbs, abs)) throw new Error(`edit path escapes repo: ${rel}`);
  const segs = path.relative(rootAbs, abs).split(/[\\/]+/);
  if (segs.some((s) => s.toLowerCase() === ".git")) throw new Error(`edit path touches .git: ${rel}`);
  const rootReal = fs.realpathSync.native(rootAbs);
  if (!within(rootReal, realAncestor(abs))) throw new Error(`edit path escapes repo through a link: ${rel}`);
  return abs;
}

/**
 * Find `find` exactly once in `cur`. CRLF-safe: models see LF text, so when an LF `find` is absent from a
 * CRLF file, the CRLF form of both `find` and `replace` is tried (still exactly-once).
 */
export function replaceOnce(cur: string, find: string, replace: string, file: string): string {
  if (find === "") throw new Error(`find text is empty in ${file}`);
  let f = find;
  let r = replace;
  if (!cur.includes(f) && cur.includes("\r\n") && f.includes("\n") && !f.includes("\r\n")) {
    f = f.replace(/\n/g, "\r\n");
    r = r.replace(/\r?\n/g, "\r\n");
  }
  const first = cur.indexOf(f);
  if (first < 0) throw new Error(`find text not present in ${file}`);
  if (cur.indexOf(f, first + 1) >= 0) throw new Error(`find text is ambiguous (occurs more than once) in ${file}`);
  return cur.slice(0, first) + r + cur.slice(first + f.length);
}

export class Backup {
  private saved = new Map<string, string | null>(); // null = file did not exist
  constructor(readonly root: string) {}
  remember(abs: string) {
    if (this.saved.has(abs)) return;
    this.saved.set(abs, fs.existsSync(abs) ? fs.readFileSync(abs, "utf8") : null);
  }
  restore() {
    for (const [abs, content] of this.saved) {
      if (content === null) removeFile(this.root, abs);
      else writeFile(this.root, abs, content);
    }
  }
  get files(): string[] {
    return [...this.saved.keys()];
  }
}

/** Remember the current content of every file a candidate would touch (for --dry rollback). */
export function snapshot(root: string, edits: Edit[]): Backup {
  const b = new Backup(root);
  for (const e of edits) b.remember(safeJoin(root, e.path));
  return b;
}

/** Apply edits in place. On any failure, roll back everything and report why. */
export function applyEdits(root: string, edits: Edit[]): ApplyResult & { backup: Backup } {
  const backup = new Backup(root);
  const touched: string[] = [];
  try {
    for (const e of edits) {
      const abs = safeJoin(root, e.path);
      assertWritable(root, abs); // I-1
      backup.remember(abs);
      if ("content" in e) writeFile(root, abs, e.content);
      else {
        if (!fs.existsSync(abs)) throw new Error(`file not found: ${e.path}`);
        writeFile(root, abs, replaceOnce(fs.readFileSync(abs, "utf8"), e.find, e.replace, e.path));
      }
      const rel = path.relative(root, abs).split(path.sep).join("/");
      if (!touched.includes(rel)) touched.push(rel);
    }
    return { ok: true, touched, backup };
  } catch (err) {
    backup.restore();
    return { ok: false, reason: (err as Error).message, touched, backup };
  }
}
