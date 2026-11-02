/**
 * The edit format the model must produce, and how it is applied.
 *
 * Small models are unreliable at unified diffs. A JSON list of exact find/replace edits
 * (or whole-file writes for new/small files) is far more robust, and a failed `find`
 * is a cheap, early rejection ("apply" stage of the cascade).
 *
 * Applying is in-place with a backup, so a failed candidate is rolled back exactly.
 */
import fs from "node:fs";
import path from "node:path";

export type Edit =
  | { path: string; find: string; replace: string }
  | { path: string; content: string }; // whole-file write (create or overwrite)

export type ApplyResult =
  | { ok: true; touched: string[] }
  | { ok: false; reason: string; touched: string[] };

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

function safeJoin(root: string, rel: string): string {
  const abs = path.resolve(root, rel);
  const rootAbs = path.resolve(root) + path.sep;
  if (!abs.startsWith(rootAbs)) throw new Error(`edit path escapes repo: ${rel}`);
  return abs;
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
      if (content === null) {
        if (fs.existsSync(abs)) fs.rmSync(abs);
      } else fs.writeFileSync(abs, content);
    }
  }
  get files(): string[] {
    return [...this.saved.keys()];
  }
}

/** Apply edits in place. On any failure, roll back everything and report why. */
export function applyEdits(root: string, edits: Edit[]): ApplyResult & { backup: Backup } {
  const backup = new Backup(root);
  const touched: string[] = [];
  try {
    for (const e of edits) {
      const abs = safeJoin(root, e.path);
      backup.remember(abs);
      if ("content" in e) {
        fs.mkdirSync(path.dirname(abs), { recursive: true });
        fs.writeFileSync(abs, e.content);
      } else {
        if (!fs.existsSync(abs)) throw new Error(`file not found: ${e.path}`);
        const cur = fs.readFileSync(abs, "utf8");
        const first = cur.indexOf(e.find);
        if (first < 0) throw new Error(`find text not present in ${e.path}`);
        if (cur.indexOf(e.find, first + 1) >= 0) throw new Error(`find text is ambiguous (occurs more than once) in ${e.path}`);
        fs.writeFileSync(abs, cur.slice(0, first) + e.replace + cur.slice(first + e.find.length));
      }
      touched.push(path.relative(root, abs));
    }
    return { ok: true, touched, backup };
  } catch (err) {
    backup.restore();
    return { ok: false, reason: (err as Error).message, touched, backup };
  }
}
