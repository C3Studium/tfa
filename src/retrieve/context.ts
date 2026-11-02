/**
 * Context for the model, within a hard character budget.
 * P0 strategy: run the type-checker, parse error locations, and include a window of
 * each referenced file around each error. Files the task text names are included too.
 * (P1 replaces this with version-aware retrieval over the whole library.)
 */
import fs from "node:fs";
import path from "node:path";
import { depSummary } from "../verify/detect.js";

export type Snippet = { file: string; from: number; to: number; text: string };

const TSC_LINE = /^(.+?)\((\d+),(\d+)\): error (TS\d+): (.*)$/gm;

export function parseTscErrors(log: string): { file: string; line: number; code: string; message: string }[] {
  const out: { file: string; line: number; code: string; message: string }[] = [];
  for (const m of log.matchAll(TSC_LINE)) out.push({ file: m[1].trim(), line: Number(m[2]), code: m[4], message: m[5] });
  return out;
}

export function fileWindow(repo: string, rel: string, line: number, radius = 40): Snippet | null {
  const abs = path.resolve(repo, rel);
  if (!fs.existsSync(abs)) return null;
  const lines = fs.readFileSync(abs, "utf8").split("\n");
  const from = Math.max(1, line - radius);
  const to = Math.min(lines.length, line + radius);
  const text = lines
    .slice(from - 1, to)
    .map((l, i) => `${String(from + i).padStart(5)}| ${l}`)
    .join("\n");
  return { file: rel, from, to, text };
}

/** Files mentioned by path in the task text (e.g. "src/app/page.tsx"). */
export function filesMentioned(repo: string, text: string): string[] {
  const found = new Set<string>();
  for (const m of text.matchAll(/[\w./-]+\.(?:tsx?|jsx?|mjs|cjs|json|scss|css|md)/g)) {
    const rel = m[0].replace(/^\.\//, "");
    if (fs.existsSync(path.resolve(repo, rel))) found.add(rel);
  }
  return [...found];
}

export function buildContext(repo: string, task: string, tscLog: string | undefined, budgetChars: number): { text: string; files: string[] } {
  const parts: string[] = [];
  const files = new Set<string>();
  parts.push(`Repository: ${path.basename(repo)}\nDependencies: ${depSummary(repo)}\n`);

  if (tscLog) {
    const errs = parseTscErrors(tscLog);
    if (errs.length) {
      parts.push(`Type-check errors (${errs.length}):\n` + errs.slice(0, 30).map((e) => `${e.file}:${e.line} ${e.code} ${e.message}`).join("\n"));
      const seen = new Set<string>();
      for (const e of errs.slice(0, 12)) {
        const key = `${e.file}:${Math.floor(e.line / 60)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const w = fileWindow(repo, e.file, e.line);
        if (w) {
          files.add(w.file);
          parts.push(`--- ${w.file} (lines ${w.from}-${w.to}) ---\n${w.text}`);
        }
      }
    } else parts.push("Type-check output (no parseable errors):\n" + tscLog.slice(-3000));
  }

  for (const rel of filesMentioned(repo, task)) {
    if (files.has(rel)) continue;
    const abs = path.resolve(repo, rel);
    const content = fs.readFileSync(abs, "utf8");
    if (content.length > 12_000) {
      const w = fileWindow(repo, rel, 1, 150);
      if (w) parts.push(`--- ${rel} (first ${w.to} lines of a long file) ---\n${w.text}`);
    } else parts.push(`--- ${rel} ---\n` + content.split("\n").map((l, i) => `${String(i + 1).padStart(5)}| ${l}`).join("\n"));
    files.add(rel);
  }

  let text = parts.join("\n\n");
  if (text.length > budgetChars) text = text.slice(0, budgetChars) + "\n…(context truncated to budget)…";
  return { text, files: [...files] };
}
