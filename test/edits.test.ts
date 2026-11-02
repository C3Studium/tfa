import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseEdits, applyEdits } from "../src/llm/edits.js";

describe("parseEdits", () => {
  it("parses a bare JSON array", () => {
    const e = parseEdits('[{"path":"a.ts","find":"x","replace":"y"}]');
    expect(e).toHaveLength(1);
  });
  it("tolerates code fences and leading prose", () => {
    const e = parseEdits('Here you go:\n```json\n[{"path":"a.ts","content":"hello"}]\n```');
    expect(e[0]).toEqual({ path: "a.ts", content: "hello" });
  });
  it("rejects output without an array", () => {
    expect(() => parseEdits("I cannot do that.")).toThrow();
  });
  it("rejects edits missing fields", () => {
    expect(() => parseEdits('[{"path":"a.ts"}]')).toThrow(/neither/);
  });
});

describe("applyEdits", () => {
  const mk = () => {
    const d = fs.mkdtempSync(path.join(os.tmpdir(), "tfa-edits-"));
    fs.writeFileSync(path.join(d, "a.ts"), "const a = 1;\nconst b = 2;\n");
    return d;
  };
  it("applies find/replace and whole-file writes", () => {
    const d = mk();
    const r = applyEdits(d, [
      { path: "a.ts", find: "const b = 2;", replace: "const b = 3;" },
      { path: "new/c.ts", content: "export const c = 1;\n" },
    ]);
    expect(r.ok).toBe(true);
    expect(fs.readFileSync(path.join(d, "a.ts"), "utf8")).toContain("const b = 3;");
    expect(fs.existsSync(path.join(d, "new/c.ts"))).toBe(true);
  });
  it("rolls back everything when a later edit fails", () => {
    const d = mk();
    const r = applyEdits(d, [
      { path: "a.ts", find: "const a = 1;", replace: "const a = 9;" },
      { path: "a.ts", find: "does not exist", replace: "" },
    ]);
    expect(r.ok).toBe(false);
    expect(fs.readFileSync(path.join(d, "a.ts"), "utf8")).toContain("const a = 1;");
  });
  it("rejects ambiguous finds", () => {
    const d = mk();
    fs.writeFileSync(path.join(d, "a.ts"), "x\nx\n");
    const r = applyEdits(d, [{ path: "a.ts", find: "x", replace: "y" }]);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/ambiguous/);
  });
  it("refuses paths that escape the repo", () => {
    const d = mk();
    const r = applyEdits(d, [{ path: "../evil.ts", content: "x" }]);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/escapes/);
  });
  it("backup.restore removes files it created", () => {
    const d = mk();
    const r = applyEdits(d, [{ path: "n.ts", content: "x" }]);
    expect(r.ok).toBe(true);
    r.backup.restore();
    expect(fs.existsSync(path.join(d, "n.ts"))).toBe(false);
  });

  it("I-8: rejects absolute, drive-letter, UNC, NUL, .git and empty paths", () => {
    const d = mk();
    for (const p of ["/etc/passwd", "C:\\Windows\\x.ts", "c:x.ts", "\\\\server\\share\\x.ts", "a\0b.ts", ".git/config", "src/../.git/hooks/pre-commit", "", "   "]) {
      const r = applyEdits(d, [{ path: p, content: "x" }]);
      expect(r.ok, p).toBe(false);
    }
    expect(fs.existsSync(path.join(d, ".git"))).toBe(false);
  });
  it("I-8: rejects an escape through a directory link inside the repo", () => {
    const d = mk();
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), "tfa-outside-"));
    fs.symlinkSync(outside, path.join(d, "linked"), process.platform === "win32" ? "junction" : "dir");
    const r = applyEdits(d, [{ path: "linked/evil.ts", content: "x" }]);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/link/);
    expect(fs.existsSync(path.join(outside, "evil.ts"))).toBe(false);
  });
  it("I-8: rejects an empty find", () => {
    const d = mk();
    const r = applyEdits(d, [{ path: "a.ts", find: "", replace: "y" }]);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/empty/);
  });
  it("CRLF-safe: an LF find matches a CRLF file exactly once and keeps CRLF", () => {
    const d = mk();
    fs.writeFileSync(path.join(d, "w.ts"), "const a = 1;\r\nconst b = 2;\r\n");
    const r = applyEdits(d, [{ path: "w.ts", find: "const a = 1;\nconst b = 2;", replace: "const a = 1;\nconst b = 3;" }]);
    expect(r.ok).toBe(true);
    expect(fs.readFileSync(path.join(d, "w.ts"), "utf8")).toBe("const a = 1;\r\nconst b = 3;\r\n");
  });
  it("touched paths are repo-relative with forward slashes and deduplicated", () => {
    const d = mk();
    const r = applyEdits(d, [
      { path: "sub/x.ts", content: "a" },
      { path: "sub/x.ts", find: "a", replace: "b" },
    ]);
    expect(r.ok && r.touched).toEqual(["sub/x.ts"]);
  });
});

