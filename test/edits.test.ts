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
});
