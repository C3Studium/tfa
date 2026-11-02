import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { HumanOwnedPathError, assertWritable, humanOwned, writeFile, appendFile } from "../src/files.js";
import { applyEdits } from "../src/llm/edits.js";
import { TFA_ROOT } from "../src/config.js";

function repo(name = "app"): string {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "tfa-guard-"));
  fs.writeFileSync(path.join(d, "package.json"), JSON.stringify({ name }));
  return d;
}

describe("I-1 write guard", () => {
  it("refuses .tfa/bench/** in every repo", () => {
    const d = repo();
    for (const rel of [".tfa/bench/tasks/t1.json", ".tfa/bench/holdout/h.json", ".TFA/Bench/runs/r.json", ".tfa/x/../bench/y.json", ".tfa/bench"]) {
      expect(() => assertWritable(d, path.join(d, rel)), rel).toThrow(HumanOwnedPathError);
    }
    expect(() => writeFile(d, path.join(d, ".tfa/bench/tasks/t.json"), "{}")).toThrow(/I-1/);
    expect(fs.existsSync(path.join(d, ".tfa/bench"))).toBe(false);
    // neighbours are fine
    expect(humanOwned(d, path.join(d, ".tfa/benchmarks.md"))).toBeNull();
    expect(humanOwned(d, path.join(d, ".tfa/trajectories/t.jsonl"))).toBeNull();
  });

  it("refuses src/verify/** and src/bench/** in the tfa repository itself", () => {
    expect(() => assertWritable(TFA_ROOT, path.join(TFA_ROOT, "src/verify/cascade.ts"))).toThrow(HumanOwnedPathError);
    expect(() => assertWritable(TFA_ROOT, path.join(TFA_ROOT, "src/bench/run.ts"))).toThrow(HumanOwnedPathError);
    expect(() => assertWritable(TFA_ROOT, path.join(TFA_ROOT, "SRC/Verify/new.ts"))).toThrow(HumanOwnedPathError);
    expect(() => assertWritable(TFA_ROOT, path.join(TFA_ROOT, "src/think/../verify/x.ts"))).toThrow(HumanOwnedPathError);
    expect(humanOwned(TFA_ROOT, path.join(TFA_ROOT, "src/think/fix.ts"))).toBeNull();
    expect(humanOwned(TFA_ROOT, path.join(TFA_ROOT, "proposals/p.patch"))).toBeNull();
  });

  it("does not guard src/verify in an unrelated target repo (ASK-HUMAN Q3, option A)", () => {
    const d = repo("my-next-app");
    expect(humanOwned(d, path.join(d, "src/verify/page.tsx"))).toBeNull();
  });

  it("follows links: a junction/symlink into a human-owned dir is refused", () => {
    const d = repo();
    fs.mkdirSync(path.join(d, ".tfa/bench"), { recursive: true });
    fs.symlinkSync(path.join(d, ".tfa/bench"), path.join(d, "innocent"), process.platform === "win32" ? "junction" : "dir");
    expect(() => appendFile(d, path.join(d, "innocent/tasks.json"), "x")).toThrow(HumanOwnedPathError);
  });

  it("applyEdits rejects a candidate that touches a human-owned path and leaves the repo untouched", () => {
    const d = repo("tfa");
    fs.mkdirSync(path.join(d, "src/verify"), { recursive: true });
    fs.writeFileSync(path.join(d, "src/verify/cascade.ts"), "export const strict = true;\n");
    fs.writeFileSync(path.join(d, "a.ts"), "const a = 1;\n");
    const r = applyEdits(d, [
      { path: "a.ts", find: "const a = 1;", replace: "const a = 2;" },
      { path: "src/verify/cascade.ts", find: "strict = true", replace: "strict = false" },
    ]);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toMatch(/human-owned/);
    expect(fs.readFileSync(path.join(d, "src/verify/cascade.ts"), "utf8")).toContain("strict = true");
    expect(fs.readFileSync(path.join(d, "a.ts"), "utf8")).toContain("const a = 1;"); // rolled back
  });

  it("static check: no fs mutation in src/ outside src/files.ts and the human-owned dirs", () => {
    const MUTATORS = /\bfs\.(writeFileSync|appendFileSync|rmSync|unlinkSync|renameSync|mkdirSync|copyFileSync|cpSync|rmdirSync|symlinkSync|createWriteStream|writeFile|appendFile|rm|unlink|rename|mkdir|copyFile|truncateSync|truncate)\b/;
    const offenders: string[] = [];
    const walk = (dir: string) => {
      for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const abs = path.join(dir, e.name);
        const rel = path.relative(TFA_ROOT, abs).split(path.sep).join("/");
        if (e.isDirectory()) {
          if (rel === "src/verify" || rel === "src/bench") continue; // human-owned: their own rules
          walk(abs);
        } else if (rel.endsWith(".ts") && rel !== "src/files.ts") {
          fs.readFileSync(abs, "utf8").split("\n").forEach((line, i) => {
            if (MUTATORS.test(line)) offenders.push(`${rel}:${i + 1}: ${line.trim()}`);
          });
        }
      }
    };
    walk(path.join(TFA_ROOT, "src"));
    expect(offenders).toEqual([]);
  });
});
