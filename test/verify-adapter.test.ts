import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { makeFixtureRepo } from "./helpers.js";
import { TransitionalVerifier, aggregate, stagesUpTo } from "../src/think/verify-adapter.js";
import { DEFAULT_CONFIG } from "../src/config.js";
import type { Candidate, Edit, StageOutcome } from "../src/types.js";

const cand = (edits: Edit[]): Candidate => ({ id: "cand_T", subtaskId: "sub_T", edits, createdBy: "worker" });
const FIX: Edit = { path: "src/price.ts", find: "const t: string = total(items);", replace: "const t: number = total(items);" };
const WRONG: Edit = { path: "src/price.ts", find: "const t: string = total(items);", replace: "const t: boolean = total(items);" };

describe("Verifier (contract §7) via the transitional adapter", () => {
  it("apply is always the first stage; a failed apply stops the cascade", async () => {
    const repo = makeFixtureRepo();
    const r = await new TransitionalVerifier(DEFAULT_CONFIG).run(cand([{ path: "src/price.ts", find: "nope", replace: "x" }]), repo);
    expect(r.ok).toBe(false);
    expect(r.failedAt).toBe("apply");
    expect(r.stages.map((s) => s.stage)).toEqual(["apply"]);
  });

  it("I-4: skipped stages are reported skipped and never ok; the verified edit stays in place", async () => {
    const repo = makeFixtureRepo();
    const seen: StageOutcome[] = [];
    const r = await new TransitionalVerifier(DEFAULT_CONFIG, (s) => seen.push(s)).run(cand([FIX]), repo);
    expect(r.ok).toBe(true);
    expect(r.stages.map((s) => s.stage)).toEqual(["apply", "lint", "typecheck", "test", "build"]);
    for (const s of r.stages.filter((s) => s.skipped)) expect(s.ok).toBe(false);
    expect(r.stages.filter((s) => s.skipped).map((s) => s.stage)).toEqual(["lint", "test", "build"]);
    expect(seen.map((s) => s.stage)).toEqual(["apply", "lint", "typecheck", "test", "build"]);
    expect(fs.readFileSync(path.join(repo, "src/price.ts"), "utf8")).toContain("const t: number");
  }, 120_000);

  it("a failing stage rolls the candidate back exactly", async () => {
    const repo = makeFixtureRepo();
    const before = fs.readFileSync(path.join(repo, "src/price.ts"), "utf8");
    const r = await new TransitionalVerifier(DEFAULT_CONFIG).run(cand([WRONG]), repo);
    expect(r.ok).toBe(false);
    expect(r.failedAt).toBe("typecheck");
    expect(fs.readFileSync(path.join(repo, "src/price.ts"), "utf8")).toBe(before);
  }, 120_000);

  it("fail closed: when every stage is skipped, nothing is verified", async () => {
    const repo = fs.mkdtempSync(path.join(os.tmpdir(), "tfa-empty-"));
    fs.writeFileSync(path.join(repo, "a.txt"), "a\n");
    const r = await new TransitionalVerifier(DEFAULT_CONFIG).run(cand([{ path: "a.txt", content: "b\n" }]), repo);
    expect(r.ok).toBe(false);
    expect(r.stages.slice(1).every((s) => s.skipped && !s.ok)).toBe(true);
    expect(fs.readFileSync(path.join(repo, "a.txt"), "utf8")).toBe("a\n"); // rolled back
  });

  it("upTo limits the stages run; stages() lists detected commands", async () => {
    const repo = makeFixtureRepo();
    const v = new TransitionalVerifier(DEFAULT_CONFIG);
    expect(v.stages(repo).map((c) => c.stage)).toEqual(["typecheck"]);
    const r = await v.run(cand([FIX]), repo, { upTo: "typecheck" });
    expect(r.stages.map((s) => s.stage)).toEqual(["apply", "lint", "typecheck"]);
    expect(stagesUpTo(["lint", "typecheck", "test"], "apply")).toEqual([]);
  }, 120_000);

  it("aggregate never counts a skipped stage as a pass", () => {
    const sk = (stage: StageOutcome["stage"]): StageOutcome => ({ stage, ok: false, skipped: true, ms: 0, log: "" });
    expect(aggregate([{ stage: "apply", ok: true, ms: 0, log: "" }, sk("lint"), sk("test")], 2)).toEqual({ ok: false });
    expect(aggregate([{ stage: "apply", ok: true, ms: 0, log: "" }, sk("lint"), { stage: "test", ok: true, ms: 1, log: "" }], 2)).toEqual({ ok: true });
    expect(aggregate([{ stage: "apply", ok: true, ms: 0, log: "" }, { stage: "lint", ok: false, ms: 1, log: "" }], 1)).toEqual({ ok: false, failedAt: "lint" });
  });
});
