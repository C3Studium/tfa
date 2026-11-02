import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { makeFixtureRepo, mockLlm } from "./helpers.js";
import { fix } from "../src/think/fix.js";
import { LlmClient } from "../src/llm/client.js";
import { DEFAULT_CONFIG, type Config } from "../src/config.js";

const GOOD = JSON.stringify([{ path: "src/price.ts", find: "const t: string = total(items);", replace: "const t: number = total(items);" }]);
const WRONG_BUT_APPLIES = JSON.stringify([{ path: "src/price.ts", find: "const t: string = total(items);", replace: "const t: string = String(total(items)) as unknown as number;" }]);
const UNPARSEABLE = "Sure! Change the type of t to number.";

function cfgFor(url: string): Config {
  const cfg = structuredClone(DEFAULT_CONFIG);
  cfg.slots.worker.url = url;
  return cfg;
}

describe("tfa fix end to end (mock model)", () => {
  it("fixes the type error first try, keeps edits, writes a trajectory and a diff", async () => {
    const repo = makeFixtureRepo();
    const llm = await mockLlm([GOOD]);
    try {
      const cfg = cfgFor(llm.url);
      const lines: string[] = [];
      const r = await fix({ repo, task: "Fix the type error in src/price.ts", cfg, client: new LlmClient("worker", cfg.slots.worker), log: (l) => lines.push(l) });
      expect(r.ok).toBe(true);
      expect(r.attempts).toBe(1);
      expect(r.diff).toContain("-  const t: string = total(items);");
      expect(r.diff).toContain("+  const t: number = total(items);");
      expect(fs.readFileSync(path.join(repo, "src/price.ts"), "utf8")).toContain("const t: number");
      const traj = fs.readFileSync(r.trajectory, "utf8").trim().split("\n").map((l) => JSON.parse(l));
      expect(traj.map((e) => e.t)).toEqual(["task", "context", "prompt", "response", "apply", "stage", "stage", "stage", "stage", "result"]);
      // The prompt carried the type-check error and the file window.
      const prompt = traj.find((e) => e.t === "prompt");
      expect(prompt.messages[1].content).toContain("TS2322");
      expect(prompt.messages[1].content).toContain("src/price.ts");
    } finally {
      llm.close();
    }
  }, 180_000);

  it("rolls back a wrong answer, feeds the failure back, and succeeds on retry", async () => {
    const repo = makeFixtureRepo();
    const llm = await mockLlm([UNPARSEABLE, WRONG_BUT_APPLIES, GOOD]);
    try {
      const cfg = cfgFor(llm.url);
      const r = await fix({ repo, task: "Fix the type error", cfg, client: new LlmClient("worker", cfg.slots.worker) });
      expect(r.ok).toBe(true);
      expect(r.attempts).toBe(3);
      // Second request carries the parse complaint; third carries the typecheck failure log.
      const req2 = llm.requests[1] as { messages: { role: string; content: string }[] };
      expect(req2.messages.at(-1)!.content).toMatch(/not a valid edit list/);
      const req3 = llm.requests[2] as { messages: { role: string; content: string }[] };
      expect(req3.messages.at(-1)!.content).toMatch(/typecheck stage failed/);
      expect(req3.messages.at(-1)!.content).toMatch(/TS2322|error TS/);
      expect(fs.readFileSync(path.join(repo, "src/price.ts"), "utf8")).toContain("const t: number = total(items);");
    } finally {
      llm.close();
    }
  }, 300_000);

  it("gives up cleanly after maxAttempts and leaves the repo untouched", async () => {
    const repo = makeFixtureRepo();
    const before = fs.readFileSync(path.join(repo, "src/price.ts"), "utf8");
    const llm = await mockLlm([WRONG_BUT_APPLIES]);
    try {
      const cfg = cfgFor(llm.url);
      const r = await fix({ repo, task: "Fix the type error", cfg, attempts: 2, client: new LlmClient("worker", cfg.slots.worker) });
      expect(r.ok).toBe(false);
      expect(r.attempts).toBe(2);
      expect(r.failedAt).toBe("typecheck");
      expect(fs.readFileSync(path.join(repo, "src/price.ts"), "utf8")).toBe(before);
    } finally {
      llm.close();
    }
  }, 300_000);

  it("--dry: verifies, then rolls back", async () => {
    const repo = makeFixtureRepo();
    const before = fs.readFileSync(path.join(repo, "src/price.ts"), "utf8");
    const llm = await mockLlm([GOOD]);
    try {
      const cfg = cfgFor(llm.url);
      const r = await fix({ repo, task: "Fix the type error", cfg, client: new LlmClient("worker", cfg.slots.worker), keep: false });
      expect(r.ok).toBe(true);
      expect(fs.readFileSync(path.join(repo, "src/price.ts"), "utf8")).toBe(before);
      expect(r.diff).toContain("const t: number");
    } finally {
      llm.close();
    }
  }, 180_000);
});
