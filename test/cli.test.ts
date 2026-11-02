import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { makeFixtureRepo, mockLlm } from "./helpers.js";
import { run, parseArgs, type Io } from "../src/cli.js";

const GOOD = JSON.stringify([{ path: "src/price.ts", find: "const t: string = total(items);", replace: "const t: number = total(items);" }]);

function io(env: NodeJS.ProcessEnv = {}): Io & { text: () => string } {
  const lines: string[] = [];
  return { out: (s) => lines.push(s), err: (s) => lines.push(s), env, text: () => lines.join("\n") };
}
const envFor = (url: string) => ({ TFA_WORKER_URL: url, TFA_SENIOR_URL: url, TFA_EMBED_URL: url, TFA_RERANK_URL: url });

describe("tfa CLI (P0/5)", () => {
  it("parses flags anywhere, with and without values", () => {
    const a = parseArgs(["repo", "--attempts", "3", "fix", "it", "--dry", "--up-to=test"], ["attempts"]);
    expect(a.pos).toEqual(["repo", "fix", "it"]);
    expect(a.flags.get("attempts")).toBe("3");
    expect(a.flags.get("dry")).toBe(true);
    expect(a.flags.get("up-to")).toBe("test");
  });

  it("help and unknown commands", async () => {
    const h = io();
    expect(await run([], h)).toBe(0);
    expect(h.text()).toContain("tfa doctor");
    const u = io();
    expect(await run(["frobnicate"], u)).toBe(1);
    expect(u.text()).toContain("unknown command");
  });

  it("models prints serve commands", async () => {
    const o = io();
    expect(await run(["models"], o)).toBe(0);
    expect(o.text()).toContain("llama-server");
  });

  it("doctor: ready when the repo is git, a stage can run and the worker answers", async () => {
    const repo = makeFixtureRepo();
    const llm = await mockLlm([""]);
    try {
      const o = io(envFor(llm.url));
      expect(await run(["doctor", repo], o)).toBe(0);
      expect(o.text()).toMatch(/typecheck\s+npm run typecheck --silent\s+\[scripts\.typecheck\]/);
      expect(o.text()).toMatch(/lint\s+— none \(skipped; never counted as a pass\)/);
      expect(o.text()).toMatch(/worker\s+http:\/\/127\.0\.0\.1:\d+\s+✓ reachable/);
      expect(o.text()).toContain("✓ ready for tfa fix");
    } finally {
      llm.close();
    }
  });

  it("doctor: not ready for a non-git folder with nothing to verify and no worker", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tfa-doc-"));
    const o = io({ TFA_WORKER_URL: "http://127.0.0.1:1" });
    expect(await run(["doctor", dir], o)).toBe(1);
    expect(o.text()).toContain("NOT a git repository");
    expect(o.text()).toContain("no stage can run");
    expect(o.text()).toMatch(/worker\s+\S+\s+✗ not reachable/);
  });

  it("verify: fails at typecheck on the broken fixture, passes after the fix; reports skipped stages", async () => {
    const repo = makeFixtureRepo();
    const a = io();
    expect(await run(["verify", repo], a)).toBe(1);
    expect(a.text()).toContain("✗ failed at typecheck");
    const p = path.join(repo, "src/price.ts");
    fs.writeFileSync(p, fs.readFileSync(p, "utf8").replace("const t: string", "const t: number"));
    const b = io();
    expect(await run(["verify", repo], b)).toBe(0);
    expect(b.text()).toContain("(1 ran, 3 skipped)");
    expect(await run(["verify", repo, "--up-to", "nonsense"], io())).toBe(1);
  }, 120_000);

  it("fix: verified change, diff, and a BenchRun-shaped acceptance record", async () => {
    const repo = makeFixtureRepo();
    const llm = await mockLlm([GOOD]);
    try {
      const o = io(envFor(llm.url));
      expect(await run(["fix", repo, "Fix", "the", "type", "error", "--attempts", "2"], o)).toBe(0);
      const text = o.text();
      expect(text).toContain("✓ verified on attempt 1");
      expect(text).toContain("+  const t: number = total(items);");
      const json = JSON.parse(text.slice(text.indexOf("{", text.indexOf("record (save as")), text.lastIndexOf("}") + 1));
      expect(json).toMatchObject({ set: "dev", results: [{ ok: true, attempts: 1, firstTry: true, corrections: 0 }], summary: { passRate: 1, firstTryRate: 1 } });
      expect(json.id).toMatch(/^run_/);
      expect(json.results[0].taskId).toMatch(/^task_/);
      expect(fs.existsSync(path.join(repo, ".tfa", "trajectories", `${json.results[0].taskId}.jsonl`))).toBe(true);
    } finally {
      llm.close();
    }
  }, 180_000);

  it("fix: exit 2 when the worker is not running; usage errors exit 1", async () => {
    const repo = makeFixtureRepo();
    const o = io({ TFA_WORKER_URL: "http://127.0.0.1:1" });
    expect(await run(["fix", repo, "do it"], o)).toBe(2);
    expect(o.text()).toContain("worker not reachable");
    expect(await run(["fix", repo], io())).toBe(1);
    expect(await run(["fix", repo, "x", "--attempts", "0"], io())).toBe(1);
  });
});
