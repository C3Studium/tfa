import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { DEFAULT_CONFIG, deepMerge, loadConfig } from "../src/config.js";
import { base32Time, canonicalJson, hashJson, newId } from "../src/ids.js";
import { SLOTS } from "../src/types.js";

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "tfa-cfg-"));
}

describe("config (contract §10)", () => {
  it("defaults define every slot and a cheapest-first cascade without apply", () => {
    for (const s of SLOTS) expect(DEFAULT_CONFIG.slots[s].url).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
    expect(DEFAULT_CONFIG.cascade).toEqual(["lint", "typecheck", "test", "build"]);
    expect(DEFAULT_CONFIG.budgets.fast).toEqual({ maxCandidates: 1, maxWallMs: 60_000 });
    expect(DEFAULT_CONFIG.notebook.recencyDecay).toBe(0.995);
    expect(DEFAULT_CONFIG.bench).toMatchObject({ devCount: 20, holdoutCount: 10, contaminationThreshold: 0.95 });
  });

  it("precedence: project override > tool config > defaults; env overrides slot URLs only", () => {
    const toolRoot = tmp();
    const repo = tmp();
    fs.writeFileSync(path.join(toolRoot, "tfa.config.json"), JSON.stringify({ schema: 1, slots: { worker: { url: "http://10.0.0.1:1", maxTokens: 111 } }, budgets: { contextTokens: 5000 } }));
    fs.mkdirSync(path.join(repo, ".tfa"));
    // BOM + CRLF, as Notepad writes it
    fs.writeFileSync(path.join(repo, ".tfa", "config.json"), "﻿{\r\n  \"slots\": { \"worker\": { \"maxTokens\": 222 } },\r\n  \"cascade\": [\"typecheck\"]\r\n}\r\n");
    const env = { TFA_WORKER_URL: "http://127.0.0.1:9999/", TFA_MAX_ATTEMPTS: "99" };
    const cfg = loadConfig(repo, { toolRoot, env });
    expect(cfg.slots.worker.url).toBe("http://127.0.0.1:9999"); // env wins for URLs
    expect(cfg.slots.worker.maxTokens).toBe(222); // project wins over tool
    expect(cfg.slots.worker.temperature).toBe(DEFAULT_CONFIG.slots.worker.temperature); // deep merge keeps defaults
    expect(cfg.budgets.contextTokens).toBe(5000); // tool wins over defaults
    expect(cfg.budgets.coreTokens).toBe(2000);
    expect(cfg.cascade).toEqual(["typecheck"]); // arrays replace
    expect(cfg.sources.project).toContain("config.json");
    expect(cfg.sources.env).toEqual(["TFA_WORKER_URL"]);
    // non-URL env vars are ignored: the attempt budget still comes from config
    expect(cfg.budgets.slow.routine.maxCandidates).toBe(DEFAULT_CONFIG.budgets.slow.routine.maxCandidates);
  });

  it("rejects an invalid cascade and a non-URL slot", () => {
    const toolRoot = tmp();
    const repo = tmp();
    fs.writeFileSync(path.join(toolRoot, "tfa.config.json"), JSON.stringify({ cascade: ["apply", "typecheck"] }));
    expect(() => loadConfig(repo, { toolRoot, env: {} })).toThrow(/implicit/);
    fs.writeFileSync(path.join(toolRoot, "tfa.config.json"), JSON.stringify({ slots: { embed: { url: "localhost" } } }));
    expect(() => loadConfig(repo, { toolRoot, env: {} })).toThrow(/slots\.embed\.url/);
  });

  it("the repository's own tfa.config.json loads", () => {
    const cfg = loadConfig(tmp(), { env: {} });
    expect(cfg.sources.tool).toMatch(/tfa\.config\.json$/);
  });

  it("deepMerge does not mutate its inputs", () => {
    const a = { x: { y: 1, z: [1, 2] } };
    const m = deepMerge(a, { x: { z: [3] } });
    expect(m).toEqual({ x: { y: 1, z: [3] } });
    expect(a.x.z).toEqual([1, 2]);
  });
});

describe("ids (contract §3)", () => {
  it("are kind-prefixed, fixed-width and sortable by time", () => {
    const a = newId("task", 1_700_000_000_000);
    const b = newId("task", 1_700_000_000_001);
    expect(a).toMatch(/^task_[0-9A-HJKMNP-TV-Z]{14}$/);
    expect(a < b).toBe(true);
    expect(base32Time(0)).toBe("0000000000");
    expect(newId("cand")).not.toBe(newId("cand"));
  });
  it("canonical JSON hashes are key-order independent", () => {
    expect(canonicalJson({ b: 1, a: { d: 2, c: 3 } })).toBe('{"a":{"c":3,"d":2},"b":1}');
    expect(hashJson({ a: 1, b: 2 })).toBe(hashJson({ b: 2, a: 1 }));
  });
});
