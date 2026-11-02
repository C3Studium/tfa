import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Trajectory, readTrajectory, tail, stripAnsi, LOG_TAIL_BYTES, EV_TYPES, type Ev } from "../src/log.js";
import { LoggedReader, HttpReader } from "../src/llm/client.js";
import { DEFAULT_CONFIG } from "../src/config.js";
import { mockLlm, slotsAt } from "./helpers.js";

const repo = () => fs.mkdtempSync(path.join(os.tmpdir(), "tfa-log-"));

describe("trajectory log (contract §5)", () => {
  it("lives at .tfa/trajectories/<taskId>.jsonl; first line is the task event with the schema marker", () => {
    const r = repo();
    const t = Trajectory.create(r, { text: "do it", origin: "user" });
    expect(t.id).toMatch(/^task_/);
    expect(path.relative(r, t.file).split(path.sep).join("/")).toBe(`.tfa/trajectories/${t.id}.jsonl`);
    const first = JSON.parse(fs.readFileSync(t.file, "utf8").split("\n")[0]);
    expect(first).toMatchObject({ t: "task", id: t.id, text: "do it", origin: "user", schema: 1 });
    expect(() => Trajectory.create(r, { text: "again", origin: "user", id: t.id })).toThrow(); // never overwrites
  });

  it("is append-only: earlier bytes never change", () => {
    const t = Trajectory.create(repo(), { text: "x", origin: "queue" });
    t.append({ t: "decompose", at: 1, subtasks: [{ id: "sub_A", text: "x", touches: [] }] });
    const before = fs.readFileSync(t.file);
    t.append({ t: "ask", at: 2, subtaskId: "sub_A", question: "which repo?" });
    const after = fs.readFileSync(t.file);
    expect(after.subarray(0, before.length).equals(before)).toBe(true);
    expect(after.length).toBeGreaterThan(before.length);
    expect(Object.getOwnPropertyNames(Trajectory.prototype).filter((m) => /write|rewrite|truncate|set/i.test(m))).toEqual([]);
  });

  it("accepts every contract event type and round-trips them", () => {
    const t = Trajectory.create(repo(), { text: "all", origin: "bench" });
    const feats = { simTask: 0, simSkill: 0, skillCovered: false, footprintKnown: 0, selfEstimate: 0.5, size: 3 };
    const evs: Ev[] = [
      { t: "decompose", at: 1, subtasks: [{ id: "sub_A", text: "a", touches: ["src/a.ts"] }] },
      { t: "gate", at: 2, subtaskId: "sub_A", decision: "slow", features: feats, score: 0.2, threshold: 0.85 },
      { t: "retrieve", at: 3, subtaskId: "sub_A", query: "q", hits: [{ chunkId: "chunk_1", score: 1, cited: true }], tokens: 10 },
      { t: "prompt", at: 4, subtaskId: "sub_A", attempt: 1, slot: "worker", messages: [{ role: "user", content: "q" }], promptHash: "h" },
      { t: "response", at: 5, subtaskId: "sub_A", attempt: 1, slot: "senior", text: "[]", ms: 9, promptTokens: 7, completionTokens: 3 },
      { t: "candidate", at: 6, subtaskId: "sub_A", candidateId: "cand_1", edits: [{ path: "a", content: "b" }] },
      { t: "stage", at: 7, candidateId: "cand_1", stage: "typecheck", ok: true, ms: 5, log: "ok" },
      { t: "stage", at: 7, candidateId: "cand_1", stage: "lint", ok: false, skipped: true, ms: 0, log: "skipped" },
      { t: "verify", at: 8, candidateId: "cand_1", ok: true, totalMs: 5 },
      { t: "reflect", at: 9, subtaskId: "sub_A", lessons: [], playbook: [] },
      { t: "save", at: 10, subtaskId: "sub_A", noteOps: [] },
      { t: "ask", at: 11, subtaskId: "sub_A", question: "?" },
      { t: "correction", at: 12, subtaskId: "sub_A", candidateId: "cand_1", humanDiff: "-a\n+b" },
    ];
    for (const e of evs) t.append(e);
    const res = t.result({ ok: true, subtaskId: "sub_A", candidateId: "cand_1", diff: "d" });
    const back = readTrajectory(t.file);
    expect(back.map((e) => e.t)).toEqual(["task", ...evs.map((e) => e.t), "result"]);
    expect(new Set(back.map((e) => e.t))).toEqual(new Set(EV_TYPES));
    // costs come from the events: tokens from responses, senior calls, non-skipped stages
    expect(res.costs).toMatchObject({ promptTokens: 7, completionTokens: 3, seniorCalls: 1, stagesRun: 1 });
    expect(() => t.append({ t: "nonsense" } as unknown as Ev)).toThrow(/unknown/);
    expect(() => t.append({ t: "task", id: "x", at: 0, repo: "", text: "", origin: "user" })).toThrow();
  });

  it("stage logs are ANSI-stripped tails of at most 6 kB", () => {
    const t = Trajectory.create(repo(), { text: "x", origin: "user" });
    const big = "\u001b[31merror\u001b[0m TS2322 ✗ " + "ř".repeat(20_000) + "\r\nEND";
    t.append({ t: "stage", at: 1, candidateId: "cand_1", stage: "test", ok: false, ms: 1, log: big });
    const ev = readTrajectory(t.file).find((e) => e.t === "stage") as Extract<Ev, { t: "stage" }>;
    expect(Buffer.byteLength(ev.log, "utf8")).toBeLessThanOrEqual(LOG_TAIL_BYTES);
    expect(ev.log).not.toContain("\u001b");
    expect(ev.log.endsWith("END")).toBe(true);
    expect(ev.log.startsWith("…(truncated)…")).toBe(true);
    expect(stripAnsi("\u001b[1;32mok\u001b[0m \u001b]8;;http://x\u0007link\u001b]8;;\u0007")).toBe("ok link");
    expect(tail("short")).toBe("short");
  });

  it("reads CRLF files and a {t:schema} header; refuses a newer schema", () => {
    const r = repo();
    const f = path.join(r, "t.jsonl");
    fs.writeFileSync(f, '{"t":"schema","schema":1}\r\n{"t":"task","id":"task_1","at":1,"repo":"r","text":"x","origin":"user"}\r\n\r\n');
    expect(readTrajectory(f).map((e) => e.t)).toEqual(["task"]);
    fs.writeFileSync(f, '{"t":"task","id":"task_1","at":1,"repo":"r","text":"x","origin":"user","schema":2}\n');
    expect(() => readTrajectory(f)).toThrow(/newer/);
  });

  it("I-6 end to end: LoggedReader into a Trajectory writes prompt then response before returning", async () => {
    const llm = await mockLlm(["hi"]);
    try {
      const t = Trajectory.create(repo(), { text: "x", origin: "user" });
      const reader = new LoggedReader(new HttpReader(slotsAt(DEFAULT_CONFIG, llm.url).slots), t, "sub_Z");
      await reader.chat("worker", [{ role: "user", content: "hello" }], { attempt: 1 });
      const evs = readTrajectory(t.file); // read immediately after return
      expect(evs.map((e) => e.t)).toEqual(["task", "prompt", "response"]);
      expect(t.costs()).toMatchObject({ promptTokens: 100, completionTokens: 20 });
    } finally {
      llm.close();
    }
  });
});
