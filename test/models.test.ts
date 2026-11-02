import { describe, it, expect } from "vitest";
import { MODELS, modelsHelp, serveCommand, downloadCommand, DRAFT_MAX } from "../src/models.js";
import { DEFAULT_CONFIG } from "../src/config.js";
import { SLOTS } from "../src/types.js";

describe("tfa models (P0/6)", () => {
  it("covers every slot, with senior A (GLM-4.7-Flash) and B (Qwen3.6-35B-A3B)", () => {
    for (const s of SLOTS) expect(MODELS.some((m) => m.slot === s), s).toBe(true);
    const seniors = MODELS.filter((m) => m.slot === "senior");
    expect(seniors.map((m) => m.variant)).toEqual(["A", "B"]);
    expect(seniors[0].file).toMatch(/GLM-4\.7-Flash/);
    expect(seniors[1].file).toMatch(/Qwen3\.6-35B-A3B/);
    expect(MODELS.find((m) => m.slot === "worker")!.file).toMatch(/Qwen3\.5-9B-Q6_K/);
  });

  it("serve ports match the configured slot URLs", () => {
    for (const m of MODELS.filter((m) => m.port > 0)) expect(DEFAULT_CONFIG.slots[m.slot].url).toBe(`http://127.0.0.1:${m.port}`);
    expect(DEFAULT_CONFIG.slots.draft.url).toBe(DEFAULT_CONFIG.slots.worker.url); // draft lives inside the worker
  });

  it("seniors keep experts on the CPU; the worker loads the draft and can save its prefix slot", () => {
    for (const m of MODELS.filter((m) => m.slot === "senior")) expect(serveCommand(m)).toContain("--n-cpu-moe");
    const worker = MODELS.find((m) => m.slot === "worker")!;
    const draft = MODELS.find((m) => m.slot === "draft")!;
    const cmd = serveCommand(worker, draft);
    expect(cmd).toContain(`--model-draft models/draft/${draft.file}`);
    expect(cmd).toContain(`--draft-max ${DRAFT_MAX}`);
    expect(cmd).toContain("--slot-save-path");
    expect(cmd).toContain("--cache-type-k q8_0");
    expect(serveCommand(draft)).toMatch(/not served separately/);
    expect(serveCommand(MODELS.find((m) => m.slot === "embed")!)).toContain("--embedding");
    expect(serveCommand(MODELS.find((m) => m.slot === "rerank")!)).toContain("--reranking");
  });

  it("download commands put each model in its own folder (senior A and B separate)", () => {
    const dirs = MODELS.map((m) => downloadCommand(m).split("--local-dir ")[1]);
    expect(new Set(dirs).size).toBe(MODELS.length);
    expect(dirs).toContain("models/senior-a");
  });

  it("help text lists every model", () => {
    const h = modelsHelp();
    for (const m of MODELS) expect(h).toContain(m.file);
  });
});
