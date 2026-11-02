import { describe, it, expect } from "vitest";
import { mockLlm, slotsAt, fakeEmbedding } from "./helpers.js";
import { HttpReader, LoggedReader, type PromptEvent, type ResponseEvent, type Reader } from "../src/llm/client.js";
import { DEFAULT_CONFIG } from "../src/config.js";

describe("HttpReader (contract §4)", () => {
  it("chat sends cache_prompt, id_slot, stream:false, max_tokens, temperature to /v1/chat/completions", async () => {
    const llm = await mockLlm(["hello"]);
    try {
      const r = new HttpReader(slotsAt(DEFAULT_CONFIG, llm.url).slots);
      const out = await r.chat("worker", [{ role: "user", content: "hi" }], { temperature: 0.3 });
      expect(out.text).toBe("hello");
      expect(out.promptTokens).toBe(100);
      expect(llm.calls[0].path).toBe("/v1/chat/completions");
      expect(llm.requests[0]).toMatchObject({ cache_prompt: true, id_slot: 0, stream: false, max_tokens: DEFAULT_CONFIG.slots.worker.maxTokens, temperature: 0.3 });
    } finally {
      llm.close();
    }
  });

  it("embed batches ≤ 64, preserves order and caches by content hash", async () => {
    const llm = await mockLlm([""]);
    try {
      const r = new HttpReader(slotsAt(DEFAULT_CONFIG, llm.url).slots);
      const texts = Array.from({ length: 130 }, (_, i) => `text ${i}`);
      const v = await r.embed(texts);
      expect(v).toHaveLength(130);
      expect(Array.from(v[77])).toEqual(fakeEmbedding("text 77").map((x) => Math.fround(x)));
      const batches = llm.calls.filter((c) => c.path === "/v1/embeddings").map((c) => (c.body as { input: string[] }).input.length);
      expect(batches).toEqual([64, 64, 2]);
      await r.embed(["text 5", "new one"]);
      const last = llm.calls.at(-1)!.body as { input: string[] };
      expect(last.input).toEqual(["new one"]); // "text 5" came from the cache
    } finally {
      llm.close();
    }
  });

  it("rerank returns scores aligned with the input documents", async () => {
    const llm = await mockLlm([""]);
    try {
      const r = new HttpReader(slotsAt(DEFAULT_CONFIG, llm.url).slots);
      const s = await r.rerank("price total", ["nothing here", "price and total", "total only"]);
      expect(s).toEqual([0, 2, 1]);
      expect(llm.calls[0]).toMatchObject({ path: "/rerank", body: { query: "price total", top_n: 3 } });
      expect(await r.rerank("q", [])).toEqual([]);
    } finally {
      llm.close();
    }
  });

  it("healthy: true on 200, false while loading (503) and when unreachable", async () => {
    const ok = await mockLlm([""]);
    const loading = await mockLlm([""], { status: 503 });
    try {
      const slots = slotsAt(DEFAULT_CONFIG, ok.url).slots;
      slots.senior.url = loading.url;
      slots.embed.url = "http://127.0.0.1:1";
      const r = new HttpReader(slots);
      expect(await r.healthy("worker")).toBe(true);
      expect(await r.healthy("senior")).toBe(false);
      expect(await r.healthy("embed")).toBe(false);
    } finally {
      ok.close();
      loading.close();
    }
  });

  it("slot save/restore posts the filename to /slots/{id}?action=", async () => {
    const llm = await mockLlm([""]);
    try {
      const r = new HttpReader(slotsAt(DEFAULT_CONFIG, llm.url).slots);
      await r.slotAction("worker", 0, "save", "prefix.bin");
      expect(llm.calls[0]).toMatchObject({ path: "/slots/0?action=save", body: { filename: "prefix.bin" } });
    } finally {
      llm.close();
    }
  });

  it("HTTP errors carry slot and status", async () => {
    const llm = await mockLlm([""]);
    try {
      const slots = slotsAt(DEFAULT_CONFIG, llm.url).slots;
      slots.rerank.url = llm.url + "/nope";
      await expect(new HttpReader(slots).rerank("q", ["d"])).rejects.toThrow(/rerank HTTP 404/);
    } finally {
      llm.close();
    }
  });
});

describe("LoggedReader (I-6)", () => {
  it("writes prompt then response before returning, with hash, slot, attempt and subtask", async () => {
    const llm = await mockLlm(["answer"]);
    try {
      const events: (PromptEvent | ResponseEvent)[] = [];
      const r = new LoggedReader(new HttpReader(slotsAt(DEFAULT_CONFIG, llm.url).slots), { append: (e) => events.push(e) }, "sub_X");
      const out = await r.chat("worker", [{ role: "user", content: "q" }], { attempt: 2 });
      expect(out.text).toBe("answer");
      expect(events.map((e) => e.t)).toEqual(["prompt", "response"]);
      expect(events[0]).toMatchObject({ subtaskId: "sub_X", attempt: 2, slot: "worker" });
      expect((events[0] as PromptEvent).promptHash).toMatch(/^[0-9a-f]{64}$/);
      expect(events[1]).toMatchObject({ text: "answer", promptTokens: 100, completionTokens: 20 });
    } finally {
      llm.close();
    }
  });

  it("keeps the prompt event when the call fails", async () => {
    const failing: Reader = {
      chat: async () => {
        throw new Error("boom");
      },
      embed: async () => [],
      rerank: async () => [],
      healthy: async () => false,
    };
    const events: { t: string }[] = [];
    const r = new LoggedReader(failing, { append: (e) => events.push(e) }, "sub_Y");
    await expect(r.chat("senior", [{ role: "user", content: "q" }])).rejects.toThrow("boom");
    expect(events.map((e) => e.t)).toEqual(["prompt"]);
  });
});
