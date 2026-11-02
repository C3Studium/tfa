/**
 * Model I/O — contract §4 (model server contract) and §7 `Reader`.
 *
 * All slots speak the OpenAI-compatible HTTP API of llama.cpp `llama-server`; code names slots, never models.
 *  - chat:   POST /v1/chat/completions  {messages, max_tokens, temperature, stream:false, cache_prompt:true, id_slot}
 *            `cache_prompt` reuses the KV prefix and `id_slot` pins the system+core prefix to one server slot
 *            (Pack G §2: prefix caching; prefill is compute-bound, Pack G §2.2(e)).
 *  - embed:  POST /v1/embeddings  {input: string[]}  batches ≤ 64, cached by content hash
 *  - rerank: POST /rerank  {query, documents[], top_n} → {index, relevance_score}[]
 *  - health: GET /health
 *  - slots:  POST /slots/{id}?action=save|restore  {filename}
 *
 * I-6: every chat call made through `LoggedReader` writes `prompt` and `response` events (contract §5)
 * synchronously, before it returns.
 */
import { hashJson, sha256 } from "../ids.js";
import type { Hash, Id, Ms, Msg, Slot, SlotConfig } from "../types.js";

export type ChatOpts = {
  temperature?: number;
  maxTokens?: number;
  stop?: string[];
  /** llama-server slot to pin the cached prefix to (default 0; servers run with --parallel 1). */
  idSlot?: number;
  /** Attempt number, recorded in the trajectory (I-6). */
  attempt?: number;
  /** Subtask the call belongs to; overrides the LoggedReader's default. */
  subtaskId?: Id;
  signal?: AbortSignal;
};

export type ChatResult = { text: string; ms: number; promptTokens?: number; completionTokens?: number };

/** Contract §7. */
export interface Reader {
  chat(slot: Slot, messages: Msg[], o?: ChatOpts): Promise<ChatResult>;
  embed(texts: string[]): Promise<Float32Array[]>;
  rerank(q: string, docs: string[]): Promise<number[]>;
  healthy(slot: Slot): Promise<boolean>;
}

export const EMBED_BATCH = 64; // contract §4

export class HttpError extends Error {
  constructor(
    readonly slot: Slot,
    readonly status: number,
    body: string,
  ) {
    super(`${slot} HTTP ${status}: ${body.slice(0, 500)}`);
  }
}

type FetchLike = typeof fetch;

export class HttpReader implements Reader {
  private embedCache = new Map<Hash, Float32Array>();
  private readonly fetch: FetchLike;
  private readonly timeoutMs: number;

  constructor(
    readonly slots: Record<Slot, SlotConfig>,
    o: { fetch?: FetchLike; timeoutMs?: number } = {},
  ) {
    this.fetch = o.fetch ?? fetch;
    this.timeoutMs = o.timeoutMs ?? 30 * 60_000; // a long CPU prefill on the senior can take minutes
  }

  private async post(slot: Slot, route: string, body: unknown, signal?: AbortSignal): Promise<unknown> {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), this.timeoutMs);
    const onAbort = () => ctl.abort();
    signal?.addEventListener("abort", onAbort);
    try {
      const res = await this.fetch(`${this.slots[slot].url}${route}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: ctl.signal,
      });
      if (!res.ok) throw new HttpError(slot, res.status, await res.text());
      return await res.json();
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }
  }

  async chat(slot: Slot, messages: Msg[], o: ChatOpts = {}): Promise<ChatResult> {
    const sc = this.slots[slot];
    const t0 = Date.now();
    const body: Record<string, unknown> = {
      model: slot, // llama-server ignores it; logged server-side
      messages,
      max_tokens: o.maxTokens ?? sc.maxTokens,
      temperature: o.temperature ?? sc.temperature,
      stream: false,
      cache_prompt: true,
      id_slot: o.idSlot ?? 0,
    };
    if (o.stop?.length) body.stop = o.stop;
    const json = (await this.post(slot, "/v1/chat/completions", body, o.signal)) as {
      choices?: { message?: { content?: string } }[];
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    };
    return {
      text: json.choices?.[0]?.message?.content ?? "",
      ms: Date.now() - t0,
      promptTokens: json.usage?.prompt_tokens,
      completionTokens: json.usage?.completion_tokens,
    };
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    const out: (Float32Array | undefined)[] = texts.map((t) => this.embedCache.get(sha256(t)));
    const missing = texts.map((t, i) => ({ t, i })).filter(({ i }) => !out[i]);
    for (let b = 0; b < missing.length; b += EMBED_BATCH) {
      const batch = missing.slice(b, b + EMBED_BATCH);
      const json = (await this.post("embed", "/v1/embeddings", { input: batch.map((x) => x.t) })) as { data?: { index: number; embedding: number[] }[] };
      const data = json.data ?? [];
      if (data.length !== batch.length) throw new Error(`embed: expected ${batch.length} vectors, got ${data.length}`);
      for (const d of data) {
        const v = Float32Array.from(d.embedding);
        const { t, i } = batch[d.index];
        out[i] = v;
        this.embedCache.set(sha256(t), v);
      }
    }
    return out as Float32Array[];
  }

  async rerank(q: string, docs: string[]): Promise<number[]> {
    if (docs.length === 0) return [];
    const json = (await this.post("rerank", "/rerank", { query: q, documents: docs, top_n: docs.length })) as {
      results?: { index: number; relevance_score: number }[];
    };
    const scores = new Array<number>(docs.length).fill(Number.NEGATIVE_INFINITY);
    for (const r of json.results ?? []) scores[r.index] = r.relevance_score;
    return scores;
  }

  async healthy(slot: Slot, timeoutMs = 3000): Promise<boolean> {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const r = await this.fetch(`${this.slots[slot].url}/health`, { signal: ctl.signal });
      return r.ok; // llama-server answers 503 while the model is loading
    } catch {
      return false;
    } finally {
      clearTimeout(t);
    }
  }

  /** Persist / restore the cached prefix of a server slot across restarts (contract §4). */
  async slotAction(slot: Slot, idSlot: number, action: "save" | "restore", filename: string): Promise<unknown> {
    return this.post(slot, `/slots/${idSlot}?action=${action}`, { filename });
  }
}

// ---- I-6: logging wrapper ----

export type PromptEvent = { t: "prompt"; at: Ms; subtaskId: Id; attempt: number; slot: Slot; messages: Msg[]; promptHash: Hash };
export type ResponseEvent = { t: "response"; at: Ms; subtaskId: Id; attempt: number; slot: Slot; text: string; ms: number; promptTokens?: number; completionTokens?: number };

/** Anything that can append trajectory events synchronously (the Trajectory writer in src/log.ts). */
export interface EventSink {
  append(ev: PromptEvent | ResponseEvent): void;
}

export function promptHash(slot: Slot, messages: Msg[]): Hash {
  return hashJson({ slot, messages });
}

/**
 * The only Reader the rest of the system should hold during a task: every chat call writes its
 * `prompt` event before the request and its `response` event before returning (I-6).
 * A failed request leaves the `prompt` event in place; the caller's `result` event records the error.
 */
export class LoggedReader implements Reader {
  constructor(
    readonly inner: Reader,
    readonly sink: EventSink,
    readonly subtaskId: Id,
  ) {}

  async chat(slot: Slot, messages: Msg[], o: ChatOpts = {}): Promise<ChatResult> {
    const subtaskId = o.subtaskId ?? this.subtaskId;
    const attempt = o.attempt ?? 0;
    this.sink.append({ t: "prompt", at: Date.now(), subtaskId, attempt, slot, messages: structuredClone(messages), promptHash: promptHash(slot, messages) });
    const r = await this.inner.chat(slot, messages, o);
    this.sink.append({ t: "response", at: Date.now(), subtaskId, attempt, slot, text: r.text, ms: r.ms, promptTokens: r.promptTokens, completionTokens: r.completionTokens });
    return r;
  }
  embed(texts: string[]): Promise<Float32Array[]> {
    return this.inner.embed(texts);
  }
  rerank(q: string, docs: string[]): Promise<number[]> {
    return this.inner.rerank(q, docs);
  }
  healthy(slot: Slot): Promise<boolean> {
    return this.inner.healthy(slot);
  }
}
