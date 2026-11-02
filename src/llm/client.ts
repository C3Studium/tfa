/**
 * Minimal OpenAI-compatible chat client for llama.cpp `llama-server`.
 * - `cache_prompt: true` asks llama.cpp to reuse the KV cache for a shared prefix
 *   (system prompt + core block), which is the main fast-path latency win.
 * - No SDK dependency: Node's fetch is enough and keeps the slot swappable.
 */
import type { ModelSlot } from "../config.js";

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type ChatResult = {
  text: string;
  promptTokens?: number;
  completionTokens?: number;
  ms: number;
};

export type ChatOptions = {
  temperature?: number;
  maxTokens?: number;
  /** Stop sequences. */
  stop?: string[];
  /** Pin requests sharing a prefix to one llama.cpp slot so the cache is reused. */
  slotId?: number;
  signal?: AbortSignal;
};

export class LlmClient {
  constructor(readonly slot: ModelSlot) {}

  async chat(messages: ChatMessage[], opts: ChatOptions = {}): Promise<ChatResult> {
    const t0 = Date.now();
    const body: Record<string, unknown> = {
      model: this.slot.name,
      messages,
      temperature: opts.temperature ?? this.slot.temperature,
      max_tokens: opts.maxTokens ?? this.slot.maxTokens,
      stream: false,
      cache_prompt: true,
    };
    if (opts.stop?.length) body.stop = opts.stop;
    if (opts.slotId !== undefined) body.id_slot = opts.slotId;

    const res = await fetch(`${this.slot.url}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: opts.signal,
    });
    if (!res.ok) throw new Error(`LLM ${this.slot.name} HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
    const json = (await res.json()) as {
      choices: { message: { content: string } }[];
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    };
    return {
      text: json.choices?.[0]?.message?.content ?? "",
      promptTokens: json.usage?.prompt_tokens,
      completionTokens: json.usage?.completion_tokens,
      ms: Date.now() - t0,
    };
  }

  /** True if the server answers /health (llama.cpp) or /v1/models. */
  async healthy(timeoutMs = 3000): Promise<boolean> {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const r = await fetch(`${this.slot.url}/health`, { signal: ctl.signal });
      if (r.ok) return true;
      const m = await fetch(`${this.slot.url}/v1/models`, { signal: ctl.signal });
      return m.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(t);
    }
  }
}
