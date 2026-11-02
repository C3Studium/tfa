/**
 * `tfa models` — model files and llama-server commands for the target machine (RTX 3080 12 GB, 32 GB DDR5).
 * Build prompt P0 task 6. Hugging Face repo ids and GGUF file names change; the human confirms them at install
 * time (build prompt §4 "Before P0"). Ports match DEFAULT_CONFIG.slots (checked by test/models.test.ts).
 *
 * Memory arithmetic is Pack G §2.2 (derivation, not a paper result):
 *   worker Q6_K: 9B × 6.56 bpw / 8 ≈ 7.38 GB; + draft ≈ 0.9 GB + buffers ≈ 1 GB in ~11.2 GiB usable VRAM
 *   leaves ≈ 2 GB of KV → at 32k context the KV cache must be q8_0, not f16 (Pack G §2.2(b)).
 *   senior 35B-A3B MoE at ~4.8 bpw ≈ 21 GB in 32 GB RAM; experts must be computed where they live (CPU),
 *   never streamed over PCIe (Pack G §2.2(c)) → --n-cpu-moe keeps every expert tensor on the CPU.
 */
import type { Slot } from "./types.js";

export type ModelSpec = {
  slot: Slot;
  /** Distinguishes alternatives for one slot (senior A / B). */
  variant?: string;
  purpose: string;
  hfRepo: string;
  file: string;
  approxGB: number;
  /** 0 = loaded inside another server (the draft). */
  port: number;
  serverArgs: string[];
  notes: string;
};

// Speculative decoding (Pack G §2.4): γ starts at 5 for greedy code, range 2–10 — arXiv:2211.17192 §3.5
export const DRAFT_MAX = 5; // arXiv:2211.17192 §3.5
export const DRAFT_MIN = 2; // arXiv:2211.17192 §3.5 (lower end of the useful γ range)

const COMMON_GPU = ["-ngl 99", "--flash-attn on", "--jinja", "--parallel 1"]; // FlashAttention: Pack G §2.4 (arXiv:2205.14135)

export const MODELS: ModelSpec[] = [
  {
    slot: "worker",
    purpose: "resident reader in VRAM: drafts edits, reflects, parses",
    hfRepo: "unsloth/Qwen3.5-9B-GGUF",
    file: "Qwen3.5-9B-Q6_K.gguf",
    approxGB: 7.4,
    port: 8080,
    serverArgs: [...COMMON_GPU, "--ctx-size 32768", "--cache-type-k q8_0 --cache-type-v q8_0", "--slot-save-path models/slots"],
    notes:
      "Q6_K ≈ 7.38 GB (Pack G §2.2(a)). KV q8_0 so 32k context fits next to the draft. Drop to Q5_K_M (≈ 6.4 GB) if VRAM is short. " +
      "Keep the draft only if the measured acceptance α exceeds the cost ratio c over ≥ 10k tokens (arXiv:2211.17192 §3.1, Corollary 3.9).",
  },
  {
    slot: "draft",
    purpose: "speculative-decoding draft for the worker (same tokenizer family)",
    hfRepo: "unsloth/Qwen3.5-0.8B-GGUF",
    file: "Qwen3.5-0.8B-Q8_0.gguf",
    approxGB: 0.9,
    port: 0,
    serverArgs: [],
    notes: "Loaded by the worker (--model-draft). Any Qwen3.5 model of 2B or less works if this exact file is not published. Pack G §2.4: draft ≈ 1/50–1/100 of the target.",
  },
  {
    slot: "senior",
    variant: "A",
    purpose: "senior A: GLM-4.7-Flash, MoE, experts in RAM; slow path for novel subtasks",
    hfRepo: "unsloth/GLM-4.7-Flash-GGUF",
    file: "GLM-4.7-Flash-UD-Q4_K_XL.gguf",
    approxGB: 19,
    port: 8081,
    serverArgs: [...COMMON_GPU, "--n-cpu-moe 999", "--ctx-size 32768", "--cache-type-k q8_0 --cache-type-v q8_0"],
    notes: "MIT licence. Not the worker's family, so the worker cannot draft for it. A vs B is decided by the P2 dev-set benchmark (contract §12).",
  },
  {
    slot: "senior",
    variant: "B",
    purpose: "senior B: Qwen3.6-35B-A3B, MoE (~3B active), experts in RAM",
    hfRepo: "unsloth/Qwen3.6-35B-A3B-GGUF",
    file: "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    approxGB: 22,
    port: 8081,
    serverArgs: [...COMMON_GPU, "--n-cpu-moe 999", "--ctx-size 32768", "--cache-type-k q8_0 --cache-type-v q8_0"],
    notes:
      "Same family as the worker. Decode bound ≈ 34 tok/s on dual-channel DDR5-5600 (Pack G §2.2(c)). CPU prefill takes about 48 s per 8k-token prompt (Pack G §2.2(e)), so keep its prompts short. " +
      "Use Q3_K_XL (≈ 17 GB) if RAM is tight.",
  },
  {
    slot: "embed",
    purpose: "embeddings for the Library (P1)",
    hfRepo: "Qwen/Qwen3-Embedding-0.6B-GGUF",
    file: "Qwen3-Embedding-0.6B-Q8_0.gguf",
    approxGB: 0.7,
    port: 8082,
    serverArgs: ["--embedding", "--pooling last", "-ngl 99", "--ctx-size 8192"],
    notes: "Alternative: nomic-ai/nomic-embed-code-GGUF.",
  },
  {
    slot: "rerank",
    purpose: "cross-encoder reranker for the Library (P1)",
    hfRepo: "gpustack/bge-reranker-v2-m3-GGUF",
    file: "bge-reranker-v2-m3-Q8_0.gguf",
    approxGB: 0.6,
    port: 8083,
    serverArgs: ["--reranking", "-ngl 99", "--ctx-size 8192"],
    notes: "Serves POST /rerank. A Qwen3-Reranker GGUF is an alternative if your llama.cpp build supports it.",
  },
];

const dirOf = (m: ModelSpec) => (m.variant ? `${m.slot}-${m.variant.toLowerCase()}` : m.slot);

export function downloadCommand(m: ModelSpec): string {
  return `hf download ${m.hfRepo} ${m.file} --local-dir models/${dirOf(m)}`;
}

export function serveCommand(m: ModelSpec, draft?: ModelSpec): string {
  if (m.port === 0) return "(loaded by the worker; not served separately)";
  const d = draft ? ` --model-draft models/${dirOf(draft)}/${draft.file} --draft-max ${DRAFT_MAX} --draft-min ${DRAFT_MIN} -ngld 99` : "";
  return `llama-server -m models/${dirOf(m)}/${m.file} --host 127.0.0.1 --port ${m.port} ${m.serverArgs.join(" ")}${d}`;
}

export function modelsHelp(): string {
  const draft = MODELS.find((m) => m.slot === "draft");
  const L: string[] = [];
  L.push("Model slots for RTX 3080 12 GB / 32 GB RAM (contract §4). Confirm repo ids and GGUF file names on Hugging Face first:");
  L.push("  https://huggingface.co/models?search=Qwen3.5-9B%20GGUF  ·  https://huggingface.co/models?search=GLM-4.7-Flash%20GGUF  ·  https://huggingface.co/models?search=Qwen3.6-35B-A3B%20GGUF");
  L.push("");
  L.push('Prerequisites: a CUDA build of llama.cpp (llama-server on PATH), `pip install -U "huggingface_hub[cli]"` (older installs: `huggingface-cli download` takes the same arguments).');
  L.push("Run the commands from the tfa folder; models/ is gitignored. Older llama.cpp builds take -fa instead of --flash-attn on.");
  L.push("");
  for (const m of MODELS) {
    L.push(`[${m.slot}${m.variant ? " " + m.variant : ""}]  ${m.purpose}  (~${m.approxGB} GB)`);
    L.push(`  download: ${downloadCommand(m)}`);
    L.push(`  serve:    ${serveCommand(m, m.slot === "worker" ? draft : undefined)}`);
    L.push(`  note:     ${m.notes}`);
    L.push("");
  }
  L.push("Layout: worker (8080, with draft) always up; embed (8082) and rerank (8083) from P1; senior A or B (8081, never both)");
  L.push("started for slow-path work and stopped when idle (contract §4 senior lifecycle).");
  L.push("Check: curl http://127.0.0.1:8080/health   then: tfa doctor <repo>");
  return L.join("\n");
}
