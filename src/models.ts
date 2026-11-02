/**
 * Model slots for the target machine (RTX 3080 12 GB, 32 GB DDR5) and the llama-server
 * commands to serve them. Exact Hugging Face repo ids change; `tfa models` prints the
 * commands and the search URLs to confirm them at install time.
 */
export type ModelSpec = {
  slot: string;
  purpose: string;
  hfRepo: string;
  file: string;
  approxGB: number;
  port: number;
  serverArgs: string[];
  notes: string;
};

export const MODELS: ModelSpec[] = [
  {
    slot: "worker",
    purpose: "fast resident reader: drafts, reflects, parses (VRAM)",
    hfRepo: "unsloth/Qwen3.5-9B-GGUF",
    file: "Qwen3.5-9B-Q6_K.gguf",
    approxGB: 7.5,
    port: 8080,
    serverArgs: [
      "-ngl 99",
      "--ctx-size 32768",
      "--cache-type-k q8_0 --cache-type-v q8_0",
      "--parallel 1",
      "--slot-save-path .tfa/slots",
      "--jinja",
      "--flash-attn on",
    ],
    notes: "Q6_K keeps quality; drop to Q5_K_M if you need more context room. Pair with the draft model below for 2–3x decoding.",
  },
  {
    slot: "draft",
    purpose: "speculative-decoding draft for the worker (same tokenizer family)",
    hfRepo: "unsloth/Qwen3.5-0.8B-GGUF",
    file: "Qwen3.5-0.8B-Q8_0.gguf",
    approxGB: 0.9,
    port: 0,
    serverArgs: ["(passed to the worker as: --model-draft <file> --draft-max 16 --draft-min 4 -ngld 99)"],
    notes: "Only useful with a same-family target. If the exact 0.8B GGUF is not published, any Qwen3.5 ≤2B works.",
  },
  {
    slot: "senior",
    purpose: "slow strong reader for novel subtasks (experts in RAM, ~3B active)",
    hfRepo: "unsloth/Qwen3.6-35B-A3B-GGUF",
    file: "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    approxGB: 22,
    port: 8081,
    serverArgs: [
      "-ngl 99",
      "--n-cpu-moe 999", // all expert tensors on CPU/RAM; attention + shared on GPU
      "--ctx-size 32768",
      "--cache-type-k q8_0 --cache-type-v q8_0",
      "--parallel 1",
      "--jinja",
      "--flash-attn on",
    ],
    notes: "Reported SWE-bench Verified ~73% (vendor). Same family as the worker → the worker can draft for it. If RAM is tight use Q3_K_XL (~17 GB). Alternate: zai-org GLM-4.7-Flash GGUF (~19 GB, MIT).",
  },
  {
    slot: "embed",
    purpose: "embeddings for the library (P1)",
    hfRepo: "Qwen/Qwen3-Embedding-0.6B-GGUF",
    file: "Qwen3-Embedding-0.6B-Q8_0.gguf",
    approxGB: 0.7,
    port: 8082,
    serverArgs: ["--embedding", "-ngl 99", "--ctx-size 8192", "--pooling last"],
    notes: "Alternative: nomic-ai/nomic-embed-code-GGUF.",
  },
];

export function downloadCommand(m: ModelSpec): string {
  return `huggingface-cli download ${m.hfRepo} ${m.file} --local-dir models/${m.slot}`;
}

export function serveCommand(m: ModelSpec, withDraft?: ModelSpec): string {
  if (m.port === 0) return "(not served separately)";
  const draft = withDraft ? ` --model-draft models/${withDraft.slot}/${withDraft.file} --draft-max 16 --draft-min 4 -ngld 99` : "";
  return `llama-server -m models/${m.slot}/${m.file} --port ${m.port} --host 127.0.0.1 ${m.serverArgs.join(" ")}${draft}`;
}

export function modelsHelp(): string {
  const lines: string[] = [];
  lines.push("Model slots for RTX 3080 12 GB / 32 GB RAM. Confirm exact GGUF file names on Hugging Face before downloading:");
  lines.push("  https://huggingface.co/models?search=Qwen3.5-9B%20GGUF   https://huggingface.co/models?search=Qwen3.6-35B-A3B%20GGUF");
  lines.push("");
  lines.push("Prerequisites: llama.cpp release with CUDA (llama-server on PATH), `pip install huggingface_hub[cli]`.");
  lines.push("");
  const draft = MODELS.find((m) => m.slot === "draft");
  for (const m of MODELS) {
    lines.push(`[${m.slot}]  ${m.purpose}  (~${m.approxGB} GB)`);
    lines.push(`  download: ${downloadCommand(m)}`);
    lines.push(`  serve:    ${serveCommand(m, m.slot === "worker" ? draft : undefined)}`);
    lines.push(`  note:     ${m.notes}`);
    lines.push("");
  }
  lines.push("Serving layout: worker (8080) stays up; senior (8081) is started for slow-path work and stopped when idle.");
  lines.push("Check: curl http://127.0.0.1:8080/health   then: tfa doctor");
  return lines.join("\n");
}
