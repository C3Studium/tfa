/**
 * I-9: the single place a prompt is assembled. The order is fixed — system, core, context, task — so the
 * prefix (system + Notebook core block) is byte-identical across calls and llama-server reuses its KV cache
 * (contract §4 "Prompt structure"). Conversation turns (retries, feedback) are appended after the task and
 * never reorder the sections before them.
 */
import fs from "node:fs";
import path from "node:path";
import type { Msg } from "../../types.js";

export type PromptParts = {
  /** 1. static per module */
  system: string;
  /** 2. Notebook core block (core.md), ≤ coreTokens; "" until P3/P4 fill it */
  core: string;
  /** 3. retrieved context, already packed under budgets.contextTokens */
  context: string;
  /** 4. the task */
  task: string;
  /** 4. conversation after the task: earlier answers and feedback */
  conversation?: Msg[];
};

export const SECTION = { core: "## Notebook core", context: "## Context", task: "## Task" } as const;

export function buildPrompt(p: PromptParts): Msg[] {
  // System and core share the one system message: several chat templates (Qwen, GLM) accept a single
  // system message at the start only. The core section is always present, so the prefix shape is stable.
  const system = `${p.system.trimEnd()}\n\n${SECTION.core}\n${p.core.trim() || "(empty)"}`;
  const user = `${SECTION.context}\n${p.context.trim() || "(none)"}\n\n${SECTION.task}\n${p.task.trim()}`;
  return [{ role: "system", content: system }, { role: "user", content: user }, ...(p.conversation ?? [])];
}

/** Read `<repo>/.tfa/notebook/core.md` verbatim, minus its schema frontmatter (contract §6.1, §11). */
export function loadCore(repo: string): string {
  const f = path.join(repo, ".tfa", "notebook", "core.md");
  if (!fs.existsSync(f)) return "";
  const raw = fs.readFileSync(f, "utf8").replace(/^﻿/, "").replace(/\r\n/g, "\n");
  return raw.replace(/^---\n[\s\S]*?\n---\n?/, "").trim();
}
