import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildPrompt, loadCore, SECTION } from "../src/llm/prompts/index.js";

describe("buildPrompt (I-9: system, core, context, task)", () => {
  const parts = { system: "SYS", core: "- prefer server components", context: "src/a.ts:1 error TS2322", task: "Fix it." };

  it("snapshot of the fixed order", () => {
    expect(buildPrompt(parts)).toMatchInlineSnapshot(`
      [
        {
          "content": "SYS

      ## Notebook core
      - prefer server components",
          "role": "system",
        },
        {
          "content": "## Context
      src/a.ts:1 error TS2322

      ## Task
      Fix it.",
          "role": "user",
        },
      ]
    `);
  });

  it("sections appear in order and the prefix is stable as the conversation grows", () => {
    const a = buildPrompt(parts);
    const b = buildPrompt({ ...parts, conversation: [{ role: "assistant", content: "[]" }, { role: "user", content: "typecheck failed" }] });
    const text = a.map((m) => m.content).join("\n");
    const pos = [text.indexOf("SYS"), text.indexOf(SECTION.core), text.indexOf(SECTION.context), text.indexOf(SECTION.task)];
    expect([...pos].sort((x, y) => x - y)).toEqual(pos);
    expect(pos.every((p) => p >= 0)).toBe(true);
    expect(b.slice(0, 2)).toEqual(a); // prefix unchanged → KV cache reuse
    expect(b.slice(2).map((m) => m.role)).toEqual(["assistant", "user"]);
  });

  it("an empty core still produces the core section (stable prefix shape)", () => {
    expect(buildPrompt({ ...parts, core: "" })[0].content).toContain(`${SECTION.core}\n(empty)`);
  });
});

describe("loadCore", () => {
  it("reads core.md without schema frontmatter, CRLF-safe; empty when absent", () => {
    const repo = fs.mkdtempSync(path.join(os.tmpdir(), "tfa-core-"));
    expect(loadCore(repo)).toBe("");
    fs.mkdirSync(path.join(repo, ".tfa", "notebook"), { recursive: true });
    fs.writeFileSync(path.join(repo, ".tfa", "notebook", "core.md"), "---\r\nschema: 1\r\n---\r\n- use zod for input\r\n");
    expect(loadCore(repo)).toBe("- use zod for input");
  });
});
