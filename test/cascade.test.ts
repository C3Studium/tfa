import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { makeFixtureRepo } from "./helpers.js";
import { detectCommands } from "../src/verify/detect.js";
import { runCascade } from "../src/verify/cascade.js";
import { DEFAULT_CONFIG } from "../src/config.js";
import { parseTscErrors, buildContext } from "../src/retrieve/context.js";

describe("cascade on the fixture repo", () => {
  it("detects typecheck from scripts and skips stages with no command", () => {
    const repo = makeFixtureRepo();
    const cmds = detectCommands(repo, ["lint", "typecheck", "test", "build"]);
    expect(cmds.map((c) => c.stage)).toEqual(["typecheck"]);
    expect(cmds[0].source).toBe("scripts.typecheck");
  });

  it("fails at typecheck on the broken fixture and passes once fixed", async () => {
    const repo = makeFixtureRepo();
    const r1 = await runCascade(repo, DEFAULT_CONFIG);
    expect(r1.ok).toBe(false);
    expect(r1.failedAt).toBe("typecheck");
    const errs = parseTscErrors(r1.stages.find((s) => s.stage === "typecheck")!.log);
    expect(errs.length).toBe(1);
    expect(errs[0].file).toBe("src/price.ts");
    expect(errs[0].code).toBe("TS2322");

    const ctx = buildContext(repo, "fix the type error", r1.stages[1].log, 20_000);
    expect(ctx.files).toContain("src/price.ts");
    expect(ctx.text).toContain("Type-check errors (1)");

    const p = path.join(repo, "src/price.ts");
    fs.writeFileSync(p, fs.readFileSync(p, "utf8").replace("const t: string = total(items);", "const t: number = total(items);"));
    const r2 = await runCascade(repo, DEFAULT_CONFIG);
    expect(r2.ok).toBe(true);
    expect(r2.stages.filter((s) => s.skipped).map((s) => s.stage)).toEqual(["lint", "test", "build"]);
  }, 120_000);
});
