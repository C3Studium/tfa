/**
 * Detect which verifier commands a target repo supports, from its package.json.
 * Explicit scripts win; otherwise fall back to the tool binaries present.
 * A stage with no command is skipped (reported as "skipped", not as a pass).
 */
import fs from "node:fs";
import path from "node:path";
import type { CascadeStage } from "../config.js";

export type StageCommand = { stage: CascadeStage; cmd: string; source: string };

type Pkg = { scripts?: Record<string, string>; dependencies?: Record<string, string>; devDependencies?: Record<string, string> };

export function readPkg(repo: string): Pkg | null {
  const p = path.join(repo, "package.json");
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, "utf8")) as Pkg;
  } catch {
    return null;
  }
}

export function detectCommands(repo: string, stages: CascadeStage[]): StageCommand[] {
  const pkg = readPkg(repo);
  const scripts = pkg?.scripts ?? {};
  const deps = { ...(pkg?.dependencies ?? {}), ...(pkg?.devDependencies ?? {}) };
  const has = (d: string) => d in deps;
  const hasScript = (s: string) => s in scripts;
  const out: StageCommand[] = [];

  for (const stage of stages) {
    let cmd: string | undefined;
    let source = "";
    switch (stage) {
      case "lint":
        if (hasScript("lint")) (cmd = "npm run lint --silent"), (source = "scripts.lint");
        else if (has("eslint")) (cmd = "npx eslint . --max-warnings=1000000"), (source = "eslint");
        else if (has("biome") || has("@biomejs/biome")) (cmd = "npx biome check ."), (source = "biome");
        break;
      case "typecheck":
        if (hasScript("typecheck")) (cmd = "npm run typecheck --silent"), (source = "scripts.typecheck");
        else if (hasScript("type-check")) (cmd = "npm run type-check --silent"), (source = "scripts.type-check");
        else if (has("typescript") || fs.existsSync(path.join(repo, "tsconfig.json"))) (cmd = "npx tsc --noEmit -p tsconfig.json"), (source = "tsc");
        break;
      case "test":
        if (hasScript("test") && !/no test specified/.test(scripts.test)) {
          // Prefer a non-watch invocation.
          if (has("vitest")) (cmd = "npx vitest run"), (source = "vitest");
          else if (has("jest")) (cmd = "npx jest --ci"), (source = "jest");
          else (cmd = "npm test --silent"), (source = "scripts.test");
        } else if (has("vitest")) (cmd = "npx vitest run"), (source = "vitest");
        else if (has("jest")) (cmd = "npx jest --ci"), (source = "jest");
        break;
      case "build":
        if (hasScript("build")) (cmd = "npm run build --silent"), (source = "scripts.build");
        break;
      case "browser":
        if (has("@playwright/test") && (hasScript("test:e2e") || hasScript("e2e")))
          (cmd = `npm run ${hasScript("test:e2e") ? "test:e2e" : "e2e"} --silent`), (source = "playwright");
        break;
      case "apply":
        break;
    }
    if (cmd) out.push({ stage, cmd, source });
  }
  return out;
}

/** Summarise dependencies with versions: the seed of version-aware retrieval (P1). */
export function depSummary(repo: string, max = 40): string {
  const pkg = readPkg(repo);
  if (!pkg) return "(no package.json)";
  const all = { ...(pkg.dependencies ?? {}), ...(pkg.devDependencies ?? {}) };
  const keys = Object.keys(all).sort();
  const important = keys.filter((k) => /^(next|react|react-dom|typescript|@types\/react|sass|framer-motion|three|@react-three|medusa|@medusajs|prisma|drizzle|zod|vitest|jest|eslint|tailwindcss|playwright)/.test(k));
  const rest = keys.filter((k) => !important.includes(k));
  const pick = [...important, ...rest].slice(0, max);
  return pick.map((k) => `${k}@${all[k]}`).join(", ") + (keys.length > max ? `, … (+${keys.length - max})` : "");
}
