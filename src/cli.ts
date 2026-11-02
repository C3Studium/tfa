#!/usr/bin/env node
/**
 * tfa — thinking-first assistant (P0)
 *
 *   tfa doctor [repo]          check node/git/llama-server and what the cascade would run
 *   tfa verify [repo]          run the verifier cascade only
 *   tfa fix <repo> "<task>"    draft → apply → verify → retry; keeps verified edits in place
 *   tfa models                 download and serve commands for the 3080 setup
 */
import fs from "node:fs";
import path from "node:path";
import { loadConfig } from "./config.js";
import { HttpReader } from "./llm/client.js";
import { modelsHelp } from "./models.js";
import { fix } from "./think/fix.js";
import { runCascade, formatStage } from "./verify/cascade.js";
import { detectCommands, readPkg } from "./verify/detect.js";
import { exec } from "./verify/exec.js";

const out = (s = "") => process.stdout.write(s + "\n");

async function main(argv: string[]) {
  const [cmd, ...rest] = argv;
  switch (cmd) {
    case "doctor":
      return doctor(path.resolve(rest[0] ?? "."));
    case "verify":
      return verify(path.resolve(rest[0] ?? "."));
    case "fix": {
      const repo = rest[0];
      const task = rest.slice(1).join(" ").trim();
      if (!repo || !task) return usage(1);
      return runFix(path.resolve(repo), task, rest.includes("--dry") ? false : true);
    }
    case "models":
      return out(modelsHelp());
    default:
      return usage(cmd ? 1 : 0);
  }
}

function usage(code: number) {
  out(`tfa — thinking-first assistant (P0)

  tfa doctor [repo]           check environment and detected verifier commands
  tfa verify [repo]           run the verifier cascade on a repo
  tfa fix <repo> "<task>"     make a verified change (add --dry to roll back after verifying)
  tfa models                  model download/serve commands for the 3080

Config: tfa.config.json (tool), <repo>/.tfa/config.json overrides; env TFA_<SLOT>_URL overrides slot URLs.`);
  process.exit(code);
}

async function doctor(repo: string) {
  const cfg = loadConfig(repo);
  out(`node ${process.version} on ${process.platform}`);
  const git = await exec("git --version", repo, 10_000);
  out(git.code === 0 ? `git  ${git.out.trim()}` : "git  MISSING (required)");
  const inRepo = fs.existsSync(path.join(repo, ".git"));
  out(`repo ${repo}  ${inRepo ? "(git repository)" : "(NOT a git repository — tfa fix will refuse)"}`);
  const pkg = readPkg(repo);
  out(pkg ? `package.json found` : `package.json not found — cascade will have nothing to run`);
  out("");
  out("cascade stages detected:");
  const cmds = detectCommands(repo, cfg.cascade);
  for (const s of cfg.cascade) {
    const c = cmds.find((x) => x.stage === s);
    out(`  ${s.padEnd(9)} ${c ? c.cmd + "   [" + c.source + "]" : "— none (skipped)"}`);
  }
  out("");
  for (const name of ["worker", "senior"] as const) {
    const slot = cfg.slots[name];
    const ok = await new HttpReader(cfg.slots).healthy(name);
    out(`${name.padEnd(7)} ${slot.url}  ${ok ? "reachable" : "not reachable (start llama-server; see `tfa models`)"}`);
  }
  out("");
  out(`trajectories → ${path.join(repo, ".tfa", "trajectories")}`);
}

async function verify(repo: string) {
  const cfg = loadConfig(repo);
  out(`verifying ${repo}`);
  const r = await runCascade(repo, cfg, { onStage: (s) => out(formatStage(s)) });
  if (!r.ok) {
    const f = r.stages.find((s) => !s.ok)!;
    out(`\n✗ failed at ${f.stage}\n${f.log.slice(-3000)}`);
    process.exit(1);
  }
  out(`\n✓ all stages passed in ${(r.totalMs / 1000).toFixed(1)}s`);
}

async function runFix(repo: string, task: string, keep: boolean) {
  const cfg = loadConfig(repo);
  const reader = new HttpReader(cfg.slots);
  if (!(await reader.healthy("worker"))) {
    out(`worker not reachable at ${cfg.slots.worker.url}. Start it first (see \`tfa models\`).`);
    process.exit(2);
  }
  const r = await fix({ repo, task, cfg, reader, keep, log: out });
  if (r.ok) {
    out(`\nverified change ${keep ? "kept in the working tree" : "rolled back (--dry)"}; diff:\n`);
    out(r.diff ?? "(empty diff)");
    out(`\nnext: review, then commit. Trajectory: ${r.trajectory}`);
  } else {
    out(`\nfailed at ${r.failedAt}. Trajectory: ${r.trajectory}`);
    process.exit(1);
  }
}

main(process.argv.slice(2)).catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
