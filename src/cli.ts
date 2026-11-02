#!/usr/bin/env node
/**
 * tfa — thinking-first assistant (P0)
 *
 *   tfa doctor [repo]                        environment, detected cascade, slot health, guard
 *   tfa verify [repo] [--up-to <stage>]      run the verifier cascade on the working tree
 *   tfa fix <repo> "<task>" [--dry] [--attempts N]
 *                                            draft → apply → verify → retry; keeps verified edits
 *   tfa models                               download and llama-server commands for the 3080
 *
 * `run()` is exported so tests drive the CLI in-process; the process entry is at the bottom.
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadConfig, type LoadedConfig } from "./config.js";
import { HUMAN_OWNED_ALWAYS, HUMAN_OWNED_IN_TFA, isTfaRepo } from "./files.js";
import { hashJson, newId } from "./ids.js";
import { HttpReader } from "./llm/client.js";
import { loadCore } from "./llm/prompts/index.js";
import { trajectoryDir } from "./log.js";
import { modelsHelp } from "./models.js";
import { fix, formatOutcome, type FixOutcome } from "./think/fix.js";
import { TransitionalVerifier } from "./think/verify-adapter.js";
import { STAGES, SLOTS, type Stage } from "./types.js";

export type Io = { out: (s: string) => void; err: (s: string) => void; env: NodeJS.ProcessEnv };

const USAGE = `tfa — thinking-first assistant (P0)

  tfa doctor [repo]                         check environment, detected verifier stages, model slots
  tfa verify [repo] [--up-to <stage>]       run the verifier cascade on a repo
  tfa fix <repo> "<task>" [--dry] [--attempts N]
                                            make a verified change (--dry rolls back after verifying)
  tfa models                                model download/serve commands for the RTX 3080

Config: tfa.config.json (tool) < <repo>/.tfa/config.json (project); env TFA_<SLOT>_URL overrides slot URLs.`;

type Parsed = { pos: string[]; flags: Map<string, string | true> };

export function parseArgs(argv: string[], withValue: string[]): Parsed {
  const pos: string[] = [];
  const flags = new Map<string, string | true>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const [k, v] = a.slice(2).split("=", 2);
      if (v !== undefined) flags.set(k, v);
      else if (withValue.includes(k)) flags.set(k, argv[++i] ?? "");
      else flags.set(k, true);
    } else pos.push(a);
  }
  return { pos, flags };
}

export async function run(argv: string[], io: Io = defaultIo()): Promise<number> {
  const [cmd, ...rest] = argv;
  try {
    switch (cmd) {
      case "doctor":
        return await doctor(path.resolve(rest[0] ?? "."), io);
      case "verify": {
        const a = parseArgs(rest, ["up-to"]);
        const upTo = a.flags.get("up-to");
        if (upTo !== undefined && (typeof upTo !== "string" || !STAGES.includes(upTo as Stage))) {
          io.err(`--up-to must be one of: ${STAGES.join(", ")}`);
          return 1;
        }
        return await verify(path.resolve(a.pos[0] ?? "."), io, upTo as Stage | undefined);
      }
      case "fix": {
        const a = parseArgs(rest, ["attempts"]);
        const [repo, ...words] = a.pos;
        const task = words.join(" ").trim();
        if (!repo || !task) {
          io.err(USAGE);
          return 1;
        }
        const attempts = a.flags.has("attempts") ? Number(a.flags.get("attempts")) : undefined;
        if (attempts !== undefined && (!Number.isInteger(attempts) || attempts < 1)) {
          io.err("--attempts must be a positive integer");
          return 1;
        }
        return await runFix(path.resolve(repo), task, io, { keep: !a.flags.has("dry"), attempts });
      }
      case "models":
        io.out(modelsHelp());
        return 0;
      case undefined:
      case "help":
      case "--help":
      case "-h":
        io.out(USAGE);
        return 0;
      default:
        io.err(`unknown command: ${cmd}\n\n${USAGE}`);
        return 1;
    }
  } catch (err) {
    io.err(err instanceof Error ? err.message : String(err));
    return 1;
  }
}

function tryCmd(cmd: string, args: string[], cwd: string): string | null {
  try {
    return execFileSync(cmd, args, { cwd, encoding: "utf8", windowsHide: true, stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return null;
  }
}

async function doctor(repo: string, io: Io): Promise<number> {
  const { out } = io;
  let ready = true;
  const major = Number(process.versions.node.split(".")[0]);
  out(`node   ${process.version} on ${process.platform}${major < 20 ? "  ✗ Node ≥ 20 required" : ""}`);
  if (major < 20) ready = false;
  const git = tryCmd("git", ["--version"], process.cwd());
  out(git ? `git    ${git}` : "git    ✗ MISSING (required)");
  if (!git) ready = false;

  if (!fs.existsSync(repo)) {
    out(`repo   ${repo}  ✗ does not exist`);
    return 1;
  }
  const inRepo = fs.existsSync(path.join(repo, ".git"));
  out(`repo   ${repo}  ${inRepo ? "(git repository)" : "✗ NOT a git repository — tfa fix will refuse"}`);
  if (!inRepo) ready = false;
  else {
    const dirty = tryCmd("git", ["status", "--porcelain"], repo);
    if (dirty) out(`       working tree has ${dirty.split(/\r?\n/).length} changed path(s); tfa fix keeps them and diffs only the files it edits`);
  }

  const cfg = loadConfig(repo, { env: io.env });
  out(`config ${[cfg.sources.tool && `tool ${cfg.sources.tool}`, cfg.sources.project && `project ${cfg.sources.project}`, cfg.sources.env.length && `env ${cfg.sources.env.join(",")}`].filter(Boolean).join("; ") || "defaults"}`);
  out("");

  out("cascade (apply is always first):");
  const cmds = new TransitionalVerifier(cfg).stages(repo);
  for (const s of cfg.cascade) {
    const c = cmds.find((x) => x.stage === s);
    out(`  ${s.padEnd(9)} ${c ? `${c.cmd}   [${c.source}]` : "— none (skipped; never counted as a pass)"}`);
  }
  if (cmds.length === 0) {
    out("  ✗ no stage can run: nothing can be verified in this repo, so tfa fix will refuse every candidate");
    ready = false;
  }
  out("");

  out("model slots:");
  const reader = new HttpReader(cfg.slots);
  for (const s of SLOTS) {
    const sc = cfg.slots[s];
    if (s === "draft" && sc.url === cfg.slots.worker.url) {
      out(`  ${s.padEnd(7)} served inside the worker (--model-draft)`);
      continue;
    }
    const ok = await reader.healthy(s);
    const need = s === "worker" ? "required for tfa fix" : s === "senior" ? "started on demand from P2" : "used from P1";
    const mark = s === "worker" ? "✗" : "–";
    out(`  ${s.padEnd(7)} ${sc.url.padEnd(24)} ${ok ? "✓ reachable" : `${mark} not reachable: ${need}; see \`tfa models\``}`);
    if (s === "worker" && !ok) ready = false;
  }
  out("");

  const core = loadCore(repo);
  out(`notebook core  ${core ? `${core.length} chars` : "(empty until P3/P4)"}`);
  out(`trajectories → ${trajectoryDir(repo)}`);
  out(`human-owned (I-1): ${[...HUMAN_OWNED_ALWAYS, ...(isTfaRepo(repo) ? HUMAN_OWNED_IN_TFA : [])].map((p) => p + "**").join(", ")}`);
  out("");
  out(ready ? "✓ ready for tfa fix" : "✗ not ready for tfa fix (see ✗ above)");
  return ready ? 0 : 1;
}

async function verify(repo: string, io: Io, upTo?: Stage): Promise<number> {
  const cfg = loadConfig(repo, { env: io.env });
  io.out(`verifying ${repo}${upTo ? ` up to ${upTo}` : ""}`);
  const v = new TransitionalVerifier(cfg, (s) => io.out(formatOutcome(s)));
  const r = await v.run({ id: newId("cand"), subtaskId: newId("sub"), edits: [], createdBy: "worker" }, repo, { upTo });
  const ran = r.stages.filter((s) => s.stage !== "apply" && !s.skipped).length;
  const skipped = r.stages.filter((s) => s.skipped).length;
  if (r.ok) {
    io.out(`\n✓ passed in ${(r.totalMs / 1000).toFixed(1)}s (${ran} ran, ${skipped} skipped)`);
    return 0;
  }
  if (!r.failedAt) {
    io.out(`\n✗ nothing verified: all ${skipped} stage(s) skipped`);
    return 1;
  }
  const f = r.stages.find((s) => s.stage === r.failedAt)!;
  io.out(`\n✗ failed at ${f.stage}\n${f.log.slice(-3000)}`);
  return 1;
}

/** P0 acceptance record in the contract §6.5 BenchRun shape, for the human to save as bench/runs/p0-first.json. */
export function p0Record(r: FixOutcome, cfg: LoadedConfig) {
  const { sources: _sources, ...plain } = cfg;
  return {
    id: newId("run"),
    at: Date.now(),
    set: "dev" as const,
    config: hashJson(plain),
    results: [{ taskId: r.taskId, ok: r.ok, attempts: r.attempts, firstTry: r.ok && r.attempts === 1, corrections: 0, costs: r.costs }],
    summary: { passRate: r.ok ? 1 : 0, firstTryRate: r.ok && r.attempts === 1 ? 1 : 0, meanCost: r.costs },
  };
}

async function runFix(repo: string, task: string, io: Io, o: { keep: boolean; attempts?: number }): Promise<number> {
  const cfg = loadConfig(repo, { env: io.env });
  const reader = new HttpReader(cfg.slots);
  if (!(await reader.healthy("worker"))) {
    io.err(`worker not reachable at ${cfg.slots.worker.url}. Start it first (see \`tfa models\`), then \`tfa doctor ${repo}\`.`);
    return 2;
  }
  const r = await fix({ repo, task, cfg, reader, keep: o.keep, attempts: o.attempts, log: io.out });
  io.out("");
  if (r.ok) {
    io.out(`verified change ${o.keep ? "kept in the working tree" : "rolled back (--dry)"}; diff:\n`);
    io.out(r.diff || "(empty diff)");
  } else io.out(`✗ ${r.note ?? `failed at ${r.failedAt}`}`);
  io.out(`\nattempts ${r.attempts} · wall-clock ${(r.costs.wallMs / 1000).toFixed(1)}s · tokens ${r.costs.promptTokens}+${r.costs.completionTokens} · trajectory ${r.trajectory}`);
  io.out(`\nrecord (save as bench/runs/p0-first.json for the P0 acceptance):\n${JSON.stringify(p0Record(r, cfg), null, 2)}`);
  if (r.ok) io.out(`\nnext: review the diff, run your build, then commit.`);
  return r.ok ? 0 : 1;
}

function defaultIo(): Io {
  return { out: (s) => process.stdout.write(s + "\n"), err: (s) => process.stderr.write(s + "\n"), env: process.env };
}

function isMain(): boolean {
  if (!process.argv[1]) return false;
  try {
    const a = fs.realpathSync(process.argv[1]);
    const b = fileURLToPath(import.meta.url);
    return process.platform === "win32" ? a.toLowerCase() === b.toLowerCase() : a === b;
  } catch {
    return false;
  }
}

if (isMain()) {
  run(process.argv.slice(2)).then((code) => {
    process.exitCode = code;
  });
}
