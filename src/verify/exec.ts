/**
 * Cross-platform command runner with a hard timeout and captured output.
 * Uses the shell so `npx`, `npm run` and Windows `.cmd` shims all resolve.
 */
import { spawn } from "node:child_process";

export type ExecResult = { code: number | null; out: string; ms: number; timedOut: boolean };

export function exec(cmd: string, cwd: string, timeoutMs: number, env: NodeJS.ProcessEnv = {}): Promise<ExecResult> {
  return new Promise((resolve) => {
    const t0 = Date.now();
    const child = spawn(cmd, { cwd, shell: true, env: { ...process.env, CI: "1", FORCE_COLOR: "0", ...env }, windowsHide: true });
    const chunks: Buffer[] = [];
    let total = 0;
    const MAX = 2_000_000; // keep logs bounded
    const push = (b: Buffer) => {
      if (total < MAX) {
        chunks.push(b);
        total += b.length;
      }
    };
    child.stdout?.on("data", push);
    child.stderr?.on("data", push);
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      killTree(child.pid);
    }, timeoutMs);
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code, out: Buffer.concat(chunks).toString("utf8"), ms: Date.now() - t0, timedOut });
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({ code: -1, out: String(err), ms: Date.now() - t0, timedOut });
    });
  });
}

function killTree(pid: number | undefined) {
  if (!pid) return;
  try {
    if (process.platform === "win32") spawn("taskkill", ["/pid", String(pid), "/T", "/F"], { windowsHide: true });
    else process.kill(-pid, "SIGKILL");
  } catch {
    try {
      process.kill(pid, "SIGKILL");
    } catch {
      /* already gone */
    }
  }
}

/** Keep only the informative tail of a long log for prompts and trajectories. */
export function tailLog(out: string, maxChars = 6000): string {
  const s = out.replace(/\u001b\[[0-9;]*m/g, "");
  return s.length <= maxChars ? s : "…(truncated)…\n" + s.slice(-maxChars);
}
