import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";
import { execSync } from "node:child_process";

/** A tiny TypeScript project with one deliberate type error, as a git repo. */
export function makeFixtureRepo(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tfa-fixture-"));
  fs.writeFileSync(
    path.join(dir, "package.json"),
    JSON.stringify({ name: "fixture", private: true, version: "0.0.0", scripts: { typecheck: "node node_modules/typescript/bin/tsc --noEmit -p tsconfig.json" }, devDependencies: { typescript: "^5.6.0" } }, null, 2),
  );
  fs.writeFileSync(path.join(dir, "tsconfig.json"), JSON.stringify({ compilerOptions: { strict: true, noEmit: true, target: "ES2022", module: "ESNext", moduleResolution: "Bundler", types: [] }, include: ["src"] }, null, 2));
  fs.mkdirSync(path.join(dir, "src"));
  fs.writeFileSync(
    path.join(dir, "src/price.ts"),
    `export type Item = { name: string; priceCzk: number; qty: number };

export function total(items: Item[]): number {
  let sum = 0;
  for (const it of items) sum += it.priceCzk * it.qty;
  return sum;
}

export function label(items: Item[]): string {
  const t: string = total(items); // deliberate error: number is not assignable to string
  return t + " Kč";
}
`,
  );
  // Reuse this project's typescript so the fixture needs no install.
  // A directory junction needs no privileges on Windows; no file symlinks are used.
  const tfaRoot = path.resolve(__dirname, "..");
  fs.mkdirSync(path.join(dir, "node_modules"), { recursive: true });
  const link = path.join(dir, "node_modules", "typescript");
  fs.symlinkSync(path.join(tfaRoot, "node_modules", "typescript"), link, process.platform === "win32" ? "junction" : "dir");
  execSync("git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init", { cwd: dir, stdio: "ignore" });
  return dir;
}

/**
 * Mock llama-server: OpenAI-compatible chat endpoint that returns scripted answers in order.
 * Records every request so tests can assert on the reflection messages.
 */
export function mockLlm(answers: string[]): Promise<{ url: string; close: () => void; requests: unknown[] }> {
  const requests: unknown[] = [];
  let i = 0;
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      if (req.url === "/health") return void res.writeHead(200).end('{"status":"ok"}');
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        requests.push(JSON.parse(body || "{}"));
        const text = answers[Math.min(i, answers.length - 1)];
        i++;
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: text } }], usage: { prompt_tokens: 100, completion_tokens: 20 } }));
      });
    });
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as { port: number };
      resolve({ url: `http://127.0.0.1:${addr.port}`, close: () => server.close(), requests });
    });
  });
}
