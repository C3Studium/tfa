import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";

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
  fs.writeFileSync(path.join(dir, ".gitignore"), "node_modules/\n");
  // Reuse this project's typescript so the fixture needs no install.
  // A directory junction needs no privileges on Windows; no file symlinks are used.
  const tfaRoot = path.resolve(__dirname, "..");
  fs.mkdirSync(path.join(dir, "node_modules"), { recursive: true });
  const link = path.join(dir, "node_modules", "typescript");
  fs.symlinkSync(path.join(tfaRoot, "node_modules", "typescript"), link, process.platform === "win32" ? "junction" : "dir");
  execSync("git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init", { cwd: dir, stdio: "ignore" });
  return dir;
}

export type MockCall = { method: string; path: string; body: unknown };

/** Deterministic 8-d embedding of a text (for the mock embed slot). */
export function fakeEmbedding(text: string): number[] {
  const h = createHash("sha256").update(text).digest();
  return Array.from({ length: 8 }, (_, i) => h[i] / 255);
}

/**
 * Mock llama-server. Speaks the contract §4 endpoints:
 *  POST /v1/chat/completions → scripted answers in order (the last one repeats)
 *  POST /v1/embeddings       → fakeEmbedding per input
 *  POST /rerank              → relevance = number of query words in the document
 *  POST /slots/{id}?action=… → {id_slot, filename}
 *  GET  /health              → 200, or `status` if set
 * `requests` holds chat request bodies (in order); `calls` holds every call.
 */
export function mockLlm(answers: string[], o: { status?: number } = {}): Promise<{ url: string; close: () => void; requests: unknown[]; calls: MockCall[] }> {
  const requests: unknown[] = [];
  const calls: MockCall[] = [];
  let i = 0;
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const url = req.url ?? "/";
      if (url === "/health") {
        calls.push({ method: req.method ?? "GET", path: url, body: null });
        return void res.writeHead(o.status ?? 200).end(JSON.stringify({ status: o.status ? "loading model" : "ok" }));
      }
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        const body = JSON.parse(raw || "{}") as Record<string, unknown>;
        calls.push({ method: req.method ?? "POST", path: url, body });
        const send = (v: unknown) => {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(v));
        };
        if (url === "/v1/chat/completions") {
          requests.push(body);
          const text = answers[Math.min(i, answers.length - 1)];
          i++;
          return send({ choices: [{ message: { role: "assistant", content: text } }], usage: { prompt_tokens: 100, completion_tokens: 20 } });
        }
        if (url === "/v1/embeddings") {
          const input = body.input as string[];
          return send({ data: input.map((t, index) => ({ index, embedding: fakeEmbedding(t) })) });
        }
        if (url === "/rerank") {
          const q = String(body.query).toLowerCase().split(/\s+/);
          const docs = body.documents as string[];
          const results = docs.map((d, index) => ({ index, relevance_score: q.filter((w) => d.toLowerCase().includes(w)).length })).sort((a, b) => b.relevance_score - a.relevance_score);
          return send({ results });
        }
        const m = /^\/slots\/(\d+)\?action=(save|restore)$/.exec(url);
        if (m) return send({ id_slot: Number(m[1]), filename: body.filename });
        res.writeHead(404).end("not found");
      });
    });
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as { port: number };
      resolve({ url: `http://127.0.0.1:${addr.port}`, close: () => server.close(), requests, calls });
    });
  });
}

/** Slots config pointing every slot at one mock server. */
export function slotsAt<T extends { slots: Record<string, { url: string }> }>(cfg: T, url: string): T {
  const c = structuredClone(cfg);
  for (const s of Object.values(c.slots)) s.url = url;
  return c;
}
