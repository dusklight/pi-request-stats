/**
 * Smoke + regression tests for the extension itself (extensions/pi-request-stats/index.ts).
 *
 * The pi runtime is supplied by the peer dependency; these tests drive the extension with a
 * stub ExtensionAPI/ExtensionContext, which is enough to check what it registers and how it
 * attributes usage it reconstructs from session history.
 */
import { test, describe, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// loadPricing() runs at import time, so make HOME/cwd hermetic BEFORE importing the extension.
const scratch = mkdtempSync(join(tmpdir(), "pi-request-stats-ext-"));
const savedHome = process.env.HOME;
const savedEnvKeys = ["PI_REQUEST_STATS_PRICING", "PI_REQUEST_STATS_PRICING_MODE", "PI_REQUEST_STATS_COST", "PI_REQUEST_STATS_NO_COST"];
const savedEnv: Record<string, string | undefined> = {};
process.env.HOME = join(scratch, "empty-home");
for (const key of savedEnvKeys) {
  savedEnv[key] = process.env[key];
  delete process.env[key];
}

const { default: extension } = await import("../extensions/pi-request-stats/index.ts");

after(() => {
  process.env.HOME = savedHome;
  for (const key of savedEnvKeys) {
    if (savedEnv[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnv[key];
  }
  rmSync(scratch, { recursive: true, force: true });
});

type Handler = (event: unknown, ctx: unknown) => Promise<unknown> | unknown;

interface Harness {
  hooks: Map<string, Handler[]>;
  commands: Map<string, { description?: string; handler: Handler }>;
  flags: Map<string, { description?: string; type?: string }>;
  notices: string[];
  statuses: string[];
  ctx: unknown;
}

function stubModel(model: string, provider: string) {
  return { id: model, provider, reasoningLevels: [], contextWindow: 200_000, maxTokens: 8192 };
}

function harness(
  branch: unknown[],
  options: { model?: string; provider?: string } = {}
): Harness {
  const hooks = new Map<string, Handler[]>();
  const commands = new Map<string, { description?: string; handler: Handler }>();
  const flags = new Map<string, { description?: string; type?: string }>();
  const notices: string[] = [];
  const statuses: string[] = [];

  // The status bar colors its text, so ui.theme.fg/bg must be real functions.
  const theme = new Proxy({}, { get: () => (text: string) => text });
  const ui = new Proxy(
    {
      notify: (text: string) => notices.push(text),
      setStatus: (_key: string, text: string) => statuses.push(text),
      theme,
    },
    { get: (target, prop) => (prop in target ? (target as Record<string, unknown>)[String(prop)] : () => undefined) }
  );

  const sessionManager = new Proxy(
    { getBranch: () => branch, getEntries: () => branch, getSessionId: () => "test-session" },
    { get: (target, prop) => (prop in target ? (target as Record<string, unknown>)[String(prop)] : () => undefined) }
  );

  const ctx = {
    cwd: scratch,
    model: stubModel(options.model ?? "gpt-6.1-sol", options.provider ?? "openai"),
    sessionManager,
    ui,
    mode: "print",
    isIdle: () => true,
    abort: () => {},
    shutdown: () => {},
  };

  const api = {
    on: (name: string, handler: Handler) => hooks.set(name, [...(hooks.get(name) ?? []), handler]),
    registerCommand: (name: string, definition: { description?: string; handler: Handler }) =>
      commands.set(name, definition),
    registerFlag: (name: string, definition: { description?: string; type?: string }) => flags.set(name, definition),
    getFlag: () => undefined,
  };

  extension(api as never);
  return { hooks, commands, flags, notices, statuses, ctx };
}

const iso = (ms: number) => new Date(ms).toISOString();
const T0 = Date.parse("2026-10-18T12:00:00.000Z");

const zeroCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 };

const userEntry = {
  type: "message",
  id: "e1",
  parentId: null,
  timestamp: iso(T0),
  message: { role: "user", content: [{ type: "text", text: "summarize this repo" }] },
};

const assistantEntry = {
  type: "message",
  id: "e2",
  parentId: "e1",
  timestamp: iso(T0 + 5000),
  message: {
    role: "assistant",
    provider: "openai",
    model: "gpt-6.1-sol",
    timestamp: T0,
    content: [
      { type: "thinking", thinking: "looking at the files" },
      { type: "text", text: "here is the summary" },
      { type: "toolCall", id: "t1", name: "read", arguments: { path: "README.md" } },
    ],
    usage: { input: 1000, output: 500, cacheRead: 2000, cacheWrite: 1000, totalTokens: 4500, cost: zeroCost },
  },
};

const compactionEntry = {
  type: "compaction",
  id: "e3",
  parentId: "e2",
  timestamp: iso(T0 + 9000),
  summary: "conversation compacted",
  firstKeptEntryId: "e2",
  tokensBefore: 4500,
  // pi writes no model/provider here (appendCompaction), only usage.
  usage: { input: 3000, output: 400, cacheRead: 0, cacheWrite: 0, totalTokens: 3400, cost: zeroCost },
};

describe("extension registration", () => {
  test("registers the documented hooks, commands and CLI flags", async () => {
    const h = harness([userEntry, assistantEntry, compactionEntry]);
    for (const name of [
      "session_start",
      "agent_start",
      "message_start",
      "message_update",
      "message_end",
      "tool_execution_start",
      "tool_execution_update",
      "tool_execution_end",
      "agent_settled",
      "session_compact",
      "session_before_compact",
    ]) {
      assert.ok(h.hooks.has(name), `missing hook: ${name}`);
    }
    assert.deepEqual([...h.commands.keys()].sort(), ["cost", "stats"]);
    assert.deepEqual([...h.flags.keys()].sort(), ["pi-request-stats-no-cost", "pi-request-stats-pricing"]);
  });

  test("/stats and /cost produce output without throwing", async () => {
    const h = harness([userEntry, assistantEntry, compactionEntry]);
    const start = h.hooks.get("session_start")![0];
    await start({}, h.ctx);
    await h.commands.get("stats")!.handler("", h.ctx);
    await h.commands.get("cost")!.handler("", h.ctx);
    const [stats, cost] = h.notices;
    assert.match(stats, /Requests: 1/);
    assert.match(stats, /Compactions: 1/);
    assert.match(cost, /Session cost:/);
    assert.match(cost, /\/cost\s+session cost by model/);
  });
});

describe("usage reconstructed from history", () => {
  test("a compaction's usage is priced as the model that was running, not an unknown model", async () => {
    const h = harness([userEntry, assistantEntry, compactionEntry]);
    await h.hooks.get("session_start")![0]({}, h.ctx);
    await h.commands.get("cost")!.handler("", h.ctx);
    const cost = h.notices.join("\n");
    assert.match(cost, /openai\/gpt-6\.1-sol: 2 model calls/, "assistant usage + compaction usage on one row");
    assert.doesNotMatch(cost, /unknown model/, "compaction usage must not land in an unattributed row");
  });

  test("with no earlier model to copy, the currently selected model is used", async () => {
    const h = harness([compactionEntry], { model: "gpt-6-astra", provider: "openai" });
    await h.hooks.get("session_start")![0]({}, h.ctx);
    await h.commands.get("cost")!.handler("", h.ctx);
    const cost = h.notices.join("\n");
    assert.match(cost, /openai\/gpt-6-astra: 1 model call/);
    assert.doesNotMatch(cost, /unknown model/);
  });

  test("tokens and compactions are counted from history so a resumed session reports numbers", async () => {
    const h = harness([userEntry, assistantEntry, compactionEntry]);
    await h.hooks.get("session_start")![0]({}, h.ctx);
    await h.commands.get("stats")!.handler("", h.ctx);
    const stats = h.notices.join("\n");
    assert.match(stats, /Session tokens: 4,500 total/);
    assert.match(stats, /🧠 thinking: .*→ .*tok\/s/);
    assert.match(stats, /💬 answer:/);
  });
});
