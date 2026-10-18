/**
 * Pi Stat Plugin
 *
 * Shows how long each request took, split into model generation time (thinking,
 * answering, tool-call arguments) and tool execution time, the tokens consumed,
 * the tokens/second throughput for thinking and answering separately, the money
 * those tokens cost, and how many compactions have occurred in the current session.
 *
 * Tool execution wall time is measured and excluded from the throughput numbers,
 * so slow tools (a bash command waiting on a long build, for example) no longer
 * drag the reported tokens/second down.
 *
 * Cost comes from pi's reported usage when the provider prices the model, and from
 * the extension's own `pricing.json` otherwise — which is what makes local models
 * (llama.cpp reports no cost) show a figure too. See cost.ts for the file layout.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import {
  addCost,
  costDetail,
  emptyCost,
  findModelPricing,
  formatCost,
  formatCostShort,
  loadPricing,
  normalize,
  priceUsage,
  rateLines,
  resolveMode,
  type CostBreakdown,
  type CostSource,
  type EstimateUsage,
  type ModelMatch,
  type ModelPricing,
  type PricingConfig,
} from "./cost.ts";

type Phase = "thinking" | "text" | "toolArgs";

interface UsageLike {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  totalTokens: number;
  reasoning?: number;
}

interface TokenCounts {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  total: number;
}

/** One priced usage record, kept so costs can be recomputed when prices change. */
interface UsageRecord {
  provider?: string;
  model?: string;
  usage: EstimateUsage;
}

/** Where a cost figure came from, plus the price list that produced it. */
interface CostOrigin {
  source: CostSource;
  pricing?: ModelPricing;
  modeName?: string;
  longContext?: boolean;
  fallback?: boolean;
}

interface ModelCostTotals {
  calls: number;
  tokens: number;
  cost: CostBreakdown;
  origin: CostOrigin;
  sources: Set<CostSource>;
}

/** Per-phase durations (ms) and token counts for one request or the whole session. */
interface PhaseStats {
  thinkingMs: number;
  textMs: number;
  toolArgsMs: number;
  thinkingTokens: number;
  textTokens: number;
  toolArgsTokens: number;
}

/** Character counts streamed so far for one in-flight assistant message. */
interface PhaseChars {
  thinking: number;
  text: number;
  toolArgs: number;
}

const PHASES: Phase[] = ["thinking", "text", "toolArgs"];
const PHASE_MS_KEY: Record<Phase, keyof PhaseStats> = {
  thinking: "thinkingMs",
  text: "textMs",
  toolArgs: "toolArgsMs",
};
const PHASE_TOKENS_KEY: Record<Phase, keyof PhaseStats> = {
  thinking: "thinkingTokens",
  text: "textTokens",
  toolArgs: "toolArgsTokens",
};

/**
 * Roughly how many characters one token of each phase occupies.
 *
 * Only needed when the provider does not report reasoning tokens (see
 * `splitOutputTokens`), where streamed characters are the only token signal left.
 * These figures are fitted over thousands of real assistant messages on local
 * chat models: thinking ~3.2 characters/token, answers ~3.5, tool-call JSON ~3.2
 * (JSON is denser because of punctuation and short keys). Tokenizers differ, so the
 * run's own measured rate takes over as soon as a message completes.
 */
const PHASE_CHARS_PER_TOKEN: Record<Phase, number> = {
  thinking: 3.2,
  text: 3.5,
  toolArgs: 3.2,
};

/** Per-phase token estimates derived from streamed characters. */
function charTokenWeights(chars: PhaseChars): Record<Phase, number> {
  return {
    thinking: chars.thinking / PHASE_CHARS_PER_TOKEN.thinking,
    text: chars.text / PHASE_CHARS_PER_TOKEN.text,
    toolArgs: chars.toolArgs / PHASE_CHARS_PER_TOKEN.toolArgs,
  };
}

function createTokenCounts(): TokenCounts {
  return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 };
}

function createPhaseStats(): PhaseStats {
  return {
    thinkingMs: 0,
    textMs: 0,
    toolArgsMs: 0,
    thinkingTokens: 0,
    textTokens: 0,
    toolArgsTokens: 0,
  };
}

function createPhaseChars(): PhaseChars {
  return { thinking: 0, text: 0, toolArgs: 0 };
}

function usageToCounts(usage: UsageLike | undefined): TokenCounts {
  const counts = createTokenCounts();
  if (!usage) return counts;
  counts.input = usage.input;
  counts.output = usage.output;
  counts.cacheRead = usage.cacheRead;
  counts.cacheWrite = usage.cacheWrite;
  counts.total = usage.totalTokens;
  return counts;
}

function addTokens(target: TokenCounts, source: TokenCounts): void {
  target.input += source.input;
  target.output += source.output;
  target.cacheRead += source.cacheRead;
  target.cacheWrite += source.cacheWrite;
  target.total += source.total;
}

function addPhaseStats(target: PhaseStats, source: PhaseStats): void {
  for (const phase of PHASES) {
    target[PHASE_MS_KEY[phase]] += source[PHASE_MS_KEY[phase]];
    target[PHASE_TOKENS_KEY[phase]] += source[PHASE_TOKENS_KEY[phase]];
  }
}

/**
 * Split an assistant message's output tokens across thinking / text / tool-call args.
 *
 * `estimated` says whether thinking tokens had to be guessed from streamed characters
 * because the provider did not report a usable reasoning count.
 */
function splitOutputTokens(
  usage: UsageLike | undefined,
  chars: PhaseChars
): { stats: PhaseStats; estimated: boolean } {
  const stats = createPhaseStats();
  const output = usage?.output ?? 0;
  if (output <= 0) return { stats, estimated: false };

  // `reasoning` is a number on providers that expose a breakdown; pi fills in 0 for the
  // many OpenAI-compatible engines (llama.cpp, vLLM, ...) that report none, so 0 must be
  // read as "not reported", never as "the model did no thinking". Those engines also fold
  // thinking tokens into `output`, exactly like OpenAI does.
  const reported = Math.min(Math.max(Math.round(usage?.reasoning ?? 0), 0), output);
  if (reported > 0) {
    // The provider reports thinking tokens exactly; split the rest by streamed characters.
    stats.thinkingTokens = reported;
    const rest = output - reported;
    const weights = charTokenWeights(chars);
    weights.thinking = 0;
    if (rest > 0) {
      // Nothing visible was streamed, so the remainder is tool-call arguments.
      if (weights.text === 0 && weights.toolArgs === 0) stats.toolArgsTokens += rest;
      else distribute(stats, rest, weights, "text");
    }
    return { stats, estimated: false };
  }

  if (chars.thinking === 0) {
    // Nothing thinking-like was streamed, so thinking really is zero tokens.
    const weights = charTokenWeights(chars);
    distribute(stats, output, weights, "text");
    return { stats, estimated: false };
  }

  // No reasoning count to trust, but the model clearly thought: size every phase from its
  // streamed characters and rescale to the reported `output`. Without this the thinking
  // bucket keeps its measured wall time while holding zero tokens, which makes the
  // thinking tok/s meaningless.
  distribute(stats, output, charTokenWeights(chars), "thinking");
  return { stats, estimated: true };
}

function distribute(
  stats: PhaseStats,
  amount: number,
  weights: Record<Phase, number>,
  fallback: Phase
): void {
  if (amount <= 0) return;
  const total = weights.thinking + weights.text + weights.toolArgs;
  if (total <= 0) {
    stats[PHASE_TOKENS_KEY[fallback]] += amount;
    return;
  }
  let assigned = 0;
  for (const phase of PHASES) {
    const share = Math.round((amount * weights[phase]) / total);
    stats[PHASE_TOKENS_KEY[phase]] += share;
    assigned += share;
  }
  // Push rounding drift onto the biggest phase.
  let biggest = fallback;
  for (const phase of PHASES) {
    if (weights[phase] > weights[biggest]) biggest = phase;
  }
  stats[PHASE_TOKENS_KEY[biggest]] += amount - assigned;
}

/** Attribute a message's whole generation window across phases by estimated token share. */
function addMessageWindowByChars(stats: PhaseStats, ms: number, chars: PhaseChars): void {
  const weights = charTokenWeights(chars);
  const total = weights.thinking + weights.text + weights.toolArgs;
  for (const phase of PHASES) {
    const share = total > 0 ? (ms * weights[phase]) / total : phase === "text" ? ms : 0;
    stats[PHASE_MS_KEY[phase]] += share;
  }
}

/** Character counts of an assistant message's content blocks, for resumed sessions. */
function charsFromContent(content: unknown): PhaseChars {
  const chars = createPhaseChars();
  if (!Array.isArray(content)) return chars;
  for (const block of content as Array<Record<string, unknown>>) {
    if (block.type === "thinking" && typeof block.thinking === "string") {
      chars.thinking += block.thinking.length;
    } else if (block.type === "text" && typeof block.text === "string") {
      chars.text += block.text.length;
    } else if (block.type === "toolCall") {
      chars.toolArgs += JSON.stringify(block.arguments ?? {}).length;
    }
  }
  return chars;
}

export default function (pi: ExtensionAPI) {
  let agentStartTime: number | null = null;
  let lastDurationMs: number | null = null;
  let lastRunTokens: TokenCounts = createTokenCounts();
  let lastRunGen: PhaseStats = createPhaseStats();
  let lastRunToolMs = 0;
  let compactionCount = 0;
  let requestCount = 0;

  // State for the request currently being processed.
  let runTokens: TokenCounts = createTokenCounts();
  let runGen: PhaseStats = createPhaseStats();
  let runChars: PhaseChars = createPhaseChars();
  let runToolMs = 0;

  // Session-wide totals.
  let sessionTokens: TokenCounts = createTokenCounts();
  let sessionGen: PhaseStats = createPhaseStats();
  let sessionToolMs = 0;
  let sessionActiveMs = 0;
  let lastStatusUpdate = 0;

  // Cost: prices come from pricing.json unless the provider already reported a cost.
  let pricingConfig: PricingConfig = loadPricing(process.cwd());
  let pricingExplicitPath: string | undefined = undefined;
  let runCost: CostBreakdown = emptyCost();
  let lastRunCost: CostBreakdown | null = null;
  let lastRunOrigin: CostOrigin | null = null;
  let sessionCost: CostBreakdown = emptyCost();
  let sessionOtherCost: CostBreakdown = emptyCost();
  let costByModel = new Map<string, ModelCostTotals>();
  let runCostSources = new Set<CostSource>();
  let lastRunCostSources = new Set<CostSource>();
  let modelsWithoutPricing = new Set<string>();
  let missingPricingHintShown = false;
  let runRecords: UsageRecord[] = [];
  let lastRunRecords: UsageRecord[] = [];

  // Whether thinking tokens had to be estimated because the engine reports no reasoning count.
  let runReasoningEstimated = false;
  let lastRunReasoningEstimated = false;
  let sessionReasoningEstimated = false;

  // In-flight assistant message (streaming).
  let msgPhase: Phase | "idle" = "idle";
  let msgPhaseStart = 0;
  let msgStartMs = 0;
  let msgChars: PhaseChars = createPhaseChars();
  let msgDur: PhaseStats = createPhaseStats();

  // In-flight tool execution window (covers parallel calls as one window).
  const activeToolCalls = new Set<string>();
  let toolWindowStart = 0;

  function toolMsLive(now: number): number {
    return runToolMs + (activeToolCalls.size > 0 && toolWindowStart > 0 ? now - toolWindowStart : 0);
  }

  function resetCostState() {
    runCost = emptyCost();
    lastRunCost = null;
    lastRunOrigin = null;
    runCostSources = new Set();
    lastRunCostSources = new Set();
    sessionCost = emptyCost();
    sessionOtherCost = emptyCost();
    costByModel = new Map();
    modelsWithoutPricing = new Set();
    missingPricingHintShown = false;
    runRecords = [];
    lastRunRecords = [];
  }

  function modelLabel(provider?: string, model?: string): string {
    if (!model) return "unknown model";
    return provider ? `${provider}/${model}` : model;
  }

  /** Price one usage record and fold it into the per-model totals. */
  function accountUsage(record: UsageRecord, bucket: "request" | "other"): CostOrigin {
    const label = modelLabel(record.provider, record.model);
    const priced = priceUsage(pricingConfig, record.provider, record.model, record.usage);
    const target = bucket === "request" ? sessionCost : sessionOtherCost;
    addCost(target, priced.cost);

    let totals = costByModel.get(label);
    if (!totals) {
      totals = { calls: 0, tokens: 0, cost: emptyCost(), origin: priced, sources: new Set() };
      costByModel.set(label, totals);
    }
    totals.calls += 1;
    totals.tokens +=
      record.usage.totalTokens ??
      record.usage.input + record.usage.output + record.usage.cacheRead + record.usage.cacheWrite;
    addCost(totals.cost, priced.cost);
    totals.sources.add(priced.source);
    if (priced.source !== "unknown" && totals.origin.source === "unknown") totals.origin = priced;
    if (priced.source === "unknown" && record.model) modelsWithoutPricing.add(label);
    return priced;
  }

  /** Sum the cost of a set of usage records without touching session totals. */
  function sumRecords(records: UsageRecord[]): {
    cost: CostBreakdown;
    origin: CostOrigin | null;
    sources: Set<CostSource>;
  } {
    const cost = emptyCost();
    const sources = new Set<CostSource>();
    let origin: CostOrigin | null = null;
    for (const record of records) {
      const priced = priceUsage(pricingConfig, record.provider, record.model, record.usage);
      addCost(cost, priced.cost);
      sources.add(priced.source);
      if (priced.source !== "unknown") origin = priced;
    }
    return { cost, origin, sources };
  }

  /** Rebuild every cost figure from history; used after /cost mode or reload. */
  function recomputeCosts(ctx: ExtensionContext) {
    sessionCost = emptyCost();
    sessionOtherCost = emptyCost();
    costByModel = new Map();
    modelsWithoutPricing = new Set();
    missingPricingHintShown = false;
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "usage" && entry.usage) {
        accountUsage({ provider: entry.provider, model: entry.model, usage: entry.usage as EstimateUsage }, "other");
      } else if (entry.type === "compaction" && entry.usage) {
        accountUsage({ model: entry.model, usage: entry.usage as EstimateUsage }, "other");
      } else if (entry.type === "message" && entry.message?.role === "assistant") {
        const message = entry.message as { provider?: string; model?: string; usage?: EstimateUsage };
        if (message.usage) {
          accountUsage({ provider: message.provider, model: message.model, usage: message.usage }, "request");
        }
      }
    }
    if (lastRunRecords.length > 0) {
      const summed = sumRecords(lastRunRecords);
      lastRunCost = summed.cost;
      lastRunOrigin = summed.origin;
      lastRunCostSources = summed.sources;
    }
  }

  /** Suffix naming the price list a cost estimate came from. */
  function originSuffix(origin: CostOrigin | null, sources?: Set<CostSource>): string {
    const known = sources ? [...sources].filter((source) => source !== "unknown" && source !== "off") : [];
    if (known.length > 1) {
      return ` (${known.join(" + ")} costs combined)`;
    }
    if (!origin) return "";
    if (origin.source === "reported") return " (reported by provider)";
    if (origin.source === "estimated" && origin.pricing) {
      const note = origin.fallback ? `${origin.pricing.key} @ ${origin.modeName}, fallback prices` : origin.pricing.key + " @ " + origin.modeName;
      return ` (${note}${origin.longContext ? ", long context" : ""})`;
    }
    return "";
  }

  /** Fold the elapsed time of the current streaming phase into its bucket. */
  function closePhase(now: number) {
    if (msgPhase === "idle") return;
    msgDur[PHASE_MS_KEY[msgPhase]] += Math.max(0, now - msgPhaseStart);
    msgPhase = "idle";
  }

  // Reconstruct compaction count, token totals, timing and cost from session history on startup
  pi.on("session_start", async (_event, ctx) => {
    pricingExplicitPath = (pi.getFlag("pi-stats-pricing") as string | undefined) || undefined;
    pricingConfig = loadPricing(ctx.cwd, pricingExplicitPath);
    // --pi-stats-no-cost wins over the price files, and stays off for the whole session
    // unless /cost on re-enables it (a later /cost reload would apply the files again).
    if (pi.getFlag("pi-stats-no-cost") === true) {
      pricingConfig.costEnabled = false;
      pricingConfig.costEnabledSource = "--pi-stats-no-cost";
    }
    resetCostState();
    compactionCount = 0;
    requestCount = 0;
    agentStartTime = null;
    lastDurationMs = null;
    lastRunTokens = createTokenCounts();
    lastRunGen = createPhaseStats();
    lastRunToolMs = 0;
    runTokens = createTokenCounts();
    runGen = createPhaseStats();
    runChars = createPhaseChars();
    runToolMs = 0;
    runReasoningEstimated = false;
    lastRunReasoningEstimated = false;
    sessionReasoningEstimated = false;
    sessionTokens = createTokenCounts();
    sessionGen = createPhaseStats();
    sessionToolMs = 0;
    sessionActiveMs = 0;
    resetStreamingState();
    activeToolCalls.clear();
    toolWindowStart = 0;

    let lastAssistantEnd: number | null = null;
    let toolBatchEnd: number | null = null;
    // Approximate per-request spans: a user prompt starts one, the next prompt ends it.
    let runStart: number | null = null;
    let runEnd: number | null = null;
    // Usage of the last run in the history, so its cost survives into /stats.
    let historyRunRecords: UsageRecord[] = [];

    const flushToolBatch = () => {
      if (lastAssistantEnd !== null && toolBatchEnd !== null && toolBatchEnd > lastAssistantEnd) {
        sessionToolMs += toolBatchEnd - lastAssistantEnd;
      }
      toolBatchEnd = null;
    };

    const flushRun = () => {
      if (runStart !== null && runEnd !== null && runEnd > runStart) {
        sessionActiveMs += runEnd - runStart;
      }
      runStart = null;
      runEnd = null;
    };

    for (const entry of ctx.sessionManager.getBranch()) {
      const entryMs = Date.parse(entry.timestamp);

      if (entry.type === "compaction") {
        compactionCount++;
        if (entry.usage) {
          accountUsage({ model: entry.model, usage: entry.usage as EstimateUsage }, "other");
        }
        flushToolBatch();
        lastAssistantEnd = null;
        if (runStart === null) runStart = entryMs;
        runEnd = Math.max(runEnd ?? entryMs, entryMs);
        continue;
      }

      if (entry.type === "usage" && entry.usage) {
        accountUsage({ provider: entry.provider, model: entry.model, usage: entry.usage as EstimateUsage }, "other");
        continue;
      }

      if (entry.type !== "message") continue;
      const message = entry.message;

      if (message.role === "assistant") {
        flushToolBatch();
        if (runStart === null) runStart = entryMs;
        runEnd = Math.max(runEnd ?? entryMs, entryMs);
        const msgStart = message.timestamp ?? entryMs;
        const genMs = Math.max(0, entryMs - msgStart);
        const chars = charsFromContent(message.content);
        const split = splitOutputTokens(message.usage, chars);
        const stats = split.stats;
        if (split.estimated) sessionReasoningEstimated = true;
        // History has no per-phase timestamps; approximate by splitting on streamed characters.
        addMessageWindowByChars(stats, genMs, chars);
        addPhaseStats(sessionGen, stats);
        addTokens(sessionTokens, usageToCounts(message.usage));
        if (message.usage) {
          const record: UsageRecord = {
            provider: (message as { provider?: string }).provider,
            model: (message as { model?: string }).model,
            usage: message.usage as EstimateUsage,
          };
          accountUsage(record, "request");
          historyRunRecords.push(record);
        }
        lastAssistantEnd = entryMs;
      } else if (message.role === "toolResult") {
        const endedAt = (message as { timestamp?: number }).timestamp ?? entryMs;
        // A batch of parallel tool results is one execution window: take the last one.
        toolBatchEnd = toolBatchEnd === null ? endedAt : Math.max(toolBatchEnd, endedAt);
        runEnd = Math.max(runEnd ?? entryMs, entryMs);
      } else {
        flushToolBatch();
        flushRun();
        if (message.role === "user") {
          requestCount++;
          runStart = entryMs; // the prompt time marks the start of that request
          historyRunRecords = [];
        }
        lastAssistantEnd = null;
      }
    }
    flushToolBatch();
    flushRun();
    // The reconstructed history's final run becomes "last request" for the cost readout.
    lastRunRecords = historyRunRecords;
    if (lastRunRecords.length > 0) {
      const summed = sumRecords(lastRunRecords);
      lastRunCost = summed.cost;
      lastRunOrigin = summed.origin;
      lastRunCostSources = summed.sources;
    }

    // A typo in a price file would otherwise only show up as silently missing costs.
    if (pricingConfig.errors.length > 0) {
      ctx.ui.notify(
        [`pi-stats pricing: ${pricingConfig.errors.length} problem${pricingConfig.errors.length !== 1 ? "s" : ""} in the price files.`].concat(
          pricingConfig.errors.slice(0, 4).map((error) => `⚠️ ${error}`),
          [pricingConfig.errors.length > 4 ? "⚠️ … more with /cost" : "", "Fix the file and run /cost reload."].filter(Boolean)
        ).join("\n"),
        "warning"
      );
    }

    updateStatus(ctx);
  });

  function resetStreamingState() {
    msgPhase = "idle";
    msgPhaseStart = 0;
    msgStartMs = 0;
    msgChars = createPhaseChars();
    msgDur = createPhaseStats();
  }

  // Track when the agent begins processing a request
  pi.on("agent_start", async (_event, ctx) => {
    agentStartTime = Date.now();
    requestCount++;
    runTokens = createTokenCounts();
    runGen = createPhaseStats();
    runChars = createPhaseChars();
    runToolMs = 0;
    runReasoningEstimated = false;
    runCost = emptyCost();
    runRecords = [];
    lastStatusUpdate = 0;
    resetStreamingState();
    activeToolCalls.clear();
    toolWindowStart = 0;
    updateStatus(ctx);
  });

  // Time the generation of each assistant message, phase by phase
  pi.on("message_start", async (event, ctx) => {
    if (event.message.role !== "assistant") return;
    const now = Date.now();
    msgPhase = "idle";
    msgPhaseStart = now;
    msgStartMs = now;
    msgChars = createPhaseChars();
    updateStatus(ctx);
  });

  pi.on("message_update", async (event, ctx) => {
    if (agentStartTime === null || event.message.role !== "assistant") return;

    const now = Date.now();
    const se = event.assistantMessageEvent;

    /** Start (or keep) the given generation phase, closing the previous one. */
    const enter = (phase: Phase) => {
      if (msgPhase === phase) return;
      closePhase(now);
      msgPhase = phase;
      msgPhaseStart = now;
    };

    switch (se?.type) {
      case "thinking_start":
        enter("thinking");
        break;
      case "thinking_delta":
        if (typeof se.delta === "string") msgChars.thinking += se.delta.length;
        enter("thinking");
        break;
      case "text_start":
        enter("text");
        break;
      case "text_delta":
        if (typeof se.delta === "string") msgChars.text += se.delta.length;
        enter("text");
        break;
      case "toolcall_start":
        enter("toolArgs");
        break;
      case "toolcall_delta":
        if (typeof se.delta === "string") msgChars.toolArgs += se.delta.length;
        enter("toolArgs");
        break;
      case "thinking_end":
      case "text_end":
      case "toolcall_end":
        closePhase(now);
        break;
      default:
        break; // start / done / error: no phase to attribute
    }

    // Throttled refresh of the live throughput readout.
    if (now - lastStatusUpdate < 1000) return;
    lastStatusUpdate = now;
    updateStatus(ctx);
  });

  // Accumulate token usage and generation time as assistant messages complete
  pi.on("message_end", async (event, ctx) => {
    if (agentStartTime === null) return;
    if (event.message.role !== "assistant") return;

    const now = Date.now();
    closePhase(now);

    const usage = event.message.usage as UsageLike | undefined;
    const counts = usageToCounts(usage);
    const split = splitOutputTokens(usage, msgChars);
    const stats = split.stats;
    if (split.estimated) {
      runReasoningEstimated = true;
      sessionReasoningEstimated = true;
    }
    // Fold this message's measured per-phase durations in.
    addPhaseStats(stats, msgDur);
    // Providers that deliver the whole message at once emit no phase timings:
    // attribute the message window by character share instead.
    if (msgDur.thinkingMs + msgDur.textMs + msgDur.toolArgsMs === 0 && msgStartMs > 0) {
      addMessageWindowByChars(stats, Math.max(0, now - msgStartMs), msgChars);
    }
    for (const phase of PHASES) {
      runChars[phase] += msgChars[phase];
    }

    addTokens(runTokens, counts);
    addTokens(sessionTokens, counts);
    addPhaseStats(runGen, stats);
    addPhaseStats(sessionGen, stats);

    if (usage) {
      const record: UsageRecord = {
        provider: (event.message as { provider?: string }).provider,
        model: (event.message as { model?: string }).model,
        usage: usage as EstimateUsage,
      };
      const priced = accountUsage(record, "request");
      addCost(runCost, priced.cost);
      runCostSources.add(priced.source);
      runRecords.push(record);
      lastRunOrigin = priced;
    }
    resetStreamingState();
    updateStatus(ctx);
  });

  // Measure tool execution wall time so it can be excluded from throughput
  pi.on("tool_execution_start", async (event, ctx) => {
    // Nested calls run inside the parent tool's window; counting them would double-count.
    if (event.parentToolCallId) return;
    if (activeToolCalls.size === 0) toolWindowStart = Date.now();
    activeToolCalls.add(event.toolCallId);
    if (agentStartTime !== null) updateStatus(ctx);
  });

  // Tools stream output while they run, which keeps the "tools" readout ticking live
  pi.on("tool_execution_update", async (event, ctx) => {
    if (event.parentToolCallId || agentStartTime === null) return;
    const now = Date.now();
    if (now - lastStatusUpdate < 1000) return;
    lastStatusUpdate = now;
    updateStatus(ctx);
  });

  pi.on("tool_execution_end", async (event, ctx) => {
    if (event.parentToolCallId) return;
    if (!activeToolCalls.delete(event.toolCallId)) return;
    if (activeToolCalls.size === 0 && toolWindowStart > 0) {
      runToolMs += Math.max(0, Date.now() - toolWindowStart);
      toolWindowStart = 0;
    }
    if (agentStartTime !== null) updateStatus(ctx);
  });

  // Track when the agent has fully settled (includes retries, compactions, etc.)
  pi.on("agent_settled", async (_event, ctx) => {
    const now = Date.now();
    closePhase(now);
    // Fold any leftover partial-message timings in before finalizing the request.
    addPhaseStats(runGen, msgDur);
    resetStreamingState();
    if (activeToolCalls.size > 0 && toolWindowStart > 0) {
      runToolMs += Math.max(0, now - toolWindowStart);
      toolWindowStart = 0;
      activeToolCalls.clear();
    }

    if (agentStartTime !== null) {
      lastDurationMs = now - agentStartTime;
      agentStartTime = null;
      sessionActiveMs += lastDurationMs;
      sessionToolMs += runToolMs;
      lastRunTokens = runTokens;
      lastRunGen = runGen;
      lastRunToolMs = runToolMs;
      lastRunCost = runCost;
      lastRunRecords = runRecords;
      lastRunCostSources = runCostSources;
      lastRunReasoningEstimated = runReasoningEstimated;
      runTokens = createTokenCounts();
      runGen = createPhaseStats();
      runChars = createPhaseChars();
      runToolMs = 0;
      runCost = emptyCost();
      runRecords = [];

      const genMs = genMsOf(lastRunGen);
      const lines = [
        `Request completed in ${formatDuration(lastDurationMs)} ` +
          `(${requestCount} request${requestCount !== 1 ? "s" : ""}, ` +
          `${compactionCount} compaction${compactionCount !== 1 ? "s" : ""}) ` +
          `— generation ${formatDuration(genMs)}, tools ${formatDuration(lastRunToolMs)} excluded`,
      ];

      if (lastRunTokens.output > 0) {
        lines.push(
          `🧠 thinking ${formatNumber(lastRunGen.thinkingTokens)} tok @ ${formatRate(lastRunGen.thinkingTokens, lastRunGen.thinkingMs)} tok/s (${formatDuration(lastRunGen.thinkingMs)})`,
          `💬 answer ${formatNumber(lastRunGen.textTokens)} tok @ ${formatRate(lastRunGen.textTokens, lastRunGen.textMs)} tok/s (${formatDuration(lastRunGen.textMs)})`,
          `⚙️ tool-call args ${formatNumber(lastRunGen.toolArgsTokens)} tok @ ${formatRate(lastRunGen.toolArgsTokens, lastRunGen.toolArgsMs)} tok/s (${formatDuration(lastRunGen.toolArgsMs)})`
        );
        const outTok =
          lastRunGen.thinkingTokens + lastRunGen.textTokens + lastRunGen.toolArgsTokens;
        lines.push(
          `⚡ generation total ${formatNumber(outTok || lastRunTokens.output)} tok @ ${formatRate(outTok || lastRunTokens.output, genMs)} tok/s`
        );
      }
      lines.push(...reasoningEstimateNote(lastRunReasoningEstimated));

      const otherMs = lastDurationMs - genMs - lastRunToolMs;
      if (otherMs > 1000) {
        lines.push(`⏳ other (model latency, retries, compaction): ${formatDuration(otherMs)}`);
      }
      lines.push(...lastRunCostLines());

      ctx.ui.notify(lines.join("\n"), "info");
    }

    updateStatus(ctx);
  });

  // Cost of the summary generated by a compaction
  pi.on("session_compact", async (event, ctx) => {
    const entry = event.compactionEntry as { model?: string; usage?: EstimateUsage };
    if (!entry.usage) return;
    const record: UsageRecord = { model: entry.model, usage: entry.usage };
    const priced = accountUsage(record, "other");
    if (agentStartTime !== null) {
      addCost(runCost, priced.cost);
      runCostSources.add(priced.source);
      runRecords.push(record);
      lastRunOrigin = priced;
    }
    updateStatus(ctx);
  });

  // Count compactions as they happen
  pi.on("session_before_compact", async (_event, ctx) => {
    compactionCount++;
    const durationStr = lastDurationMs !== null ? formatDuration(lastDurationMs) : "N/A";
    ctx.ui.setStatus(
      "pi-stat",
      ctx.ui.theme.fg("dim", ` 📦 compaction #${compactionCount}  ⏱ ${durationStr}`)
    );
  });

  // Register /stats command to show stats on demand
  pi.registerCommand("stats", {
    description: "Show session timing, token throughput and compaction stats",
    handler: async (_args, ctx) => {
      const lines: string[] = [
        `Requests: ${requestCount}`,
        `Compactions: ${compactionCount}`,
      ];

      if (lastDurationMs !== null) {
        lines.push("");
        lines.push(
          `Last request: ${formatDuration(lastDurationMs)} = ${formatDuration(genMsOf(lastRunGen))} generating + ${formatDuration(lastRunToolMs)} tools (excluded from tok/s)`
        );
        lines.push(...phaseLines(lastRunGen));
        lines.push(...reasoningEstimateNote(lastRunReasoningEstimated));
      }

      if (agentStartTime !== null) {
        const elapsed = Date.now() - agentStartTime;
        const live = livePhaseStats();
        const liveGenMs = genMsOf(live.stats) || elapsed - live.toolMs;
        lines.push("");
        lines.push(
          `Current request: ${formatDuration(elapsed)} (running...) = ${formatDuration(liveGenMs)} generating + ${formatDuration(live.toolMs)} tools`
        );
        lines.push(...phaseLines(live.stats, " (est.)"));
        lines.push(...reasoningEstimateNote(runReasoningEstimated));
      }

      lines.push("");
      lines.push(
        `Session active time: ${formatDuration(sessionActiveMs)} = ${formatDuration(genMsOf(sessionGen))} generating + ${formatDuration(sessionToolMs)} tools`
      );
      lines.push(...phaseLines(sessionGen, " (session)"));
      lines.push(...reasoningEstimateNote(sessionReasoningEstimated));
      lines.push(
        `Session tokens: ${formatNumber(sessionTokens.total)} total (${formatNumber(sessionTokens.input)} in / ${formatNumber(sessionTokens.output)} out, cache ${formatNumber(sessionTokens.cacheRead)} read / ${formatNumber(sessionTokens.cacheWrite)} write)`
      );
      lines.push("");
      if (lastRunCost && pricingConfig.costEnabled) {
        lines.push(`Last request cost: ${lastRunCostLine()}`);
      }
      lines.push(...sessionCostLines());

      ctx.ui.notify(lines.join("\n"), "info");
    },
  });

  // Register /cost command: money view of the session, plus price-list control
  pi.registerCommand("cost", {
    description: "Show estimated cost per model, and switch or reload the pricing configuration",
    handler: async (args, ctx) => {
      const parts = (args ?? "").trim().split(/\s+/).filter(Boolean);
      const sub = (parts[0] ?? "").toLowerCase();

      if (sub === "help" || sub === "?") {
        ctx.ui.notify(costHelp().join("\n"), "info");
        return;
      }

      // Master switch for cost estimation: on, off, toggle, or a look at the current state.
      if (sub === "on" || sub === "enable" || sub === "off" || sub === "disable" || sub === "toggle" || sub === "status") {
        const want = sub === "on" || sub === "enable" ? true : sub === "off" || sub === "disable" ? false : undefined;
        pricingConfig.costEnabled = want === undefined ? !pricingConfig.costEnabled : want;
        if (pricingConfig.costEnabled) pricingConfig.costEnabledSource = "/cost on";
        else pricingConfig.costEnabledSource = "/cost off";
        // Re-price the whole session so the totals match the new setting immediately.
        recomputeCosts(ctx);
        const lines = [
          pricingConfig.costEnabled
            ? "Cost estimation is on — figures come from the price files below, or from the provider when it reports them."
            : "Cost estimation is off — no cost is calculated or shown (turn back on: /cost on)",
        ];
        lines.push(...sessionCostLines());
        lines.push(...pricingConfigLines());
        ctx.ui.notify(lines.join("\n"), "info");
        updateStatus(ctx);
        return;
      }

      if (sub === "modes") {
        const pricing = relevantPricing(ctx);
        if (!pricing) {
          ctx.ui.notify(
            "The current model has no prices configured. Use /cost rates to list what is available.",
            "warning"
          );
          return;
        }
        const resolved = resolveMode(pricing.pricing, pricingConfig.mode);
        ctx.ui.notify(
          [
            `${pricing.pricing.name} (${pricing.pricing.key}): modes ${pricing.pricing.modeNames.join(", ")}`,
            `currently using "${resolved?.modeName ?? "none"}"${pricing.fallback ? " (fallback prices for this model)" : ""}`,
            "switch with: /cost mode <name>",
          ].join("\n"),
          "info"
        );
        return;
      }

      if (sub === "mode") {
        const name = normalize(parts[1] ?? "");
        if (!name) {
          ctx.ui.notify("Usage: /cost mode <name> — list names with /cost modes", "warning");
          return;
        }
        const pricing = relevantPricing(ctx);
        if (pricing && !pricing.pricing.modes[name]) {
          ctx.ui.notify(
            `"${name}" is not a mode of ${pricing.pricing.key}. Available: ${pricing.pricing.modeNames.join(", ")}`,
            "warning"
          );
          return;
        }
        pricingConfig.mode = name;
        recomputeCosts(ctx);
        const lines = [`Estimating with mode "${name}" now.`];
        if (lastRunCost) {
          lines.push(`Last request cost: ${lastRunCostLine()}`);
        }
        lines.push(...sessionCostLines());
        ctx.ui.notify(lines.join("\n"), "info");
        updateStatus(ctx);
        return;
      }

      if (sub === "rates") {
        const asked = parts[1] ? normalize(parts[1]) : undefined;
        const all = [...pricingConfig.models.values()];
        const targets = asked
          ? all.filter((pricing) => {
              const key = normalize(pricing.key);
              return key === asked || key.includes(asked);
            })
          : all;
        if (targets.length === 0) {
          ctx.ui.notify(
            `No configured model matches "${asked}". Priced models: ${[...new Set(all.map((pricing) => pricing.key))].join(", ") || "none"}`,
            "warning"
          );
          return;
        }
        const lines: string[] = [];
        for (const pricing of targets) {
          // List only the active mode unless a specific model was asked for.
          lines.push(...rateLines(pricingConfig, pricing, asked ? undefined : pricingConfig.mode));
        }
        if (pricingConfig.fallbackModel) {
          lines.push(`(models without their own entry use "${pricingConfig.fallbackLabel}" prices)`);
        }
        ctx.ui.notify(lines.join("\n"), "info");
        return;
      }

      if (sub === "reload") {
        pricingExplicitPath = parts[1] ?? pricingExplicitPath;
        pricingConfig = loadPricing(ctx.cwd, pricingExplicitPath);
        recomputeCosts(ctx);
        ctx.ui.notify([...pricingConfigLines(), ...sessionCostLines()].join("\n"), "info");
        updateStatus(ctx);
        return;
      }

      const lines = [...sessionCostLines(), "", ...pricingConfigLines()];
      const match = relevantPricing(ctx);
      if (match) {
        lines.push("", ...rateLines(pricingConfig, match.pricing));
        if (match.fallback) {
          lines.push(`note: ${match.pricing.key} prices are a fallback for this model; add its own entry or aliases for exact figures.`);
        }
      } else if (pricingConfig.costEnabled) {
        lines.push("", "The current model has no prices configured, so its cost shows as $0.");
      }
      lines.push("", ...costHelp());
      ctx.ui.notify(lines.join("\n"), "info");
    },
  });

  // Show the prices of the current model at a glance
  pi.registerFlag("pi-stats-pricing", {
    description: "Path to a pi-stats pricing JSON file (overrides pricing.json and PI_STATS_PRICING)",
    type: "string",
  });

  // Start a session without any money readout
  pi.registerFlag("pi-stats-no-cost", {
    description: "Disable pi-stats cost estimation for this run (same as /cost off, or PI_STATS_COST=off)",
    type: "boolean",
  });

  function genMsOf(stats: PhaseStats): number {
    return stats.thinkingMs + stats.textMs + stats.toolArgsMs;
  }

  /** "$X — breakdown (source of the rates)" for the request that just finished. */
  function lastRunCostLine(): string {
    return `${formatCost(lastRunCost.total)} — ${costDetail(lastRunCost)}${originSuffix(lastRunOrigin, lastRunCostSources)}`;
  }

  /** One line pricing the request that just finished. */
  function lastRunCostLines(): string[] {
    if (!lastRunCost || !pricingConfig.costEnabled) return [];
    if (lastRunCost.total > 0) {
      return [`💵 cost ${lastRunCostLine()}`];
    }
    if (modelsWithoutPricing.size > 0 && !missingPricingHintShown) {
      // Only nag once per session: an unpriced model repeats this on every request.
      missingPricingHintShown = true;
      return [
        `💵 no cost estimate: add prices for ${[...modelsWithoutPricing].join(", ")} to pi-stats.pricing.json (see /cost rates)`,
      ];
    }
    return [];
  }

  /** One line explaining that thinking tokens (and their tok/s) are estimates. */
  function reasoningEstimateNote(estimated: boolean): string[] {
    if (!estimated) return [];
    return [
      "ℹ️ the engine reports no reasoning token count: thinking tokens, and therefore thinking tok/s, are estimated from streamed characters",
    ];
  }

  /** Session cost split by model, plus where the prices came from. */
  function sessionCostLines(): string[] {
    const lines: string[] = [];
    if (!pricingConfig.costEnabled) {
      // Deliberately no "add prices" hint: nothing is missing, the readout is switched off.
      lines.push("Session cost: estimation disabled — enable with /cost on (or \"cost\": true in pi-stats.pricing.json)");
      return lines;
    }
    const total = sessionCost.total + sessionOtherCost.total;
    lines.push(
      `Session cost: ${formatCost(total)} = ${formatCost(sessionCost.total)} requests + ${formatCost(sessionOtherCost.total)} compaction/summary usage`
    );
    for (const [label, totals] of costByModel) {
      lines.push(
        `  ${label}: ${totals.calls} model call${totals.calls !== 1 ? "s" : ""}, ${formatNumber(totals.tokens)} tok → ${formatCost(totals.cost.total)}${originSuffix(totals.origin, totals.sources)}`
      );
    }
    if (modelsWithoutPricing.size > 0) {
      lines.push(
        `  no prices configured for: ${[...modelsWithoutPricing].join(", ")} (add them to pi-stats.pricing.json)`
      );
    }
    return lines;
  }

  function pricingConfigLines(): string[] {
    const lines = [
      `Prices: ${pricingConfig.sources.length > 0 ? pricingConfig.sources.join("  <  ") : "no pricing.json found"}`,
      `mode=${pricingConfig.mode}  currency=${pricingConfig.currency}  per ${formatNumber(pricingConfig.perTokens)} tokens  multiplier=${pricingConfig.multiplier}  preferReportedCost=${pricingConfig.preferReportedCost}`,
      `cost estimation: ${pricingConfig.costEnabled ? "on" : "OFF"} (set by ${pricingConfig.costEnabledSource})`,
      `models priced: ${pricingConfig.models.size > 0 ? [...new Set([...pricingConfig.models.values()].map((pricing) => pricing.key))].join(", ") : "none"}`,
      `fallback for unpriced models: ${pricingConfig.fallbackLabel || "none"}`,
    ];
    for (const error of pricingConfig.errors) lines.push(`⚠️ ${error}`);
    return lines;
  }

  /** Model of the last request, or the one currently selected. */
  function relevantPricing(ctx: ExtensionContext): ModelMatch | undefined {
    const last = [...costByModel.keys()].pop();
    if (last) {
      const slash = last.indexOf("/");
      const provider = slash > 0 ? last.slice(0, slash) : undefined;
      const model = slash > 0 ? last.slice(slash + 1) : last;
      const found = findModelPricing(pricingConfig, provider, model);
      if (found) return found;
    }
    const model = ctx.model as { id?: string; provider?: string } | undefined;
    return findModelPricing(pricingConfig, model?.provider, model?.id ?? undefined);
  }

  function costHelp(): string[] {
    return [
      "/cost                 session cost by model, and the pricing files in use",
      "/cost modes           processing modes available for the current model",
      "/cost mode <name>     estimate with another price list (standard, batch, flex, fast, ultrafast)",
      "/cost rates [model]   resolved prices for every model, or just one",
      "/cost reload [path]   re-read pricing.json (optionally from another path) and recompute",
      "/cost off | on        stop / resume cost estimation (flip with /cost toggle, status with /cost status)",
      "                      off also works in the files (\"cost\": false) and the environment (PI_STATS_COST=off)",
    ];
  }

  /** Live phase stats for the in-flight request, with the current message estimated in. */
  function livePhaseStats(): { stats: PhaseStats; toolMs: number } {
    const now = Date.now();
    const stats = createPhaseStats();
    addPhaseStats(stats, runGen);

    const openMs = msgPhase !== "idle" ? now - msgPhaseStart : 0;
    addPhaseStats(stats, msgDur);
    if (msgPhase !== "idle") stats[PHASE_MS_KEY[msgPhase]] += openMs;

    // Fold the partially streamed message in using the tokens-per-character rate that the
    // same phase already showed in this request; that keeps each rate's numerator and
    // denominator about the same phase. Before anything completed, fall back to the
    // per-phase characters-per-token priors.
    const runCharTotal = runChars.thinking + runChars.text + runChars.toolArgs;
    const runTokenTotal = runGen.thinkingTokens + runGen.textTokens + runGen.toolArgsTokens;
    for (const phase of PHASES) {
      const phaseTokens = runGen[PHASE_TOKENS_KEY[phase]];
      const phaseChars = runChars[phase];
      const tokensPerChar =
        phaseChars > 0 && phaseTokens > 0
          ? phaseTokens / phaseChars
          : runCharTotal > 0
            ? (runTokenTotal || runTokens.output) / runCharTotal
            : 1 / PHASE_CHARS_PER_TOKEN[phase];
      stats[PHASE_TOKENS_KEY[phase]] += msgChars[phase] * tokensPerChar;
    }

    return { stats, toolMs: toolMsLive(now) };
  }

  function phaseLines(stats: PhaseStats, suffix = ""): string[] {
    const total = stats.thinkingTokens + stats.textTokens + stats.toolArgsTokens;
    if (total <= 0) return [`No generated tokens yet${suffix}`];
    const genMs = genMsOf(stats);
    return [
      `  🧠 thinking: ${formatNumber(stats.thinkingTokens)} tok in ${formatDuration(stats.thinkingMs)} → ${formatRate(stats.thinkingTokens, stats.thinkingMs)} tok/s`,
      `  💬 answer:   ${formatNumber(stats.textTokens)} tok in ${formatDuration(stats.textMs)} → ${formatRate(stats.textTokens, stats.textMs)} tok/s`,
      `  ⚙️ tool args: ${formatNumber(stats.toolArgsTokens)} tok in ${formatDuration(stats.toolArgsMs)} → ${formatRate(stats.toolArgsTokens, stats.toolArgsMs)} tok/s`,
      `  ⚡ generating${suffix}: ${formatNumber(total)} tok in ${formatDuration(genMs)} → ${formatRate(total, genMs)} tok/s`,
    ];
  }

  function statusText(): string {
    const compactionStr = compactionCount === 0 ? "" : `  📦 ${compactionCount}`;
    const stats = agentStartTime !== null ? livePhaseStats() : null;
    const shown = stats ? stats.stats : lastRunGen;
    const toolMs = stats ? stats.toolMs : lastRunToolMs;
    // Money is cumulative, so the readout always shows the session total (matching pi's footer);
    // per-request figures are in the completion notification and /stats.
    const costTotal = sessionCost.total + sessionOtherCost.total;

    const elapsed =
      agentStartTime !== null
        ? Date.now() - agentStartTime
        : lastDurationMs !== null
          ? lastDurationMs
          : null;

    const timeStr = elapsed !== null ? formatDuration(elapsed) : "—";
    const thinkingStr =
      shown.thinkingTokens > 0
        ? `  🧠 ${formatRate(shown.thinkingTokens, shown.thinkingMs)}`
        : "  🧠 —";
    const textStr =
      shown.textTokens > 0 ? `  💬 ${formatRate(shown.textTokens, shown.textMs)}` : "  💬 —";
    const toolStr = toolMs > 0 ? `  🔧 ${formatDuration(toolMs)}` : "";
    const costStr = costTotal > 0 ? `  💵 ${formatCostShort(costTotal)}` : "";

    return ` ⏱ ${timeStr}${thinkingStr}${textStr} tk/s${toolStr}${costStr}${compactionStr}`;
  }

  function updateStatus(ctx: ExtensionContext) {
    ctx.ui.setStatus("pi-stat", ctx.ui.theme.fg("dim", statusText()));
  }
}

function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }

  const seconds = ms / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = (seconds % 60).toFixed(1);
  return `${minutes}m ${remainingSeconds}s`;
}

function formatRate(tokens: number, ms: number): string {
  if (tokens <= 0) return "0";
  if (ms <= 0) return "—";
  const rate = tokens / (ms / 1000);
  return rate >= 100 ? String(Math.round(rate)) : rate.toFixed(1);
}

function formatNumber(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}
