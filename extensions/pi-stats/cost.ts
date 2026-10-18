/**
 * Cost estimation for pi-stats.
 *
 * Prices come from a JSON configuration file. They use the same unit and shape as
 * pi's own model catalog (`cost.input` / `output` / `cacheRead` / `cacheWrite` per
 * one million tokens, with optional long-context tiers), so the numbers can be
 * copied straight from https://developers.openai.com/api/docs/pricing or from a pi
 * `models.json` entry.
 *
 * Resolution order (later files win, model by model):
 *   1. pricing.json next to this file (bundled defaults)
 *   2. ~/.pi/agent/pi-stats.pricing.json (user override)
 *   3. <cwd>/.pi/pi-stats.pricing.json   (project override)
 *   4. $PI_STATS_PRICING, or the path given to /cost reload <path>
 *
 * Estimation itself can be switched off — `"cost": false` in any of those files,
 * `PI_STATS_COST=off` / `PI_STATS_NO_COST=1` in the environment, or `/cost off` at
 * runtime. With it off every usage record prices to zero, so the status bar drops the
 * money readout and no "add prices for this model" hint is shown.
 */

import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export interface Rates {
  /** Uncached input tokens. */
  input: number;
  /** Cache hits (pi's `usage.cacheRead`). */
  cachedInput: number;
  /** Cache creations (pi's `usage.cacheWrite`). */
  cacheWrite: number;
  /** Generated tokens, reasoning included (pi's `usage.output`). */
  output: number;
}

export interface CostMode {
  base: Rates;
  /** Higher rates that apply once a request's input tokens exceed `aboveTokens`. */
  tiers: Array<{ aboveTokens: number; rates: Rates }>;
}

export interface ModelPricing {
  key: string;
  name: string;
  modes: Record<string, CostMode>;
  modeNames: string[];
}
export interface CostBreakdown {
  input: number;
  cachedInput: number;
  cacheWrite: number;
  output: number;
  total: number;
}

export interface EstimateUsage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  cacheWrite1h?: number;
  cost?: Partial<CostBreakdown>;
}

/** Where a cost figure came from. `off` means estimation was disabled by the user. */
export type CostSource = "reported" | "estimated" | "unknown" | "off";

export interface ModelMatch {
  pricing: ModelPricing;
  /** True when the model has no entry of its own and `fallbackModel` prices were used. */
  fallback: boolean;
}

export interface PricingConfig {
  currency: string;
  /** Number of tokens a price applies to. Default 1,000,000. */
  perTokens: number;
  /** Selected processing mode: standard, batch, flex, fast, ultrafast, ... */
  mode: string;
  /** Use the provider/pi-reported cost when it is greater than zero. */
  preferReportedCost: boolean;
  /** Multiplier applied to every estimated component (e.g. 1.1 regional uplift). */
  multiplier: number;
  /** Model key used to price models that have no entry of their own (empty disables it). */
  fallbackModel: string;
  /** How to print that key (as written in the config). */
  fallbackLabel: string;
  /** Master switch for cost estimation; off means every figure is $0. */
  costEnabled: boolean;
  /** What last set `costEnabled` (file path, env var name, or "default" / command name). */
  costEnabledSource: string;
  models: Map<string, ModelPricing>;
  /** Files that were read, in precedence order. */
  sources: string[];
  /** Parse problems worth telling the user about. */
  errors: string[];
}

export const EMPTY_COST: CostBreakdown = {
  input: 0,
  cachedInput: 0,
  cacheWrite: 0,
  output: 0,
  total: 0,
};

export function emptyCost(): CostBreakdown {
  return { ...EMPTY_COST };
}

export function addCost(target: CostBreakdown, source: CostBreakdown): void {
  target.input += source.input;
  target.cachedInput += source.cachedInput;
  target.cacheWrite += source.cacheWrite;
  target.output += source.output;
  target.total += source.total;
}

/** Normalize names so `cacheWrite`, `cache-write`, `cache_write` and `Cache Writes` all match. */
export function normalize(value: string): string {
  return value
    .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

const RATE_ALIASES: Record<keyof Rates, string[]> = {
  input: ["input", "input-tokens", "input-price", "uncached-input", "prompt", "prompt-tokens", "prompt-price"],
  cachedInput: [
    "cached-input",
    "cache-read",
    "cache-read-input",
    "input-cached",
    "cached",
    "cached-tokens",
    "cache",
    "cache-hit",
    "cache-read-tokens",
  ],
  cacheWrite: [
    "cache-write",
    "cache-write-input",
    "cache-writes",
    "cache-creation",
    "cache-writing",
    "input-cache-write",
    "cache-write-tokens",
  ],
  output: ["output", "output-tokens", "output-price", "completion", "completion-tokens", "generated", "answer"],
};

const THRESHOLD_KEYS = [
  "above-tokens",
  "input-tokens-above",
  "above",
  "over",
  "threshold",
  "context-tokens-above",
  "over-input-tokens",
];

const META_KEYS = new Set([
  "name",
  "provider",
  "providers",
  "aliases",
  "matches",
  "note",
  "notes",
  "comment",
  "source",
  "retrieved",
  "documentation",
  "url",
  "docs",
  "example",
]);

/** Keys that switch cost estimation on or off, in any pricing file (normalized, so camelCase spelling works too). */
const COST_FLAG_KEYS = ["cost", "costs", "cost-enabled", "cost-estimates", "cost-estimation", "estimate-cost", "estimation"];

const TRUTHY = new Set(["1", "true", "yes", "on", "enable", "enabled", "y"]);
const FALSY = new Set(["0", "false", "no", "off", "disable", "disabled", "n"]);

/**
 * Read a cost on/off flag from a pricing file. Returns undefined when the file says
 * nothing about it, and reports a problem for a value that cannot be read as a flag.
 */
function readCostFlag(data: Record<string, unknown>, path: string, errors: string[]): boolean | undefined {
  for (const key of COST_FLAG_KEYS) {
    const found = Object.keys(data).find((candidate) => normalize(candidate) === key);
    if (found === undefined) continue;
    const raw = data[found];
    if (typeof raw === "boolean") return raw;
    if (typeof raw === "string") {
      const text = normalize(raw);
      if (TRUTHY.has(text)) return true;
      if (FALSY.has(text)) return false;
    }
    errors.push(`${path}: "${found}" must be true/false (or "on"/"off"), got ${JSON.stringify(raw)}`);
    return undefined;
  }
  return undefined;
}

/** Same for an environment variable; undefined when unset or empty. */
function readEnvFlag(name: string, errors: string[]): boolean | undefined {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return undefined;
  const text = normalize(raw);
  if (TRUTHY.has(text)) return true;
  if (FALSY.has(text)) return false;
  errors.push(`${name}: expected on/off (1/0, true/false), got "${raw}"`);
  return undefined;
}

function toNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value.replace(/[$,\s]/g, ""));
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

/** Flattened, normalized view of an object's numeric fields. */
function numericFields(value: unknown): Map<string, number> {
  const fields = new Map<string, number>();
  if (!value || typeof value !== "object" || Array.isArray(value)) return fields;
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    if (key.startsWith("_")) continue;
    const num = toNumber(raw);
    if (num !== undefined) fields.set(normalize(key), num);
  }
  return fields;
}

function pickRate(fields: Map<string, number>, aliases: string[]): number {
  for (const alias of aliases) {
    const found = fields.get(alias);
    if (found !== undefined) return found;
  }
  return 0;
}

/** Read one rate set; returns null when the object holds none of the four prices. */
function parseRates(value: unknown): Rates | null {
  const fields = numericFields(value);
  if (fields.size === 0) return null;
  const hasAny = (Object.keys(RATE_ALIASES) as Array<keyof Rates>).some((field) =>
    RATE_ALIASES[field].some((alias) => fields.has(alias))
  );
  if (!hasAny) return null;
  return {
    input: pickRate(fields, RATE_ALIASES.input),
    cachedInput: pickRate(fields, RATE_ALIASES.cachedInput),
    cacheWrite: pickRate(fields, RATE_ALIASES.cacheWrite),
    output: pickRate(fields, RATE_ALIASES.output),
  };
}

/** Extract tiers from a `long` / `longContext` object or a pi-style `tiers` array. */
function parseTiers(entry: Record<string, unknown>, modelKey: string, errors: string[]): CostMode["tiers"] {
  const tiers: CostMode["tiers"] = [];

  const oneTier = (raw: unknown, label: string): void => {
    const fields = numericFields(raw);
    const rates = parseRates(raw);
    if (!rates) return;
    const above = thresholdOf(fields);
    if (above === undefined) {
      errors.push(`${modelKey}: "${label}" has no aboveTokens threshold — its rates are ignored`);
      return;
    }
    tiers.push({ aboveTokens: above, rates });
  };

  for (const [key, raw] of Object.entries(entry)) {
    if (key.startsWith("_")) continue;
    const name = normalize(key);
    if (name === "long" || name === "long-context" || name === "long-context-rates") {
      oneTier(raw, key);
    } else if (name === "tiers" && Array.isArray(raw)) {
      for (const tier of raw) oneTier(tier, "tiers entry (needs " + THRESHOLD_KEYS[0] + ")");
    }
  }
  return tiers;
}

function thresholdOf(fields: Map<string, number>): number | undefined {
  for (const alias of THRESHOLD_KEYS) {
    const found = fields.get(alias);
    if (found !== undefined) return found;
  }
  return undefined;
}

function addMode(modes: Record<string, CostMode>, name: string, value: unknown, modelKey: string, errors: string[]) {
  const rates = parseRates(value);
  if (!rates) return;
  const tiers = parseTiers(value as Record<string, unknown>, `${modelKey}/${name}`, errors);
  const key = normalize(name) || "standard";
  modes[key] = { base: rates, tiers };
}

/** Accept several spellings of a model entry and normalize them to modes + tiers. */
function parseModel(modelKey: string, entry: unknown, errors: string[], allowEmpty = false): ModelPricing | null {
  if (!entry || typeof entry !== "object") return null;
  const raw = entry as Record<string, unknown>;
  const modes: Record<string, CostMode> = {};
  const name = typeof raw.name === "string" && raw.name.trim() ? raw.name.trim() : modelKey;

  // Grouped form: { modes: { standard: {...}, fast: {...} } } (also "rates"/"prices").
  for (const [key, value] of Object.entries(raw)) {
    if (key.startsWith("_")) continue;
    const name2 = normalize(key);
    if ((name2 === "modes" || name2 === "rates" || name2 === "prices") && value && typeof value === "object") {
      for (const [modeName, modeValue] of Object.entries(value as Record<string, unknown>)) {
        addMode(modes, modeName, modeValue, modelKey, errors);
      }
    }
  }

  // Flat form: { standard: {...}, batch: {...} }, or pi's catalog form { cost: {...}, tiers: [...] }.
  if (Object.keys(modes).length === 0) {
    for (const [key, value] of Object.entries(raw)) {
      if (key.startsWith("_")) continue;
      const name2 = normalize(key);
      if (META_KEYS.has(name2) || name2 === "tiers" || name2 === "per-tokens" || name2 === "unit") continue;
      if (name2 === "cost") {
        addMode(modes, "standard", value, modelKey, errors);
        const tiers = parseTiers({ tiers: (value as Record<string, unknown>)?.tiers } as Record<string, unknown>, modelKey, errors);
        if (modes.standard && tiers.length) modes.standard.tiers = tiers;
        continue;
      }
      if (value && typeof value === "object" && !Array.isArray(value)) addMode(modes, key, value, modelKey, errors);
    }
  }

  // Simplest form: rates listed directly on the entry, optionally with a `long` block.
  if (Object.keys(modes).length === 0) {
    addMode(modes, "standard", raw, modelKey, errors);
  }

  const modeNames = Object.keys(modes);
  if (modeNames.length === 0) {
    if (!allowEmpty) {
      errors.push(`${modelKey}: no prices found — expected input / cachedInput / cacheWrite / output per 1M tokens`);
    }
    return null;
  }
  return { key: modelKey, name, modes, modeNames };
}

/** Extra model ids that share an entry's prices ("aliases" or "matches"). */
function aliasListOf(entry: unknown): string[] {
  const raw = (entry as Record<string, unknown>)?.aliases ?? (entry as Record<string, unknown>)?.matches;
  if (!Array.isArray(raw)) return [];
  return raw.filter((alias) => typeof alias === "string" && (alias as string).trim() !== "").map((alias) => normalize(alias as string));
}

function extensionDir(): string {
  try {
    return dirname(fileURLToPath(import.meta.url));
  } catch {
    return process.cwd();
  }
}

interface RawFile {
  path: string;
  data: Record<string, unknown>;
}

function readJson(path: string, errors: string[]): RawFile | null {
  let absolute: string;
  try {
    absolute = resolve(path);
  } catch {
    return null;
  }
  if (!existsSync(absolute)) return null;
  try {
    const data = JSON.parse(readFileSync(absolute, "utf8"));
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      errors.push(`${absolute}: expected a JSON object`);
      return null;
    }
    return { path: absolute, data: data as Record<string, unknown> };
  } catch (error) {
    errors.push(`${absolute}: ${(error as Error).message}`);
    return null;
  }
}

/**
 * Load and merge pricing files. `explicitPath` (from $PI_STATS_PRICING or a
 * command) has the highest precedence.
 */
export function loadPricing(cwd: string, explicitPath?: string): PricingConfig {
  const errors: string[] = [];
  const files: RawFile[] = [];
  const candidates = [
    join(extensionDir(), "pricing.json"),
    join(homedir(), ".pi", "agent", "pi-stats.pricing.json"),
    join(cwd, ".pi", "pi-stats.pricing.json"),
  ];
  const extra = explicitPath || process.env.PI_STATS_PRICING;
  if (extra) candidates.push(extra);

  for (const candidate of candidates) {
    const file = readJson(candidate, errors);
    if (file) files.push(file);
  }

  const config: PricingConfig = {
    currency: "USD",
    perTokens: 1_000_000,
    mode: "standard",
    preferReportedCost: true,
    multiplier: 1,
    fallbackModel: "",
    fallbackLabel: "",
    costEnabled: true,
    costEnabledSource: "default",
    models: new Map(),
    sources: files.map((file) => file.path),
    errors,
  };

  // Alias entries resolve after all files are parsed, so they may point at a model
  // defined in another file.
  const pendingAliases: Array<{ alias: string; target: string }> = [];

  for (const file of files) {
    const data = file.data;
    const currency = typeof data.currency === "string" ? data.currency.trim() : undefined;
    if (currency) config.currency = currency;
    const perTokens = toNumber(data.perTokens ?? data.per);
    if (perTokens && perTokens > 0) config.perTokens = perTokens;
    const mode = typeof data.mode === "string" ? normalize(data.mode) : undefined;
    if (mode) config.mode = mode;
    if (typeof data.preferReportedCost === "boolean") config.preferReportedCost = data.preferReportedCost;
    const multiplier = toNumber(data.multiplier);
    if (multiplier !== undefined && multiplier > 0) config.multiplier = multiplier;
    if (typeof data.fallbackModel === "string") {
      const fallback = data.fallbackModel.trim();
      config.fallbackModel = fallback ? normalize(fallback) : "";
      config.fallbackLabel = fallback ? (config.models.get(config.fallbackModel)?.key ?? fallback) : "";
    }
    // Later files win, so a project file can turn money readouts off for one directory.
    const costFlag = readCostFlag(data, file.path, errors);
    if (costFlag !== undefined) {
      config.costEnabled = costFlag;
      config.costEnabledSource = file.path;
    }

    const hasWrapperKey =
      typeof data.currency !== "undefined" ||
      typeof data.mode !== "undefined" ||
      typeof data.preferReportedCost !== "undefined" ||
      typeof data.fallbackModel !== "undefined" ||
      costFlag !== undefined;
    const models = data.models ?? (hasWrapperKey ? undefined : data);
    if (!models || typeof models !== "object" || Array.isArray(models)) continue;
    for (const [key, entry] of Object.entries(models as Record<string, unknown>)) {
      if (key.startsWith("_") || COST_FLAG_KEYS.includes(normalize(key))) continue;
      const target = normalize(key);

      // "my-local-model": "gpt-6.1-sol" — price this id with another entry's rates.
      if (typeof entry === "string") {
        pendingAliases.push({ alias: target, target: normalize(entry) });
        continue;
      }

      const pricing = parseModel(key, entry, errors, aliasListOf(entry).length > 0);
      if (pricing) config.models.set(target, pricing);

      const aliases = aliasListOf(entry);
      if (aliases.length === 0) continue;
      if (pricing) {
        // Priced entry that also lists extra ids using the same rates.
        for (const alias of aliases) config.models.set(alias, pricing);
      } else {
        // Alias-only entry: reuse the rates registered under this key earlier.
        for (const alias of aliases) pendingAliases.push({ alias, target });
      }
    }
  }

  for (const { alias, target } of pendingAliases) {
    const found = config.models.get(target);
    if (found) {
      config.models.set(alias, found);
    } else if (!config.models.has(alias)) {
      errors.push(`"${alias}": no prices under "${target}" — define that model first, or give rates directly`);
    }
  }

  const envMode = process.env.PI_STATS_PRICING_MODE;
  if (envMode) config.mode = normalize(envMode);
  // The environment beats the files, so one shell setting overrides a whole project tree.
  const envCost = readEnvFlag("PI_STATS_COST", errors);
  if (envCost !== undefined) {
    config.costEnabled = envCost;
    config.costEnabledSource = "PI_STATS_COST";
  }
  if (readEnvFlag("PI_STATS_NO_COST", errors) === true) {
    config.costEnabled = false;
    config.costEnabledSource = "PI_STATS_NO_COST";
  }
  // The fallback entry may live in a file read after the one that named it.
  if (config.fallbackModel) {
    config.fallbackLabel = config.models.get(config.fallbackModel)?.key ?? config.fallbackLabel;
  }
  if (explicitPath && !files.some((file) => resolve(file.path) === resolve(explicitPath))) {
    errors.push(`${explicitPath}: not found or not valid JSON`);
  }
  return config;
}

/** Find pricing for a provider/model pair, tolerating quantization and tag suffixes. */
export function findModelPricing(
  config: PricingConfig,
  provider: string | undefined,
  model: string | undefined
): ModelMatch | undefined {
  if (!model) {
    return config.fallbackModel && config.models.has(config.fallbackModel)
      ? { pricing: config.models.get(config.fallbackModel)!, fallback: true }
      : undefined;
  }
  const candidates = [provider ? `${provider}/${model}` : "", model].filter(Boolean).map(normalize);

  for (const candidate of candidates) {
    const exact = config.models.get(candidate);
    if (exact) return { pricing: exact, fallback: false };
  }

  // Longest configured key that the model id starts with (or vice versa), so
  // "gpt-6.1-sol:q4_K_M", "gpt-6.1-sol-2026-04-01" and "openai/gpt-6.1-sol" all match.
  let best: ModelPricing | undefined;
  let bestLength = 0;
  for (const candidate of candidates) {
    for (const [key, pricing] of config.models) {
      const longer = candidate.startsWith(key) ? candidate : key.startsWith(candidate) ? key : undefined;
      const shorter = longer === candidate ? key : candidate;
      if (!longer || shorter.length < 4) continue;
      if (longer.length === shorter.length || longer[shorter.length] === "-") {
        if (shorter.length > bestLength) {
          best = pricing;
          bestLength = shorter.length;
        }
      }
    }
  }
  if (best) return { pricing: best, fallback: false };

  const fallback = config.fallbackModel ? config.models.get(config.fallbackModel) : undefined;
  return fallback ? { pricing: fallback, fallback: true } : undefined;
}

/** Pick the mode entry to use, falling back to standard, then to the first mode. */
export function resolveMode(pricing: ModelPricing, mode: string): { modeName: string; modeEntry: CostMode } | undefined {
  const names = pricing.modeNames;
  const chosen = pricing.modes[mode] ? mode : pricing.modes.standard ? "standard" : names[0];
  const entry = pricing.modes[chosen];
  return entry ? { modeName: chosen, modeEntry: entry } : undefined;
}

function chooseRates(modeEntry: CostMode, inputTokens: number): { rates: Rates; long: boolean } {
  let rates = modeEntry.base;
  let matched = -1;
  let long = false;
  for (const tier of modeEntry.tiers) {
    if (inputTokens > tier.aboveTokens && tier.aboveTokens > matched) {
      rates = tier.rates;
      matched = tier.aboveTokens;
      long = true;
    }
  }
  return { rates, long };
}

/** Estimate one usage record from the configured prices. */
export function estimateCost(
  config: PricingConfig,
  pricing: ModelPricing,
  usage: EstimateUsage,
  mode?: string
): { cost: CostBreakdown; modeName: string; longContext: boolean } {
  const resolved = resolveMode(pricing, mode ?? config.mode);
  const cost = emptyCost();
  if (!resolved) return { cost, modeName: config.mode, longContext: false };

  const selected = chooseRates(resolved.modeEntry, usage.input + usage.cacheRead + usage.cacheWrite);
  const scale = (config.multiplier * 1) / config.perTokens;
  // Pi bills one-hour cache writes at 2x base input (Anthropic's rate); mirror that.
  const longWrite = Math.min(usage.cacheWrite1h ?? 0, usage.cacheWrite);
  const shortWrite = usage.cacheWrite - longWrite;

  cost.input = selected.rates.input * usage.input * scale;
  cost.cachedInput = selected.rates.cachedInput * usage.cacheRead * scale;
  cost.cacheWrite =
    (selected.rates.cacheWrite * shortWrite + selected.rates.input * 2 * longWrite) * scale;
  cost.output = selected.rates.output * usage.output * scale;
  cost.total = cost.input + cost.cachedInput + cost.cacheWrite + cost.output;
  return { cost, modeName: resolved.modeName, longContext: selected.long };
}

export function costFromReported(usage: EstimateUsage): CostBreakdown {
  const reported = usage.cost;
  if (!reported) return emptyCost();
  const cost: CostBreakdown = {
    input: reported.input ?? 0,
    cachedInput: reported.cacheRead ?? 0,
    cacheWrite: reported.cacheWrite ?? 0,
    output: reported.output ?? 0,
    total: reported.total ?? (reported.input ?? 0) + (reported.cacheRead ?? 0) + (reported.cacheWrite ?? 0) + (reported.output ?? 0),
  };
  return cost;
}

/**
 * Cost of one message: the provider's own figure when it has one (hosted models
 * already price through pi's catalog), otherwise the configured estimate.
 */
export function priceUsage(
  config: PricingConfig,
  provider: string | undefined,
  model: string | undefined,
  usage: EstimateUsage
): {
  cost: CostBreakdown;
  source: CostSource;
  pricing?: ModelPricing;
  modeName?: string;
  longContext?: boolean;
  fallback?: boolean;
} {
  // Estimation switched off: no figure, and deliberately not "unknown", so callers do
  // not start asking the user to add prices for models that are simply not being priced.
  if (!config.costEnabled) return { cost: emptyCost(), source: "off" };
  if (config.preferReportedCost && (usage.cost?.total ?? 0) > 0) {
    return { cost: costFromReported(usage), source: "reported" };
  }
  const match = findModelPricing(config, provider, model);
  if (!match) return { cost: emptyCost(), source: "unknown" };
  const estimate = estimateCost(config, match.pricing, usage);
  return {
    cost: estimate.cost,
    source: "estimated",
    pricing: match.pricing,
    modeName: estimate.modeName,
    longContext: estimate.longContext,
    fallback: match.fallback,
  };
}

/** "$0.0042" for small amounts, "$1.234" mid-range, "$123.45" large. */
export function formatCost(value: number): string {
  const abs = Math.abs(value);
  if (abs === 0) return "$0";
  if (abs >= 100) return `$${value.toFixed(2)}`;
  if (abs >= 1) return `$${value.toFixed(3)}`;
  return `$${value.toFixed(4)}`;
}

/** Compact form for the status bar: "$0.042", "$1.23", "$123". */
export function formatCostShort(value: number): string {
  const abs = Math.abs(value);
  if (abs === 0) return "$0";
  if (abs >= 100) return `$${value.toFixed(0)}`;
  if (abs >= 1) return `$${value.toFixed(2)}`;
  if (abs >= 0.01) return `$${value.toFixed(3)}`;
  return `$${value.toFixed(4)}`;
}

export function costDetail(cost: CostBreakdown): string {
  return (
    `${formatCost(cost.input)} in / ${formatCost(cost.cachedInput)} cached / ` +
    `${formatCost(cost.cacheWrite)} cache write / ${formatCost(cost.output)} out`
  );
}

/** Resolved prices for one model, for `/cost rates`. Pass `onlyMode` to print one mode. */
export function rateLines(config: PricingConfig, pricing: ModelPricing, onlyMode?: string): string[] {
  const lines: string[] = [];
  const names = onlyMode && pricing.modes[onlyMode] ? [onlyMode] : pricing.modeNames;
  for (const modeName of names) {
    const entry = pricing.modes[modeName];
    const mark = modeName === config.mode ? "*" : " ";
    lines.push(
      `${mark} ${modeName}: ${pricing.name} — ${formatRate(entry.base, config.perTokens)}` +
        (config.multiplier !== 1 ? ` (x${config.multiplier})` : "")
    );
    for (const tier of entry.tiers) {
      lines.push(`    over ${tier.aboveTokens.toLocaleString("en-US")} input tok: ${formatRate(tier.rates, config.perTokens)}`);
    }
  }
  return lines;
}

function formatRate(rates: Rates, perTokens: number): string {
  const suffix = perTokens === 1_000_000 ? "/M tok" : `/${perTokens.toLocaleString("en-US")} tok`;
  return (
    `in $${rates.input} cached $${rates.cachedInput} ` +
    `cache write $${rates.cacheWrite} out $${rates.output} ${suffix}`
  );
}
