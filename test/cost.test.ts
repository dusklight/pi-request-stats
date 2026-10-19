/**
 * Unit tests for the pricing/cost logic in extensions/pi-request-stats/cost.ts — the part of the
 * extension where a silent mistake turns into a wrong money figure, and which needs no
 * pi runtime to exercise.
 *
 * Run with `npm test` (node --test). Uses only node built-ins.
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  addCost,
  costDetail,
  costFromReported,
  emptyCost,
  estimateCost,
  findModelPricing,
  formatCost,
  formatCostShort,
  loadPricing,
  normalize,
  priceUsage,
  rateLines,
  resolveMode,
  type EstimateUsage,
  type PricingConfig,
} from "../extensions/pi-request-stats/cost.ts";

// ---------------------------------------------------------------- test fixtures

const workDir = mkdtempSync(join(tmpdir(), "pi-request-stats-"));

/** loadPricing() reads $HOME/.pi/agent/... — point HOME at an empty dir so tests are hermetic. */
const savedEnv: Record<string, string | undefined> = {};
const envKeys = ["HOME", "PI_REQUEST_STATS_PRICING", "PI_REQUEST_STATS_PRICING_MODE", "PI_REQUEST_STATS_COST", "PI_REQUEST_STATS_NO_COST"];

before(() => {
  for (const key of envKeys) savedEnv[key] = process.env[key];
  process.env.HOME = join(workDir, "empty-home");
  mkdirSync(process.env.HOME, { recursive: true });
  for (const key of envKeys.slice(1)) delete process.env[key];
});

after(() => {
  for (const key of envKeys) {
    if (savedEnv[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnv[key];
  }
  rmSync(workDir, { recursive: true, force: true });
});

/** Write a pricing file into the scratch dir and load it as the highest-precedence source. */
function configFrom(name: string, data: unknown): PricingConfig {
  const path = join(workDir, name);
  writeFileSync(path, JSON.stringify(data));
  return loadPricing(workDir, path);
}

function usage(over: Partial<EstimateUsage> = {}): EstimateUsage {
  return { input: 1000, output: 800, cacheRead: 5000, cacheWrite: 2000, ...over };
}

const ROUND = 1e-12;
function close(actual: number, expected: number, message?: string): void {
  assert.ok(Math.abs(actual - expected) < ROUND, `${message ?? "value"}: ${actual} != ${expected}`);
}

// ------------------------------------------------------------------- normalize

describe("normalize", () => {
  test("folds camelCase, spaces and punctuation to one spelling", () => {
    assert.equal(normalize("cachedInput"), "cached-input");
    assert.equal(normalize("Cache Writes"), "cache-writes");
    assert.equal(normalize("cache_write_input"), "cache-write-input");
    assert.equal(normalize("  GPT-6.1 Sol  "), "gpt-6-1-sol");
  });
});

// ---------------------------------------------------------------- loadPricing

describe("loadPricing", () => {
  test("the bundled price file loads cleanly and carries the documented models and modes", () => {
    const config = configFrom("bundled-check.json", { models: {} });
    assert.deepEqual(config.errors, [], `bundled pricing errors: ${config.errors.join("; ")}`);
    assert.equal(config.currency, "USD");
    assert.equal(config.perTokens, 1_000_000);
    assert.equal(config.mode, "standard");
    assert.equal(config.preferReportedCost, true);
    for (const key of ["gpt-6-1-sol", "gpt-6-astra", "gpt-6-luna"]) {
      assert.ok(config.models.has(key), `bundled file is missing "${key}"`);
    }
    const sol = config.models.get("gpt-6-1-sol")!;
    assert.deepEqual(
      [...sol.modeNames].sort(),
      ["batch", "fast", "flex", "standard", "ultrafast"],
      "gpt-6.1-sol modes"
    );
    assert.equal(sol.modes.standard.base.input, 2);
    assert.equal(sol.modes.standard.base.cachedInput, 0.1);
    assert.equal(sol.modes.standard.base.cacheWrite, 2.5);
    assert.equal(sol.modes.standard.base.output, 10);
    assert.equal(sol.modes.standard.tiers[0].aboveTokens, 272_000);
    assert.equal(sol.modes.standard.tiers[0].rates.input, 4);
  });

  test("a project file overrides one model and keeps the rest of the bundled list", () => {
    configFrom("base.json", { models: { "test-keep-zzz": { input: 1, output: 2 } } });
    mkdirSync(join(workDir, ".pi"), { recursive: true });
    writeFileSync(
      join(workDir, ".pi", "pi-request-stats.pricing.json"),
      JSON.stringify({ models: { "test-keep-zzz": { input: 7, output: 9 } } })
    );
    const config = loadPricing(workDir);
    const match = findModelPricing(config, undefined, "test-keep-zzz");
    assert.ok(match && !match.fallback);
    assert.equal(match.pricing.modes.standard.base.input, 7);
    assert.equal(match.pricing.modes.standard.base.output, 9);
    assert.ok(config.models.has("gpt-6-1-sol"), "bundled entries survive a project override");
  });

  test("an explicit path that does not exist is reported instead of silently ignored", () => {
    const config = loadPricing(workDir, join(workDir, "nope.json"));
    assert.ok(config.errors.some((error) => error.includes("nope.json")));
  });

  test("top-level settings are read from the files", () => {
    const config = configFrom("settings.json", {
      currency: "EUR",
      perTokens: 1000,
      mode: "Fast",
      multiplier: 1.1,
      preferReportedCost: false,
      fallbackModel: "",
      models: { "test-set-zzz": { input: 1, output: 2 } },
    });
    assert.equal(config.currency, "EUR");
    assert.equal(config.perTokens, 1000);
    assert.equal(config.mode, "fast", "mode is normalized");
    assert.equal(config.multiplier, 1.1);
    assert.equal(config.preferReportedCost, false);
    assert.equal(config.fallbackModel, "");
  });

  test("rate field spellings from pi's catalog and the pricing page all resolve", () => {
    const config = configFrom("spellings.json", {
      models: {
        "test-alias-zzz": {
          "Prompt": 1,
          "Cache Read Input": 0.25,
          "Cache Writes": 1.25,
          "Completion Tokens": 4,
        },
      },
    });
    const base = config.models.get("test-alias-zzz")!.modes.standard.base;
    assert.equal(base.input, 1);
    assert.equal(base.cachedInput, 0.25);
    assert.equal(base.cacheWrite, 1.25);
    assert.equal(base.output, 4);
  });

  test("a flat entry (rates with no mode wrapper) becomes the standard mode", () => {
    const config = configFrom("flat.json", { models: { "test-flat-zzz": { input: 0.8, output: 3.2 } } });
    const pricing = config.models.get("test-flat-zzz")!;
    assert.deepEqual(pricing.modeNames, ["standard"]);
    assert.equal(pricing.modes.standard.base.output, 3.2);
    assert.equal(pricing.name, "test-flat-zzz", "a model without a display name falls back to its key");
  });

  test("long-context tiers arrive from both the pi-style array and the single-object form", () => {
    const config = configFrom("tiers.json", {
      models: {
        "test-tier-a-zzz": {
          input: 2,
          output: 10,
          tiers: [{ aboveTokens: 272000, input: 4, output: 15 }],
        },
        "test-tier-b-zzz": { input: 2, output: 10, long: { inputTokensAbove: 1000, input: 3, output: 6 } },
      },
    });
    assert.deepEqual(
      config.models.get("test-tier-a-zzz")!.modes.standard.tiers.map((tier) => tier.aboveTokens),
      [272000]
    );
    const alt = config.models.get("test-tier-b-zzz")!.modes.standard.tiers;
    assert.deepEqual(alt.map((tier) => tier.aboveTokens), [1000]);
    assert.equal(alt[0].rates.input, 3);
  });

  test("a tier without a threshold is ignored with an error instead of applying to everything", () => {
    const config = configFrom("bad-tier.json", {
      models: { "test-badtier-zzz": { input: 2, output: 10, tiers: [{ input: 4, output: 15 }] } },
    });
    assert.equal(config.models.get("test-badtier-zzz")!.modes.standard.tiers.length, 0);
    assert.ok(config.errors.some((error) => error.includes("no aboveTokens threshold")));
  });
});

// ---------------------------------------------------------------- model lookup

describe("findModelPricing", () => {
  const config = () =>
    configFrom("lookup.json", {
      fallbackModel: "test-base-zzz",
      models: {
        "test-base-zzz": { name: "Base", input: 2, output: 10 },
        "test-alias-zzz": "test-base-zzz",
        "test-arr-zzz": { name: "Arr", input: 1, output: 2, aliases: ["test-arr-alt-zzz"] },
      },
    });

  test("exact ids match without the fallback flag", () => {
    const match = findModelPricing(config(), "openai", "test-base-zzz");
    assert.ok(match && !match.fallback);
    assert.equal(match.pricing.key, "test-base-zzz");
  });

  test("quantization and version suffixes still resolve to the entry", () => {
    const withConfig = config();
    for (const model of ["test-base-zzz:Q4_K_M", "test-base-zzz-2026-04-01", "test-base-zzz"]) {
      const match = findModelPricing(withConfig, undefined, model);
      assert.ok(match, `no match for ${model}`);
      assert.equal(match.pricing.key, "test-base-zzz", model);
      assert.equal(match.fallback, false, `${model} matched its own entry, not the fallback`);
    }
  });

  test('an id mapped with "id": "other-id" shares that entry\'s rates', () => {
    const withConfig = config();
    const alias = findModelPricing(withConfig, undefined, "test-alias-zzz");
    const base = findModelPricing(withConfig, undefined, "test-base-zzz");
    assert.ok(alias && base);
    assert.equal(alias.pricing.name, base.pricing.name);
    assert.equal(alias.fallback, false);
  });

  test("an entry listing aliases is reachable under those aliases too", () => {
    const match = findModelPricing(config(), undefined, "test-arr-alt-zzz");
    assert.ok(match && !match.fallback);
    assert.equal(match.pricing.name, "Arr");
  });

  test("an unlisted model borrows fallbackModel rates and is flagged as such", () => {
    const match = findModelPricing(config(), undefined, "totally-unlisted-model-zzz");
    assert.ok(match);
    assert.equal(match.fallback, true);
    assert.equal(match.pricing.key, "test-base-zzz");
  });

  test('an unlisted model with "fallbackModel": "" has no price at all', () => {
    const noFallback = configFrom("nofb.json", {
      fallbackModel: "",
      models: { "test-only-zzz": { input: 1, output: 2 } },
    });
    assert.equal(findModelPricing(noFallback, undefined, "totally-unlisted-model-zzz"), undefined);
  });

  test("no model id at all falls back when one is configured", () => {
    const match = findModelPricing(config(), undefined, undefined);
    assert.ok(match && match.fallback);
    assert.equal(match.pricing.key, "test-base-zzz");
  });
});

// ---------------------------------------------------------------------- modes

describe("resolveMode", () => {
  test("falls back to standard, then to the first mode", () => {
    const config = configFrom("modes.json", {
      models: {
        "test-modes-zzz": { input: 1, output: 2, modes: { standard: { input: 3, output: 4 }, fast: { input: 5, output: 6 } } },
        "test-nostd-zzz": { modes: { batch: { input: 1, output: 2 } } },
      },
    });
    assert.equal(resolveMode(config.models.get("test-modes-zzz")!, "fast")!.modeName, "fast");
    assert.equal(resolveMode(config.models.get("test-modes-zzz")!, "ultrafast")!.modeName, "standard");
    assert.equal(resolveMode(config.models.get("test-nostd-zzz")!, "standard")!.modeName, "batch");
  });

  test("bundled GPT-6 Luna has no ultrafast prices, so that mode resolves to standard", () => {
    const config = configFrom("luna.json", { models: {} });
    const luna = config.models.get("gpt-6-luna")!;
    assert.ok(!luna.modes.ultrafast, "Luna should not carry ultrafast rates");
    assert.equal(resolveMode(luna, "ultrafast")!.modeName, "standard");
  });
});

// --------------------------------------------------------------- estimateCost

describe("estimateCost", () => {
  const priced = () =>
    configFrom("estimate.json", {
      models: {
        "test-price-zzz": {
          input: 2,
          cachedInput: 0.1,
          cacheWrite: 2.5,
          output: 10,
          modes: {
            standard: {
              input: 2,
              cachedInput: 0.1,
              cacheWrite: 2.5,
              output: 10,
              tiers: [{ aboveTokens: 272000, input: 4, cachedInput: 0.2, cacheWrite: 5, output: 15 }],
            },
            fast: { input: 4, cachedInput: 0.2, cacheWrite: 5, output: 20 },
          },
        },
      },
    });

  test("each component is priced separately and the total is their sum", () => {
    const config = priced();
    const match = findModelPricing(config, undefined, "test-price-zzz")!;
    const result = estimateCost(config, match.pricing, usage());
    close(result.cost.input, 0.002);
    close(result.cost.cachedInput, 0.0005);
    close(result.cost.cacheWrite, 0.005);
    close(result.cost.output, 0.008);
    close(result.cost.total, 0.0155);
    assert.equal(result.modeName, "standard");
    assert.equal(result.longContext, false);
  });

  test("the long-context tier applies once input + cache read + cache write pass the threshold", () => {
    const config = priced();
    const match = findModelPricing(config, undefined, "test-price-zzz")!;
    const under = estimateCost(config, match.pricing, usage({ input: 269_000, cacheRead: 2000, cacheWrite: 1000 }));
    const over = estimateCost(config, match.pricing, usage({ input: 300_000, cacheRead: 0, cacheWrite: 0, output: 0 }));
    assert.equal(under.longContext, false);
    assert.equal(over.longContext, true);
    close(over.cost.input, 4 * 300_000 / 1_000_000);
  });

  test("1-hour cache writes are charged at 2x base input, the rest at the cache-write rate", () => {
    const config = priced();
    const match = findModelPricing(config, undefined, "test-price-zzz")!;
    const result = estimateCost(config, match.pricing, usage({ cacheWrite: 1000, cacheWrite1h: 400 }));
    close(result.cost.cacheWrite, (2.5 * 600 + 2 * 2 * 400) / 1_000_000);
  });

  test("a flat entry with a top-level `long` block prices short and long context correctly", () => {
    const config = configFrom("longblock.json", {
      models: { "test-longblk-zzz": { input: 2, cachedInput: 0.1, cacheWrite: 2.5, output: 10, long: { aboveTokens: 272000, input: 4, cachedInput: 0.2, cacheWrite: 5, output: 15 } } },
    });
    const pricing = config.models.get("test-longblk-zzz")!;
    assert.deepEqual(pricing.modeNames, ["standard"], "a `long` block must not become a price mode");
    const short = estimateCost(config, pricing, usage());
    const long = estimateCost(config, pricing, usage({ input: 300_000, cacheRead: 0, cacheWrite: 0, output: 0 }));
    assert.equal(short.longContext, false);
    close(short.cost.input, 2 * 1000 / 1_000_000);
    assert.equal(long.longContext, true);
    close(long.cost.input, 4 * 300_000 / 1_000_000);
  });

  test("a top-level tiers array beside flat rates attaches to the standard mode", () => {
    const config = configFrom("flatxtiers.json", {
      models: { "test-fxt-zzz": { input: 2, output: 10, tiers: [{ aboveTokens: 1000, input: 3, output: 6 }] } },
    });
    const pricing = config.models.get("test-fxt-zzz")!;
    assert.deepEqual(pricing.modeNames, ["standard"]);
    assert.deepEqual(pricing.modes.standard.tiers.map((tier) => tier.aboveTokens), [1000]);
  });

  test("the multiplier scales every component", () => {
    const config = priced();
    config.multiplier = 1.1;
    const match = findModelPricing(config, undefined, "test-price-zzz")!;
    close(estimateCost(config, match.pricing, usage()).cost.total, 0.0155 * 1.1);
  });

  test("an explicit mode argument overrides the configured mode", () => {
    const config = priced();
    const match = findModelPricing(config, undefined, "test-price-zzz")!;
    const result = estimateCost(config, match.pricing, usage(), "fast");
    assert.equal(result.modeName, "fast");
    close(result.cost.input, 4 * 1000 / 1_000_000);
  });

  test("perTokens is respected by non-USD-per-million price files", () => {
    const config = configFrom("per1k.json", { perTokens: 1000, models: { "test-per1k-zzz": { input: 2, output: 10 } } });
    const match = findModelPricing(config, undefined, "test-per1k-zzz")!;
    close(estimateCost(config, match.pricing, usage({ cacheRead: 0, cacheWrite: 0 })).cost.input, 2 * 1000 / 1000);
  });
});

// ------------------------------------------------------ cost source selection

describe("priceUsage", () => {
  test("a provider-reported cost is used as-is and marked as reported", () => {
    const config = configFrom("reported.json", { models: { "test-rep-zzz": { input: 2, output: 10 } } });
    const result = priceUsage(config, undefined, "test-rep-zzz", usage({ cost: { input: 0.01, cacheRead: 0.003, cacheWrite: 0.004, output: 0.02, total: 0.037 } }));
    assert.equal(result.source, "reported");
    close(result.cost.total, 0.037);
    close(result.cost.cachedInput, 0.003, "pi spells the cache-hit figure cacheRead");
    assert.equal(result.pricing, undefined, "reported figures do not borrow local rates");
  });

  test("a reported cost without a total is summed from its components", () => {
    const reported = costFromReported(usage({ cost: { input: 0.01, cacheRead: 0.02, cacheWrite: 0.004, output: 0.02 } }));
    close(reported.total, 0.054);
    assert.equal(costFromReported(usage()).total, 0, "no reported cost means no figure");
  });

  test('preferReportedCost: false derives every figure from the local prices', () => {
    const config = configFrom("prefer.json", {
      preferReportedCost: false,
      models: { "test-pref-zzz": { input: 2, output: 10 } },
    });
    const result = priceUsage(config, undefined, "test-pref-zzz", usage({ cost: { total: 5 } }));
    assert.equal(result.source, "estimated");
    assert.equal(result.fallback, false);
  });

  test("a model with no entry and no fallback reports source unknown and $0", () => {
    const config = configFrom("unknown.json", { fallbackModel: "", models: { "test-unk-zzz": { input: 1, output: 2 } } });
    const result = priceUsage(config, undefined, "totally-unlisted-model-zzz", usage());
    assert.equal(result.source, "unknown");
    assert.equal(result.cost.total, 0);
  });

  test("zero reported cost falls through to estimation (local servers report none)", () => {
    const config = configFrom("zero.json", { models: { "test-zero-zzz": { input: 2, output: 10 } } });
    const result = priceUsage(config, undefined, "test-zero-zzz", usage({ cost: { total: 0 } }));
    assert.equal(result.source, "estimated");
    assert.ok(result.cost.total > 0);
  });
});

// ------------------------------------------------------- the cost on/off switch

describe("cost estimation switch", () => {
  test('"cost": false in a file turns estimation off and says where the setting came from', () => {
    const config = configFrom("off.json", { cost: false, models: { "test-off-zzz": { input: 1, output: 2 } } });
    assert.equal(config.costEnabled, false);
    assert.ok(config.costEnabledSource.includes("off.json"));
    const result = priceUsage(config, undefined, "test-off-zzz", usage());
    assert.equal(result.source, "off");
    assert.equal(result.cost.total, 0);
  });

  test('the flag also accepts "cost": "off" and the costEnabled spelling', () => {
    assert.equal(configFrom("off2.json", { cost: "off", models: {} }).costEnabled, false);
    assert.equal(configFrom("off3.json", { costEnabled: false, models: {} }).costEnabled, false);
    assert.equal(configFrom("on.json", { cost: "on", models: {} }).costEnabled, true);
  });

  test("the environment outranks the files", () => {
    const off = configFrom("envoff.json", { cost: false, models: { "test-env-zzz": { input: 1, output: 2 } } });
    assert.equal(off.costEnabled, false);

    process.env.PI_REQUEST_STATS_COST = "on";
    assert.equal(configFrom("envoff2.json", { cost: false, models: {} }).costEnabled, true);

    process.env.PI_REQUEST_STATS_COST = "off";
    assert.equal(configFrom("envon.json", { cost: true, models: {} }).costEnabled, false);

    delete process.env.PI_REQUEST_STATS_COST;
    process.env.PI_REQUEST_STATS_NO_COST = "1";
    const shorthand = configFrom("envshorthand.json", { cost: true, models: { "test-env-zzz": { input: 1, output: 2 } } });
    assert.equal(shorthand.costEnabled, false);
    assert.equal(shorthand.costEnabledSource, "PI_REQUEST_STATS_NO_COST");
    delete process.env.PI_REQUEST_STATS_NO_COST;
  });

  test('an unparseable flag value is reported as an error, not guessed at', () => {
    const config = configFrom("badflag.json", { cost: "maybe", models: {} });
    assert.ok(config.errors.some((error) => error.includes("cost")));
  });
});

// -------------------------------------------------------------------- reading

describe("money formatting and readouts", () => {
  test("formatCost scales its precision to the amount", () => {
    assert.equal(formatCost(0), "$0");
    assert.equal(formatCost(0.0042), "$0.0042");
    assert.equal(formatCost(1.234), "$1.234");
    assert.equal(formatCost(1.2356), "$1.236");
    assert.equal(formatCost(123.456), "$123.46");
  });

  test("formatCostShort is compact enough for the status bar", () => {
    assert.equal(formatCostShort(0), "$0");
    assert.equal(formatCostShort(0.042), "$0.042");
    assert.equal(formatCostShort(1.234), "$1.23");
    assert.equal(formatCostShort(1234.5), "$1235");
  });

  test("costDetail names every component", () => {
    const cost = { input: 0.086, cachedInput: 0.014, cacheWrite: 0.075, output: 0.019, total: 0.194 };
    assert.equal(costDetail(cost), "$0.0860 in / $0.0140 cached / $0.0750 cache write / $0.0190 out");
  });

  test("addCost accumulates components and totals", () => {
    const target = emptyCost();
    addCost(target, { input: 1, cachedInput: 2, cacheWrite: 3, output: 4, total: 10 });
    addCost(target, { input: 0.5, cachedInput: 0, cacheWrite: 0, output: 0.5, total: 1 });
    assert.equal(target.input, 1.5);
    assert.equal(target.total, 11);
  });

  test("rateLines marks the active mode and shows tier rates", () => {
    const config = configFrom("rates.json", {
      models: {
        "test-rate-zzz": {
          name: "Rate Test",
          modes: {
            standard: { input: 2, cachedInput: 0.1, cacheWrite: 2.5, output: 10, tiers: [{ aboveTokens: 272000, input: 4, cachedInput: 0.2, cacheWrite: 5, output: 15 }] },
          },
        },
      },
    });
    const lines = rateLines(config, config.models.get("test-rate-zzz")!);
    assert.match(lines[0], /^\* standard: Rate Test — in \$2 cached \$0\.1 cache write \$2\.5 out \$10 \/M tok$/);
    assert.match(lines[1], /over 272,000 input tok/);
    assert.deepEqual(rateLines(config, config.models.get("test-rate-zzz")!, "standard"), lines);
  });
});
