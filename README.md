> [!NOTE]
> `INITIAL_REQUIREMENTS.md` was used for the initial prompt and implementation, then the tokens-per-second feature, `README.md`, and `.gitignore` were added afterwards, all locally using `llama.cpp` with `Qwen 3.8 Flash Next Q4` and `pi`.
>
> Subsequent features and updates were implemented using `Strata` with `Qwen 3.8 Flash Next ISTA-DASLab IQ3_S`.

# pi-stats

A [pi](https://pi.dev) extension that reports request timing, token throughput, compaction counts,
and estimated request cost for your coding sessions.

## What it shows

- **Status bar** — live elapsed time with thinking and answering rates while a request runs,
  plus the tool time excluded so far, the estimated cost so far, and the compaction count:
  ```
   ⏱ 12.3s  🧠 42.1  💬 68.3 tk/s  🔧 4.2s  💵 $1.42  📦 2
  ```
- **Request-complete notification** — one line per phase, each with its own rate:
  ```
  Request completed in 12.3s (3 requests, 2 compactions) — generation 8.1s, tools 4.2s excluded
  🧠 thinking 341 tok @ 42.1 tok/s (8.1s)
  💬 answer 556 tok @ 68.3 tok/s (8.1s)
  ⚙️ tool-call args 129 tok @ 55.0 tok/s (2.3s)
  ⚡ generation total 1,026 tok @ 91.0 tok/s
  💵 cost $0.1940 — $0.0860 in / $0.0140 cached / $0.0750 cache write / $0.0190 out (gpt-6.1-sol @ standard)
  ```
- **`/stats` command** — on-demand breakdown:
  - Requests and compactions in the session
  - Last request: wall time split into generating + tools, then the per-phase rates
  - Current request progress and live per-phase rates (while running)
  - Session active time split into generating + tools, and session-wide per-phase throughput
  - Total session tokens (input / output, cache read / write)
  - Cost of the last request and of the session, broken down per model

Counts, throughput, tool time and cost are reconstructed from session history on startup, so
resumed sessions report meaningful numbers immediately. For history, per-phase timing is
approximated by splitting each message's generation window across its content blocks,
since transcripts store message start/end times but not per-block times.

While a request runs, the in-flight message is counted with the tokens-per-character rate the
*same phase* already showed earlier in that request, so a live rate never divides a partial
token count by the whole request's phase time (that mismatch used to display ~5 tok/s for a
thinking phase running at ~85 tok/s).

## Why the timing is split

Wall-clock time from the start to the end of a request includes everything the agent does,
not just the model generating tokens: a `bash` call waiting on a build or test run, a
`read` of a large file, retries, and compaction. Counting all of that against generated
tokens makes the throughput look far slower than the inference engine actually is.

The extension therefore tracks three generation phases and tool time separately:

| Bucket | What it covers |
| --- | --- |
| 🧠 thinking | streamed thinking/reasoning blocks |
| 💬 answer | streamed text (the reply) |
| ⚙️ tool args | streaming of tool-call arguments (still model generation) |
| 🔧 tools | wall time spent executing tools — **excluded** from every tok/s figure |
| ⏳ other | model latency before the first token, retries, compaction |

Parallel tool calls count as one window, and nested tool calls (a tool invoking other tools)
are not counted twice. Thinking tokens come from `usage.reasoning` when the provider reports a
non-zero count. Many OpenAI-compatible engines (`llama.cpp`/`llama-server`, vLLM, and most
proxies) report no reasoning breakdown at all — pi fills that in as `0`, while their `output`
token count already includes thinking — so a reported `0` is read as "not reported" and every
phase is instead sized from the characters streamed for it, rescaled to the reported output
tokens and weighted by per-phase characters-per-token priors measured on real sessions
(thinking ≈ 3.2, answer ≈ 3.5, tool JSON ≈ 3.2). `/stats` and the completion notification say
plainly when thinking tokens — and therefore thinking tok/s — are estimates.

## Estimated cost

Every request is priced from a JSON price list, so costs are available even when the
provider reports none (a local `llama.cpp` server, for example, reports zero). When the
provider *does* report a cost — which pi already computes for hosted models — that figure is
used as-is and the price list only splits it into components; set
`"preferReportedCost": false` to derive every figure from the local prices instead.

Price lists are read in this order, and later files win model by model:

| File | Purpose |
| --- | --- |
| `extensions/pi-stats/pricing.json` | bundled prices (GPT-6.1 Sol, Astra and Luna from the [OpenAI pricing page](https://developers.openai.com/api/docs/pricing)) |
| `~/.pi/agent/pi-stats.pricing.json` | your own models and overrides |
| `<cwd>/.pi/pi-stats.pricing.json` | per-project overrides |
| `$PI_STATS_PRICING`, `--pi-stats-pricing <path>`, `/cost reload <path>` | explicit file |

All rates are per **1,000,000 tokens** in one currency, and use the same field names as pi's
own model catalog, so an entry can be pasted straight from a pi `models.json` file. The
bundled list also carries the price modes from that page — Standard, Batch and Flex (50%),
Fast (2x) and Ultrafast (6x, where the page lists it) — each with its long-context tier (the
`>272K input tokens` rates, applied per request once its input tokens pass the threshold).
The bundled numbers were read from the page on 2026-10-18. A user override looks like this:

```jsonc
// ~/.pi/agent/pi-stats.pricing.json
{
  "mode": "standard",          // which price list to estimate with
  "fallbackModel": "gpt-6.1-sol", // prices for models that have no entry of their own
  "models": {
    "gpt-6.1-sol": {
      "name": "GPT-6.1 Sol (OpenAI)",
      "modes": {
        "standard": {
          "input": 2, "cachedInput": 0.1, "cacheWrite": 2.5, "output": 10,
          "tiers": [{ "aboveTokens": 272000, "input": 4, "cachedInput": 0.2, "cacheWrite": 5, "output": 15 }]
        },
        "fast": { "input": 4, "cachedInput": 0.2, "cacheWrite": 5, "output": 20 }
      }
    },
    // same prices under another id — handy for renamed or locally served models
    "llama-server/qwen3.8-flash-next-iq3_s": "gpt-6.1-sol",
    // flat form: just the rates of one mode
    "gpt-4.1-mini": { "input": 0.8, "cachedInput": 0.2, "cacheWrite": 1, "output": 3.2 }
  }
}
```

Field names are matched loosely (`cachedInput`, `cacheRead`, `cached_input`, `Cache Writes`,
`writeOutputTokens`, `reasoningOutputTokens`, … all work), model ids are matched loosely too
(`gpt-6.1-sol:Q4_K_M` and `gpt-6.1-sol-2026-04-01` both find `gpt-6.1-sol`), and a model with
no entry at all is priced with `fallbackModel` — shown as `fallback prices` so it is obvious
that the figure is borrowed. Unpriced models leave a one-off hint naming the id to add.

### Switching cost estimation off

If you want the timing and throughput readouts without the money, there is a single switch.
Any of these turns estimation off; `/cost on` turns it back on:

| How | Scope | |
| --- | --- | --- |
| `/cost off` · `/cost on` · `/cost toggle` | current session | flips the setting and re-prices (or zeroes) the session totals immediately |
| `"cost": false` in any price file | that file's scope | also accepted as `"costEnabled"`, `"costEstimates"`, `"estimateCost"`, or `"cost": "off"`; a project file can switch money off for one directory |
| `PI_STATS_COST=off` | environment | `on` switches it back on; `PI_STATS_NO_COST=1` is a shorthand for off |
| `pi -e . --pi-stats-no-cost` | one run | same as the environment setting, for a single session |

Precedence runs files → environment → `/cost` command, so a project file never outvotes an
explicit `PI_STATS_COST` or a `/cost on` in the session you are in.

With estimation off, every usage record prices to $0, the `💵` figure disappears from the status
bar, the per-request cost line is left out of the completion notification, and `/stats` and
`/cost` print `Session cost: estimation disabled` — instead of the usual "add prices for this
model" hint, since nothing is missing. The price files are still read, so `/cost rates`,
`/cost modes` and `/cost mode <name>` keep working and the figures come straight back on
`/cost on`.

### `/cost` command

```
/cost                  cost so far by model, the price files in use, and the active mode
/cost modes            price modes available for the model of the last request
/cost mode <name>      estimate with another mode (standard, batch, flex, fast, ultrafast)
/cost rates [model]    resolved prices for every model, or just one
/cost reload [path]    re-read the price files and recompute the session cost
/cost off | on         stop / resume cost estimation (/cost toggle flips it, /cost status reports it)
```

Changing the mode or reloading re-prices the whole session from the usage records in the
session history, so the numbers always reflect the current price list. Reloading also re-applies
the `"cost"` flag from the files, which is why a `/cost off` followed by a `/cost reload`
can turn estimation back on.

## Install (local use)

From this directory:

```bash
pi install .
```

Or load it for a single run without touching settings:

```bash
pi -e .
```

Then start `pi` normally. Use `/stats` anytime to print the full report, and `/cost` to see
the estimated spend and the prices behind it (or `/cost off` to drop the money readout — see
above).

Then start `pi` normally. Use `/stats` anytime to print the full report, and `/cost` to see
the estimated spend and the prices behind it.

## Notes

- Written in TypeScript; Pi loads it directly via `jiti` — no build step.
- Costs are estimates unless the provider reports them. Rates for hosted models are only as
  current as the price list in `pricing.json`; update it when the provider changes prices.
  Local servers are priced by borrowing another model's rates (`fallbackModel` or an alias),
  which is a comparison figure rather than money spent.
