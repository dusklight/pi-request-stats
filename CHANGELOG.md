# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-10-19

Initial public release of **`pi-request-stats`**, a Pi extension that reports, for every request:
how the time split between model generation and tool execution, tokens-per-second for each phase,
compaction counts, and an estimated cost from a JSON price list.

### Added

- **Request timing split by what produced it** — 🧠 thinking, 💬 answer and ⚙️ tool-call
  argument streaming are measured as model generation, while 🔧 tool execution wall time is
  measured separately and excluded from every tokens-per-second figure. ⏳ "other" covers
  latency before the first token, retries and compaction.
- **Per-phase tokens-per-second** in the live status bar, in the request-complete notification
  and in `/stats`, with a session-wide throughput roll-up.
- **Thinking-token estimation** for engines that report no reasoning breakdown
  (`llama.cpp`/`llama-server`, vLLM, most OpenAI-compatible proxies). A reported `0` is read as
  "not reported"; phases are sized from streamed characters, rescaled to the reported output
  tokens and weighted by measured characters-per-token priors. Every readout says when thinking
  tokens — and so thinking tok/s — are estimates.
- **Compaction counting** and a `📦` compaction counter in the status bar.
- **`/stats` command** — on-demand report: request and compaction counts, the last request's and
  the session's wall time split into generating + tools, live and session-wide per-phase rates,
  and session token totals (input / output, cache read / write).
- **Estimated request cost** from layered JSON price files: bundled `pricing.json`,
  `~/.pi/agent/pi-request-stats.pricing.json`, `<cwd>/.pi/pi-request-stats.pricing.json`, and an explicit file
  via `$PI_REQUEST_STATS_PRICING`, `--pi-request-stats-pricing <path>` or `/cost reload <path>` (later files win,
  model by model). Rates are per 1M tokens and use pi's own catalog field names; field spellings
  and model ids are matched loosely, entries may carry `aliases`, one id may be mapped onto
  another entry, and `fallbackModel` prices unlisted models.
- **Price modes** (standard, batch, flex, fast, ultrafast) and **long-context tiers**
  (`>272K` input tokens), switchable with `/cost mode <name>`.
- **`/cost` command**: session cost by model, `/cost modes`, `/cost mode <name>`,
  `/cost rates [model]`, `/cost reload [path]`, `/cost off | on | toggle | status`.
  Changing a mode or reloading re-prices the whole session from the usage records in history.
- **Cost estimation switch** — `/cost off`, `"cost": false` in any price file,
  `PI_REQUEST_STATS_COST=off` / `PI_REQUEST_STATS_NO_COST=1`, or `pi -e . --pi-request-stats-no-cost`. Precedence is
  files → environment → command.
- **Honest labelling of every money figure**: `≈$0.19` for estimates from the price file, a bare
  `$0.19` for costs billed by the provider, explicit wording naming the price entry whose rates
  were used, and a distinct form for the borrowed-rates case.
- **History replay on startup** — requests, compactions, throughput, tool time and cost are
  reconstructed from the session transcript, so a resumed session reports meaningful numbers at
  once.
- **CLI flags** `--pi-request-stats-pricing <path>` and `--pi-request-stats-no-cost`.
- **Project hygiene** — MIT `LICENSE`, package metadata (`license`, `author`, `repository`,
  `files`, `engines`), `pi-package` keyword and an explicit `pi.extensions` manifest entry,
  `tsconfig.json` with a strict `npm run typecheck`, `npm test`, and this changelog.

### Known limits

- Costs are estimates unless the provider reports them; bundled rates are a snapshot of the
  OpenAI pricing page taken 2026-10-18 and need refreshing when prices change.
- Local servers are priced by borrowing another model's rates (`fallbackModel` or an alias): a
  comparison figure, not money spent, and labelled as such.
- Per-phase timing for historical messages is approximate: transcripts store message start/end
  times, not per-content-block times.
