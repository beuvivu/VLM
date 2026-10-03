# VLM scheduled result updates

> Execution: inline with pytest first, then independent whole-branch review.

**Goal:** Refresh actual draw results automatically at each product's Vietnam-time windows, recover missed runs and expose honest freshness.

**Design:** Reuse the existing sync APIs/pipelines and add a shared UTC+7 schedule plus durable runner. The API owns storage and runs the updater in its lifespan; a CLI runs the same engine once for GitHub Actions. Source budgets and stale/empty fallback preserve progress under blocking. Date-only Keno/Bingo18 records never certify a particular intraday slot.

**Constraints:** Python >=3.11; no new runtime dependencies; no manufactured results/IDs; Max3D/+ shared draw; Max4D excluded; independent per-product retry, one writer per storage owner; Linux/Windows compatibility.

## Tasks

- [x] Test scheduled windows: weekly days, Lotto 13/21, fast 06:00–22:15, UTC conversion, midnight/startup recovery, archive exclusion.
- [x] Implement `vlm.updates.schedule`: last completed slot, polling interval and freshness assessment (date precision retained).
- [x] Test and implement durable `UpdateRunner`: independent failure/retry, bounded attempts, duplicate-run prevention, atomic journal/status, restart recovery.
- [x] Integrate API lifespan, status endpoint and shared sync locks; CLI once/daemon. Results persisted before reporting success; prize failure does not discard result success.
- [x] Add finite budgets and empty/stale fallbacks to existing source chains, with regression tests.
- [x] Replace Docker's six-hour loop with API-owned updater. Add GitHub recovery schedule and preserve state/results outside cache; update README and operations guide.
- [ ] PR CI and merge only after green. Local full pytest/Ruff, API lifecycle tests and package build passed; final review in progress.

Bounded live smoke on 03/10/2026 20:26 ICT: Power remained #1405 dated 01/10, no new records; optional finance timed out. Freshness correctly reported `behind_schedule` and CLI exited 1. No claim that blocked/delayed sources have supplied the latest draw. Docker runtime could not be exercised in this workspace (Docker is unavailable).

## Review focus

Network stall must not starve another game; HTTP 200 with old results must remain stale; failure/restart must retain committed records; no two writers may open the DuckDB file; UTC cron/naive timestamps must not shift Vietnam dates.
