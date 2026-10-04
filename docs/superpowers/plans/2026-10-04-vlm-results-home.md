# VLM Results Homepage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the results/predictions/comparisons homepage and retain the existing analysis at forecast.html.
**Architecture:** A read-only Python snapshot joins validated results with immutable pre-draw ledgers. The static builder emits two pages plus JSON; JavaScript controls filters and refresh. Both scheduled jobs deploy the refreshed snapshot.
**Tech Stack:** Python 3.11+, existing NumPy/Pydantic/DuckDB, static HTML/CSS/JavaScript, GitHub Pages.
**Spec:** docs/VLM_RESULTS_HOME.md

## Global Constraints
- Preserve existing JSON consumers and forecast page functionality.
- Official provenance and missing finance values must remain explicit.
- Never recompute historical predictions from current models; match exact product/ID/date.
- Active seven products; Max3D+ shares Max3D results and has its own prize catalogue.
- VLA lavender/purple/teal, full light/dark, Vietnamese, 375/414/1440px, 44px controls.
- Read-only snapshot; no model learning or network in the builder.

## Review Focus
- Repeated/forced/late ledger entries must never replace the first valid pre-draw forecast.
- A result arriving after cache loss or through fallback must preserve provenance and exact draw pairing.
- Missing prize data must never be reported as zero winners or a previous draw jackpot.
- Mobile/dark/keyboard state remains usable during JSON refresh and failures.
- Results and batch jobs cannot deploy stale builds in the wrong order or lose commits.

### Task 1: Read-only dashboard data and honest settlement
**Files:** Create src/vlm/web/__init__.py, src/vlm/web/dashboard.py; test tests/test_vlm_dashboard.py.
**Interfaces:** Consumes DrawRecord, game rules, seed/cache/journal, ML/legacy ledgers. Produces build_dashboard(data_dir: Path, seed_dir: Path, directory: Path, journal: Path | None = None, now: datetime | None = None) -> dict and compare_prediction(prediction: dict, draw: DrawRecord) -> dict.
- [ ] Write controlled fixtures for Mega/Power/Lotto/Max/Keno/Bingo results and forecasts; assert product/ID/date guards, first valid frozen issue, no late/forced entries, nullable finance and exclusions. Use hand-derived hits and tier names, leading zeroes and repeated dice.
- [ ] Run pytest tests/test_vlm_dashboard.py -q. Expected: FAIL because dashboard module missing.
- [ ] Implement bounded recent result extraction and full rule catalogues, immutable ledger ranking, next forecast fallback labels, exact settlement.
- [ ] Run pytest tests/test_vlm_dashboard.py -q. Expected: PASS. Commit feat: build source-backed results and forecast comparisons.

### Task 2: Two-page responsive dashboard
**Files:** Modify scripts/build_site.py; create scripts/site_assets/index.html, styles.css, dashboard.js; extend tests/test_vlm_dashboard.py integration fixture.
**Interfaces:** Consumes build_dashboard dictionary. Produces index.html, forecast.html, data/dashboard.json, assets, preserving summary/forecast/scoreboard/ledger paths.
- [ ] Add real builder integration assertions: both navigable pages and usable result/forecast/comparison JSON without checkpoint. Run test. Expected: FAIL before page migration.
- [ ] Rename the existing generated page to forecast.html and create VLA homepage with result/product filters, draw selection, prize details, Top5 forecast, comparison cards, loading/stale/error state, theme and refresh. Link both pages.
- [ ] Run pytest tests/test_vlm_dashboard.py -q. Expected: PASS. Build real seed + journal and inspect browser at 1440/375/414px, both themes, controls and JSON refresh. Expected: no horizontal overflow or console errors; selections persist; matches clearly labelled. Commit feat: add results homepage and preserve forecast analysis.

### Task 3: Automatic publication and end-to-end verification
**Files:** Modify .github/workflows/results.yml, update.yml, README.md, docs/VLM_RESULTS_HOME.md as needed.
**Interfaces:** Consumes builder CLI and updater ledgers. Produces refreshed Pages in both scheduled workflows.
- [ ] Make results job generate/record legacy and ML forecasts after source update, preserving results on model failure; build and deploy Pages independently of individual product error. Share concurrency and cache strategy. Batch update also uses both-page builder.
- [ ] Update usage/schedules/docs and run full pytest suite, Ruff F/B023, real build, wheel build. Expected: all checks pass (optional backend skips explicitly reported).
- [ ] Commit ci: publish dashboard with every result update. Run a whole-branch independent review with plan/spec and precise diff; fix important findings with RED→GREEN, full suite. Publish a draft PR, verify green CI, merge per ongoing user authorization, and verify Pages deploy. Expected: main contains both pages with actual data.
