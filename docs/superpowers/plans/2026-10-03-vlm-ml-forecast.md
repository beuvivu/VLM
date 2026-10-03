# VLM ML Forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver causal ML ensemble forecasts with incremental feedback and automatic updates after persisted draws.

**Architecture:** Add a bounded, persistent ML pipeline alongside the existing forecaster. Convert node predictions into proper set/digit laws, score before learning, and separate experimental backtests from timed live evidence.

**Tech Stack:** Python 3.11+, NumPy, scikit-learn, optional XGBoost/LightGBM, Pydantic/FastAPI, pytest.

**Spec:** `docs/VLM_ML_DESIGN.md`

## Global Constraints

- Preserve package version 4.0.0 and existing forecast API/CLI compatibility.
- No fabricated missing draws or future labels in features, fitting, calibration, or tuning.
- Bootstrap at most 1.000 recent draws; bounded tree buffer; eight GRU snapshots.
- Proper joint probabilities; live gate e-value 140, at least 100 scored live draws, positive recent log-score.
- JSON gzip atomic checkpoints, no pickle; persist results before forecast work.
- Optional ml extras must never silently skip a requested backend.

## Review Focus

- Repaired old IDs or a corrected existing result must invalidate learned weights/evidence.
- All-zero/all-one labels and missing classes must still produce finite proper probabilities.
- Gaps, duplicate or descending IDs, future dates must not be treated as complete sequential history.
- Concurrent updater/API calls must not race checkpoint or score ledger; model failure retains results.
- Interrupted checkpoints and an expired/unverified target must not manufacture live evidence.

### Task 1: Causal features and probability laws

**Files:** Create `src/vlm/forecast/{__init__,features,distribution}.py`; test `tests/test_vlm_forecast_features.py`.
**Interfaces:** `FeatureState(spec).snapshot(next_id, next_date) -> ndarray`, `observe(row, draw_id, draw_date)`; `Law(spec, values, mixture).log_likelihood(row, bonus=0)`, `marginals()`, `top(count, max_nodes=20000)`.

- [x] Write failing causal prefix, ID-gap, shape/range and hand-derived small-space probability tests.
- [x] Run `pytest tests/test_vlm_forecast_features.py -q`; expected failures before implementation.
- [x] Implement streaming features and exact normalized set/digit mixture laws, bounded top search.
- [x] Run the same test command; expected all pass.
- [x] Commit task code/tests.

### Task 2: Trainable experts and persistent feedback pipeline

**Files:** Create `src/vlm/forecast/{models,pipeline}.py`; test `tests/test_vlm_forecast_models.py`, `tests/test_vlm_forecast_pipeline.py`.
**Interfaces:** `OnlineLogistic.predict/learn`, `GRU.predict/loss_and_gradients/learn`, `TreeExpert.fit/predict`, `MLForecaster(product, config).update(series)`, `report(top_n, budget)`, `save/load`.
**Consumes:** FeatureState and Law from Task 1.

- [x] Write failing finite-difference GRU, planted signal, probability, score-before-learn, reload/idempotence, corrected-prefix and incomplete-target tests.
- [x] Run the two new files; expected failures for missing implementation.
- [x] Implement real BPTT/Adam, bounded RF plus optional boosters, fixed-share proper-score feedback, reliability/Brier, bounded bootstrap and secure checkpoints.
- [x] Run the same files; expected all pass, optional adapters pass with installed extras.
- [x] Commit task code/tests and dependency extras.

### Task 3: Automatic learning and user interfaces

**Files:** Create `src/vlm/forecast/{service,cli,api}.py`; modify config, API main, update service/runner, workflow caches, `.env.example`, `pyproject.toml`; test `tests/test_vlm_forecast_integration.py`.
**Interfaces:** `refresh_models(state, product) -> dict`; `vlm-forecast next/update/benchmark`; `/ml/forecast/{product}` and update status learning result.
**Consumes:** MLForecaster, current load_series and record_window, shared synchronization.

- [x] Write failing integration tests for real source result -> saved learned state, CLI/API, model error isolation, lock contention and expired forecast rejection.
- [x] Run integration tests; expected failures before wiring.
- [x] Implement synchronized persisted updates off event loop, APIs/CLI, restart/correction warnings, deterministic budget portfolio.
- [x] Run integration tests and existing updater/forecast/API tests; expected all pass.
- [x] Commit task code/tests.

### Task 4: Real-data evaluation and completion

**Files:** Create `scripts/ml_report.py`, `docs/VLM_ML_FORECAST.md`, `reports/VLM_ML_2026-10-03.md`; update README.
**Interfaces:** Reproducible bootstrap/backtest report for every product; exact settings/snapshot provenance and no-edge limits.

- [x] Run benchmark with RF/GRU and separate boosters on actual stored data; expected finite scores and explicit gate/evidence limitations.
- [x] Run bare `pytest -q`, Ruff, package build and CLI smoke; expected all pass.
- [x] Request independent whole-branch review, resolve important findings with failing regression tests, run full suite.
- [x] Commit reports and publish reviewed branch/PR; follow existing authorization to complete integration after green CI.
