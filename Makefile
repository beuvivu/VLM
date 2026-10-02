.PHONY: install dev init doctor site test lint sync seed prizes official-data products products-sync import-v130 checklist import-pages forecast forecast-update forecast-report products-report market report bao-report analyze inference backtest ml serve docker-build docker-up docker-job

PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

install:         ## bộ cài: .venv + gói + dữ liệu + bộ dự báo + kiểm tra (Windows: install.cmd)
	bash install.sh

dev:             ## như install, thêm pytest/ruff/mypy và chạy test
	bash install.sh --dev

init:            ## nạp dữ liệu đi kèm vào kho, học bộ dự báo, kiểm tra
	$(PY) -m vietlott_engine.cli init

doctor:          ## kiểm tra Python, gói, dữ liệu, kho, bộ dự báo, mạng
	$(PY) -m vietlott_engine.cli doctor

site:            ## site/: trang tĩnh + JSON từ trạng thái bộ dự báo (như trên GitHub Pages)
	$(PY) scripts/build_site.py

test:
	$(PY) -m pytest

lint:
	ruff check --select F,B023 src scripts tests

seed:            ## load the bundled snapshot (offline)
	$(PY) -m vietlott_engine.cli sync --game all --source file --path data/seed

sync:            ## incremental sync from the configured source
	$(PY) -m vietlott_engine.cli sync --game all

prizes:          ## official prize tables (winners per tier + jackpot) → store
	$(PY) -m vietlott_engine.cli prizes --game all --source canonical

official-data:   ## rebuild data/seed: official records + community archive (+ V130=… 1.3.0 snapshot)
	$(PY) scripts/build_official_dataset.py --vendor .vendor --update $(if $(V130),--v130-dir $(V130),)

products:        ## coverage of all seven products
	$(PY) -m vietlott_engine.cli products list

products-sync:   ## Keno, Bingo18, Max 3D, Max 3D Pro (SOURCE=auto|vietlott|nhanaz|mirror|canonical|v130)
	$(PY) -m vietlott_engine.cli products sync --source $(or $(SOURCE),auto)

import-v130:     ## import a Vietlott Quant Engine 1.3.0 snapshot: make import-v130 V130=/path/to/Vietlott-Quant-Engine-1.3.0
	$(PY) -m vietlott_engine.cli products sync --source v130 --path $(V130)
	$(PY) -m vietlott_engine.cli prizes --game all --source v130 --path $(V130)

forecast:        ## self-learning forecaster: learn new draws, forecast the next one, record it (PRODUCT=all|mega645|…|max4d)
	$(PY) -m vietlott_engine.cli forecast next --product $(or $(PRODUCT),all)

forecast-update: ## after a draw: learn it and score the recorded forecasts; then the live track record
	$(PY) -m vietlott_engine.cli forecast update --product $(or $(PRODUCT),all)
	$(PY) -m vietlott_engine.cli forecast scoreboard

forecast-report: ## reports/forecast.{md,json}: forecaster on the bundled history + size/power simulation
	$(PY) scripts/forecast_report.py

checklist:       ## reports/verification_checklist.{md,csv}: vietlott.vn pages worth opening by hand (Cloudflare is passed by a person)
	$(PY) scripts/verification_checklist.py

import-pages:    ## compare (DRY=1) or import pages saved from vietlott.vn: make import-pages PAGES=~/Downloads/vietlott [DRY=1]
	$(PY) -m vietlott_engine.cli products import-pages --path $(PAGES) $(if $(DRY),--dry-run,)

products-report: ## reports/products.{md,json}: odds, randomness tests, Max 3D digit cross-check
	$(PY) scripts/products_report.py

market:          ## payout share, tickets sold, crowd calibration, sales model, Lotto rolldowns → data/calibration
	$(PY) -m vietlott_engine.cli market

report:          ## reports/vietlott_v3.{md,json}
	$(PY) scripts/market_report.py

bao-report:      ## reports/bao_vietlott.{md,json}: bao catalogue, prize tables, strategy comparison, Max 3D
	$(PY) scripts/bao_report.py

analyze:
	$(PY) -m vietlott_engine.cli analyze --game all

inference:       ## fairness certificate (power/TOST, hierarchical Bayes, e-processes, change points)
	$(PY) -m vietlott_engine.cli inference --game all --sims 2000 --out reports

backtest:
	$(PY) -m vietlott_engine.cli backtest --game all --tickets 5 --out reports \
		--strategies random hot cold overdue bayes markov gcn anti_popularity wheel

ml:
	$(PY) -m vietlott_engine.cli ml --game all

serve:
	$(PY) -m vietlott_engine.cli serve --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d api scheduler

docker-job:
	docker compose --profile jobs run --rm backtest
