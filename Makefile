.PHONY: install ingest run test eval lint docker

install:      ## Install dependencies
	pip install -r requirements.txt

ingest:       ## Build the search index from data/docs
	python -m app.rag.ingest

run:          ## Build index if needed and start the server
	python run.py

test:         ## Run unit tests
	pytest -q

eval:         ## Run the evaluation quality gate
	python -m eval.run_eval

lint:         ## Lint the codebase
	ruff check app eval tests

docker:       ## Build and run with Docker Compose
	docker compose up --build
