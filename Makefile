SHELL := /bin/zsh

PYTHON ?= .venv/bin/python
HOST ?= 127.0.0.1
PORT ?= 7860

.PHONY: run dev

run:
	uv run --python $(PYTHON) gui_app.py --host $(HOST) --port $(PORT)

dev: run
