ifneq ("$(wildcard .venv/bin/python)","")
PYTHON ?= .venv/bin/python
else
PYTHON ?= python3
endif
PYTHON_GIL ?= 1

BACKEND_SRC := backend/src
TUI_SRC := tui/src

.PHONY: help backend-test tui-test test dev-help

help:
	@printf '%s\n' \
		'Available targets:' \
		'  python        $(PYTHON)' \
		'  python_gil    $(PYTHON_GIL)' \
		'  backend-test  Run Python backend smoke tests' \
		'  tui-test      Run Python TUI tests' \
		'  test          Run both smoke suites' \
		'  dev-help      Show the root development launcher help'

backend-test:
	PYTHON_GIL=$(PYTHON_GIL) PYTHONPATH=$(BACKEND_SRC) $(PYTHON) -m unittest discover -s backend/tests -v

tui-test:
	PYTHON_GIL=$(PYTHON_GIL) PYTHONPATH=$(TUI_SRC) $(PYTHON) -m unittest discover -s tui/tests -v

test: backend-test tui-test

dev-help:
	PYTHON_GIL=$(PYTHON_GIL) $(PYTHON) scripts/dev.py --help
