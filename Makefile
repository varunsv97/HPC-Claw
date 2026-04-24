ifneq ("$(wildcard .venv/bin/python)","")
PYTHON ?= .venv/bin/python
else
PYTHON ?= python3
endif
PYTHON_GIL ?= 1

BACKEND_SRC := backend/src

.PHONY: help test dev-help prototype-doctor prototype-cluster prototype-env tui-run

help:
	@printf '%s\n' \
		'Available targets:' \
		'  python        $(PYTHON)' \
		'  python_gil    $(PYTHON_GIL)' \
		'  test          Run backend tests' \
		'  prototype-doctor  Print installation/guardrail report' \
		'  prototype-cluster Run login-node cluster discovery' \
		'  prototype-env     Run module/environment discovery' \
		'  tui-run       Launch the Rust prototype TUI' \
		'  dev-help      Show the development launcher help'

test:
	PYTHON_GIL=$(PYTHON_GIL) PYTHONPATH=$(BACKEND_SRC) $(PYTHON) -m unittest discover -s backend/tests -v

dev-help:
	PYTHON_GIL=$(PYTHON_GIL) $(PYTHON) scripts/dev.py --help

prototype-doctor:
	PYTHON_GIL=$(PYTHON_GIL) PYTHONPATH=$(BACKEND_SRC) $(PYTHON) -m hpc_assistant_backend doctor

prototype-cluster:
	PYTHON_GIL=$(PYTHON_GIL) PYTHONPATH=$(BACKEND_SRC) $(PYTHON) -m hpc_assistant_backend discover-cluster

prototype-env:
	PYTHON_GIL=$(PYTHON_GIL) PYTHONPATH=$(BACKEND_SRC) $(PYTHON) -m hpc_assistant_backend discover-env

tui-run:
	cargo run --manifest-path tui/Cargo.toml
