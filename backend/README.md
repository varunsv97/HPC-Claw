# Backend Runtime

This package contains the Python backend runtime surface for `hpc-assistant`.
Current scope in this workspace:

- user-space installable package metadata and CLI entrypoints
- Deep Agents and LangGraph runtime wiring
- policy-aware HPC tool wrappers for Slurm, Lmod, and filesystem inspection
- approval-gated mutation tools surfaced through Deep Agents interrupt metadata
- a threaded localhost HTTP service for the local Textual TUI flow
