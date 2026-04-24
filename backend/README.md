# Backend Runtime

This package contains the Python backend prototype for `hpc-assistant`.

Current focus:

- safe cluster discovery from login nodes
- software/module environment discovery
- optional, explicit probe jobs for compute-node hardware inspection
- OpenCode tool shims for selected deterministic operations
- experimental PGOA orchestration on top of the shared service layer

The backend is designed to be installable in user space on a shared HPC
cluster. That is why the current prototype avoids adding larger agent
frameworks unless they provide clear operational value.

## Primary CLI

Once installed, the package exposes:

```bash
hpc-assistant-backend doctor
hpc-assistant-backend discover-cluster
hpc-assistant-backend discover-env
hpc-assistant-backend probe-cluster --partition <name> --yes
```

OpenCode helpers:

```bash
hpc-assistant-backend show-opencode-tools
hpc-assistant-backend sync-opencode-tools
```

Experimental:

```bash
hpc-assistant-backend run-pgoa --workload-id <id> --job-script-path <path> --primary-kpi elapsed_s --kpi-unit seconds
```
