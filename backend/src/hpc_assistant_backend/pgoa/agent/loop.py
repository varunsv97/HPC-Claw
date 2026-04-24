"""PGOAAgent: OpenAI function-calling agent loop."""

from __future__ import annotations

import json
import logging
from typing import Any

from hpc_assistant_backend.config import AssistantSettings, load_settings
from hpc_assistant_backend.pgoa.agent.prompts import (
    build_initial_user_message,
    build_system_prompt,
)
from hpc_assistant_backend.pgoa.agent.tools import TOOL_SCHEMAS
from hpc_assistant_backend.pgoa.cluster_discovery import (
    collect_login_node_info,
    is_stale,
    run_probe_jobs,
    ApprovalCallback,
)
from hpc_assistant_backend.pgoa.schema import ClusterProfile, DeltaReport, KPIMetrics, OptimizationResult
from hpc_assistant_backend.pgoa.services import (
    analyze_run,
    apply_binding_change,
    build_kpi,
    collect_likwid_profile,
    collect_ncu_profile,
    collect_slurm_profile,
    compare_runs,
    record_slurm_action,
)
from hpc_assistant_backend.pgoa.store import ExperimentStore

log = logging.getLogger(__name__)


class PGOAAgent:
    def __init__(
        self,
        store: ExperimentStore,
        llm_client: Any,
        model: str = "gpt-4o",
        max_iterations: int = 5,
        kpi_threshold_pct: float = 2.0,
        settings: AssistantSettings | None = None,
        probe_approval_callback: ApprovalCallback | None = None,
    ) -> None:
        self._store = store
        self._client = llm_client
        self._model = model
        self._max_iterations = max_iterations
        self._kpi_threshold_pct = kpi_threshold_pct
        self._settings = settings or load_settings()
        self._cluster_profile: ClusterProfile = self._init_cluster_profile(
            probe_approval_callback
        )

    # ------------------------------------------------------------------
    # Cluster discovery
    # ------------------------------------------------------------------

    def _init_cluster_profile(
        self, approval_callback: ApprovalCallback | None
    ) -> ClusterProfile:
        """Collect cluster hardware info at startup; use cache when fresh."""
        # Phase 1: login-node discovery (always fast, no jobs)
        fresh = collect_login_node_info(self._settings)
        cluster_name = fresh.cluster_name

        # Try to load existing cache and merge probe data into it
        cached = self._store.load_cluster_profile(cluster_name)
        if cached is not None and not is_stale(cached):
            log.info("Using cached cluster profile for %s (age OK)", cluster_name)
            profile = cached
        else:
            if cached is not None:
                log.info("Cluster profile for %s is stale, refreshing", cluster_name)
            # Preserve probe data from stale cache so we don't re-run probes unnecessarily
            if cached is not None:
                probed_hw = {
                    p.name: p.hardware
                    for p in cached.partitions
                    if p.hardware_from_probe
                }
                merged_parts = []
                for p in fresh.partitions:
                    if p.name in probed_hw:
                        merged_parts.append(
                            p.model_copy(
                                update={"hardware": probed_hw[p.name], "hardware_from_probe": True}
                            )
                        )
                    else:
                        merged_parts.append(p)
                profile = fresh.model_copy(update={"partitions": merged_parts})
            else:
                profile = fresh
            self._store.save_cluster_profile(profile)

        # Phase 2: probe jobs (gated by approval callback)
        if approval_callback is not None:
            needs_probe = [p.name for p in profile.partitions if not p.hardware_from_probe]
            if needs_probe:
                profile = run_probe_jobs(
                    profile,
                    self._settings,
                    partitions=needs_probe,
                    approval_callback=approval_callback,
                )
                self._store.save_cluster_profile(profile)

        return profile

    def run(
        self,
        workload_id: str,
        job_script_path: str,
        primary_kpi: str,
        kpi_unit: str,
        lower_is_better: bool = True,
    ) -> OptimizationResult:
        system_prompt = build_system_prompt(
            primary_kpi=primary_kpi,
            kpi_unit=kpi_unit,
            workload_id=workload_id,
            kpi_threshold_pct=self._kpi_threshold_pct,
            cluster_profile=self._cluster_profile,
        )

        # Read job script for the initial message (best-effort)
        try:
            from pathlib import Path

            job_script_content = Path(job_script_path).read_text(encoding="utf-8")
        except OSError:
            job_script_content = "(could not read job script)"

        user_message = build_initial_user_message(
            workload_id=workload_id,
            job_script_path=job_script_path,
            job_script_content=job_script_content,
            kpi_metric=primary_kpi,
            kpi_unit=kpi_unit,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        all_deltas: list[DeltaReport] = []
        best_run_id: str | None = None
        convergence_reason = "max_iterations_reached"
        iterations_run = 0
        last_compare_result: dict | None = None

        for iteration in range(self._max_iterations):
            iterations_run = iteration + 1
            log.info(
                "PGOA iteration %d/%d for workload %s",
                iterations_run,
                self._max_iterations,
                workload_id,
            )

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )

            choice = response.choices[0]
            messages.append(choice.message.model_dump(exclude_unset=True))

            # No tool calls → agent decided to stop
            if not choice.message.tool_calls:
                convergence_reason = "agent_no_tool_calls"
                break

            # Execute tool calls
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:
                    result = {"error": f"Invalid JSON arguments: {exc}"}
                else:
                    result = self._dispatch(tool_name, args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

                # Track deltas for convergence check
                if tool_name == "compare_runs" and "kpi_delta_pct" in result:
                    last_compare_result = result

            # Convergence check after compare_runs
            if last_compare_result is not None:
                delta_pct = abs(last_compare_result.get("kpi_delta_pct", 0.0))
                direction = last_compare_result.get("kpi_direction", "neutral")

                # Reconstruct DeltaReport for tracking
                try:
                    delta_report = DeltaReport(
                        from_run_id=last_compare_result["from_run_id"],
                        to_run_id=last_compare_result["to_run_id"],
                        action_applied=last_compare_result.get("action_applied", ""),
                        kpi_delta_pct=last_compare_result["kpi_delta_pct"],
                        kpi_direction=last_compare_result["kpi_direction"],
                        secondary_deltas=last_compare_result.get("secondary_deltas", {}),
                    )
                    all_deltas.append(delta_report)
                    if direction == "improved" or best_run_id is None:
                        best_run_id = last_compare_result["to_run_id"]
                except Exception:
                    pass

                if delta_pct < self._kpi_threshold_pct:
                    convergence_reason = "below_threshold"
                    break

                last_compare_result = None  # reset for next iteration

        # Derive baseline run_id from the first recorded delta (from_run_id is always baseline)
        baseline_run_id = all_deltas[0].from_run_id if all_deltas else None
        final_best_run_id = best_run_id or baseline_run_id or ""

        # Compute total improvement from baseline to best run
        try:
            assert baseline_run_id and final_best_run_id
            baseline_bundle = self._store.load_bundle(workload_id, baseline_run_id)
            best_bundle = self._store.load_bundle(workload_id, final_best_run_id)
            total_improvement = (
                (baseline_bundle.kpi.value - best_bundle.kpi.value)
                / abs(baseline_bundle.kpi.value)
                * 100.0
                if baseline_bundle.kpi.value != 0
                else 0.0
            )
        except Exception:
            total_improvement = 0.0

        return OptimizationResult(
            workload_id=workload_id,
            iterations_run=iterations_run,
            best_run_id=final_best_run_id,
            total_kpi_improvement_pct=total_improvement,
            convergence_reason=convergence_reason,
            all_deltas=all_deltas,
        )

    # ------------------------------------------------------------------
    # Tool dispatch — calls services.py directly; no intermediate registry
    # ------------------------------------------------------------------

    def _make_kpi(self, args: dict) -> KPIMetrics:
        return build_kpi(
            args["kpi_metric"],
            args["kpi_value"],
            args["kpi_unit"],
            lower_is_better=bool(args.get("lower_is_better", True)),
        )

    def _dispatch(self, tool_name: str, args: dict) -> dict:
        try:
            return self._call_service(tool_name, args)
        except Exception as exc:
            log.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
            return {"error": str(exc)}

    def _call_service(self, tool_name: str, args: dict) -> dict:
        if tool_name == "collect_slurm_profile":
            kpi = self._make_kpi(args)
            bundle = collect_slurm_profile(
                self._settings,
                self._store,
                workload_id=args["workload_id"],
                run_id=args["run_id"],
                job_id=int(args["job_id"]),
                kpi=kpi,
            )
            return {
                "ok": True,
                "run_id": args["run_id"],
                "slurm": bundle.slurm.model_dump() if bundle.slurm else None,
            }
        if tool_name == "collect_ncu_profile":
            kpi = self._make_kpi(args)
            bundle = collect_ncu_profile(
                self._store,
                workload_id=args["workload_id"],
                run_id=args["run_id"],
                ncu_csv_path=args["ncu_csv_path"],
                kpi=kpi,
            )
            return {
                "ok": True,
                "run_id": args["run_id"],
                "roofline": bundle.compute.roofline_position if bundle.compute else None,
            }
        if tool_name == "collect_likwid_profile":
            kpi = self._make_kpi(args)
            bundle = collect_likwid_profile(
                self._store,
                workload_id=args["workload_id"],
                run_id=args["run_id"],
                likwid_output_path=args["likwid_output_path"],
                kpi=kpi,
            )
            return {
                "ok": True,
                "run_id": args["run_id"],
                "cpu_perf": bundle.cpu_perf.model_dump() if bundle.cpu_perf else None,
            }
        if tool_name == "analyze_bottlenecks":
            report = analyze_run(
                self._store,
                workload_id=args["workload_id"],
                run_id=args["run_id"],
            )
            return {
                "run_id": report.run_id,
                "primary_bottleneck": report.primary_bottleneck,
                "details": report.details,
                "recommended_action_hint": report.recommended_action_hint,
            }
        if tool_name == "propose_slurm_action":
            proposal = record_slurm_action(
                self._store,
                workload_id=args["workload_id"],
                run_id=args["run_id"],
                action_name=args["action_name"],
                parameters=args["parameters"],
                rationale=args["rationale"],
                expected_improvement_pct=args.get("expected_improvement_pct"),
            )
            return {"ok": True, "proposal": proposal.model_dump()}
        if tool_name == "apply_slurm_action":
            try:
                output_path, changes = apply_binding_change(
                    self._settings,
                    self._store,
                    workload_id=args["workload_id"],
                    run_id=args["run_id"],
                    job_script_path=args["job_script_path"],
                    output_script_path=args["output_script_path"],
                    ntasks_per_node=args.get("ntasks_per_node"),
                    ntasks_per_socket=args.get("ntasks_per_socket"),
                    cpus_per_task=args.get("cpus_per_task"),
                    mem_bind=args.get("mem_bind"),
                    cpu_bind=args.get("cpu_bind"),
                    exclusive=bool(args.get("exclusive", False)),
                )
            except ValueError as exc:
                return {"ok": False, "validation_errors": [p.strip() for p in str(exc).split(";")]}
            return {"ok": True, "new_script_path": output_path, "changes_applied": changes}
        if tool_name == "compare_runs":
            delta = compare_runs(
                self._store,
                workload_id=args["workload_id"],
                from_run_id=args["from_run_id"],
                to_run_id=args["to_run_id"],
                action_applied=args["action_applied"],
            )
            return {
                "from_run_id": delta.from_run_id,
                "to_run_id": delta.to_run_id,
                "kpi_delta_pct": delta.kpi_delta_pct,
                "kpi_direction": delta.kpi_direction,
                "secondary_deltas": delta.secondary_deltas,
            }
        return {"error": f"Unknown tool: {tool_name!r}"}
