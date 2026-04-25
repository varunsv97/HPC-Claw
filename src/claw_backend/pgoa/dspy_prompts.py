"""DSPy signatures and modules for structured PGOA reasoning.

Three decision points in the optimization loop are handled here:

1. :class:`BottleneckToHypothesis`
   Given a bottleneck report + cluster context, produce a concrete optimization
   hypothesis and decide whether a **source-code edit** is required vs. a
   Slurm-script-only change.

2. :class:`HypothesisToEditPrompt`
   Convert the hypothesis into a complete, self-contained instruction for
   OpenCode so it can apply the code change without any additional context.

3. :class:`MetricsDeltaEvaluation`
   After the post-edit run, compare before/after KPI and decide whether the
   hypothesis was confirmed, rejected, or inconclusive — and propose the next
   step.

Usage::

    from claw_backend.pgoa.dspy_prompts import configure_dspy, get_modules

    configure_dspy(settings)
    b2h, h2ep, delta_eval = get_modules()

    pred = b2h(
        bottleneck_type="memory_bound_gpu",
        bottleneck_details=...,
        recommended_hint=...,
        cluster_context=...,
        profile_summary=...,
    )
    if pred.edit_needed:
        pred2 = h2ep(hypothesis=pred.hypothesis, ...)
        # send pred2.opencode_prompt to OpenCode
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from claw_backend.config import AssistantSettings

log = logging.getLogger(__name__)

# DSPy is an optional dependency — import it at module level so Signature subclasses
# work, but wrap in a try/except so the rest of the backend can import this file
# without dspy installed.  Callers that actually invoke get_modules() or
# configure_dspy() will see a clear ImportError if dspy is missing.
try:
    import dspy
    _DSPY_AVAILABLE = True
except ImportError:
    dspy = None  # type: ignore[assignment]
    _DSPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DSPy Signatures — only defined when dspy is available
# ---------------------------------------------------------------------------

if _DSPY_AVAILABLE:
    class BottleneckToHypothesis(dspy.Signature):  # type: ignore[misc]
        """Given a performance bottleneck, generate a concrete, testable optimization hypothesis.

        Determine whether the fix requires source code changes (edit_needed=True) or
        only a Slurm job-script change (edit_needed=False).

        Bottleneck types that almost always require source edits:
          memory_bound_gpu, compute_bound_gpu, latency_bound_gpu,
          cpu_memory_bound, cpu_compute_bound

        Bottleneck types that can usually be fixed via Slurm script only:
          mpi_binding, high_rss

        Return edit_needed=False for none_detected and insufficient_data.
        """

        bottleneck_type: str = dspy.InputField(
            desc="Primary bottleneck category, e.g. memory_bound_gpu"
        )
        bottleneck_details: str = dspy.InputField(
            desc="JSON object with metric summaries from the BottleneckReport details dict"
        )
        recommended_hint: str = dspy.InputField(
            desc="recommended_action_hint string from the BottleneckReport"
        )
        cluster_context: str = dspy.InputField(
            desc=(
                "Hardware summary: CPU arch, GPU model, peak memory bandwidth, "
                "NUMA topology, available toolchains"
            )
        )
        profile_summary: str = dspy.InputField(
            desc=(
                "Key metrics from the ProfileBundle: KPI value+unit, Slurm elapsed_s, "
                "GPU sm_occupancy_pct, memory_bw_utilization_pct, roofline_position"
            )
        )

        hypothesis: str = dspy.OutputField(
            desc=(
                "Specific, testable optimization hypothesis in one sentence. "
                "Example: 'Switching matrix multiplications from FP32 to BF16 will reduce "
                "memory traffic and improve throughput by ~20%.'"
            )
        )
        rationale: str = dspy.OutputField(
            desc="Two-sentence explanation of why this hypothesis follows from the bottleneck data"
        )
        edit_needed: bool = dspy.OutputField(
            desc=(
                "True if the hypothesis requires a source code change; "
                "False if a Slurm-script-only change (binding, memory flags) suffices"
            )
        )
        expected_improvement_pct: float = dspy.OutputField(
            desc="Realistic expected speedup as a percentage (e.g. 15 means 15%)"
        )

    class HypothesisToEditPrompt(dspy.Signature):  # type: ignore[misc]
        """Convert an optimization hypothesis into a complete OpenCode edit instruction.

        The output ``opencode_prompt`` must be fully self-contained: it must include
        the bottleneck context, the specific change to make, which files to inspect
        first, and any hardware constraints to respect.  OpenCode will not have
        access to any other context.
        """

        hypothesis: str = dspy.InputField(
            desc="The specific optimization hypothesis to implement"
        )
        rationale: str = dspy.InputField(
            desc="Why this hypothesis is expected to improve performance"
        )
        bottleneck_type: str = dspy.InputField(
            desc="Bottleneck category, e.g. memory_bound_gpu"
        )
        edit_roots: str = dspy.InputField(
            desc=(
                "Comma-separated list of source directories and files that OpenCode "
                "is allowed to edit, relative to the project root"
            )
        )
        hardware_context: str = dspy.InputField(
            desc=(
                "Target hardware constraints: CPU arch (e.g. x86_64 with AVX-512), "
                "GPU model (e.g. NVIDIA A100 80GB), peak memory bandwidth"
            )
        )

        opencode_prompt: str = dspy.OutputField(
            desc=(
                "Complete, self-contained instruction for OpenCode. "
                "Must include: (1) what bottleneck was detected and relevant metrics, "
                "(2) the exact code change to apply, "
                "(3) which files/directories to start from, "
                "(4) hardware-specific constraints (precision, intrinsics, etc.), "
                "(5) what NOT to change (keep backward compatibility, no API changes). "
                "Write it as a direct imperative to OpenCode, not as a description."
            )
        )
        target_files_hint: str = dspy.OutputField(
            desc=(
                "Comma-separated list of files most likely to need edits "
                "(OpenCode will verify these exist)"
            )
        )

    class MetricsDeltaEvaluation(dspy.Signature):  # type: ignore[misc]
        """Evaluate whether a code edit confirmed or refuted the optimization hypothesis.

        Use the KPI delta direction and magnitude to judge effectiveness.
        A confirmed hypothesis improved performance beyond the noise threshold.
        A rejected hypothesis degraded or did not change performance.
        Inconclusive means the change was within noise or data was insufficient.
        """

        delta_json: str = dspy.InputField(
            desc=(
                "JSON object with fields: from_run_id, to_run_id, "
                "kpi_delta_pct (negative = faster when lower_is_better), "
                "kpi_direction (improved/degraded/neutral), secondary_deltas"
            )
        )
        hypothesis: str = dspy.InputField(
            desc="The hypothesis that motivated the edit"
        )
        bottleneck_type: str = dspy.InputField(
            desc="The bottleneck category that was targeted"
        )
        expected_improvement_pct: float = dspy.InputField(
            desc="The improvement % that was predicted before the edit"
        )

        verdict: str = dspy.OutputField(
            desc="Exactly one of: confirmed, rejected, inconclusive"
        )
        explanation: str = dspy.OutputField(
            desc="One sentence explaining why the verdict was reached, citing specific numbers"
        )
        next_hypothesis: str = dspy.OutputField(
            desc=(
                "Next optimization hypothesis to try, or empty string if "
                "converged / no more useful directions to explore"
            )
        )
        should_rollback: bool = dspy.OutputField(
            desc=(
                "True if the edit degraded performance and should be reverted "
                "before the next iteration"
            )
        )


# ---------------------------------------------------------------------------
# Module instances (initialized lazily after configure_dspy)
# ---------------------------------------------------------------------------

_bottleneck_to_hypothesis: "dspy.ChainOfThought | None" = None
_hypothesis_to_edit_prompt: "dspy.Predict | None" = None
_delta_evaluation: "dspy.ChainOfThought | None" = None


def configure_dspy(settings: AssistantSettings) -> None:
    """Configure the global DSPy LM to use the same backend as the PGOA agent.

    Must be called once before :func:`get_modules`.
    Raises :exc:`ImportError` if dspy is not installed.
    """
    if not _DSPY_AVAILABLE:
        raise ImportError(
            "DSPy is required for structured prompt reasoning. "
            "Install it with: pip install 'dspy>=2.6.0'"
        )
    lm = dspy.LM(  # type: ignore[union-attr]
        model=f"openai/{settings.openai_model}",
        api_key=settings.openai_api_key or "",
        base_url=settings.openai_base_url,
        timeout=settings.openai_timeout_seconds,
        cache=False,
    )
    dspy.configure(lm=lm)  # type: ignore[union-attr]
    log.debug("DSPy configured with model=%s", settings.openai_model)


def get_modules() -> tuple[Any, Any, Any]:
    """Return ``(bottleneck_to_hypothesis, hypothesis_to_edit_prompt, delta_evaluation)``.

    Module instances are created once and reused.  Call :func:`configure_dspy`
    first — DSPy will raise if no LM is configured.
    Raises :exc:`ImportError` if dspy is not installed.
    """
    if not _DSPY_AVAILABLE:
        raise ImportError(
            "DSPy is required for structured prompt reasoning. "
            "Install it with: pip install 'dspy>=2.6.0'"
        )
    global _bottleneck_to_hypothesis, _hypothesis_to_edit_prompt, _delta_evaluation
    if _bottleneck_to_hypothesis is None:
        _bottleneck_to_hypothesis = dspy.ChainOfThought(BottleneckToHypothesis)  # type: ignore[union-attr]
        _hypothesis_to_edit_prompt = dspy.Predict(HypothesisToEditPrompt)  # type: ignore[union-attr]
        _delta_evaluation = dspy.ChainOfThought(MetricsDeltaEvaluation)  # type: ignore[union-attr]
    return _bottleneck_to_hypothesis, _hypothesis_to_edit_prompt, _delta_evaluation
