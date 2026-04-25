import { tool } from "@opencode/tool"
import { runPythonTool } from "./_python"

export const collect_profile = tool({
  description: "Collect a performance profile for a completed Slurm job and store it in the experiment store. Optionally merges Nsight Compute (NCU) CSV and/or LIKWID text output. Returns the run_id needed for subsequent pgoa_* calls.",
  args: {
    workload_id: tool.schema.string().describe("Unique workload identifier."),
    job_id: tool.schema.number().int().describe("Slurm job ID."),
    kpi_metric: tool.schema.string().describe("Primary KPI name (e.g. elapsed_s)."),
    kpi_value: tool.schema.string().describe("Current KPI value (numeric string)."),
    kpi_unit: tool.schema.string().describe("Unit for the KPI (e.g. seconds)."),
    run_type: tool.schema.string().describe("\"baseline\" or \"iteration\".").optional(),
    iteration: tool.schema.number().int().describe("Iteration number (if run_type=iteration).").optional(),
    ncu_csv_path: tool.schema.string().describe("Path to Nsight Compute --csv output file.").optional(),
    likwid_output_path: tool.schema.string().describe("Path to likwid-perfctr text output file.").optional(),
    lower_is_better: tool.schema.boolean().describe("True when lower KPI is better.").optional(),
  },
  async execute(args, context) {
    return await runPythonTool("pgoa_collect_profile", args, context)
  },
})

export const analyze = tool({
  description: "Analyze a stored ProfileBundle and return the primary bottleneck (memory_bound_gpu, compute_bound_gpu, mpi_binding, high_rss, etc.) with a recommended action hint.",
  args: {
    workload_id: tool.schema.string().describe("Workload identifier."),
    run_id: tool.schema.string().describe("Run ID returned by pgoa_collect_profile."),
  },
  async execute(args, context) {
    return await runPythonTool("pgoa_analyze", args, context)
  },
})

export const apply_binding = tool({
  description: "Validate Slurm binding parameters and rewrite the job script's #SBATCH directives in-place or to a new output path. Returns the output path and a list of changes applied.",
  args: {
    workload_id: tool.schema.string().describe("Workload identifier."),
    run_id: tool.schema.string().describe("Run ID for provenance tracking."),
    job_script_path: tool.schema.string().describe("Absolute path to source job script."),
    output_script_path: tool.schema.string().describe("Absolute path for the modified script."),
    ntasks_per_node: tool.schema.number().int().describe("Value for --ntasks-per-node.").optional(),
    ntasks_per_socket: tool.schema.number().int().describe("Value for --ntasks-per-socket.").optional(),
    cpus_per_task: tool.schema.number().int().describe("Value for --cpus-per-task.").optional(),
    mem_bind: tool.schema.string().describe("Slurm mem-bind policy (local/none/prefer/bind).").optional(),
    cpu_bind: tool.schema.string().describe("Slurm cpu-bind policy (cores/threads/sockets/rank/none).").optional(),
    exclusive: tool.schema.boolean().describe("Add --exclusive flag.").optional(),
  },
  async execute(args, context) {
    return await runPythonTool("pgoa_apply_binding", args, context)
  },
})

export const compare_runs = tool({
  description: "Compute a KPI delta between two stored runs and classify the change as improved / degraded / neutral. Returns kpi_delta_pct and secondary metric deltas.",
  args: {
    workload_id: tool.schema.string().describe("Workload identifier."),
    from_run_id: tool.schema.string().describe("Baseline run ID."),
    to_run_id: tool.schema.string().describe("Iteration run ID to compare against baseline."),
    action_applied: tool.schema.string().describe("Short description of the change applied."),
  },
  async execute(args, context) {
    return await runPythonTool("pgoa_compare_runs", args, context)
  },
})

export const store_info = tool({
  description: "List all experiment runs for a workload (baseline first, then iterations sorted by iteration number). Returns run metadata without loading profile data.",
  args: {
    workload_id: tool.schema.string().describe("Workload identifier."),
  },
  async execute(args, context) {
    return await runPythonTool("pgoa_store_info", args, context)
  },
})

