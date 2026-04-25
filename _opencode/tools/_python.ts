import { spawn } from "child_process"
import * as path from "path"
import * as fs from "fs"

function findPython(root: string): string {
  const candidates = [
    path.join(root, ".venv", "bin", "python"),
    path.join(root, "backend", ".venv", "bin", "python"),
    "python3",
  ]
  for (const p of candidates) {
    if (p.startsWith("/") || p.startsWith(".")) {
      if (fs.existsSync(p)) return p
    } else {
      return p
    }
  }
  return "python3"
}

/**
 * Read configuration from agent.toml in the project root.
 *
 * Returns an object with:
 *   readonlyPaths  — JSON string for HPC_ASSISTANT_READONLY_PATHS
 *   dataPaths      — JSON string for HPC_ASSISTANT_DATA_PATHS
 *   agentTomlPath  — absolute path to the found agent.toml, or ""
 */
function readAgentToml(root: string): {
  readonlyPaths: string
  dataPaths: string
  agentTomlPath: string
} {
  const empty = { readonlyPaths: "", dataPaths: "", agentTomlPath: "" }
  const tomlPath = path.join(root, "agent.toml")
  if (!fs.existsSync(tomlPath)) return empty

  // We can't easily parse TOML in plain Node without a library, so we use a
  // minimal regex-based extractor for the two arrays we need.  OpenCode plugins
  // run in an isolated sandbox without npm packages, so we cannot import toml.
  try {
    const raw = fs.readFileSync(tomlPath, "utf-8")

    function extractArray(section: string, key: string): string[] {
      // Match [section] ... key = [ ... ] (handles multi-line arrays)
      const sectionRe = new RegExp(`\\[${section}\\][^[]*?${key}\\s*=\\s*\\[([^\\]]*?)\\]`, "s")
      const m = sectionRe.exec(raw)
      if (!m) return []
      return m[1]
        .split(",")
        .map((s) => s.trim().replace(/^["']|["']$/g, ""))
        .filter((s) => s.length > 0)
    }

    const readonly = extractArray("paths", "readonly")
    const data = extractArray("paths", "data")

    return {
      readonlyPaths: readonly.length > 0 ? JSON.stringify(readonly) : "",
      dataPaths: data.length > 0 ? JSON.stringify(data) : "",
      agentTomlPath: tomlPath,
    }
  } catch {
    return empty
  }
}

/**
 * Read the readonly_paths list from the repo-local opencode.json, if present.
 *
 * opencode.json format (user-controlled):
 *   {
 *     "readonly_paths": ["Makefile", "src/generated/**", "*.lock"]
 *   }
 *
 * Returns a JSON-serialised string suitable for HPC_ASSISTANT_READONLY_PATHS,
 * or an empty string when no patterns are configured.
 *
 * @deprecated  Prefer agent.toml; this is kept for backward compatibility.
 */
function readReadonlyPaths(root: string): string {
  const configPath = path.join(root, "opencode.json")
  if (!fs.existsSync(configPath)) return ""
  try {
    const raw = fs.readFileSync(configPath, "utf-8")
    const cfg = JSON.parse(raw) as Record<string, unknown>
    const patterns = cfg["readonly_paths"]
    if (!Array.isArray(patterns) || patterns.length === 0) return ""
    // Validate: keep only non-empty strings
    const strings = patterns.filter((p): p is string => typeof p === "string" && p.length > 0)
    return strings.length > 0 ? JSON.stringify(strings) : ""
  } catch {
    return ""
  }
}

export async function runPythonTool(
  toolName: string,
  args: Record<string, unknown>,
  context: { root: string },
): Promise<unknown> {
  const root = context.root
  const python = findPython(root)
  const pythonPath = path.join(root, "backend", "src")

  // Prefer agent.toml; fall back to opencode.json for readonly_paths
  const agentCfg = readAgentToml(root)
  const readonlyPaths = agentCfg.readonlyPaths || readReadonlyPaths(root)
  const dataPaths = agentCfg.dataPaths
  const agentTomlPath = agentCfg.agentTomlPath

  const env: Record<string, string> = {
    ...process.env as Record<string, string>,
    PYTHONPATH: pythonPath,
    HPC_ASSISTANT_FILESYSTEM_ROOTS:
      process.env.HPC_ASSISTANT_FILESYSTEM_ROOTS || root,
  }
  if (readonlyPaths) {
    env["HPC_ASSISTANT_READONLY_PATHS"] = readonlyPaths
  }
  if (dataPaths) {
    env["HPC_ASSISTANT_DATA_PATHS"] = dataPaths
  }
  if (agentTomlPath) {
    env["HPC_ASSISTANT_AGENT_TOML_PATH"] = agentTomlPath
  }

  const output = await new Promise<{ exitCode: number; stdout: string; stderr: string }>(
    (resolve, reject) => {
      const child = spawn(
        python,
        [
          "-m",
          "hpc_assistant_backend.opencode_tools",
          "run",
          toolName,
          JSON.stringify(args ?? {}),
        ],
        { cwd: root, env },
      )
      let stdout = ""
      let stderr = ""
      child.stdout.on("data", (chunk) => { stdout += String(chunk) })
      child.stderr.on("data", (chunk) => { stderr += String(chunk) })
      child.on("error", reject)
      child.on("close", (code) => { resolve({ exitCode: code ?? 1, stdout, stderr }) })
    },
  )

  if (output.exitCode !== 0) {
    throw new Error(
      output.stderr.trim() || `python tool ${toolName} exited with code ${output.exitCode}`,
    )
  }

  try {
    return JSON.parse(output.stdout)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`python tool ${toolName} returned invalid JSON: ${message}`)
  }
}
