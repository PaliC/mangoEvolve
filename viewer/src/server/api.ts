/**
 * API handlers for MangoEvolve Viewer
 */

import { join, resolve } from "path";
import { readdir, readFile, stat, exists } from "fs/promises";
import type {
  ExperimentMeta,
  ExperimentData,
  TrialSummary,
  TrialData,
  CircleData,
  AllCircles,
  RootLogEntry,
} from "../types";
import { extractCircles } from "./circles";

// Resolve experiments directory relative to project root
const PROJECT_DIR = resolve(import.meta.dir, "../../..");
const EXPERIMENTS_DIR = join(PROJECT_DIR, "experiments");

/**
 * Parse run_id (YYYYMMDD_HHMMSS) to ISO timestamp
 */
function parseRunId(runId: string): string | null {
  try {
    const match = runId.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
    if (!match) return null;
    const [, year, month, day, hour, min, sec] = match;
    return `${year}-${month}-${day}T${hour}:${min}:${sec}`;
  } catch {
    return null;
  }
}

/**
 * Extract experiment metadata from directory name
 */
async function getExperimentMeta(expDir: string): Promise<ExperimentMeta | null> {
  const name = expDir.split("/").pop()!;
  const fullPath = join(EXPERIMENTS_DIR, name);

  try {
    const stats = await stat(fullPath);
    if (!stats.isDirectory()) return null;
  } catch {
    return null;
  }

  // Extract run_id from name (last two underscore-separated parts)
  const parts = name.split("_");
  let runId = "";
  let expName = name;

  if (parts.length >= 2) {
    const lastPart = parts[parts.length - 1];
    const secondLastPart = parts[parts.length - 2];
    if (lastPart.length === 6 && secondLastPart.length === 8) {
      runId = `${secondLastPart}_${lastPart}`;
      expName = parts.slice(0, -2).join("_");
    }
  }

  // Read experiment.json if it exists
  const expJsonPath = join(fullPath, "experiment.json");
  let bestScore: number | null = null;
  let numGenerations = 0;
  let rootModel: string | null = null;

  const hasExpJson = await exists(expJsonPath);
  const hasGenerations = await exists(join(fullPath, "generations"));

  if (hasExpJson) {
    try {
      const data = JSON.parse(await readFile(expJsonPath, "utf-8")) as ExperimentData;
      bestScore = data.best_trial?.sum_radii ?? data.best_trial?.score ?? null;
      numGenerations = data.num_generations;
      // Extract root model from config
      const config = data.config as { root_llm?: { model?: string } };
      rootModel = config?.root_llm?.model ?? null;
    } catch {
      // Ignore parse errors
    }
  }

  return {
    id: name,
    name: expName,
    run_id: runId,
    timestamp: runId ? parseRunId(runId) : null,
    best_score: bestScore,
    num_generations: numGenerations,
    path: fullPath,
    complete: hasExpJson && hasGenerations,
    root_model: rootModel,
  };
}

/**
 * Get list of all experiments
 */
async function getExperiments(): Promise<ExperimentMeta[]> {
  try {
    const entries = await readdir(EXPERIMENTS_DIR);
    const experiments: ExperimentMeta[] = [];

    for (const entry of entries) {
      if (entry.startsWith(".")) continue;
      const meta = await getExperimentMeta(entry);
      if (meta) experiments.push(meta);
    }

    // Sort by run_id descending (newest first)
    experiments.sort((a, b) => b.run_id.localeCompare(a.run_id));
    return experiments;
  } catch {
    return [];
  }
}

/**
 * Get experiment data
 */
async function getExperiment(expId: string): Promise<ExperimentData | null> {
  const expJsonPath = join(EXPERIMENTS_DIR, expId, "experiment.json");
  try {
    const data = await readFile(expJsonPath, "utf-8");
    return JSON.parse(data) as ExperimentData;
  } catch {
    return null;
  }
}

/**
 * Get trials for a generation
 */
async function getGenerationTrials(expId: string, genNum: number): Promise<TrialSummary[]> {
  const genDir = join(EXPERIMENTS_DIR, expId, "generations", `gen_${genNum}`);
  const trials: TrialSummary[] = [];

  try {
    const entries = await readdir(genDir);
    for (const entry of entries) {
      if (!entry.startsWith("trial_") || !entry.endsWith(".json")) continue;
      if (entry === "trial_selections.json") continue;

      try {
        const data = JSON.parse(
          await readFile(join(genDir, entry), "utf-8")
        ) as TrialData;

        trials.push({
          trial_id: data.trial_id,
          generation: data.generation,
          parent_id: data.parent_id,
          metrics: data.metrics,
          reasoning: data.reasoning?.slice(0, 200) || "",
          timestamp: data.timestamp,
        });
      } catch {
        // Skip invalid files
      }
    }
  } catch {
    return [];
  }

  return trials;
}

/**
 * Get full trial data
 */
async function getTrial(expId: string, trialId: string): Promise<TrialData | null> {
  const parts = trialId.split("_");
  if (parts.length < 3) return null;

  const genNum = parts[1];
  const trialPath = join(
    EXPERIMENTS_DIR,
    expId,
    "generations",
    `gen_${genNum}`,
    `${trialId}.json`
  );

  try {
    const data = await readFile(trialPath, "utf-8");
    return JSON.parse(data) as TrialData;
  } catch {
    return null;
  }
}

/**
 * Get circle data for a trial
 */
async function getTrialCircles(expId: string, trialId: string): Promise<CircleData | null> {
  const parts = trialId.split("_");
  if (parts.length < 3) return null;

  const genNum = parts[1];
  const genDir = join(EXPERIMENTS_DIR, expId, "generations", `gen_${genNum}`);
  const cachePath = join(genDir, `${trialId}.circles.json`);
  const trialPath = join(genDir, `${trialId}.json`);

  // Check cache first
  try {
    if (await exists(cachePath)) {
      const cached = await readFile(cachePath, "utf-8");
      return JSON.parse(cached) as CircleData;
    }
  } catch {
    // Cache miss or parse error
  }

  // Load trial and check if valid
  let trial: TrialData;
  try {
    trial = JSON.parse(await readFile(trialPath, "utf-8")) as TrialData;
  } catch {
    return null;
  }

  if (!trial.metrics?.valid) {
    return { valid: false, error: trial.metrics?.error || "Invalid trial" };
  }

  // Extract circles using Python
  const result = await extractCircles(trial.code, PROJECT_DIR);

  // Cache the result
  try {
    await Bun.write(cachePath, JSON.stringify(result));
  } catch {
    // Ignore cache write errors
  }

  return result;
}

/**
 * Get all circles for an experiment
 */
async function getAllCircles(expId: string): Promise<AllCircles> {
  const genDir = join(EXPERIMENTS_DIR, expId, "generations");
  const results: AllCircles = {};

  try {
    const genDirs = await readdir(genDir);

    // Collect all trial IDs
    const trialIds: string[] = [];
    for (const gen of genDirs) {
      if (!gen.startsWith("gen_")) continue;

      const entries = await readdir(join(genDir, gen));
      for (const entry of entries) {
        if (entry.startsWith("trial_") && entry.endsWith(".json") && entry !== "trial_selections.json") {
          trialIds.push(entry.replace(".json", ""));
        }
      }
    }

    // Process trials in parallel (limited concurrency)
    const CONCURRENCY = 4;
    for (let i = 0; i < trialIds.length; i += CONCURRENCY) {
      const batch = trialIds.slice(i, i + CONCURRENCY);
      const promises = batch.map(async (trialId) => {
        const circles = await getTrialCircles(expId, trialId);
        if (circles) {
          results[trialId] = circles;
        }
      });
      await Promise.all(promises);
    }
  } catch {
    // Return partial results
  }

  return results;
}

/**
 * Get root LLM log
 */
async function getRootLog(expId: string): Promise<RootLogEntry[]> {
  const logPath = join(EXPERIMENTS_DIR, expId, "root_llm_log.jsonl");
  const entries: RootLogEntry[] = [];

  try {
    const content = await readFile(logPath, "utf-8");
    for (const line of content.split("\n")) {
      if (line.trim()) {
        try {
          entries.push(JSON.parse(line) as RootLogEntry);
        } catch {
          // Skip invalid lines
        }
      }
    }
  } catch {
    return [];
  }

  return entries;
}

/**
 * JSON response helper
 */
function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * Handle API requests
 */
async function handleRequest(path: string): Promise<Response> {
  // GET /api/experiments
  if (path === "/api/experiments") {
    return jsonResponse(await getExperiments());
  }

  // GET /api/experiment/:id/root-log
  if (path.endsWith("/root-log")) {
    const expId = path.split("/")[3];
    return jsonResponse(await getRootLog(expId));
  }

  // GET /api/experiment/:id/all-circles
  if (path.endsWith("/all-circles")) {
    const expId = path.split("/")[3];
    return jsonResponse(await getAllCircles(expId));
  }

  // GET /api/experiment/:id/trial/:trialId/circles
  if (path.includes("/trial/") && path.endsWith("/circles")) {
    const parts = path.split("/");
    const expId = parts[3];
    const trialId = parts[5];
    const result = await getTrialCircles(expId, trialId);
    return result ? jsonResponse(result) : jsonResponse({ error: "Not found" }, 404);
  }

  // GET /api/experiment/:id/trial/:trialId
  if (path.includes("/trial/")) {
    const parts = path.split("/");
    const expId = parts[3];
    const trialId = parts[5];
    const result = await getTrial(expId, trialId);
    return result ? jsonResponse(result) : jsonResponse({ error: "Not found" }, 404);
  }

  // GET /api/experiment/:id/gen/:genNum
  if (path.includes("/gen/")) {
    const parts = path.split("/");
    const expId = parts[3];
    const genNum = parseInt(parts[5]);
    return jsonResponse(await getGenerationTrials(expId, genNum));
  }

  // GET /api/experiment/:id
  if (path.startsWith("/api/experiment/")) {
    const expId = path.split("/")[3];
    const result = await getExperiment(expId);
    return result ? jsonResponse(result) : jsonResponse({ error: "Not found" }, 404);
  }

  return jsonResponse({ error: "Not found" }, 404);
}

export const api = {
  handleRequest,
  getExperiments,
  getExperiment,
  getGenerationTrials,
  getTrial,
  getTrialCircles,
  getAllCircles,
  getRootLog,
};
