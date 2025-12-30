/**
 * Shared types for MangoEvolve Viewer
 */

// Experiment metadata (from experiments list)
export interface ExperimentMeta {
  id: string;
  name: string;
  run_id: string;
  timestamp: string | null;
  best_score: number | null;
  num_generations: number;
  path: string;
  complete: boolean;
  root_model: string | null;
}

// Trial metrics
export interface TrialMetrics {
  valid: boolean;
  sum_radii?: number;
  score?: number;
  target_ratio?: number;
  combined_score?: number;
  eval_time: number;
  error?: string | null;
}

// Trial selection info
export interface TrialSelection {
  trial_id: string;
  reasoning: string;
  category: "performance" | "diversity" | "potential";
}

// Generation summary (from experiment.json)
export interface GenerationSummary {
  generation_num: number;
  num_trials: number;
  num_successful_trials?: number;
  best_trial_id: string | null;
  best_sum_radii?: number;
  best_score?: number;
  selected_trial_ids: string[];
  selection_reasoning: string;
  trial_selections?: TrialSelection[];
  timestamp: string;
}

// Best trial info
export interface BestTrial {
  trial_id: string;
  sum_radii?: number;
  score?: number;
  generation: number;
}

// Full experiment data (from experiment.json)
export interface ExperimentData {
  experiment_id: string;
  config: Record<string, unknown>;
  start_time: string;
  end_time: string | null;
  termination_reason: string | null;
  num_generations: number;
  total_trials: number;
  best_trial: BestTrial | null;
  generations: GenerationSummary[];
  scratchpad?: string;
}

// Trial summary (for list view)
export interface TrialSummary {
  trial_id: string;
  generation: number;
  parent_id: string | null;
  metrics: TrialMetrics;
  reasoning: string;
  timestamp: string;
}

// Full trial data
export interface TrialData {
  trial_id: string;
  generation: number;
  parent_id: string | null;
  code: string;
  metrics: TrialMetrics;
  prompt: string;
  response: string;
  reasoning: string;
  timestamp: string;
  cost_data?: unknown;
}

// Circle packing data
export interface CircleData {
  valid: boolean;
  centers?: number[][];  // [n, 2] array
  radii?: number[];      // [n] array
  sum_radii?: number;
  error?: string;
}

// All circles for an experiment
export type AllCircles = Record<string, CircleData>;

// Root LLM log entry
export interface RootLogEntry {
  turn: number;
  role: "assistant" | "user" | "system";
  content: string;
  code_executed?: string | null;
  execution_result?: string | null;
  timestamp: string;
}

// Lineage tree node (for UI)
export interface LineageNode extends TrialSummary {
  children: LineageNode[];
}
