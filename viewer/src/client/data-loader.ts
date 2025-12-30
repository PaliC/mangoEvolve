/**
 * Data loading utilities for MangoEvolve Viewer
 */

import type {
  ExperimentMeta,
  ExperimentData,
  TrialSummary,
  TrialData,
  CircleData,
  AllCircles,
  RootLogEntry,
  LineageNode,
} from "../types";

class DataLoaderClass {
  private cache = new Map<string, unknown>();

  /**
   * Fetch with caching
   */
  async fetchJSON<T>(url: string, useCache = true): Promise<T> {
    if (useCache && this.cache.has(url)) {
      return this.cache.get(url) as T;
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    if (useCache) {
      this.cache.set(url, data);
    }
    return data as T;
  }

  /**
   * Get list of experiments
   */
  async getExperiments(): Promise<ExperimentMeta[]> {
    return this.fetchJSON("/api/experiments");
  }

  /**
   * Get experiment data
   */
  async getExperiment(expId: string): Promise<ExperimentData> {
    return this.fetchJSON(`/api/experiment/${expId}`);
  }

  /**
   * Get trials for a generation
   */
  async getGenerationTrials(expId: string, genNum: number): Promise<TrialSummary[]> {
    return this.fetchJSON(`/api/experiment/${expId}/gen/${genNum}`);
  }

  /**
   * Get full trial data
   */
  async getTrial(expId: string, trialId: string): Promise<TrialData> {
    return this.fetchJSON(`/api/experiment/${expId}/trial/${trialId}`);
  }

  /**
   * Get circle coordinates for a trial
   */
  async getTrialCircles(expId: string, trialId: string): Promise<CircleData> {
    return this.fetchJSON(`/api/experiment/${expId}/trial/${trialId}/circles`, false);
  }

  /**
   * Get all circle coordinates for an experiment (preload)
   */
  async getAllCircles(expId: string): Promise<AllCircles> {
    return this.fetchJSON(`/api/experiment/${expId}/all-circles`, false);
  }

  /**
   * Get root LLM log
   */
  async getRootLog(expId: string): Promise<RootLogEntry[]> {
    return this.fetchJSON(`/api/experiment/${expId}/root-log`);
  }

  /**
   * Format date from timestamp
   */
  formatDate(timestamp: string | null): string {
    if (!timestamp) return "Unknown";
    try {
      const date = new Date(timestamp);
      return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return timestamp;
    }
  }

  /**
   * Format score
   */
  formatScore(score: number | null | undefined): string {
    if (score === null || score === undefined) return "-";
    return score.toFixed(4);
  }

  /**
   * Parse trial ID to get generation number
   */
  parseTrialId(trialId: string): { generation: number; index: number } | null {
    const parts = trialId.split("_");
    if (parts.length >= 3) {
      return {
        generation: parseInt(parts[1]),
        index: parseInt(parts[2]),
      };
    }
    return null;
  }

  /**
   * Build lineage tree from trials
   */
  buildLineageTree(allTrials: TrialSummary[]): LineageNode[] {
    const trialMap = new Map<string, LineageNode>();
    const roots: LineageNode[] = [];

    // Index all trials
    for (const trial of allTrials) {
      trialMap.set(trial.trial_id, {
        ...trial,
        children: [],
      });
    }

    // Build tree structure
    for (const trial of allTrials) {
      const node = trialMap.get(trial.trial_id)!;
      const parentId = trial.parent_id;

      if (!parentId || parentId === "root" || parentId === "null" || !trialMap.has(parentId)) {
        roots.push(node);
      } else {
        const parent = trialMap.get(parentId);
        if (parent) {
          parent.children.push(node);
        }
      }
    }

    // Sort children by score
    const sortChildren = (node: LineageNode): void => {
      node.children.sort((a, b) => {
        const scoreA = a.metrics?.sum_radii ?? a.metrics?.score ?? 0;
        const scoreB = b.metrics?.sum_radii ?? b.metrics?.score ?? 0;
        return scoreB - scoreA;
      });
      for (const child of node.children) {
        sortChildren(child);
      }
    };

    for (const root of roots) {
      sortChildren(root);
    }

    // Sort roots by score
    roots.sort((a, b) => {
      const scoreA = a.metrics?.sum_radii ?? a.metrics?.score ?? 0;
      const scoreB = b.metrics?.sum_radii ?? b.metrics?.score ?? 0;
      return scoreB - scoreA;
    });

    return roots;
  }
}

export const DataLoader = new DataLoaderClass();
