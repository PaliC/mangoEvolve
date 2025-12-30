/**
 * MangoEvolve Viewer - Main Application
 */

import { DataLoader } from "./data-loader";
import { CircleVis, type CircleInfo } from "./circle-vis";
import type {
  ExperimentMeta,
  ExperimentData,
  TrialSummary,
  TrialData,
  AllCircles,
  RootLogEntry,
} from "../types";

// Declare external libraries
declare const marked: { parse: (text: string) => string };
declare const hljs: { highlightElement: (el: Element) => void };

interface AppState {
  experiments: ExperimentMeta[];
  currentExperiment: string | null;
  currentExperimentData: ExperimentData | null;
  currentTrial: TrialData | null;
  allTrials: TrialSummary[];
  allCircles: AllCircles;
  circlesLoading: boolean;
  rootLogLoaded: boolean;
  // Log navigation state
  rootLog: RootLogEntry[];
  logFilter: string;
  logIndex: number;
  logViewAll: boolean;
  // Model filter state
  modelFilter: string;
}

interface AppElements {
  experimentsList: HTMLElement;
  noExperiment: HTMLElement;
  experimentView: HTMLElement;
  experimentName: HTMLElement;
  experimentDate: HTMLElement;
  experimentBestScore: HTMLElement;
  experimentGenerations: HTMLElement;
  tabs: NodeListOf<HTMLElement>;
  tabPanels: NodeListOf<HTMLElement>;
  generationsList: HTMLElement;
  circleCanvas: HTMLElement;
  circleInfo: HTMLElement;
  circlesLoading: HTMLElement;
  trialPlaceholder: HTMLElement;
  trialDetails: HTMLElement;
  trialTitle: HTMLElement;
  trialScore: HTMLElement;
  trialValid: HTMLElement;
  trialParent: HTMLElement;
  detailTabs: NodeListOf<HTMLElement>;
  detailPanels: NodeListOf<HTMLElement>;
  promptContent: HTMLElement;
  responseContent: HTMLElement;
  codeContent: HTMLElement;
  lineageTree: HTMLElement;
  rootLogContainer: HTMLElement;
  // Log navigation elements
  logFilters: NodeListOf<HTMLElement>;
  logPrev: HTMLElement;
  logNext: HTMLElement;
  logPosition: HTMLElement;
  logViewAll: HTMLElement;
  // Model filter element
  modelSelect: HTMLSelectElement;
}

class App {
  private state: AppState = {
    experiments: [],
    currentExperiment: null,
    currentExperimentData: null,
    currentTrial: null,
    allTrials: [],
    allCircles: {},
    circlesLoading: false,
    rootLogLoaded: false,
    // Log navigation state
    rootLog: [],
    logFilter: "all",
    logIndex: 0,
    logViewAll: true,
    // Model filter state
    modelFilter: "all",
  };

  private elements!: AppElements;

  /**
   * Initialize the app
   */
  async init(): Promise<void> {
    this.cacheElements();
    this.bindEvents();
    await this.loadExperiments();
  }

  /**
   * Cache DOM elements
   */
  private cacheElements(): void {
    this.elements = {
      experimentsList: document.getElementById("experiments-list")!,
      noExperiment: document.getElementById("no-experiment")!,
      experimentView: document.getElementById("experiment-view")!,
      experimentName: document.getElementById("experiment-name")!,
      experimentDate: document.getElementById("experiment-date")!,
      experimentBestScore: document.getElementById("experiment-best-score")!,
      experimentGenerations: document.getElementById("experiment-generations")!,
      tabs: document.querySelectorAll(".tab") as NodeListOf<HTMLElement>,
      tabPanels: document.querySelectorAll(".tab-panel") as NodeListOf<HTMLElement>,
      generationsList: document.getElementById("generations-list")!,
      circleCanvas: document.getElementById("circle-canvas")!,
      circleInfo: document.getElementById("circle-info")!,
      circlesLoading: document.getElementById("circles-loading")!,
      trialPlaceholder: document.getElementById("trial-placeholder")!,
      trialDetails: document.getElementById("trial-details")!,
      trialTitle: document.getElementById("trial-title")!,
      trialScore: document.getElementById("trial-score")!,
      trialValid: document.getElementById("trial-valid")!,
      trialParent: document.getElementById("trial-parent")!,
      detailTabs: document.querySelectorAll(".detail-tab") as NodeListOf<HTMLElement>,
      detailPanels: document.querySelectorAll(".detail-panel") as NodeListOf<HTMLElement>,
      promptContent: document.getElementById("prompt-content")!,
      responseContent: document.getElementById("response-content")!,
      codeContent: document.getElementById("code-content")!,
      lineageTree: document.getElementById("lineage-tree")!,
      rootLogContainer: document.getElementById("root-log-container")!,
      // Log navigation elements
      logFilters: document.querySelectorAll(".log-filter") as NodeListOf<HTMLElement>,
      logPrev: document.getElementById("log-prev")!,
      logNext: document.getElementById("log-next")!,
      logPosition: document.getElementById("log-position")!,
      logViewAll: document.getElementById("log-view-all")!,
      // Model filter element
      modelSelect: document.getElementById("model-select") as HTMLSelectElement,
    };
  }

  /**
   * Bind event handlers
   */
  private bindEvents(): void {
    // Main tabs
    this.elements.tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const tabId = tab.dataset.tab;
        if (tabId) this.switchTab(tabId);
      });
    });

    // Detail tabs
    this.elements.detailTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const detailId = tab.dataset.detail;
        if (detailId) this.switchDetailTab(detailId);
      });
    });

    // Log filter buttons
    this.elements.logFilters.forEach((filter) => {
      filter.addEventListener("click", () => {
        const filterType = filter.dataset.filter;
        if (filterType) this.setLogFilter(filterType);
      });
    });

    // Log navigation buttons
    this.elements.logPrev.addEventListener("click", () => this.navigateLog(-1));
    this.elements.logNext.addEventListener("click", () => this.navigateLog(1));
    this.elements.logViewAll.addEventListener("click", () => this.toggleLogViewAll());

    // Model filter
    this.elements.modelSelect.addEventListener("change", () => {
      this.state.modelFilter = this.elements.modelSelect.value;
      this.renderExperimentsList();
    });
  }

  /**
   * Load experiments list
   */
  private async loadExperiments(): Promise<void> {
    try {
      this.state.experiments = await DataLoader.getExperiments();
      this.populateModelFilter();
      this.renderExperimentsList();
    } catch (error) {
      console.error("Failed to load experiments:", error);
      this.elements.experimentsList.innerHTML = `
        <div class="loading" style="color: var(--error);">
          Failed to load experiments
        </div>
      `;
    }
  }

  /**
   * Populate model filter dropdown
   */
  private populateModelFilter(): void {
    // Get unique models from complete experiments
    const models = new Set<string>();
    for (const exp of this.state.experiments) {
      if (exp.complete && exp.root_model) {
        models.add(exp.root_model);
      }
    }

    // Sort models alphabetically
    const sortedModels = Array.from(models).sort();

    // Build options HTML
    let html = '<option value="all">All Models</option>';
    for (const model of sortedModels) {
      // Show shortened model name in display
      const displayName = this.shortenModelName(model);
      html += `<option value="${model}">${displayName}</option>`;
    }

    this.elements.modelSelect.innerHTML = html;
  }

  /**
   * Shorten model name for display
   */
  private shortenModelName(model: string): string {
    // Remove provider prefix (e.g., "google/gemini-3-flash" -> "gemini-3-flash")
    const parts = model.split("/");
    return parts.length > 1 ? parts[parts.length - 1] : model;
  }

  /**
   * Render experiments list in sidebar
   */
  private renderExperimentsList(): void {
    // Filter to only show complete experiments, and optionally by model
    let filteredExperiments = this.state.experiments.filter((exp) => exp.complete);

    if (this.state.modelFilter !== "all") {
      filteredExperiments = filteredExperiments.filter(
        (exp) => exp.root_model === this.state.modelFilter
      );
    }

    const html = filteredExperiments
      .map(
        (exp) => `
        <div class="experiment-item" data-id="${exp.id}">
          <div class="name">${exp.name}</div>
          <div class="meta">
            <span>${DataLoader.formatDate(exp.timestamp)}</span>
            <span class="score">${exp.best_score ? DataLoader.formatScore(exp.best_score) : "-"}</span>
          </div>
        </div>
      `
      )
      .join("");

    this.elements.experimentsList.innerHTML =
      html || '<div class="loading">No experiments found</div>';

    // Add click handlers
    this.elements.experimentsList.querySelectorAll(".experiment-item").forEach((item) => {
      item.addEventListener("click", () => {
        const id = (item as HTMLElement).dataset.id;
        if (id) this.selectExperiment(id);
      });
    });
  }

  /**
   * Select an experiment
   */
  private async selectExperiment(expId: string): Promise<void> {
    // Update UI
    this.elements.experimentsList.querySelectorAll(".experiment-item").forEach((item) => {
      item.classList.toggle("active", (item as HTMLElement).dataset.id === expId);
    });

    this.elements.noExperiment.classList.add("hidden");
    this.elements.experimentView.classList.remove("hidden");

    // Load experiment data
    try {
      this.state.currentExperiment = expId;
      this.state.currentExperimentData = await DataLoader.getExperiment(expId);
      this.state.rootLogLoaded = false;
      this.state.allTrials = [];
      this.state.allCircles = {};
      this.state.circlesLoading = true;
      // Reset log state
      this.state.rootLog = [];
      this.state.logFilter = "all";
      this.state.logIndex = 0;
      this.state.logViewAll = true;

      this.renderExperimentHeader();
      await this.loadAllTrials();
      this.renderGenerations();

      // Reset to generations tab
      this.switchTab("generations");

      // Preload all circles in background
      this.preloadCircles(expId);
    } catch (error) {
      console.error("Failed to load experiment:", error);
    }
  }

  /**
   * Render experiment header
   */
  private renderExperimentHeader(): void {
    const data = this.state.currentExperimentData!;
    const meta = this.state.experiments.find((e) => e.id === this.state.currentExperiment);

    this.elements.experimentName.textContent = meta?.name || this.state.currentExperiment || "";
    this.elements.experimentDate.textContent = DataLoader.formatDate(data.start_time);

    const bestScore = data.best_trial?.sum_radii ?? data.best_trial?.score;
    this.elements.experimentBestScore.textContent = bestScore
      ? DataLoader.formatScore(bestScore)
      : "-";
    this.elements.experimentGenerations.textContent = `${data.num_generations} generations`;
  }

  /**
   * Load all trials for the experiment
   */
  private async loadAllTrials(): Promise<void> {
    const data = this.state.currentExperimentData!;
    const allTrials: TrialSummary[] = [];

    for (let gen = 0; gen < data.num_generations; gen++) {
      try {
        const trials = await DataLoader.getGenerationTrials(this.state.currentExperiment!, gen);
        allTrials.push(...trials);
      } catch (error) {
        console.error(`Failed to load gen ${gen} trials:`, error);
      }
    }

    this.state.allTrials = allTrials;
  }

  /**
   * Render generations view
   */
  private renderGenerations(): void {
    const data = this.state.currentExperimentData!;
    const generations = data.generations || [];

    const html = generations
      .map((gen, idx) => {
        const genTrials = this.state.allTrials.filter((t) => t.generation === idx);
        const bestTrialId = gen.best_trial_id;
        const bestScore = gen.best_sum_radii ?? gen.best_score;

        return `
          <div class="generation-card" data-gen="${idx}">
            <div class="generation-header">
              <h4>Gen ${idx}</h4>
              <span class="best-score">${bestScore ? DataLoader.formatScore(bestScore) : "-"}</span>
            </div>
            <div class="trials-grid">
              ${genTrials
                .map((trial) => {
                  const isBest = trial.trial_id === bestTrialId;
                  const isValid = trial.metrics?.valid !== false;
                  const classes = ["trial-chip"];
                  if (isBest) classes.push("best");
                  if (!isValid) classes.push("invalid");

                  return `<span class="${classes.join(" ")}" data-trial="${trial.trial_id}">${trial.trial_id.split("_")[2]}</span>`;
                })
                .join("")}
            </div>
          </div>
        `;
      })
      .join("");

    this.elements.generationsList.innerHTML = html;

    // Add click handlers
    this.elements.generationsList.querySelectorAll(".trial-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const trialId = (chip as HTMLElement).dataset.trial;
        if (trialId) this.selectTrial(trialId);
      });
    });
  }

  /**
   * Select a trial
   */
  private async selectTrial(trialId: string): Promise<void> {
    // Update UI
    this.elements.generationsList.querySelectorAll(".trial-chip").forEach((chip) => {
      chip.classList.toggle("active", (chip as HTMLElement).dataset.trial === trialId);
    });

    this.elements.trialPlaceholder.classList.add("hidden");
    this.elements.trialDetails.classList.remove("hidden");

    try {
      // Load full trial data
      const trial = await DataLoader.getTrial(this.state.currentExperiment!, trialId);
      this.state.currentTrial = trial;
      this.renderTrialDetails(trial);

      // Render circle visualization (uses preloaded data if available)
      this.renderCircles(trialId);
    } catch (error) {
      console.error("Failed to load trial:", error);
    }
  }

  /**
   * Render trial details
   */
  private renderTrialDetails(trial: TrialData): void {
    this.elements.trialTitle.textContent = trial.trial_id;

    const score = trial.metrics?.sum_radii ?? trial.metrics?.score ?? 0;
    this.elements.trialScore.textContent = `Score: ${DataLoader.formatScore(score)}`;

    const isValid = trial.metrics?.valid !== false;
    this.elements.trialValid.textContent = isValid ? "Valid" : "Invalid";
    this.elements.trialValid.className = isValid ? "valid" : "invalid";

    // Parent link
    if (trial.parent_id && trial.parent_id !== "root" && trial.parent_id !== "null") {
      this.elements.trialParent.innerHTML = `Parent: <a data-trial="${trial.parent_id}">${trial.parent_id}</a>`;
      const link = this.elements.trialParent.querySelector("a");
      if (link) {
        link.addEventListener("click", (e) => {
          e.preventDefault();
          this.selectTrial(trial.parent_id!);
        });
      }
    } else {
      this.elements.trialParent.innerHTML = "Parent: root";
    }

    // Render content
    this.elements.promptContent.innerHTML = marked.parse(trial.prompt || "*No prompt*");
    this.elements.responseContent.innerHTML = marked.parse(trial.response || "*No response*");
    this.elements.codeContent.textContent = trial.code || "# No code";

    // Highlight code
    hljs.highlightElement(this.elements.codeContent);
  }

  /**
   * Preload all circles for an experiment
   */
  private async preloadCircles(expId: string): Promise<void> {
    // Show loading indicator
    this.elements.circlesLoading.classList.remove("hidden");

    try {
      console.log("Preloading circles...");
      const circles = await DataLoader.getAllCircles(expId);
      // Only update if still on the same experiment
      if (this.state.currentExperiment === expId) {
        this.state.allCircles = circles;
        this.state.circlesLoading = false;
        console.log(`Preloaded ${Object.keys(circles).length} circle packings`);

        // Hide loading indicator
        this.elements.circlesLoading.classList.add("hidden");

        // If we have a current trial selected, re-render with cached data
        if (this.state.currentTrial) {
          this.renderCircles(this.state.currentTrial.trial_id);
        }
      }
    } catch (error) {
      console.error("Failed to preload circles:", error);
      this.state.circlesLoading = false;
      this.elements.circlesLoading.classList.add("hidden");
    }
  }

  /**
   * Render circles from cache or show loading
   */
  private renderCircles(trialId: string): void {
    const circles = this.state.allCircles[trialId];

    if (circles) {
      if (circles.valid) {
        CircleVis.render(this.elements.circleCanvas, circles, (info: CircleInfo | null) => {
          this.elements.circleInfo.textContent = info
            ? CircleVis.formatInfo(info)
            : `Sum of radii: ${DataLoader.formatScore(circles.sum_radii)}`;
        });
        this.elements.circleInfo.textContent = `Sum of radii: ${DataLoader.formatScore(circles.sum_radii)}`;
      } else {
        CircleVis.renderError(this.elements.circleCanvas, circles.error);
      }
    } else if (this.state.circlesLoading) {
      CircleVis.renderLoading(this.elements.circleCanvas);
      this.elements.circleInfo.textContent = "Loading circles...";
    } else {
      // Circles not available and not loading - try to fetch individually
      this.loadTrialCircles(trialId);
    }
  }

  /**
   * Load and render circle visualization (fallback for individual fetch)
   */
  private async loadTrialCircles(trialId: string): Promise<void> {
    CircleVis.renderLoading(this.elements.circleCanvas);
    this.elements.circleInfo.textContent = "";

    try {
      const circles = await DataLoader.getTrialCircles(this.state.currentExperiment!, trialId);

      // Cache the result
      this.state.allCircles[trialId] = circles;

      if (circles.valid) {
        CircleVis.render(this.elements.circleCanvas, circles, (info: CircleInfo | null) => {
          this.elements.circleInfo.textContent = info
            ? CircleVis.formatInfo(info)
            : `Sum of radii: ${DataLoader.formatScore(circles.sum_radii)}`;
        });
        this.elements.circleInfo.textContent = `Sum of radii: ${DataLoader.formatScore(circles.sum_radii)}`;
      } else {
        CircleVis.renderError(this.elements.circleCanvas, circles.error);
      }
    } catch (error) {
      console.error("Failed to load circles:", error);
      CircleVis.renderError(this.elements.circleCanvas, String(error));
    }
  }

  /**
   * Switch main tab
   */
  private switchTab(tabId: string): void {
    // Update tab buttons
    this.elements.tabs.forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.tab === tabId);
    });

    // Update tab panels
    this.elements.tabPanels.forEach((panel) => {
      panel.classList.toggle("active", panel.id === `tab-${tabId}`);
    });

    // Load content if needed
    if (tabId === "lineage" && this.state.allTrials.length > 0) {
      this.renderLineageTree();
    } else if (tabId === "root-log" && !this.state.rootLogLoaded) {
      this.loadRootLog();
    }
  }

  /**
   * Switch detail tab
   */
  private switchDetailTab(detailId: string): void {
    this.elements.detailTabs.forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.detail === detailId);
    });

    this.elements.detailPanels.forEach((panel) => {
      panel.classList.toggle("active", panel.id === `detail-${detailId}`);
    });
  }

  /**
   * Render lineage tree
   */
  private renderLineageTree(): void {
    const bestTrialId = this.state.currentExperimentData?.best_trial?.trial_id;
    const roots = DataLoader.buildLineageTree(this.state.allTrials);

    const renderNode = (node: TrialSummary & { children: TrialSummary[] }): string => {
      const score = node.metrics?.sum_radii ?? node.metrics?.score ?? 0;
      const isValid = node.metrics?.valid !== false;
      const isBest = node.trial_id === bestTrialId;

      const classes = ["lineage-node-content"];
      if (!isValid) classes.push("invalid");
      if (isBest) classes.push("best");

      let html = `
        <div class="lineage-node">
          <div class="${classes.join(" ")}" data-trial="${node.trial_id}">
            <div class="trial-id">${node.trial_id}${isBest ? " *" : ""}</div>
            <div class="node-score">${isValid ? DataLoader.formatScore(score) : "Invalid"}</div>
          </div>
        </div>
      `;

      if (node.children && node.children.length > 0) {
        html += `<div class="lineage-children">`;
        for (const child of node.children) {
          html += renderNode(child as TrialSummary & { children: TrialSummary[] });
        }
        html += `</div>`;
      }

      return html;
    };

    const html = roots
      .map(
        (root) => `
        <div class="lineage-root">
          ${renderNode(root)}
        </div>
      `
      )
      .join("");

    this.elements.lineageTree.innerHTML =
      html || '<div class="placeholder-small">No lineage data</div>';

    // Add click handlers
    this.elements.lineageTree.querySelectorAll(".lineage-node-content").forEach((node) => {
      node.addEventListener("click", () => {
        // Switch to generations tab and select trial
        this.switchTab("generations");
        const trialId = (node as HTMLElement).dataset.trial;
        if (trialId) this.selectTrial(trialId);
      });
    });
  }

  /**
   * Load root LLM log
   */
  private async loadRootLog(): Promise<void> {
    this.elements.rootLogContainer.innerHTML =
      '<div class="loading">Loading conversation...</div>';

    try {
      const log = await DataLoader.getRootLog(this.state.currentExperiment!);
      this.state.rootLogLoaded = true;
      this.state.rootLog = log || [];
      this.state.logIndex = 0;
      this.state.logFilter = "all";
      this.state.logViewAll = true;

      // Update filter button states
      this.elements.logFilters.forEach((f) => {
        f.classList.toggle("active", f.dataset.filter === "all");
      });
      this.elements.logViewAll.classList.add("active");

      this.renderRootLog();
    } catch (error) {
      console.error("Failed to load root log:", error);
      this.elements.rootLogContainer.innerHTML = `
        <div class="placeholder-small" style="color: var(--error);">
          Failed to load log: ${error}
        </div>
      `;
    }
  }

  /**
   * Get filtered log entries
   */
  private getFilteredLog(): RootLogEntry[] {
    if (this.state.logFilter === "all") {
      return this.state.rootLog;
    }
    return this.state.rootLog.filter((entry) => entry.role === this.state.logFilter);
  }

  /**
   * Set log filter
   */
  private setLogFilter(filter: string): void {
    this.state.logFilter = filter;
    this.state.logIndex = 0;

    // Update filter button states
    this.elements.logFilters.forEach((f) => {
      f.classList.toggle("active", f.dataset.filter === filter);
    });

    this.renderRootLog();
  }

  /**
   * Navigate log entries
   */
  private navigateLog(direction: number): void {
    const filtered = this.getFilteredLog();
    if (filtered.length === 0) return;

    // If in view all mode, switch to single view
    if (this.state.logViewAll) {
      this.state.logViewAll = false;
      this.elements.logViewAll.classList.remove("active");
      this.state.logIndex = direction > 0 ? 0 : filtered.length - 1;
    } else {
      this.state.logIndex = Math.max(0, Math.min(filtered.length - 1, this.state.logIndex + direction));
    }

    this.renderRootLog();
  }

  /**
   * Toggle view all mode
   */
  private toggleLogViewAll(): void {
    this.state.logViewAll = !this.state.logViewAll;
    this.elements.logViewAll.classList.toggle("active", this.state.logViewAll);
    this.renderRootLog();
  }

  /**
   * Render root LLM log
   */
  private renderRootLog(): void {
    const filtered = this.getFilteredLog();

    if (filtered.length === 0) {
      this.elements.rootLogContainer.innerHTML =
        '<div class="placeholder-small">No log entries found</div>';
      this.elements.logPosition.textContent = "- / -";
      return;
    }

    // Update position indicator
    if (this.state.logViewAll) {
      this.elements.logPosition.textContent = `All (${filtered.length})`;
    } else {
      this.elements.logPosition.textContent = `${this.state.logIndex + 1} / ${filtered.length}`;
    }

    // Get entries to render
    const entries = this.state.logViewAll ? filtered : [filtered[this.state.logIndex]];

    const html = entries
      .map((entry) => {
        const role = entry.role || "unknown";
        const content = entry.content || "";

        // Parse markdown in content
        const parsedContent = marked.parse(content);

        // Render execution result if present
        let executionHtml = "";
        if (entry.execution_result) {
          executionHtml = `
            <div class="log-execution-result">
              <h5>Execution Result</h5>
              <pre>${this.escapeHtml(entry.execution_result)}</pre>
            </div>
          `;
        }

        return `
          <div class="log-entry ${role}">
            <div class="log-entry-header">
              <span class="log-entry-role">${role}</span>
              <span class="log-entry-turn">Turn ${entry.turn}</span>
            </div>
            <div class="log-entry-content">${parsedContent}</div>
            ${executionHtml}
          </div>
        `;
      })
      .join("");

    this.elements.rootLogContainer.innerHTML = html;

    // Highlight all code blocks
    this.elements.rootLogContainer.querySelectorAll("pre code").forEach((block) => {
      hljs.highlightElement(block);
    });
  }

  /**
   * Escape HTML
   */
  private escapeHtml(text: string): string {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
}

// Export for bundling
export const app = new App();
