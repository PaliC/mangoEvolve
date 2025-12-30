/**
 * Circle packing visualization using SVG
 */

import type { CircleData } from "../types";

export interface CircleInfo {
  index: number;
  x: number;
  y: number;
  radius: number;
}

const NS = "http://www.w3.org/2000/svg";

/**
 * Get color for a circle based on radius
 */
function getColor(radius: number, maxRadius: number): string {
  const ratio = radius / maxRadius;
  // Blue to green to gold gradient
  if (ratio < 0.5) {
    const t = ratio * 2;
    return `hsl(${220 - t * 80}, 70%, ${50 + t * 10}%)`;
  } else {
    const t = (ratio - 0.5) * 2;
    return `hsl(${140 - t * 100}, ${70 + t * 20}%, ${60 + t * 15}%)`;
  }
}

/**
 * Create SVG element
 */
function createSVG(): SVGSVGElement {
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 1 1");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  svg.style.width = "100%";
  svg.style.height = "100%";

  // Background
  const bg = document.createElementNS(NS, "rect");
  bg.setAttribute("x", "0");
  bg.setAttribute("y", "0");
  bg.setAttribute("width", "1");
  bg.setAttribute("height", "1");
  bg.setAttribute("fill", "#1a1a2e");
  svg.appendChild(bg);

  // Unit square boundary
  const boundary = document.createElementNS(NS, "rect");
  boundary.setAttribute("x", "0");
  boundary.setAttribute("y", "0");
  boundary.setAttribute("width", "1");
  boundary.setAttribute("height", "1");
  boundary.setAttribute("fill", "none");
  boundary.setAttribute("stroke", "#3a3a5a");
  boundary.setAttribute("stroke-width", "0.002");
  svg.appendChild(boundary);

  return svg;
}

/**
 * Render circles to container
 */
export function render(
  container: HTMLElement,
  data: CircleData,
  onHover: ((info: CircleInfo | null) => void) | null = null
): void {
  container.innerHTML = "";

  if (!data || !data.valid || !data.centers || !data.radii) {
    container.innerHTML = '<div class="placeholder-small">No valid packing data</div>';
    return;
  }

  const svg = createSVG();
  const centers = data.centers;
  const radii = data.radii;
  const maxRadius = Math.max(...radii);

  // Create group for circles
  const group = document.createElementNS(NS, "g");

  // Add circles
  for (let i = 0; i < centers.length; i++) {
    const [x, y] = centers[i];
    const r = radii[i];

    const circle = document.createElementNS(NS, "circle");
    circle.setAttribute("cx", String(x));
    circle.setAttribute("cy", String(1 - y)); // Flip Y for SVG coordinates
    circle.setAttribute("r", String(r));
    circle.setAttribute("fill", getColor(r, maxRadius));
    circle.setAttribute("fill-opacity", "0.7");
    circle.setAttribute("stroke", "white");
    circle.setAttribute("stroke-width", "0.002");
    circle.setAttribute("stroke-opacity", "0.5");
    circle.style.cursor = "pointer";
    circle.style.transition = "fill-opacity 0.2s";

    // Hover effects
    circle.addEventListener("mouseenter", () => {
      circle.setAttribute("fill-opacity", "1");
      circle.setAttribute("stroke-opacity", "1");
      if (onHover) {
        onHover({ index: i, x, y, radius: r });
      }
    });

    circle.addEventListener("mouseleave", () => {
      circle.setAttribute("fill-opacity", "0.7");
      circle.setAttribute("stroke-opacity", "0.5");
      if (onHover) {
        onHover(null);
      }
    });

    group.appendChild(circle);
  }

  svg.appendChild(group);
  container.appendChild(svg);
}

/**
 * Render loading state
 */
export function renderLoading(container: HTMLElement): void {
  container.innerHTML = '<div class="placeholder-small">Loading circles...</div>';
}

/**
 * Render error state
 */
export function renderError(container: HTMLElement, error: string | null | undefined): void {
  container.innerHTML = `<div class="placeholder-small" style="color: var(--error);">
    Error: ${error || "Failed to load circles"}
  </div>`;
}

/**
 * Format circle info for display
 */
export function formatInfo(info: CircleInfo | null): string {
  if (!info) return "";
  return `Circle ${info.index}: center (${info.x.toFixed(4)}, ${info.y.toFixed(4)}), radius ${info.radius.toFixed(4)}`;
}

export const CircleVis = {
  render,
  renderLoading,
  renderError,
  formatInfo,
};
