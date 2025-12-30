/**
 * MangoEvolve Viewer - Client Entry Point
 */

import { app } from "./app";

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  app.init().catch(console.error);
});
