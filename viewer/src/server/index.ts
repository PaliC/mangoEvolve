/**
 * MangoEvolve Viewer - Bun Server
 */

import { api } from "./api";
import { join } from "path";

const PORT = parseInt(process.env.PORT || "8080");
const PUBLIC_DIR = join(import.meta.dir, "../../public");

const server = Bun.serve({
  port: PORT,

  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname;

    // API routes
    if (path.startsWith("/api/")) {
      return api.handleRequest(path);
    }

    // Serve static files
    let filePath = join(PUBLIC_DIR, path === "/" ? "index.html" : path);

    const file = Bun.file(filePath);
    if (await file.exists()) {
      return new Response(file);
    }

    // Fallback to index.html for SPA routing
    const indexFile = Bun.file(join(PUBLIC_DIR, "index.html"));
    if (await indexFile.exists()) {
      return new Response(indexFile);
    }

    return new Response("Not Found", { status: 404 });
  },

  error(error) {
    console.error("Server error:", error);
    return new Response("Internal Server Error", { status: 500 });
  },
});

console.log(`
${"=".repeat(50)}
  MangoEvolve Circle Packing Viewer
${"=".repeat(50)}

  Open in browser: http://localhost:${server.port}

  Press Ctrl+C to stop
`);
