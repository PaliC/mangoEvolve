/**
 * Circle extraction using Python subprocess
 */

import type { CircleData } from "../types";

// Python script for circle extraction
const PYTHON_SCRIPT = `
import sys
import json

sys.path.insert(0, sys.argv[1] + '/src')

from mango_evolve.evaluation.circle_packing import run_code_with_timeout

code = sys.stdin.read()

try:
    centers, radii, sum_radii, error = run_code_with_timeout(
        code,
        timeout_seconds=60,
        n_circles=26
    )

    if error or centers is None:
        print(json.dumps({"valid": False, "error": error or "Failed to execute code"}))
    else:
        print(json.dumps({
            "valid": True,
            "centers": centers.tolist(),
            "radii": radii.tolist(),
            "sum_radii": float(sum_radii) if sum_radii else float(sum(radii))
        }))
except Exception as e:
    print(json.dumps({"valid": False, "error": str(e)}))
`;

/**
 * Extract circles from Python code using subprocess
 */
export async function extractCircles(code: string, projectDir: string): Promise<CircleData> {
  try {
    const proc = Bun.spawn(["uv", "run", "python", "-c", PYTHON_SCRIPT, projectDir], {
      cwd: projectDir,
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    });

    // Write code to stdin and close
    proc.stdin.write(code);
    proc.stdin.end();

    // Read output
    const output = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();

    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      console.error("Python stderr:", stderr);
      return { valid: false, error: `Process exited with code ${exitCode}` };
    }

    try {
      return JSON.parse(output.trim()) as CircleData;
    } catch {
      return { valid: false, error: "Failed to parse Python output" };
    }
  } catch (error) {
    return { valid: false, error: String(error) };
  }
}
