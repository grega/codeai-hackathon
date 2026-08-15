/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import { resolve } from "node:path";

const root = import.meta.dirname;

// Library build: bundles src/pipeline.ts (and its @techstark/opencv-js
// dependency) into a single ES module the main app's static/js loads
// directly via a plain <script type="module">/dynamic import — no bundler
// involved at request time, only here at build time.
export default defineConfig({
  build: {
    outDir: resolve(root, "../static/js/pipeline"),
    emptyOutDir: true,
    lib: {
      entry: resolve(root, "src/pipeline.ts"),
      formats: ["es"],
      fileName: () => "pipeline.js",
    },
  },
  test: {
    // Default to a plain Node environment: stage-level tests construct
    // cv.Mats directly from decoded pixel buffers and never touch a canvas.
    // Tests that need a real canvas (toOutputPng, the full pipeline e2e
    // test) opt into jsdom via a `// @vitest-environment jsdom` comment.
    environment: "node",
    include: ["src/**/__tests__/**/*.spec.ts"],
    setupFiles: ["src/__tests__/setup/canvasPolyfill.ts"],
    server: {
      deps: {
        // Force this CJS/UMD package through Vite's transform instead of
        // Node's native ESM interop, which mishandles its self-referential
        // `module.exports.default = cv` shape.
        inline: ["@techstark/opencv-js"],
      },
    },
  },
});
