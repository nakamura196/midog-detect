import { defineConfig } from "vite";

// COOP/COEP は付けない（ndl-layout-viz と同じ判断）。推論は WebGPU で動き、
// WASM のマルチスレッドの恩恵は小さい。付けると静的な配信先で動かしにくい。
export default defineConfig({
  base: "./",
  worker: { format: "es" },
});
