/**
 * onnxruntime-web を CDN から実行時に読む（ndl-layout-viz の ort.ts と同じ）。
 * WebGPU 用の wasm は約 26MB あり、Cloudflare Pages の 1 ファイル 25MiB を超えるため同梱しない。
 * 型だけ npm の onnxruntime-web から取る（実コードはバンドルされない）。
 */
import type * as Ort from "onnxruntime-web";

export type { Ort };
export type OrtModule = typeof Ort;

const VERSION = "1.26.0";
const CDN_DIR = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSION}/dist/`;

let mod: OrtModule | null = null;

export async function loadOrt(): Promise<OrtModule> {
  if (mod) return mod;
  const entry = CDN_DIR + "ort.webgpu.bundle.min.mjs";
  mod = (await import(/* @vite-ignore */ entry)) as OrtModule;
  mod.env.wasm.wasmPaths = CDN_DIR;
  return mod;
}
