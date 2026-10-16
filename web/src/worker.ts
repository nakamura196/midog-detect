/**
 * 推論用の Web Worker。モデルの取得（進み具合つき・ブラウザに保存）と、640 角の切片の推論を受け持つ。
 * ndl-layout-viz の worker.ts を、出力 1 つ（[1, 300, 6]）の形に縮めたもの。
 */
import { loadOrt, type Ort, type OrtModule } from "./ort";

export type ToWorker =
  | { type: "init"; webgpu: string; wasm: string; force?: "webgpu" | "wasm" }
  | { type: "run"; id: number; tensor: Float32Array; size: number };

export type FromWorker =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; backend: "webgpu" | "wasm"; f16: boolean }
  | { type: "result"; id: number; out: Float32Array; ms: number }
  | { type: "error"; message: string };

const post = (m: FromWorker, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(m, transfer);

let session: Ort.InferenceSession | null = null;
let ort: OrtModule;

/** 開発サーバは無いファイルにも画面の HTML を 200 で返す。それをモデルとして置かない */
function looksLikeHtml(b: Uint8Array): boolean {
  for (let i = 0; i < Math.min(b.length, 64); i++) {
    const c = b[i];
    if (c === 0x20 || c === 0x0a || c === 0x0d || c === 0x09 || c === 0xef || c === 0xbb || c === 0xbf) continue;
    return c === 0x3c; // "<"
  }
  return false;
}

/** モデルを取る。一度取ったものは Cache API に置き、2 回目からは読み直さない。 */
async function fetchModel(url: string): Promise<Uint8Array> {
  const cache = await caches.open("midog-web-models").catch(() => null);
  const hit = await cache?.match(url);
  const res = hit ?? (await fetch(url));
  if (!res.ok || !res.body) throw new Error(`モデルを取れませんでした: ${res.status} ${url}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const r = await reader.read();
    if (r.value) {
      chunks.push(r.value);
      loaded += r.value.length;
      post({ type: "progress", loaded, total });
    }
    if (r.done) break;
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  if (looksLikeHtml(buf)) {
    if (hit) return cache!.delete(url).then(() => fetchModel(url));
    throw new Error(`モデルが見つかりません: ${url}`);
  }
  if (!hit && cache) await cache.put(url, new Response(buf.slice())).catch(() => {});
  return buf;
}

async function init(webgpu: string, wasm: string, force?: "webgpu" | "wasm") {
  ort = await loadOrt();
  type Adapter = { features: Set<string> };
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<Adapter | null> } }).gpu;
  const adapter = force === "wasm" ? null : await gpu?.requestAdapter().catch(() => null);
  const backend = adapter ? "webgpu" : "wasm";
  // 16 ビット版は、GPU が 16 ビットの計算（shader-f16）に対応しているときだけ使う。
  // 対応していない GPU（ヘッドレスの Chromium など）では、16 ビット版はセッションを作る所で落ちる
  const f16 = !!adapter?.features.has("shader-f16");
  const bytes = await fetchModel(f16 ? webgpu : wasm);
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: [backend],
    graphOptimizationLevel: "all",
  });
  post({ type: "ready", backend, f16 });
}

async function run(id: number, tensor: Float32Array, size: number) {
  if (!session) throw new Error("モデルが読み込まれていません");
  const t0 = performance.now();
  const input = new ort.Tensor("float32", tensor, [1, 3, size, size]);
  const res = await session.run({ [session.inputNames[0]]: input });
  const out = res[session.outputNames[0]].data as Float32Array;
  post({ type: "result", id, out, ms: performance.now() - t0 }, [out.buffer]);
}

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  try {
    if (e.data.type === "init") await init(e.data.webgpu, e.data.wasm, e.data.force);
    else await run(e.data.id, e.data.tensor, e.data.size);
  } catch (err) {
    post({ type: "error", message: String((err as Error)?.message ?? err) });
  }
};
