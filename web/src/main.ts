/**
 * 画面: OpenSeadragon で標本を表示し、640 角の切片ごとに Worker で推論して、点を重ねる。
 *
 * URL の引数:
 *   ?slide=<標本>               手元の標本（scripts/make_demo.py で作る）。既定は 006
 *   ?info=<IIIF の info.json>   手元にない標本を読むとき
 *   &truth=<正解の JSON>         make_demo.py の形。info を変えたときは無ければ正解なしで動く
 *   &backend=wasm               WebGPU があっても CPU で動かす
 */
import OpenSeadragon from "openseadragon";
import "./style.css";
import meta from "./model-meta.json";
import { Cutter, loadInfo, TILE, windows, type Info, type Window } from "./slide";
import { nms, RADIUS_UM, score, type Pred, type Truth, type Mark } from "./points";
import type { FromWorker, ToWorker } from "./worker";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);
const BASE = import.meta.env.BASE_URL;
const SLIDE = params.get("slide") ?? "006";
const infoUrl = params.get("info") ?? `${BASE}iiif/${SLIDE}/info.json`;
const truthUrl = params.get("truth") ?? (params.has("info") ? null : `${BASE}data/${SLIDE}.json`);

// 標本を替えるときは読み込み直す（推論の途中の状態を持ち越さないため）
const pick = $("pick") as HTMLSelectElement;
if (params.has("info")) pick.closest("label")!.hidden = true;
else pick.value = SLIDE;
pick.onchange = () => {
  const q = new URLSearchParams(location.search);
  q.set("slide", pick.value);
  location.search = q.toString();
};

const HF = "https://huggingface.co/nakamura196/yolo26x-midog-onnx/resolve/main/";
const modelUrl = (name: string) => (import.meta.env.DEV ? `${BASE}models/${name}` : HF + name);

/** 推論で残す最低の確信度（eval_points.py の predict と同じ 0.01）。しきい値は表示のときに当てる */
const KEEP = 0.01;

interface SlideMeta {
  name: string;
  tumor: string;
  split: string | null;
  um_per_px: number;
  truth: Truth[];
}

const state = {
  info: null as Info | null,
  slide: null as SlideMeta | null,
  raw: [] as Pred[],
  preds: [] as Pred[],
  thr: meta.threshold ?? 0.5,
  scanning: null as Window | null,
  done: new Set<string>(),
  running: false,
  stop: false,
};

// テスト（Playwright）から結果を読むため
(window as unknown as { __midog: unknown }).__midog = state;

// --- Worker ------------------------------------------------------------------
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const waiting = new Map<number, (r: { out: Float32Array; ms: number }) => void>();
let nextId = 0;
worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  if (m.type === "progress") {
    $("status").textContent = `モデルを取得中 ${(m.loaded / 1e6).toFixed(0)}${m.total ? ` / ${(m.total / 1e6).toFixed(0)}` : ""} MB`;
    if (m.total) ($("bar") as HTMLProgressElement).value = m.loaded / m.total;
  } else if (m.type === "ready") {
    $("backend").textContent = `${m.backend === "webgpu" ? "WebGPU" : "CPU"}（${m.f16 ? 16 : 32} ビット版）`;
    $("status").textContent = "準備ができました";
    ($("bar") as HTMLProgressElement).value = 0;
    const b = $("run") as HTMLButtonElement;
    b.disabled = false;
    b.textContent = "検出を始める";
  } else if (m.type === "result") {
    waiting.get(m.id)?.(m);
    waiting.delete(m.id);
  } else {
    $("status").textContent = `エラー: ${m.message}`;
  }
};
const force = params.get("backend");
worker.postMessage({
  type: "init",
  webgpu: modelUrl("midog-yolo26x-fp16.onnx"),
  wasm: modelUrl("midog-yolo26x.onnx"),
  force: force === "wasm" || force === "webgpu" ? force : undefined,
} satisfies ToWorker);

function infer(tensor: Float32Array): Promise<{ out: Float32Array; ms: number }> {
  const id = nextId++;
  return new Promise((resolve) => {
    waiting.set(id, resolve);
    worker.postMessage({ type: "run", id, tensor, size: TILE } satisfies ToWorker, [tensor.buffer]);
  });
}

// --- 表示 --------------------------------------------------------------------
const viewer = OpenSeadragon({
  element: $("viewer"),
  prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@6.1.1/build/openseadragon/images/",
  showNavigator: true,
  navigatorPosition: "BOTTOM_LEFT",
  maxZoomPixelRatio: 3,
  gestureSettingsMouse: { clickToZoom: false },
});

const canvas = $("overlay") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function toScreen(x: number, y: number) {
  const p = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y));
  return { x: p.x, y: p.y };
}

let scored: { s: ReturnType<typeof score>; mf: Pred[] } | null = null;

function recompute() {
  const mf = state.preds.filter((p) => p.cls === 0 && p.score >= state.thr);
  scored = state.slide ? { s: score(state.slide.truth, mf, RADIUS_UM / state.slide.um_per_px), mf } : { s: null!, mf };
  renderResult();
  renderList();
  draw();
}

function draw() {
  const dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!state.info) return;
  const scale = toScreen(1, 0).x - toScreen(0, 0).x; // 画像 1px が画面で何 px か
  const r = Math.max(25 * scale, 5); // 正解の枠（50px 四方）の半分

  // 推論の終わった切片・推論中の切片
  if (state.running || state.done.size < windows(state.info).length) {
    for (const k of state.done) {
      const [x, y] = k.split(",").map(Number);
      const a = toScreen(x, y);
      ctx.fillStyle = "rgba(47,95,208,0.07)";
      ctx.fillRect(a.x, a.y, TILE * scale, TILE * scale);
    }
  }
  if (state.scanning) {
    const a = toScreen(state.scanning.x, state.scanning.y);
    ctx.strokeStyle = css("--accent");
    ctx.lineWidth = 2;
    ctx.strokeRect(a.x, a.y, TILE * scale, TILE * scale);
  }

  const ring = (x: number, y: number, color: string, dash = false, lw = 2.5) => {
    const p = toScreen(x, y);
    if (p.x < -r || p.y < -r || p.x > w + r || p.y > h + r) return;
    ctx.beginPath();
    ctx.setLineDash(dash ? [4, 3] : []);
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.stroke();
  };

  const showPred = ($("show-pred") as HTMLInputElement).checked;
  const showTruth = ($("show-truth") as HTMLInputElement).checked;
  const showImp = ($("show-imp") as HTMLInputElement).checked;

  if (showImp) {
    for (const p of state.preds) if (p.cls === 1 && p.score >= state.thr) ring(p.x, p.y, css("--imp"), true, 1.5);
    if (state.slide && showTruth) {
      for (const t of state.slide.truth) if (t.cls === 1) {
        const p = toScreen(t.x, t.y);
        ctx.setLineDash([]);
        ctx.strokeStyle = css("--imp");
        ctx.lineWidth = 1.5;
        ctx.strokeRect(p.x - r * 0.7, p.y - r * 0.7, r * 1.4, r * 1.4);
      }
    }
  }
  if (showTruth && state.slide) {
    state.slide.truth.forEach((t, i) => {
      if (t.cls !== 0) return;
      const fn = !scored || !state.done.size || scored.s.truth[i] === "fn";
      // 見逃し（または未推論）は赤の二重丸、当たったものは予測の緑の丸に任せ、小さな印だけ付ける
      if (fn) {
        ring(t.x, t.y, css("--fn"), false, 2.5);
        const p = toScreen(t.x, t.y);
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
  }
  if (showPred && scored) {
    scored.mf.forEach((p, i) => {
      const mark: Mark | null = state.slide ? scored!.s.pred[i] : null;
      ring(p.x, p.y, mark === "fp" ? css("--fp") : css("--tp"));
    });
  }
  ctx.setLineDash([]);
}

function renderResult() {
  const el = $("result");
  if (!scored) return void (el.textContent = "");
  const n = scored.mf.length;
  if (!state.slide) {
    el.innerHTML = `<table><tr><td>分裂像の予測</td><td>${n} 件</td></tr></table>`;
    $("legend").innerHTML = `<span style="color:var(--tp)"><i></i>分裂像の予測</span>`;
    return;
  }
  const { tp, fp, fn } = scored.s;
  const f = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "—");
  const P = tp / (tp + fp);
  const R = tp / (tp + fn);
  const F1 = (2 * tp) / (2 * tp + fp + fn);
  const partial = state.done.size < windows(state.info!).length ? "（推論の済んだ所まで。見逃しは未推論の所も含む）" : "";
  el.innerHTML = `<table>
    <tr><td>分裂像の予測</td><td>${n} 件</td></tr>
    <tr><td>当たり / 誤検出 / 見逃し</td><td>${tp} / ${fp} / ${fn}</td></tr>
    <tr><td>適合率 / 再現率</td><td>${f(P)} / ${f(R)}</td></tr>
    <tr><td>F1</td><td>${f(F1)}</td></tr></table>
    <p class="muted">${partial}</p>`;
  $("legend").innerHTML = `<span style="color:var(--tp)"><i></i>当たり</span>
    <span style="color:var(--fp)"><i></i>誤検出</span><span style="color:var(--fn)"><i></i>見逃し（正解）</span>`;
}

function renderList() {
  const ol = $("list");
  ol.innerHTML = "";
  if (!scored) return;
  const items = scored.mf.map((p, i) => ({ p, mark: state.slide ? scored!.s.pred[i] : null }));
  const fns = state.slide ? state.slide.truth.filter((t, i) => t.cls === 0 && scored!.s.truth[i] === "fn") : [];
  const add = (label: string, color: string, x: number, y: number) => {
    const li = document.createElement("li");
    li.innerHTML = `<span style="color:${color}">●</span> ${label}`;
    li.onclick = () => zoomTo(x, y);
    ol.append(li);
  };
  for (const { p, mark } of [...items].sort((a, b) => b.p.score - a.p.score).slice(0, 300)) {
    const lab = mark === "fp" ? "誤検出" : mark === "tp" ? "当たり" : "予測";
    add(`${p.score.toFixed(2)} ${lab}（${Math.round(p.x)}, ${Math.round(p.y)}）`, mark === "fp" ? "var(--fp)" : "var(--tp)", p.x, p.y);
  }
  if (state.done.size) for (const t of fns) add(`見逃し（${Math.round(t.x)}, ${Math.round(t.y)}）`, "var(--fn)", t.x, t.y);
}

function zoomTo(x: number, y: number) {
  viewer.viewport.fitBounds(viewer.viewport.imageToViewportRectangle(x - 160, y - 160, 320, 320));
}

for (const ev of ["animation", "update-viewport", "resize", "open"] as const) viewer.addHandler(ev, draw);
for (const id of ["show-pred", "show-truth", "show-imp"]) $(id).addEventListener("change", draw);

const thr = $("thr") as HTMLInputElement;
thr.value = String(state.thr);
$("thr-v").textContent = state.thr.toFixed(2);
$("thr-note").textContent = meta.threshold != null ? `既定 ${meta.threshold}（val の標本で F1 が最大になる値）` : "";
thr.addEventListener("input", () => {
  state.thr = Number(thr.value);
  $("thr-v").textContent = state.thr.toFixed(2);
  recompute();
});

// --- 推論 --------------------------------------------------------------------
async function scan() {
  const info = state.info!;
  const wins = windows(info);
  const cutter = new Cutter(info);
  state.running = true;
  state.stop = false;
  state.raw = [];
  state.done.clear();
  const btn = $("run") as HTMLButtonElement;
  btn.textContent = "止める";
  const t0 = performance.now();
  let msSum = 0;
  // 次の切片のタイルは、今の切片の推論中に取っておく
  let next = cutter.cut(wins[0]);
  for (let i = 0; i < wins.length && !state.stop; i++) {
    const win = wins[i];
    state.scanning = win;
    const tensor = await next;
    if (i + 1 < wins.length) next = cutter.cut(wins[i + 1]);
    const { out, ms } = await infer(tensor);
    msSum += ms;
    for (let k = 0; k < out.length; k += 6) {
      const s = out[k + 4];
      if (s < KEEP) continue;
      state.raw.push({ x: win.x + (out[k] + out[k + 2]) / 2, y: win.y + (out[k + 1] + out[k + 3]) / 2, score: s, cls: out[k + 5] });
    }
    state.done.add(`${win.x},${win.y}`);
    state.preds = nms(state.raw);
    ($("bar") as HTMLProgressElement).value = (i + 1) / wins.length;
    $("status").textContent = `切片 ${i + 1} / ${wins.length}・推論 ${(msSum / (i + 1)).toFixed(0)} ms/切片・経過 ${((performance.now() - t0) / 1000).toFixed(1)} 秒`;
    recompute();
  }
  state.scanning = null;
  state.running = false;
  const sec = (performance.now() - t0) / 1000;
  $("status").textContent = `${state.stop ? "止めました" : "終わりました"}: 切片 ${state.done.size} / ${wins.length}・${sec.toFixed(1)} 秒（推論 ${(msSum / Math.max(state.done.size, 1)).toFixed(0)} ms/切片、タイル ${cutter.fetched} 枚）`;
  (state as unknown as { seconds: number }).seconds = sec;
  btn.textContent = "もう一度検出する";
  recompute();
}

$("run").addEventListener("click", () => {
  if (state.running) state.stop = true;
  else scan().catch((e) => ($("status").textContent = `エラー: ${e.message ?? e}`));
});

// --- 読み込み ----------------------------------------------------------------
(async () => {
  const info = await loadInfo(infoUrl);
  state.info = info;
  viewer.open(info.json as unknown as Parameters<typeof viewer.open>[0]);
  if (truthUrl) {
    try {
      const r = await fetch(truthUrl);
      if (r.ok) state.slide = (await r.json()) as SlideMeta;
    } catch {
      /* 正解なしで続ける */
    }
  }
  const s = state.slide;
  const wins = windows(info).length;
  $("slide").innerHTML = s
    ? `標本 ${s.name}・${s.tumor}（${s.split ?? "—"}）<br>${info.width.toLocaleString()} × ${info.height.toLocaleString()} px・切片 ${wins} 枚・正解の分裂像 ${s.truth.filter((t) => t.cls === 0).length}`
    : `${info.width.toLocaleString()} × ${info.height.toLocaleString()} px・切片 ${wins} 枚`;
  recompute();
})().catch((e) => ($("status").textContent = `エラー: ${e.message ?? e}`));
