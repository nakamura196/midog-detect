/**
 * IIIF の画像から、原寸の 640 角の切片を切り出す。
 *
 * 切片の並べ方は学習の切り分け（scripts/prepare.py の starts()）と同じ:
 * 0, 640, 1280, ... と並べ、最後の切片は端にそろえる（はみ出さない）。
 * 切片は IIIF の原寸のタイル（scaleFactor 1）を取ってきて、画面の外のキャンバスに貼り合わせて作る。
 * タイルだけを使うので、静的ファイルだけのレベル 0 の配信でも、画像サーバ（レベル 1・2）でも同じに動く。
 */

export const TILE = 640;

export interface Info {
  base: string;
  width: number;
  height: number;
  tile: number;
  version: 2 | 3;
  json: Record<string, unknown>;
}

/** info.json を読む。id が仮の値（make_demo.py の example.invalid）なら、読み込んだ場所から付け直す */
export async function loadInfo(url: string): Promise<Info> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`info.json を取れませんでした: ${res.status} ${url}`);
  const json = (await res.json()) as Record<string, unknown>;
  const version = String(json["@context"] ?? "").includes("/image/2") ? 2 : 3;
  let base = String(json.id ?? json["@id"] ?? "");
  if (!base || base.includes("example.invalid")) {
    base = new URL(url, location.href).href.replace(/\/info\.json$/, "");
    json[version === 2 ? "@id" : "id"] = base;
  }
  const tiles = (json.tiles as { width: number; scaleFactors: number[] }[] | undefined)?.[0];
  if (tiles && !tiles.scaleFactors.includes(1)) throw new Error("原寸（scaleFactor 1）のタイルがありません");
  return {
    base,
    width: Number(json.width),
    height: Number(json.height),
    tile: tiles?.width ?? 512,
    version,
    json,
  };
}

export function starts(n: number): number[] {
  const s: number[] = [];
  for (let x = 0; x <= Math.max(n - TILE, 0); x += TILE) s.push(x);
  if (s[s.length - 1] + TILE < n) s.push(n - TILE);
  return s;
}

export interface Window {
  x: number;
  y: number;
}

/** 上の段から順に、左から右へ */
export function windows(info: Info): Window[] {
  const out: Window[] = [];
  for (const y of starts(info.height)) for (const x of starts(info.width)) out.push({ x, y });
  return out;
}

/** IIIF のタイルを取って切片を作る係。取ったタイルは少しだけ手元に残す（隣の切片と共有するため） */
export class Cutter {
  private cache = new Map<string, Promise<ImageBitmap>>();
  private canvas = new OffscreenCanvas(TILE, TILE);
  private ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
  fetched = 0;

  constructor(private info: Info, private keep = 48) {}

  private tileUrl(x: number, y: number, w: number, h: number): string {
    const size = this.info.version === 2 ? `${w},` : `${w},${h}`;
    return `${this.info.base}/${x},${y},${w},${h}/${size}/0/default.jpg`;
  }

  private tile(x: number, y: number): Promise<ImageBitmap> {
    const t = this.info.tile;
    const w = Math.min(t, this.info.width - x);
    const h = Math.min(t, this.info.height - y);
    const url = this.tileUrl(x, y, w, h);
    let p = this.cache.get(url);
    if (p) {
      // 使ったものを後ろへ（古いものから捨てる）
      this.cache.delete(url);
      this.cache.set(url, p);
      return p;
    }
    p = fetch(url).then(async (r) => {
      if (!r.ok) throw new Error(`タイルを取れませんでした: ${r.status} ${url}`);
      this.fetched++;
      // 色の管理で画素を変えない（学習の切片は PIL がそのまま読んだ値）
      return createImageBitmap(await r.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    });
    this.cache.set(url, p);
    while (this.cache.size > this.keep) {
      const old = this.cache.keys().next().value!;
      this.cache.get(old)!.then((b) => b.close(), () => {});
      this.cache.delete(old);
    }
    return p;
  }

  /** 切片を、モデルの入力（[3, 640, 640]、RGB、0..1）にする */
  async cut(win: Window): Promise<Float32Array> {
    const t = this.info.tile;
    const jobs: Promise<void>[] = [];
    this.ctx.fillStyle = "#fff";
    this.ctx.fillRect(0, 0, TILE, TILE);
    for (let ty = Math.floor(win.y / t) * t; ty < win.y + TILE; ty += t) {
      for (let tx = Math.floor(win.x / t) * t; tx < win.x + TILE; tx += t) {
        jobs.push(this.tile(tx, ty).then((b) => this.ctx.drawImage(b, tx - win.x, ty - win.y)));
      }
    }
    await Promise.all(jobs);
    const px = this.ctx.getImageData(0, 0, TILE, TILE).data;
    const n = TILE * TILE;
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      out[i] = px[i * 4] / 255;
      out[n + i] = px[i * 4 + 1] / 255;
      out[2 * n + i] = px[i * 4 + 2] / 255;
    }
    return out;
  }
}
