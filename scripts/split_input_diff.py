"""ブラウザと Python で確信度がずれた点について、入力の違いのどこが効いているかを分ける。

ブラウザ（CPU・32 ビット版）と Python の予測を突き合わせ、確信度の差が大きい点を選ぶ。
その点を含む切片を、同じ Python のモデル（best.pt）で次の 3 通りの入力から推論し直す。

  raw   元の TIFF から切った切片そのもの（JPEG を通さない）
  patch raw を PIL で JPEG（品質 95）に保存して読み直したもの。学習・Python の評価の入力と同じ
  tiles ブラウザと同じ IIIF のタイル（512 角、品質 95・4:2:0）を PIL で読み、貼り合わせたもの

patch と tiles の違いが「切り方」（JPEG の 8 画素の区画とタイルの境目がどこに来るか）、
raw との違いが JPEG そのものの影響。tiles とブラウザの違いは、JPEG の読み方（復号）と推論の実装の違い。

切片が重なる所は eval_points.py と同じく、切片ごとの予測を標本の座標に戻し、25px の NMS で 1 つにする。
各点の確信度は、その点から 7.5µm 以内にある分裂像の予測の最大値（無ければ 0）。

使い方:
  python3 scripts/split_input_diff.py 345 --web out/web-wasm-q95-345.json --py out/py-points-345.json
"""
import argparse
import io
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TILE = 640
IIIF_TILE = 512
NMS_PX = 25


def starts(n):
    s = list(range(0, max(n - TILE, 0) + 1, TILE))
    if s[-1] + TILE < n:
        s.append(n - TILE)
    return s


def crop_raw(tiff, x, y, tmp):
    out = Path(tmp) / f"{x}_{y}.png"
    subprocess.run(["vips", "crop", str(tiff), str(out), str(x), str(y), str(TILE), str(TILE)], check=True)
    return Image.open(out).convert("RGB")


def as_patch(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def from_tiles(iiif, x, y, w, h):
    canvas = Image.new("RGB", (TILE, TILE))
    for ty in range(y // IIIF_TILE * IIIF_TILE, y + TILE, IIIF_TILE):
        for tx in range(x // IIIF_TILE * IIIF_TILE, x + TILE, IIIF_TILE):
            tw, th = min(IIIF_TILE, w - tx), min(IIIF_TILE, h - ty)
            t = Image.open(iiif / f"{tx},{ty},{tw},{th}" / f"{tw},{th}" / "0" / "default.jpg").convert("RGB")
            canvas.paste(t, (tx - x, ty - y))
    return canvas


def nms(preds):
    keep = []
    for p in sorted(preds, key=lambda p: -p[2]):
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > NMS_PX for q in keep):
            keep.append(p)
    return keep


def nearest(preds, x, y, r):
    """(x, y) からいちばん近い、r 以内の予測の確信度（無ければ 0）。近くの別の物を拾わないよう、最大ではなく最近を取る"""
    d = [(math.hypot(px - x, py - y), s) for px, py, s in preds]
    d = [v for v in d if v[0] <= r]
    return min(d)[1] if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slide")
    ap.add_argument("--web", required=True)
    ap.add_argument("--py", required=True)
    ap.add_argument("--weights", default=str(ROOT / "runs/yolo26x-midog/weights/best.pt"))
    ap.add_argument("--min-diff", type=float, default=0.03)
    ap.add_argument("--json", help="結果を書き出す先")
    a = ap.parse_args()

    meta = json.loads((ROOT / f"web/public/data/{a.slide}.json").read_text())
    W, H, r = meta["width"], meta["height"], 7.5 / meta["um_per_px"]
    tiff = ROOT / f"data/raw/{a.slide}.tiff"
    iiif = ROOT / f"web/public/iiif/{a.slide}"

    web = [(p["x"], p["y"], p["score"]) for p in json.load(open(a.web)) if p["cls"] == 0]
    py = json.load(open(a.py))
    # 差の大きい点。compare_web.py と同じく、Python の点を確信度の高い順に、まだ使っていない最寄りのブラウザの点に当てる
    targets, used = [], set()
    for x, y, s in sorted(py, key=lambda p: -p[2]):
        best = min(((math.hypot(p[0] - x, p[1] - y), j) for j, p in enumerate(web) if j not in used), default=None)
        sw = 0.0
        if best and best[0] <= r:
            used.add(best[1])
            sw = web[best[1]][2]
        if max(s, sw) >= 0.1 and abs(s - sw) > a.min_diff:
            targets.append({"x": x, "y": y, "py": s, "web": sw})
    print(f"差が {a.min_diff} を超える点: {len(targets)}")

    xs, ys = starts(W), starts(H)
    wins = sorted({(wx, wy) for t in targets for wx in xs for wy in ys
                   if wx <= t["x"] < wx + TILE and wy <= t["y"] < wy + TILE})
    print(f"推論する切片: {len(wins)} × 3 通り（右端の列 x={xs[-1]}、下端の段 y={ys[-1]} は 16 画素の区画がタイルとずれる）")

    from ultralytics import YOLO
    model = YOLO(a.weights)
    kinds = ("raw", "patch", "tiles")
    per_win = {}  # (wx, wy) -> {kind: [(x, y, s)]}
    with tempfile.TemporaryDirectory() as tmp:
        for wx, wy in wins:
            raw = crop_raw(tiff, wx, wy, tmp)
            imgs = {"raw": raw, "patch": as_patch(raw), "tiles": from_tiles(iiif, wx, wy, W, H)}
            per_win[wx, wy] = {}
            for k, img in imgs.items():
                res = model.predict(img, imgsz=640, conf=0.01, verbose=False)[0].boxes
                per_win[wx, wy][k] = [(wx + (x1 + x2) / 2, wy + (y1 + y2) / 2, s)
                                      for (x1, y1, x2, y2), c, s in zip(res.xyxy.tolist(), res.cls.tolist(), res.conf.tolist())
                                      if int(c) == 0]
            # 画素の違い（0..255）の平均
            dp = np.abs(np.asarray(imgs["patch"], np.int16) - np.asarray(raw, np.int16)).mean()
            dt = np.abs(np.asarray(imgs["tiles"], np.int16) - np.asarray(imgs["patch"], np.int16)).mean()
            print(f"  切片 ({wx},{wy}) 画素の差の平均: patch-raw {dp:.2f} / tiles-patch {dt:.2f}")
    preds = {k: nms([p for w in per_win.values() for p in w[k]]) for k in kinds}

    print(f"\n{'x':>6} {'y':>6} | {'Python':>6} {'patch':>6} {'raw':>6} {'tiles':>6} {'ブラウザ':>6} | 切片ごと raw/patch/tiles")
    for t in targets:
        for k in kinds:
            t[k] = nearest(preds[k], t["x"], t["y"], r)
        t["windows"] = {f"{wx},{wy}": {k: nearest(nms(per_win[wx, wy][k]), t["x"], t["y"], r) for k in kinds}
                        for wx, wy in wins if wx <= t["x"] < wx + TILE and wy <= t["y"] < wy + TILE}
        detail = "  ".join(f"({w}) " + "/".join(f"{v[k]:.2f}" for k in kinds) for w, v in t["windows"].items())
        print(f"{t['x']:6.0f} {t['y']:6.0f} | {t['py']:6.3f} {t['patch']:6.3f} {t['raw']:6.3f} {t['tiles']:6.3f} {t['web']:6.3f} | {detail}")

    def mx(k1, k2):
        return max(abs(t[k1] - t[k2]) for t in targets)

    print("\n確信度の差の最大（この点の集まりで）")
    for k1, k2, what in [("py", "web", "Python vs ブラウザ（出発点）"), ("py", "patch", "検算: Python vs patch（同じはず）"),
                         ("web", "tiles", "ブラウザ vs tiles（復号と推論の実装）"), ("patch", "tiles", "patch vs tiles（切り方）"),
                         ("raw", "patch", "raw vs patch（切片の JPEG）"), ("raw", "tiles", "raw vs tiles（タイルの JPEG）")]:
        print(f"  {what}: {mx(k1, k2):.3f}")
    if a.json:
        Path(a.json).write_text(json.dumps(targets, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
