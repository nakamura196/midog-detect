#!/usr/bin/env python3
"""配布元（MIDOG++）の評価の仕方で測る: 分裂像の中心が 7.5µm 以内なら正解、F1 は標本ごとの数を足して出す。

Usage:
    python3 scripts/eval_points.py runs/yolo11x-midog/weights/best.pt > runs/yolo11x-midog/test_points.json
    python3 scripts/eval_points.py runs/yolov8x-loto-net/weights/best.pt --holdout "human neuroendocrine tumor"

--holdout <腫瘍の種類>: その種類を抜いて学習したモデル用。しきい値を選ぶ val から、その種類の標本を外す
（抜いた種類を見ずに決めたしきい値で、抜いた種類の test を測る）。

配布元の evaluation.py（DeepMicroscopy/MIDOGpp）と evalutils の score_detection に合わせたこと:
- 正解は区分「分裂像」の枠の中心。予測も区分 0（分裂像）だけを使い、似た別物の予測は捨てる
- 座標は µm に直して比べる。画素の大きさはスキャナで違う（0.226〜0.253µm）ので、
  標本ごとに TIFF の解像度タグから取る（data/spacing.json。無ければ figshare から先頭だけ取って作る）
- 突き合わせは正解の順に、7.5µm 以内でいちばん近い、まだ使っていない予測を当てる（evalutils と同じ貪欲法）
- F1 は全標本の TP・FP・FN を足してから出す（配布元の aggregate）。腫瘍の種類ごとも同じ
- 判定のしきい値は val で F1 が最大になる値を選び、test に当てる

配布元と違うこと:
- 配布元は標本 1 枚を丸ごと推論する。ここは切り分け済みの 640px 切片で推論し、標本の座標に戻す。
  切片が重なる所の二重の予測は、配布元の後処理と同じ距離の NMS（25px）で 1 つにする
- しきい値を探す範囲: 配布元は 0.50〜0.99（自分たちの検出器の出力に合わせた範囲）。YOLO の確信度は
  それより低めに出るので 0.05〜0.95 で探す
"""
import contextlib
import json
import re
import struct
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RADIUS_UM = 7.5
NMS_PX = 25  # 配布元 utils/nms_WSI.py の nms() と同じ
TILE_RE = re.compile(r"^(\d{3})_(\d{5})_(\d{5})$")

weights = Path(sys.argv[1])
holdout = sys.argv[sys.argv.index("--holdout") + 1] if "--holdout" in sys.argv else None
run_dir = weights.parent.parent

coco = json.loads((DATA / "MIDOG++.json").read_text())
split = json.loads((DATA / "midog" / "split.json").read_text())
tumor = {im["file_name"]: im["tumor_type"] for im in coco["images"]}
gt = defaultdict(list)
names = {im["id"]: im["file_name"] for im in coco["images"]}
for a in coco["annotations"]:
    if a["category_id"] == 1:
        x1, y1, x2, y2 = a["bbox"]
        gt[names[a["image_id"]]].append(((x1 + x2) / 2, (y1 + y2) / 2))
slides = {s: sorted(n for n in split if split[n] == s) for s in ("val", "test")}
assert holdout is None or holdout in tumor.values(), f"腫瘍の種類の名前が違う: {holdout}"


def tiff_um_per_px(fid):
    """TIFF の先頭（と IFD の所）だけを Range で取り、XResolution から µm/px を出す。
    figshare は S3 へ転送するので、転送先の URL を 1 回目で覚えて 2 回目以降に使う（1 回 10 秒かかるため）"""
    url = f"https://ndownloader.figshare.com/files/{fid}"
    buf = {}

    def get(a, n):
        nonlocal url
        for lo, data in buf.items():
            if lo <= a and a + n <= lo + len(data):
                return data[a - lo:a - lo + n]
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{a + max(n, 65536) - 1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            url = r.geturl()
            buf[a] = r.read()
        return buf[a][:n]

    head = get(0, 8)
    E = "<" if head[:2] == b"II" else ">"
    off = struct.unpack(E + "I", head[4:8])[0]
    k = struct.unpack(E + "H", get(off, 2))[0]
    ifd = get(off + 2, 12 * k)
    tags = {}
    for i in range(k):
        t, ty, cn, v = struct.unpack(E + "HHII", ifd[12 * i:12 * i + 12])
        tags[t] = v
    num, den = struct.unpack(E + "II", get(tags[282], 8))
    unit = tags.get(296, 2) & 0xFFFF if E == "<" else tags.get(296, 2) >> 16
    return (25400 if unit == 2 else 10000) / (num / den)


def load_spacing():
    path = DATA / "spacing.json"
    sp = json.loads(path.read_text()) if path.exists() else {}
    need = [n for s in slides.values() for n in s if n not in sp]
    if need:
        from concurrent.futures import ThreadPoolExecutor
        nb = json.loads((DATA / "Setup.ipynb").read_text())
        src = "".join("".join(c["source"]) for c in nb["cells"])
        fid = {name: f for f, name in re.findall(r"'(\d+)':\s*'(\d{3}\.tiff)'", src)}
        with ThreadPoolExecutor(4) as ex:
            for i, (n, u) in enumerate(zip(need, ex.map(lambda n: tiff_um_per_px(fid[n]), need)), 1):
                sp[n] = u
                print(f"spacing {i}/{len(need)} {n} {u:.4f}", file=sys.stderr)
                path.write_text(json.dumps(sp, indent=1, sort_keys=True))
    return sp


def predict(s):
    """切片ごとに推論し、標本の座標の (x, y, 確信度) にする。結果は runs/<モデル>/points_<分割>.json に取っておく"""
    cache = run_dir / f"points_{s}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO
        model = YOLO(str(weights))
    files = sorted((DATA / "midog" / "images" / s).glob("*.jpg"))
    raw = defaultdict(list)
    for i in range(0, len(files), 256):
        for r in model.predict([str(f) for f in files[i:i + 256]], imgsz=640, conf=0.01, batch=16,
                               verbose=False, stream=True):
            stem, ty, tx = TILE_RE.match(Path(r.path).stem).groups()
            b = r.boxes
            for (x1, y1, x2, y2), c, p in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist()):
                if int(c) == 0:
                    raw[stem + ".tiff"].append([int(tx) + (x1 + x2) / 2, int(ty) + (y1 + y2) / 2, p])
        print(f"{s} {min(i + 256, len(files))}/{len(files)}", file=sys.stderr)
    pts = {}
    for n in slides[s]:
        a = np.array(raw.get(n, []), dtype=float).reshape(-1, 3)
        if len(a):
            # 切片の重なりで同じ物に 2 つ付いた予測を、確信度の高い方に寄せる（配布元の距離 NMS）
            order = np.argsort(-a[:, 2])
            near = cKDTree(a[:, :2]).query_ball_point(a[:, :2], r=NMS_PX)
            alive = np.ones(len(a), bool)
            keep = []
            for j in order:
                if alive[j]:
                    keep.append(j)
                    alive[near[j]] = False
            a = a[keep]
        pts[n] = a.tolist()
    cache.write_text(json.dumps(pts))
    return pts


def score_slide(g, p, r):
    """evalutils.scorers.score_detection と同じ（正解の順に、近い順で未使用の予測を当てる）"""
    if not g:
        return 0, len(p), 0
    if not p:
        return 0, 0, len(g)
    p, g = np.array(p), np.array(g)
    hits = [sorted(h, key=lambda j: np.hypot(*(p[j] - g[i])))
            for i, h in enumerate(cKDTree(p).query_ball_point(g, r=r))]
    used = [False] * len(p)
    tp = fn = 0
    for h in hits:
        for j in h:
            if not used[j]:
                used[j] = True
                tp += 1
                break
        else:
            fn += 1
    return tp, used.count(False), fn


def f1(tp, fp, fn, eps=1e-6):
    return {"P": round(tp / (tp + fp + eps), 4), "R": round(tp / (tp + fn + eps), 4),
            "F1": round((2 * tp + eps) / (2 * tp + fp + fn + eps), 4), "TP": tp, "FP": fp, "FN": fn}


def evaluate(s, pts, thr, sp, skip=None):
    per_slide = {}
    for n in slides[s]:
        if tumor[n] == skip:
            continue
        u = sp[n]
        g = [(x * u, y * u) for x, y in gt[n]]
        p = [(x * u, y * u) for x, y, c in pts[n] if c > thr]
        per_slide[n] = score_slide(g, p, RADIUS_UM)
    tot = np.sum(list(per_slide.values()), axis=0)
    by_t = defaultdict(lambda: np.zeros(3, int))
    for n, v in per_slide.items():
        by_t[tumor[n]] += v
    return per_slide, f1(*map(int, tot)), {t: f1(*map(int, v)) for t, v in sorted(by_t.items())}


sp = load_spacing()
pts = {s: predict(s) for s in ("val", "test")}
grid = [round(t, 2) for t in np.arange(0.05, 0.96, 0.01)]
val_f1 = {t: evaluate("val", pts["val"], t, sp, skip=holdout)[1]["F1"] for t in grid}
thr = max(val_f1, key=val_f1.get)
per_slide, total, by_tumor = evaluate("test", pts["test"], thr, sp)
test_f1 = {t: evaluate("test", pts["test"], t, sp)[1]["F1"] for t in grid}
oracle = max(test_f1, key=test_f1.get)
slide_f1 = [f1(*v)["F1"] for n, v in per_slide.items() if sum(v)]
print(json.dumps({
    "weights": str(weights),
    "holdout": holdout,
    "rule": f"MIDOG++ evaluation.py: center within {RADIUS_UM} um, F1 from summed TP/FP/FN over slides",
    "threshold_from_val": thr,
    "val_F1": val_f1[thr],
    "test": total,
    "test_per_tumor": by_tumor,
    "test_holdout": by_tumor.get(holdout),
    "test_mean_of_slide_F1": round(float(np.mean(slide_f1)), 4),
    "test_slides": len(per_slide),
    "test_best_threshold_reference": {"threshold": oracle, "F1": test_f1[oracle]},
    "um_per_px_range": [round(min(sp[n] for n in per_slide), 4), round(max(sp[n] for n in per_slide), 4)],
}, ensure_ascii=False, indent=1))
