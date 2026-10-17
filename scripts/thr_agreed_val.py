#!/usr/bin/env python3
"""割れた正解を除いた方式ごとに、val でしきい値を選び直して test に当てる。

Usage:
    python3 scripts/thr_agreed_val.py runs/yolo26x-midog/points_val.json out/py-points-test-26x.json

f1_agreed.py の 3 通り（all / agreed / agreed2）を、val と test の両方で数える。
方式ごとに val で F1 が最大になるしきい値を選び、そのしきい値の test の F1 を出す
（f1_agreed.py は、しきい値を配布元どおりの方式で val から選んだ 0.39 に固定していた）。

val の標本は data/midog/split.json（mdx-gpu で切り分けたもの）から取る。
解像度は data/spacing.json（無い標本があれば止まる。eval_points.py で作る）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RADIUS_UM = 7.5
KINDS = ("all", "agreed", "agreed2")

pts = {"val": json.loads(Path(sys.argv[1]).read_text()), "test": json.loads(Path(sys.argv[2]).read_text())}
coco = json.loads((DATA / "MIDOG++.json").read_text())
split = json.loads((DATA / "midog" / "split.json").read_text())
sp = json.loads((DATA / "spacing.json").read_text())
names = {im["id"]: im["file_name"] for im in coco["images"]}
tumor = {im["file_name"]: im["tumor_type"] for im in coco["images"]}
ann = defaultdict(lambda: {1: [], 2: []})
for a in coco["annotations"]:
    x1, y1, x2, y2 = a["bbox"]
    ann[names[a["image_id"]]][a["category_id"]].append(((x1 + x2) / 2, (y1 + y2) / 2, tuple(a["labels"])))
slides = {s: sorted(n for n in split if split[n] == s) for s in ("val", "test")}
for s in slides:
    assert set(slides[s]) == set(pts[s]), f"{s}: 予測の標本と split.json が食い違う"
    assert all(n in sp for n in slides[s]), f"{s}: spacing.json に無い標本がある"


def match(g, p, r):
    """eval_points.score_slide と同じ貪欲法"""
    hit = [False] * len(g)
    used = [False] * len(p)
    if len(g) and len(p):
        g, p = np.array(g), np.array(p)
        for i, h in enumerate(cKDTree(p).query_ball_point(g, r=r)):
            for j in sorted(h, key=lambda j: np.hypot(*(p[j] - g[i]))):
                if not used[j]:
                    used[j] = hit[i] = True
                    break
    return hit, used


def score(s, thr):
    """{方式: [TP, FP, FN]}（標本を通した合計）"""
    out = {k: np.zeros(3, int) for k in KINDS}
    for n in slides[s]:
        u = sp[n]
        mit, imp = ann[n][1], ann[n][2]
        g = [(x * u, y * u) for x, y, _ in mit]
        split_m = [len(l) == 3 for *_, l in mit]
        p = [(x * u, y * u) for x, y, c in pts[s][n] if c > thr]
        hit, used = match(g, p, RADIUS_UM)
        fps = [q for q, k in zip(p, used) if not k]
        tp_ag = sum(h and not x for h, x in zip(hit, split_m))
        fn_ag = sum((not h) and not x for h, x in zip(hit, split_m))
        on_split_imp = 0
        isplit = [len(l) == 3 and l != (2, 2, 2) for *_, l in imp]
        if fps and any(isplit):
            ig = np.array([(x * u, y * u) for (x, y, _), x_ in zip(imp, isplit) if x_])
            on_split_imp = sum(bool(nr) for nr in cKDTree(ig).query_ball_point(np.array(fps), r=RADIUS_UM))
        out["all"] += (sum(hit), len(fps), sum(not h for h in hit))
        out["agreed"] += (tp_ag, len(fps), fn_ag)
        out["agreed2"] += (tp_ag, len(fps) - on_split_imp, fn_ag)
    return out


def f1(tp, fp, fn):
    return {"F1": round(2 * tp / max(2 * tp + fp + fn, 1), 4), "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4), "TP": int(tp), "FP": int(fp), "FN": int(fn)}


grid = [round(t, 2) for t in np.arange(0.05, 0.96, 0.01)]  # eval_points.py と同じ
tab = {s: {t: score(s, t) for t in grid} for s in ("val", "test")}
res = {}
for k in KINDS:
    vf = {t: f1(*tab["val"][t][k])["F1"] for t in grid}
    tf = {t: f1(*tab["test"][t][k])["F1"] for t in grid}
    thr = max(vf, key=vf.get)
    ora = max(tf, key=tf.get)
    res[k] = {"thr_from_val": thr, "val": f1(*tab["val"][thr][k]), "test": f1(*tab["test"][thr][k]),
              "test_at_0.39": f1(*tab["test"][0.39][k]), "test_best_thr_reference": {"thr": ora, "F1": tf[ora]},
              "val_F1_curve": vf}

print(json.dumps({"val_preds": sys.argv[1], "test_preds": sys.argv[2],
                  "slides": {s: len(v) for s, v in slides.items()}, **res}, ensure_ascii=False, indent=1))
