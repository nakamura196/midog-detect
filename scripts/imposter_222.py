#!/usr/bin/env python3
"""labels が (2,2,2) の「似た別物」が、モデルにとって一致した物と割れた物のどちらに近いかを見る。

Usage:
    python3 scripts/imposter_222.py out/py-points-test-26x.json --thr 0.39

似た別物を labels で 3 群に分ける: (2,2) 一致 / (2,2,2) / 割れた物（(2,1,2) と (1,2,2)）。
群ごとに、(a) 7.5µm 以内に誤検出（しきい値以上で、分裂像の正解に当たらなかった予測）が乗った割合、
(b) 7.5µm 以内の予測の確信度の最大値（予測が無ければ 0）の分布、を出す。
あわせて、f1_agreed.py の「一致だけ（両側）」で (2,2,2) の上の誤検出も数えなかった場合の F1 を出す。
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RADIUS_UM = 7.5

pts = json.loads(Path(sys.argv[1]).read_text())
thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.39
coco = json.loads((DATA / "MIDOG++.json").read_text())
sp = json.loads((DATA / "spacing.json").read_text())
names = {im["id"]: im["file_name"] for im in coco["images"]}
ann = defaultdict(lambda: {1: [], 2: []})
for a in coco["annotations"]:
    x1, y1, x2, y2 = a["bbox"]
    ann[names[a["image_id"]]][a["category_id"]].append(((x1 + x2) / 2, (y1 + y2) / 2, tuple(a["labels"])))
rows_csv = csv.DictReader(open(DATA / "datasets_xvalidation.csv"), delimiter=";")
test = sorted(f"{int(r['Slide']):03d}.tiff" for r in rows_csv if r["Dataset"] == "test")
assert set(test) == set(pts), "予測の標本と配布元の test が食い違う"


def group(l):
    if len(l) == 2:
        return "agreed (2,2)"
    return "(2,2,2)" if l == (2, 2, 2) else "split"


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


stats = defaultdict(lambda: {"n": 0, "fp_on": 0, "maxconf": []})
tp = fn = fp_all = fp_split = fp_222 = 0
for n in test:
    u = sp[n]
    mit, imp = ann[n][1], ann[n][2]
    g = [(x * u, y * u) for x, y, _ in mit]
    msplit = [len(l) == 3 for *_, l in mit]
    allp = np.array([(x * u, y * u, c) for x, y, c in pts[n]]).reshape(-1, 3)
    p = [tuple(q[:2]) for q in allp if q[2] > thr]
    hit, used = match(g, p, RADIUS_UM)
    tp += sum(h and not s for h, s in zip(hit, msplit))
    fn += sum((not h) and not s for h, s in zip(hit, msplit))
    fps = np.array([q for q, k in zip(p, used) if not k]).reshape(-1, 2)
    fp_all += len(fps)
    if not imp:
        continue
    ig = np.array([(x * u, y * u) for x, y, _ in imp])
    gr = [group(l) for *_, l in imp]
    # 誤検出 1 つを、近くに乗っている似た別物の群に割り当てる（割れた物を優先、次に (2,2,2)）
    if len(fps):
        for nb in cKDTree(ig).query_ball_point(fps, r=RADIUS_UM):
            gs = {gr[k] for k in nb}
            if "split" in gs:
                fp_split += 1
            elif "(2,2,2)" in gs:
                fp_222 += 1
    fp_tree = cKDTree(fps) if len(fps) else None
    all_tree = cKDTree(allp[:, :2]) if len(allp) else None
    for (x, y), gname in zip(ig, gr):
        s = stats[gname]
        s["n"] += 1
        if fp_tree is not None and fp_tree.query_ball_point((x, y), r=RADIUS_UM):
            s["fp_on"] += 1
        near = all_tree.query_ball_point((x, y), r=RADIUS_UM) if all_tree is not None else []
        s["maxconf"].append(float(allp[near, 2].max()) if near else 0.0)


def f1(tp, fp, fn):
    return {"F1": round(2 * tp / max(2 * tp + fp + fn, 1), 4), "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4), "TP": int(tp), "FP": int(fp), "FN": int(fn)}


print(json.dumps({
    "preds": sys.argv[1], "thr": thr, "slides": len(test),
    "groups": {k: {"n": v["n"], "fp_on": v["fp_on"], "fp_on_rate": round(v["fp_on"] / v["n"], 4),
                   "maxconf_mean": round(float(np.mean(v["maxconf"])), 4),
                   "maxconf_quartiles": [round(float(q), 3) for q in np.percentile(v["maxconf"], [25, 50, 75])],
                   "maxconf_ge_0.2": round(float(np.mean(np.array(v["maxconf"]) >= 0.2)), 4)}
               for k, v in sorted(stats.items())},
    "fp_on_split_imposters": fp_split, "fp_on_222_only": fp_222,
    "agreed2": f1(tp, fp_all - fp_split, fn),
    "agreed2_plus_222": f1(tp, fp_all - fp_split - fp_222, fn),
}, ensure_ascii=False, indent=1))
