#!/usr/bin/env python3
"""病理医の判定が割れた正解を除いて、test の F1 を出し直す。

Usage:
    python3 scripts/f1_agreed.py out/py-points-test-26x.json --thr 0.39

3 通りで数える（しきい値は同じ。突き合わせは eval_points.py と同じ、7.5µm 以内・貪欲法）:

- all:       配布元どおり。割れた正解も正解として数える
- agreed:    割れた分裂像（labels が 3 要素）を除く。除いた正解に当たった予測は誤検出にも数えない
             （除いた物は「数えない場所」として扱う。誤検出に回すと、除いたぶん成績が下がって見える）
- agreed2:   agreed に加えて、割れた「似た別物」（3 要素で (2,2,2) 以外）の上の誤検出も数えない
             （1 人が分裂像と判定した物を、どちらの側でも数えない）

突き合わせは割れた正解を含めた全部で 1 回だけ行い、その結果を振り分ける（除いてから突き合わせ直すと、
割れた正解に当たっていた予測が隣の一致した正解に移ることがあり、比べにくくなる）。
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
tumor = {im["file_name"]: im["tumor_type"] for im in coco["images"]}
ann = defaultdict(lambda: {1: [], 2: []})
for a in coco["annotations"]:
    x1, y1, x2, y2 = a["bbox"]
    ann[names[a["image_id"]]][a["category_id"]].append(((x1 + x2) / 2, (y1 + y2) / 2, tuple(a["labels"])))
rows_csv = csv.DictReader(open(DATA / "datasets_xvalidation.csv"), delimiter=";")
test = sorted(f"{int(r['Slide']):03d}.tiff" for r in rows_csv if r["Dataset"] == "test")
assert set(test) == set(pts), "予測の標本と配布元の test が食い違う"


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


def score(thr):
    """腫瘍の種類ごとに {方式: [TP, FP, FN]} と、除いた数"""
    out = defaultdict(lambda: {k: np.zeros(3, int) for k in ("all", "agreed", "agreed2")})
    dropped = defaultdict(lambda: np.zeros(2, int))  # 除いた分裂像, 数えなかった似た別物の上の誤検出
    for n in test:
        u = sp[n]
        mit, imp = ann[n][1], ann[n][2]
        g = [(x * u, y * u) for x, y, _ in mit]
        split = [len(l) == 3 for *_, l in mit]
        p = [(x * u, y * u) for x, y, c in pts[n] if c > thr]
        hit, used = match(g, p, RADIUS_UM)
        fps = [q for q, k in zip(p, used) if not k]
        tp_all, fn_all = sum(hit), sum(not h for h in hit)
        tp_ag = sum(h and not s for h, s in zip(hit, split))
        fn_ag = sum((not h) and not s for h, s in zip(hit, split))
        on_split_imp = 0
        isplit = [len(l) == 3 and l != (2, 2, 2) for *_, l in imp]
        if fps and any(isplit):
            ig = np.array([(x * u, y * u) for (x, y, _), s in zip(imp, isplit) if s])
            on_split_imp = sum(bool(nr) for nr in cKDTree(ig).query_ball_point(np.array(fps), r=RADIUS_UM))
        r = out[tumor[n]]
        r["all"] += (tp_all, len(fps), fn_all)
        r["agreed"] += (tp_ag, len(fps), fn_ag)
        r["agreed2"] += (tp_ag, len(fps) - on_split_imp, fn_ag)
        dropped[tumor[n]] += (sum(split), on_split_imp)
    return out, dropped


def f1(tp, fp, fn):
    return {"F1": round(2 * tp / max(2 * tp + fp + fn, 1), 4), "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4), "TP": int(tp), "FP": int(fp), "FN": int(fn)}


out, dropped = score(thr)
tot = {k: sum(r[k] for r in out.values()) for k in ("all", "agreed", "agreed2")}
# 参考: test で F1 が最大になるしきい値（方式ごと）
grid = [round(t, 2) for t in np.arange(0.10, 0.81, 0.01)]
best = {}
for k in ("all", "agreed", "agreed2"):
    sc = {}
    for t in grid:
        o, _ = score(t)
        sc[t] = f1(*sum(r[k] for r in o.values()))["F1"]
    b = max(sc, key=sc.get)
    best[k] = {"thr": b, "F1": sc[b]}

print(json.dumps({
    "preds": sys.argv[1], "thr": thr, "slides": len(test),
    "dropped_split_mitoses": int(sum(d[0] for d in dropped.values())),
    "dropped_fp_on_split_imposters": int(sum(d[1] for d in dropped.values())),
    "total": {k: f1(*v) for k, v in tot.items()},
    "test_best_thr": best,
    "by_tumor": [{"tumor": t, "split_mitoses": int(dropped[t][0]), "fp_on_split_imposters": int(dropped[t][1]),
                  **{k: f1(*r[k]) for k in r}} for t, r in sorted(out.items())],
}, ensure_ascii=False, indent=1))
