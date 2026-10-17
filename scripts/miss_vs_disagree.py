#!/usr/bin/env python3
"""見逃し（と誤検出）が、病理医の判定が割れた正解に偏るかを、test の全標本で数える。

Usage:
    python3 scripts/miss_vs_disagree.py out/py-points-test-26x.json --thr 0.39

予測は eval_points.py が作る標本ごとの (x, y, 確信度) の表（区分「分裂像」の予測だけ）。
突き合わせは eval_points.py と同じ（µm に直し、正解の順に 7.5µm 以内で近い、未使用の予測を当てる）。

正解の labels 欄（data/MIDOG++.json）: 要素が 3 つなら「最初の 2 つが食い違い、3 つ目が決めた」。
ここでは分裂像の正解のうち要素 3 つのものを「割れた」とする。
誤検出は、7.5µm 以内に「似た別物」の正解があるかと、それが割れたもの（(1,2,2)）かを数える。
(2,2,2) は割れた例に入れない（3 人とも似た別物）。
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RADIUS_UM = 7.5

pts = json.loads(Path(sys.argv[1]).read_text())
thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.39
coco = json.loads((DATA / "MIDOG++.json").read_text())
sp = json.loads((DATA / "spacing.json").read_text())  # eval_points.py が作る標本ごとの µm/px
names = {im["id"]: im["file_name"] for im in coco["images"]}
tumor = {im["file_name"]: im["tumor_type"] for im in coco["images"]}
ann = defaultdict(lambda: {1: [], 2: []})
for a in coco["annotations"]:
    x1, y1, x2, y2 = a["bbox"]
    ann[names[a["image_id"]]][a["category_id"]].append(((x1 + x2) / 2, (y1 + y2) / 2, tuple(a["labels"])))
# test は配布元の分割（data/midog/split.json は手元の試しの 3 枚だけのことがあるので使わない）
rows_csv = csv.DictReader(open(DATA / "datasets_xvalidation.csv"), delimiter=";")
test = sorted(f"{int(r['Slide']):03d}.tiff" for r in rows_csv if r["Dataset"] == "test")
assert set(test) == set(pts), "予測の標本と配布元の test が食い違う"


def match(g, p, r):
    """eval_points.score_slide と同じ貪欲法。正解ごとの当たり外れと、使った予測の印を返す"""
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


# 行: 腫瘍の種類ごとの [当たり, 当たりのうち割れた, 見逃し, 見逃しのうち割れた, 誤検出, 似た別物の上, うち割れた,
#     似た別物の正解の数, うち割れた（比べる基準）]
rows = defaultdict(lambda: np.zeros(9, int))
for n in test:
    u = sp[n]
    mit = ann[n][1]
    imp = ann[n][2]
    g = [(x * u, y * u) for x, y, _ in mit]
    p = [(x * u, y * u) for x, y, c in pts[n] if c > thr]
    hit, used = match(g, p, RADIUS_UM)
    split3 = [len(l) == 3 for *_, l in mit]
    r = rows[tumor[n]]
    r[0] += sum(h for h in hit)
    r[1] += sum(h and s for h, s in zip(hit, split3))
    r[2] += sum(not h for h in hit)
    r[3] += sum((not h) and s for h, s in zip(hit, split3))
    fps = [q for q, k in zip(p, used) if not k]
    r[4] += len(fps)
    r[7] += len(imp)
    r[8] += sum(len(l) == 3 and l != (2, 2, 2) for *_, l in imp)
    if fps and imp:
        ig = np.array([(x * u, y * u) for x, y, _ in imp])
        isplit = [len(l) == 3 and l != (2, 2, 2) for *_, l in imp]
        for near in cKDTree(ig).query_ball_point(np.array(fps), r=RADIUS_UM):
            if near:
                r[5] += 1
                r[6] += any(isplit[j] for j in near)

tot = sum(rows.values())


def line(name, r):
    tp, tps, fn, fns, fp, fpi, fpis, ni, nis = map(int, r)
    p = fisher_exact([[fns, fn - fns], [tps, tp - tps]])[1] if tp and fn else float("nan")
    return {"tumor": name, "TP": tp, "TP_split": tps, "TP_split_rate": round(tps / max(tp, 1), 3),
            "FN": fn, "FN_split": fns, "FN_split_rate": round(fns / max(fn, 1), 3), "fisher_p": float(f"{p:.2g}"),
            "FP": fp, "FP_on_imposter": fpi, "FP_on_split_imposter": fpis,
            "imposters": ni, "imposters_split": nis, "imposters_split_rate": round(nis / max(ni, 1), 3)}


print(json.dumps({"preds": sys.argv[1], "thr": thr, "slides": len(test),
                  "total": line("all", tot),
                  "by_tumor": [line(t, r) for t, r in sorted(rows.items())]},
                 ensure_ascii=False, indent=1))
