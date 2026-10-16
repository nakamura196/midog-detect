#!/usr/bin/env python3
"""配布元の test（標本単位）でモデルを測り、区分ごとの成績を JSON で出す。

Usage:
    python3 scripts/eval.py runs/yolo26x-midog/weights/best.pt > runs/yolo26x-midog/test.json
"""
import contextlib
import json
import sys

weights = sys.argv[1]
data = sys.argv[2] if len(sys.argv) > 2 else "data/midog.yaml"

# Ultralytics は読み込んだ時点の標準出力に進捗を書くので、読み込みから標準エラーへ回す。
# JSON だけを標準出力に出すため
with contextlib.redirect_stdout(sys.stderr):
    from ultralytics import YOLO
    m = YOLO(weights).val(data=data, split="test", imgsz=640, batch=16, plots=False, verbose=False)
box = m.box
per_class = {}
for i, c in enumerate(m.ap_class_index):
    p, r, ap50, ap = box.class_result(i)
    per_class[m.names[int(c)]] = {"P": round(p, 3), "R": round(r, 3),
                                  "mAP50": round(ap50, 3), "mAP50-95": round(ap, 3)}
print(json.dumps({
    "weights": weights,
    "split": "test (MIDOG++ official, slide-level)",
    "all": {"P": round(box.mp, 3), "R": round(box.mr, 3),
            "mAP50": round(box.map50, 3), "mAP50-95": round(box.map, 3)},
    "per_class": per_class,
    "speed_ms": {k: round(v, 2) for k, v in m.speed.items()},
}, ensure_ascii=False, indent=1))
