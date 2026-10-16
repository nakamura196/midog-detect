#!/usr/bin/env python3
"""1 つの腫瘍の種類を抜いた学習用の設定を書き出す（leave-one-tumor-out）。切り直しはしない。

Usage:
    python3 scripts/loto.py net            # data/loto/net/{train,val}.txt と data.yaml を作る
    python3 scripts/loto.py --count        # 腫瘍ごとの切片数を出すだけ
    python3 scripts/loto.py --name net     # MIDOG++.json での種類の名前を出すだけ

- 切片のファイル名は「標本番号_y_x.jpg」なので、標本の腫瘍の種類で振り分けられる
- 抜いた種類は train と val の両方から外す（val は最良のエポックとしきい値を選ぶのに使うため）
- test は全部の種類を残す（抜いた種類のほか、残した種類の成績も見る）
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MIDOG = DATA / "midog"

# 短い名前 → MIDOG++.json の tumor_type
TUMORS = {
    "breast": "human breast cancer",
    "lung": "canine lung cancer",
    "lymphoma": "canine lymphosarcoma",
    "mast": "canine cutaneous mast cell tumor",
    "net": "human neuroendocrine tumor",
    "sarcoma": "canine soft tissue sarcoma",
    "melanoma": "human melanoma",
}

coco = json.loads((DATA / "MIDOG++.json").read_text())
tumor = {im["file_name"][:3]: im["tumor_type"] for im in coco["images"]}


def tiles(split):
    return sorted((MIDOG / "images" / split).glob("*.jpg"))


if sys.argv[1] == "--name":
    print(TUMORS[sys.argv[2]])
    sys.exit()

if sys.argv[1:] == ["--count"]:
    for s in ("train", "val", "test"):
        c = Counter(tumor[f.name[:3]] for f in tiles(s))
        print(s, sum(c.values()), dict(sorted(c.items())))
    sys.exit()

key = sys.argv[1]
held = TUMORS[key]
out = DATA / "loto" / key
out.mkdir(parents=True, exist_ok=True)
for s in ("train", "val"):
    keep = [str(f) for f in tiles(s) if tumor[f.name[:3]] != held]
    (out / f"{s}.txt").write_text("\n".join(keep) + "\n")
    print(f"{s}: {len(keep)} / {len(tiles(s))} 切片（{held} を除く）", file=sys.stderr)
(out / "data.yaml").write_text(
    f"path: {MIDOG}\ntrain: {out / 'train.txt'}\nval: {out / 'val.txt'}\ntest: images/test\n"
    "names:\n  0: mitotic_figure\n  1: imposter\n")
print(out / "data.yaml")
