#!/usr/bin/env python3
"""MIDOG++ を 640px の切片に切り分け、YOLO の形式にする。

Usage:
    python3 scripts/prepare.py              # 全 503 枚
    python3 scripts/prepare.py --limit 3    # 動作確認（各分割から少しだけ）

入力（setup.sh が取る）:
    data/MIDOG++.json              正解（COCO に近い形。bbox は [x1, y1, x2, y2]、どれも 50px 四方）
    data/datasets_xvalidation.csv  配布元の train/test 分割（標本単位、392/111 枚）
    data/Setup.ipynb               配布元の取得用ノートブック。figshare のファイル ID が書いてある

出力:
    data/midog/images/{train,val,test}/<標本>_<y>_<x>.jpg
    data/midog/labels/{train,val,test}/<標本>_<y>_<x>.txt
    data/midog.yaml

決めたこと:
- 縮小しない。分裂像は 50px 四方しかなく、縮めると見えなくなる。原寸で 640px に切る
- 区分は 2 つ: 0 = 分裂像、1 = 分裂像に似た別物（配布元の "not mitotic figure"）。
  まぎらわしい物を別の区分として覚えさせる。成績で見るのは 0 だけ
- val は配布元の train から標本単位で 15% を抜く（腫瘍の種類ごと、seed 0）。test は配布元のまま
- 何も写っていない切片は train では一部だけ残す（--empty-ratio）。val・test は全部残す
  （標本全体での見落とし・誤検出を測るため）
- 元の TIFF は 1 枚 0.14GB・全部で約 65GB。切ったら消す（--keep-raw で残す）
"""
import argparse
import csv
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 1 枚 4,000 万画素前後。PIL の既定の上限警告を外す

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "midog"
RAW = DATA / "raw"
TILE = 640
NAMES = {0: "mitotic_figure", 1: "imposter"}

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="各分割の標本数の上限（動作確認用）")
ap.add_argument("--empty-ratio", type=float, default=0.1, help="train で残す空の切片の割合")
ap.add_argument("--val-ratio", type=float, default=0.15)
ap.add_argument("--keep-raw", action="store_true")
args = ap.parse_args()

coco = json.loads((DATA / "MIDOG++.json").read_text())
images = {im["id"]: im for im in coco["images"]}
boxes = defaultdict(list)
for a in coco["annotations"]:
    boxes[a["image_id"]].append((a["category_id"] - 1, a["bbox"]))
by_name = {im["file_name"]: im for im in coco["images"]}

# figshare のファイル ID は配布元のノートブックの辞書 {'<ID>': '<名前>.tiff'} から取る。
# API は叩かない（2026-10-08、500 件を続けて引いて 403 で止められた）
nb = json.loads((DATA / "Setup.ipynb").read_text())
src = "".join("".join(c["source"]) for c in nb["cells"])
file_id = {name: fid for fid, name in re.findall(r"'(\d+)':\s*'(\d{3}\.tiff)'", src)}

rows = list(csv.DictReader(open(DATA / "datasets_xvalidation.csv"), delimiter=";"))
rng = random.Random(0)
split = {}
by_tumor = defaultdict(list)
for r in rows:
    name = f"{int(r['Slide']):03d}.tiff"
    if r["Dataset"] == "test":
        split[name] = "test"
    else:
        by_tumor[r["Tumor"]].append(name)
for tumor, names in sorted(by_tumor.items()):
    names = sorted(names)
    rng.shuffle(names)
    k = round(len(names) * args.val_ratio)
    for i, name in enumerate(names):
        split[name] = "val" if i < k else "train"

if args.limit:
    keep, seen = {}, defaultdict(int)
    for name in sorted(split):
        if seen[split[name]] < args.limit:
            keep[name] = split[name]
            seen[split[name]] += 1
    split = keep

missing = sorted(n for n in split if n not in file_id or n not in by_name)
if missing:
    sys.exit(f"ファイル ID か正解が無い標本: {missing[:5]} ...（{len(missing)} 件）")


def starts(n):
    """0, 640, ... と並べ、最後の切片は端にそろえる（はみ出さない）"""
    s = list(range(0, max(n - TILE, 0) + 1, TILE))
    if s[-1] + TILE < n:
        s.append(n - TILE)
    return s


for s in ("train", "val", "test"):
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

stats = defaultdict(lambda: defaultdict(int))
done_log = OUT / "done.txt"
done = set(done_log.read_text().split()) if done_log.exists() else set()
for n, (name, s) in enumerate(sorted(split.items()), 1):
    if name in done:
        continue
    raw = RAW / name
    if not raw.exists():
        tmp = raw.with_suffix(".part")
        urllib.request.urlretrieve(f"https://ndownloader.figshare.com/files/{file_id[name]}", tmp)
        tmp.rename(raw)
    im = by_name[name]
    img = Image.open(raw).convert("RGB")
    W, H = img.size
    assert (W, H) == (im["width"], im["height"]), (name, img.size)
    stem = name[:-5]
    for y in starts(H):
        for x in starts(W):
            lines = []
            for c, (x1, y1, x2, y2) in boxes[im["id"]]:
                cx1, cy1 = max(x1, x), max(y1, y)
                cx2, cy2 = min(x2, x + TILE), min(y2, y + TILE)
                # 半分以上が切片に入っている枠だけ残す（境目で割れた枠の片割れを数えない）
                if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < 0.5 * (x2 - x1) * (y2 - y1):
                    continue
                lines.append(f"{c} {((cx1 + cx2) / 2 - x) / TILE:.6f} {((cy1 + cy2) / 2 - y) / TILE:.6f} "
                             f"{(cx2 - cx1) / TILE:.6f} {(cy2 - cy1) / TILE:.6f}")
            if not lines and s == "train" and rng.random() >= args.empty_ratio:
                stats[s]["empty_dropped"] += 1
                continue
            tile = f"{stem}_{y:05d}_{x:05d}"
            img.crop((x, y, x + TILE, y + TILE)).save(OUT / "images" / s / f"{tile}.jpg", quality=95)
            (OUT / "labels" / s / f"{tile}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats[s]["tiles"] += 1
            stats[s]["boxes"] += len(lines)
    if not args.keep_raw:
        raw.unlink()
    with done_log.open("a") as f:
        f.write(name + "\n")
    print(f"{n}/{len(split)} {name} {s} {W}x{H}", file=sys.stderr)

(DATA / "midog.yaml").write_text(
    f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    + "".join(f"  {k}: {v}\n" for k, v in NAMES.items()))
(OUT / "split.json").write_text(json.dumps(split, indent=1, sort_keys=True))
print(json.dumps(stats, indent=1))
