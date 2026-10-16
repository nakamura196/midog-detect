"""ブラウザのデモ（web/）に置く標本を作る。元の TIFF → IIIF のタイルと、正解の点。

- タイルは IIIF Image API 3 のレベル 0（静的ファイルだけで配れる形）。512 角。
  JPEG は品質 95・色の間引き 4:2:0 で、学習の切片（PIL の quality=95）と同じ保存の仕方にする。
  vips は品質 90 以上だと既定で色を間引かない。品質 90 や間引きなしでは、同じ標本で確信度が最大 0.18〜0.32 ずれた
  元の TIFF は 4 バンド（RGBA）なので、学習の切り分け（PIL の convert("RGB")）と同じく先頭 3 バンドを取る
- info.json の id は仮の値。画面が読み込んだ場所から付け直す
- 判定（中心が 7.5µm 以内なら正解）のため、画素の大きさも書く
- 正解は配布元の MIDOG++.json から。枠の中心の点と区分（0 = 分裂像、1 = 似た別物）

使い方:
  python3 scripts/make_demo.py data/raw/006.tiff
  → web/public/iiif/006/ と web/public/data/006.json
"""
import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiff")
    ap.add_argument("--labels", default=str(ROOT / "data/MIDOG++.json"))
    a = ap.parse_args()
    tiff = Path(a.tiff)
    name = tiff.stem
    out = ROOT / "web/public"

    with tempfile.TemporaryDirectory() as tmp:
        rgb = Path(tmp) / "rgb.v"
        subprocess.run(["vips", "extract_band", str(tiff), str(rgb), "0", "--n", "3"], check=True)
        dest = out / "iiif" / name
        subprocess.run(["vips", "dzsave", str(rgb), str(dest), "--layout", "iiif3", "--tile-size", "512",
                        "--overlap", "0", "--id", "https://example.invalid/iiif", "--suffix", ".jpg[Q=95,subsample-mode=on]"],
                       check=True)
        (out / "iiif" / "vips-properties.xml").unlink(missing_ok=True)  # dzsave が書く元の画像の情報。配らない

    # 画素の大きさ（µm）。スキャナで違う。vips の xres は 1mm あたりの画素数（eval_points.py の値と同じになる）
    xres = float(subprocess.run(["vipsheader", "-f", "xres", str(tiff)], check=True, capture_output=True, text=True).stdout)

    d = json.loads(Path(a.labels).read_text())
    im = next(i for i in d["images"] if i["file_name"] == tiff.name)
    # 手元の split.json は試しに切り分けた標本しか持たないので、test かどうかは配布元の分割から取る
    split = json.loads((ROOT / "data/midog/split.json").read_text()).get(tiff.name)
    if split is None:
        rows = csv.DictReader(open(ROOT / "data/datasets_xvalidation.csv"), delimiter=";")
        split = next((r["Dataset"] for r in rows if f"{int(r['Slide']):03d}.tiff" == tiff.name), None)
    pts = [
        {"x": (b[0] + b[2]) / 2, "y": (b[1] + b[3]) / 2, "cls": x["category_id"] - 1}
        for x in d["annotations"] if x["image_id"] == im["id"] for b in [x["bbox"]]
    ]
    meta = {"name": name, "width": im["width"], "height": im["height"], "tumor": im["tumor_type"],
            "split": split, "um_per_px": 1000 / xres, "source": "MIDOG++ (CC0), https://doi.org/10.1038/s41597-023-02327-4",
            "truth": pts}
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False))
    n_mf = sum(p["cls"] == 0 for p in pts)
    print(f"{name}: {im['width']}x{im['height']} {im['tumor_type']} ({split}) 分裂像 {n_mf} / 似た別物 {len(pts) - n_mf}")


if __name__ == "__main__":
    main()
