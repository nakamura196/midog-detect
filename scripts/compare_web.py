"""ブラウザの予測（web/scripts/check.mjs の出力）を、Python の予測（eval_points.py の points_test.json）と突き合わせる。

入力の違い: Python は元の TIFF から切った切片（JPEG 品質 95）、ブラウザは IIIF のタイル（JPEG 品質 90）を貼り合わせたもの。
同じ物を指す点は 7.5µm 以内なら同じとみなし、確信度の差も見る。

使い方:
  python3 scripts/compare_web.py out/web-points-006.json out/py-points-006.json --thr 0.39 --um 0.2263
"""
import argparse
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("web")
    ap.add_argument("py")
    ap.add_argument("--thr", type=float, default=0.39)
    ap.add_argument("--um", type=float, required=True, help="1 画素の大きさ（µm）")
    a = ap.parse_args()
    r = 7.5 / a.um
    web = [p for p in json.load(open(a.web)) if p["cls"] == 0]
    py = json.load(open(a.py))  # [x, y, 確信度]（分裂像だけ）
    for lo in (0.01, a.thr):
        w = [p for p in web if p["score"] >= lo]
        q = [p for p in py if p[2] >= lo]
        used = set()
        diffs = []
        for x, y, s in sorted(q, key=lambda p: -p[2]):
            best = min(((math.hypot(p["x"] - x, p["y"] - y), j) for j, p in enumerate(w) if j not in used), default=None)
            if best and best[0] <= r:
                used.add(best[1])
                diffs.append((best[0], abs(w[best[1]]["score"] - s)))
        dmax = max((d for d, _ in diffs), default=0)
        smax = max((s for _, s in diffs), default=0)
        print(f"確信度 {lo} 以上: Python {len(q)} 点 / ブラウザ {len(w)} 点 / 対応 {len(diffs)}"
              f"（位置の差の最大 {dmax:.1f}px、確信度の差の最大 {smax:.3f}）")


if __name__ == "__main__":
    main()
