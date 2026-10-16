"""YOLO26x（分裂像）を、ブラウザで動かす ONNX に書き出す。

ndl-layout の export_onnx.py と違い、途中の層の特徴は出さない（3 次元の演出はしない）。
Ultralytics のふつうの書き出しをそのまま使う。YOLO26 は NMS を使わないので、
出力は 1 切片につき確率の高い順の 300 枠で、JS 側は しきい値で切るだけでよい。

入力:  images [1, 3, 640, 640]  原寸の 640 角の切片（縮小しない。学習と同じ）。RGB、0..1
出力:  output0 [1, 300, 6]      x1, y1, x2, y2（切片の座標）, 確率, 区分（0 = 分裂像、1 = 似た別物）

使い方:
  ~/git/ndl/ndl-layout/.venv/bin/python scripts/export_onnx.py runs/yolo26x-midog/weights/best.pt out/midog-yolo26x.onnx
  （-fp16.onnx と、区分名・しきい値の .json も同じ場所にできる）

必要なもの: ultralytics==8.4.119 onnx==1.19.1 onnxruntime==1.23.2 onnxconverter-common==1.16.0
"""
import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import onnx
from onnxconverter_common import float16
from ultralytics import YOLO

IMGSZ = 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--points", help="eval_points.py の結果（しきい値を取る）。既定は重みの 2 つ上の test_points.json")
    a = ap.parse_args()

    yolo = YOLO(a.weights)
    tmp = yolo.export(format="onnx", imgsz=IMGSZ, opset=17, simplify=False)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)

    # しきい値は val で選んだもの（eval_points.py）。test を見て選んだ値は使わない
    points = Path(a.points) if a.points else Path(a.weights).parents[1] / "test_points.json"
    thr = json.loads(points.read_text())["threshold_from_val"] if points.exists() else None
    meta = {
        "imgsz": IMGSZ,
        "names": [yolo.names[k] for k in sorted(yolo.names)],
        "threshold": thr,
        "threshold_source": str(points) if thr is not None else None,
        "source": a.weights,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    # ブラウザ（WebGPU）用の 16 ビット版。入出力は float32 のまま残す。
    # 最後の畳み込みより後（枠の復元と上位 300 の選び出し）は 32 ビットのまま残す。
    # そのまま変換すると、そこの Cast の型が合わなくなり、読み込みで落ちる
    m32 = onnx.load(str(out))
    nodes = m32.graph.node
    last_conv = max(i for i, n in enumerate(nodes) if n.op_type == "Conv")
    keep = [n.name for n in nodes[last_conv + 1 :]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 1e-7 未満の重みを丸める警告が数百行出る
        m16 = float16.convert_float_to_float16(m32, keep_io_types=True, node_block_list=keep)
    out16 = out.with_name(out.stem + "-fp16.onnx")
    onnx.save(m16, str(out16))
    print(f"wrote {out16} ({out16.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
