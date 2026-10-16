"""export_onnx.py で作った ONNX が、Ultralytics の推論と同じ枠を出すかを確かめる。

切片は原寸の 640 角なので letterbox は要らない（JS 側も、切片をそのまま 0..1 にして渡す）。
区分・座標（3px 以内）で突き合わせる。

使い方:
  ~/git/ndl/ndl-layout/.venv/bin/python scripts/check_onnx.py runs/yolo26x-midog/weights/best.pt out/midog-yolo26x.onnx
"""
import argparse
import glob

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO

IMGSZ = 640


def load(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    assert im.size == (IMGSZ, IMGSZ), (path, im.size)
    return (np.asarray(im, dtype=np.float32) / 255).transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--images", default="data/midog/images/test/*.jpg")
    ap.add_argument("-n", type=int, default=0, help="使う切片の数（0 = 全部）")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    yolo = YOLO(a.weights)
    images = sorted(glob.glob(a.images))
    if a.n:
        images = images[: a.n]
    ref = []
    for p in images:
        r = yolo.predict(p, imgsz=IMGSZ, conf=a.conf, verbose=False)[0]
        ref.append((r.boxes.xyxy.numpy(), r.boxes.cls.numpy(), r.boxes.conf.numpy()))

    for path in a.onnx:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        n_pt = n_ox = n_match = 0
        worst = dconf = 0.0
        for p, (pt, pc, pf) in zip(images, ref):
            o = sess.run(None, {"images": load(p)})[0][0]  # [300, 6]
            o = o[o[:, 4] >= a.conf]
            n_pt += len(pt)
            n_ox += len(o)
            for b, c, f in zip(pt, pc, pf):
                same = o[o[:, 5] == c]
                if len(same) == 0:
                    continue
                d = np.abs(same[:, :4] - b).max(1)
                j = d.argmin()
                if d[j] < 3:
                    n_match += 1
                    worst = max(worst, d[j])
                    dconf = max(dconf, abs(same[j, 4] - f))
        print(f"{path}: {len(images)} 切片 / Ultralytics {n_pt} 枠 / ONNX {n_ox} 枠 / 一致 {n_match}"
              f"（座標の差の最大 {worst:.2f}px、確率の差の最大 {dconf:.4f}）")


if __name__ == "__main__":
    main()
