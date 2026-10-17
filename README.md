# midog-detect

病理画像の分裂像（mitotic figure）を YOLO で検出する試作。
データは [MIDOG++](https://doi.org/10.1038/s41597-023-02327-4)（画像と正解は CC0、figshare）。
古典籍のレイアウト検出（ndl-layout）の手順を移したもの。経緯と結果は [docs/notes/training.md](docs/notes/training.md)。

- 画面（ブラウザの中だけで推論する）: https://nakamura196.github.io/midog-detect/
- モデル（ONNX）: https://huggingface.co/nakamura196/yolo26x-midog-onnx

**研究用の試作です。医療機器ではなく、診断には使えません。**

## 使い方

```
zsh scripts/send.zsh                       # 手元から GPU サーバ（mdx-gpu）へスクリプトを送る
VENV=~/venv bash scripts/setup.sh          # GPU サーバで: 正解・分割を取り、503 標本を 640px に切る（約 65GB を順に取って消す）
VENV=~/venv bash scripts/train.sh          # GPU サーバで: v8x・11x・26x を学習し、test で測る
python3 scripts/eval_points.py runs/yolo11x-midog/weights/best.pt  # 配布元の評価（中心 7.5µm 以内の F1）
```

`data/`・`runs/` は git に入れない。

## ブラウザで動かす（web/）

標本を IIIF のタイルで表示し、640px の切片ごとにブラウザの中で推論する（onnxruntime-web、WebGPU か CPU）。

```
~/git/ndl/ndl-layout/.venv/bin/python scripts/export_onnx.py runs/yolo26x-midog/weights/best.pt out/midog-yolo26x.onnx
~/git/ndl/ndl-layout/.venv/bin/python scripts/check_onnx.py runs/yolo26x-midog/weights/best.pt out/midog-yolo26x.onnx out/midog-yolo26x-fp16.onnx
python3 scripts/make_demo.py data/raw/006.tiff   # 元の TIFF（figshare）→ web/public/iiif/006/ と正解の点（ほかの 7 標本も同じ）
cd web && npm ci && npm run dev                  # public/models/ は ../out/ への symlink
node scripts/check.mjs http://localhost:5173/ ../out/web-points-006.json --wasm   # ヘッドレスで最後まで流す
```

- `?slide=268` で標本を選ぶ（test の 8 標本: 006・100 乳がん、220 肺がん、268 リンパ腫、345 肥満細胞腫、353 神経内分泌腫瘍、458 軟部肉腫、552 黒色腫。画面の欄からも選べる）
- `?info=<IIIF の info.json>` で別の画像も読める（原寸のタイルがあれば、レベル 0 でも画像サーバでもよい）
- 研究用の試作。診断には使えない

## 出典

Aubreville M. et al. A comprehensive multi-domain dataset for mitotic figure detection.
Scientific Data 10, 484 (2023). https://doi.org/10.1038/s41597-023-02327-4

## ライセンス

コードと重みは AGPL-3.0（[LICENSE](LICENSE)）。重みは Ultralytics YOLO から作ったものなので、それに合わせている。
`web/public/` の画像のタイルと正解は MIDOG++ から作ったもので、CC0。

作成: 中村 覚（東京大学）
