#!/usr/bin/env bash
# GPU サーバで 1 回だけ流す: 仮想環境・正解・分割・取得先を用意し、画像を切り分ける。
#
# 使い方（GPU サーバ上で、tmux の中で。全部で 65GB を取るので数時間かかる）:
#   VENV=~/venv bash scripts/setup.sh              # 既存の環境を使う
#   VENV=~/venv LIMIT=2 bash scripts/setup.sh      # 動作確認（各分割 2 標本だけ）
#
# できるもの:
#   data/MIDOG++.json, data/datasets_xvalidation.csv, data/Setup.ipynb
#   data/midog/ と data/midog.yaml（scripts/prepare.py が作る）
set -euo pipefail
cd "$(dirname "$0")/.."

# 公開から 1 週間以上たった版に固定する（サプライチェーン攻撃対策）。
# ndl-layout と同じく、既存の ~/venv（Ultralytics 8.4.119）を使うなら作らない
ULTRALYTICS_VERSION=8.4.170
VENV="${VENV:-.venv}"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "ultralytics==${ULTRALYTICS_VERSION}"
fi
. "$VENV/bin/activate"
mkdir -p data

# 正解（CC0）は figshare、分割と取得先は配布元の GitHub（MIT）から。
# GitHub 側はコミットを固定する（2026-10-08 時点の main）
MIDOGPP_COMMIT=c8f4a18d261e5c9d09c04b0925d40cd7935de618
GH="https://raw.githubusercontent.com/DeepMicroscopy/MIDOGpp/${MIDOGPP_COMMIT}"
[ -f data/MIDOG++.json ] || curl -fsSL -o data/MIDOG++.json https://ndownloader.figshare.com/files/41265615
[ -f data/datasets_xvalidation.csv ] || curl -fsSL -o data/datasets_xvalidation.csv "$GH/datasets_xvalidation.csv"
[ -f data/Setup.ipynb ] || curl -fsSL -o data/Setup.ipynb "$GH/Setup.ipynb"

python3 -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ultralytics', ultralytics.__version__)"
if [ -n "${LIMIT:-}" ]; then
  python3 scripts/prepare.py --limit "$LIMIT"
else
  python3 scripts/prepare.py
fi
