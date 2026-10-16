#!/usr/bin/env bash
# 1 エポックだけ学習して時間を測る（本番の学習の前に GPU の見積もりを出すため）。
# 切り分け（setup.sh）がまだ動いていれば、終わるまで待ってから始める（最長 6 時間）。
#
# 使い方（GPU サーバ上で、tmux の中で）:
#   tmux new -d -s midog-epoch1 'VENV=~/venv bash scripts/time_epoch.sh yolo26x.pt 2>&1 | tee runs/epoch1.log'
#
# 結果: runs/epoch1-<モデル>/results.csv（time 列 = 学習と val を合わせた秒数）と、
#       このスクリプトが最後に出す切片の数・経過秒数
set -euo pipefail
cd "$(dirname "$0")/.."
. "${VENV:-.venv}/bin/activate"
MODEL="${1:-yolo26x.pt}"
BATCH="${BATCH:-16}"

i=0
while pgrep -f "python3 scripts/prepare.py" >/dev/null && [ $i -lt 360 ]; do sleep 60; i=$((i+1)); done
if pgrep -f "python3 scripts/prepare.py" >/dev/null; then echo "切り分けが 6 時間で終わらなかった"; exit 1; fi
[ -f data/midog/split.json ] || { echo "切り分けが最後まで行っていない（data/midog/split.json が無い）"; exit 1; }

for s in train val test; do echo "$s $(ls data/midog/images/$s | wc -l) 切片"; done
name="epoch1-${MODEL%.pt}"
start=$(date +%s)
echo "== $(date -u +%FT%TZ) train ${name}"
yolo detect train model="$MODEL" data=data/midog.yaml imgsz=640 \
  epochs=1 batch="$BATCH" seed=0 deterministic=True \
  project="$(pwd)/runs" name="$name" exist_ok=True plots=False
echo "== $(date -u +%FT%TZ) 経過 $(( $(date +%s) - start )) 秒（モデルの読み込み・ラベルの読み込みを含む）"
