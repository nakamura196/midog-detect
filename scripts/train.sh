#!/usr/bin/env bash
# モデルを順に学習し、配布元の test（標本単位、111 枚）で測る。GPU サーバの tmux の中で流す。
#
# 使い方（GPU サーバ上で）:
#   tmux new -s train 'bash scripts/train.sh 2>&1 | tee -a runs/train.log'
#   bash scripts/train.sh yolo26x.pt            # 1 つだけ流す
#   既存の環境を使うときは VENV=~/venv を前に付ける
#
# 比べるもの（既定）:
#   yolov8x  — 古典籍（ndl-layout）と同じ 3 世代で比べる
#   yolo11x  — YOLO11
#   yolo26x  — YOLO26（Ultralytics の最新世代）
#
# 結果: runs/<名前>/weights/best.pt と runs/<名前>/test.json
set -euo pipefail
cd "$(dirname "$0")/.."
. "${VENV:-.venv}/bin/activate"
mkdir -p runs

MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(yolov8x.pt yolo11x.pt yolo26x.pt)

EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-50}"
BATCH="${BATCH:-16}"

for m in "${MODELS[@]}"; do
  name="${m%.pt}-midog"
  if [ -f "runs/${name}/weights/best.pt" ]; then
    echo "skip ${name}（学習済み）"
  else
    echo "== $(date -u +%FT%TZ) train ${name}"
    yolo detect train model="$m" data=data/midog.yaml imgsz=640 \
      epochs="$EPOCHS" patience="$PATIENCE" batch="$BATCH" seed=0 deterministic=True \
      project="$(pwd)/runs" name="$name" exist_ok=True plots=True
  fi
  echo "== $(date -u +%FT%TZ) test ${name}"
  python3 scripts/eval.py "runs/${name}/weights/best.pt" > "runs/${name}/test.json"
  cat "runs/${name}/test.json"
done
echo "== $(date -u +%FT%TZ) done"
