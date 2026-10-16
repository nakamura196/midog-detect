#!/usr/bin/env bash
# 1 つの腫瘍の種類を抜いて学習し、配布元の評価（7.5µm 以内の F1）で測る。GPU サーバの tmux の中で流す。
#
# 使い方（GPU サーバ上で）:
#   tmux new -s midog-loto 'VENV=~/venv bash scripts/loto.sh net 2>&1 | tee -a runs/loto.log'
#   bash scripts/loto.sh lung lymphoma ...    # 順に流す。学習済みのものは飛ばす
#
# 種類の短い名前は scripts/loto.py の TUMORS（breast lung lymphoma mast net sarcoma melanoma）。
# 既存の全種類で学習したモデル（runs/yolov8x-midog）と比べるため、モデル・seed・batch・imgsz は同じにする。
# patience だけ 50 → 20 に下げる（既存の v8x は 30 エポック目が最良で、20 でも同じエポックが選ばれる）。
#
# 結果: runs/yolov8x-loto-<名前>/weights/best.pt と runs/yolov8x-loto-<名前>/test_points.json
set -euo pipefail
cd "$(dirname "$0")/.."
. "${VENV:-.venv}/bin/activate"
mkdir -p runs

MODEL="${MODEL:-yolov8x.pt}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
BATCH="${BATCH:-16}"

for key in "$@"; do
  name="${MODEL%.pt}-loto-${key}"
  data="$(python3 scripts/loto.py "$key")"
  held="$(python3 scripts/loto.py --name "$key")"
  if [ -f "runs/${name}/weights/best.pt" ]; then
    echo "skip ${name}（学習済み）"
  else
    echo "== $(date -u +%FT%TZ) train ${name}（${held} を抜く）"
    yolo detect train model="$MODEL" data="$data" imgsz=640 \
      epochs="$EPOCHS" patience="$PATIENCE" batch="$BATCH" seed=0 deterministic=True \
      project="$(pwd)/runs" name="$name" exist_ok=True plots=True
  fi
  echo "== $(date -u +%FT%TZ) eval ${name}"
  python3 scripts/eval_points.py "runs/${name}/weights/best.pt" --holdout "$held" > "runs/${name}/test_points.json"
  python3 -c "import json; d = json.load(open('runs/${name}/test_points.json')); print(json.dumps({k: d[k] for k in ('holdout', 'threshold_from_val', 'test', 'test_holdout')}, ensure_ascii=False))"
done
echo "== $(date -u +%FT%TZ) done"
