#!/usr/bin/env zsh
# 学習コードを GPU サーバへ送る（データは向こうで figshare から取る）。
#
# 使い方:
#   zsh scripts/send.zsh                 # 既定は mdx-gpu:~/midog-detect
#   zsh scripts/send.zsh <host> <dir>
set -euo pipefail
HOST="${1:-mdx-gpu}"
DEST="${2:-midog-detect}"
HERE="${0:A:h:h}"
ssh "$HOST" "mkdir -p '$DEST/scripts' '$DEST/runs'"
rsync -a "$HERE/scripts/" "$HOST:$DEST/scripts/"
echo "送りました: $HOST:$DEST/scripts/"
