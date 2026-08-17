#!/usr/bin/env bash
# OCR skill 环境初始化：uv venv(Python 3.12) + 依赖 + 模型预热（幂等）
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$SKILL_DIR/.venv}"
PY_BIN="$VENV_DIR/bin/python"

echo "[venv_setup] 1/3 venv（Python 3.12）"
if [ ! -x "$PY_BIN" ]; then
  uv venv "$VENV_DIR" --python 3.12
fi

echo "[venv_setup] 2/3 依赖安装"
uv pip install --python "$PY_BIN" \
  "rapidocr>=3.9" onnxruntime pymupdf pillow numpy pytest

echo "[venv_setup] 3/3 模型预热（首次联网下载并 SHA256 校验，之后离线）"
if [ -f "$SKILL_DIR/scripts/ocr.py" ]; then
  "$PY_BIN" "$SKILL_DIR/scripts/ocr.py" --smoke
else
  echo "[venv_setup] 跳过预热：scripts/ocr.py 尚未创建（Task 5 后重跑本脚本即可）"
fi

echo "[venv_setup] 完成"
