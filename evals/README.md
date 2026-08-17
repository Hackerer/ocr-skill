# 评测素材说明

eval 提示中的图片文件在执行时由 `tests/image_factory.py` 生成（确定性绘制，
不入库）。生成方式：
```bash
PYTHONPATH=tests .venv/bin/python - <<'PY'
from image_factory import make_text_image, make_ui_image
make_text_image("/tmp/eval_invoice.png")
make_ui_image("/tmp/eval_ui.png")
PY
```
生成后将文件路径填入 `evals.json` 的 `files` 字段再执行双跑。
真实素材（发票/截图/设计稿）可随时补充到 `evals/assets/`（不入库则标注来源）。
