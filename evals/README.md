# 评测素材说明

eval 提示中的图片/PDF 文件在执行时由 `tests/image_factory.py` 生成（确定性绘制，
不入库）。生成方式：
```bash
PYTHONPATH=tests .venv/bin/python - <<'PY'
from image_factory import make_text_image, make_ui_image, make_table_image, make_pdf
make_text_image("/tmp/eval_invoice.png")
make_ui_image("/tmp/eval_ui.png")
make_table_image("/tmp/eval_table.png")
make_pdf("/tmp/eval_two.pdf")
PY
```
生成后将文件路径填入 `evals.json` 的 `files` 字段再执行双跑。
（中文渲染需 macOS 字体或安装 Noto CJK；PDF 素材为两页 "Page Content"。）
