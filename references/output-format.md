# OCR 输出格式（按需加载）

## ocr.py 纯文本模式（默认）
多文件多页分隔头：`[文件i/共N] 文件: <名>（页 p/共M）`（N=文件总数，M=该文件页数）
低置信文本带 `⟦低置信⟧` 前缀（conf < 0.5，保留不删除）。
空行分隔段落/块。无文字时输出 `未检测到文字`。

## ocr.py --json（顶层数组，UTF-8 中文不转义）
```json
[
  {
    "file": "screenshot.png",
    "page": 1,
    "width": 1280,
    "height": 800,
    "lines": [
      {
        "text": "商品详情",
        "conf": 0.98,
        "font_size": 36.0,
        "box": [12.0, 20.0, 108.0, 52.0],
        "low_conf": false
      }
    ]
  }
]
```
字段说明：
- `box`：`[x1, y1, x2, y2]`，检测多边形外接矩形，原图像素坐标系
- `font_size`：≈ 框高，用于字号层级推断
- `low_conf`：`conf < 0.5` 时为 true
- 多页 PDF 每个页面一个对象（page 递增）；多文件依次排列

## ocr.py --table（TSV）
- 列边界 = 全部文本框 x1 的一维聚类中心（gap=12），文本按左缘 x1 就近落列
- 空单元格留空，`\t` 分隔、`\n` 分行；表头需 LLM 猜测标注
- 已知局限：右对齐数值列可能碎列（输出尾部附提示）；复杂表格（合并/斜线）请改用 --json

## analyze.py（事实清单 JSON，结论由 LLM 下；顶层恒为数组）
```json
[
  {
    "file": "ui.png",
    "palette": [{"hex": "#1F2937", "pct": 42.1, "role": "bg"}],
    "contrast_issues": [
      {"text": "保存", "box": [...], "fg": "#9CA3AF", "bg": "#F3F4F6",
       "ratio": 2.1, "wcag": "AA 未达标(需≥4.5)"}
    ],
    "font_size_clusters": [
      {"size": 36.0, "count": 3, "texts": ["商品详情"], "consistent": true}
    ],
    "alignment_notes": [
      {"type": "left_align_group", "x": 120.0, "elements": 5}
    ]
  }
]
```
- `wcag: null` = 对比度通过 AA（4.5:1）；`contrast_issues` 只含未达标块
- `role: "bg"` 为启发式（占比最大主色），仅供参考
