#!/usr/bin/env python
"""OCR CLI：图片/PDF → 阅读顺序文本 / JSON / 表格 TSV。

用法:
  ocr.py <文件...> [--json|--table] [--model-type tiny|small|medium] [--fast] [--smoke]

输出契约见 references/output-format.md。设计依据 docs/superpowers/specs/2026-08-17-ocr-skill-design.md §2/§4。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from rapidocr import ModelType, OCRVersion, RapidOCR

LOW_CONF_THRESHOLD = 0.5
PARAGRAPH_GAP_PX = 24.0    # render_text 段落分隔阈值：相邻行 y1 间距 > 该值插空行
COLUMN_GAP_PX = 12.0       # build_tsv 列聚类 gap：x1 差 > 该值分列
MAX_PDF_PIXELS = 2_000_000
MODEL_TYPES = {"tiny": ModelType.TINY, "small": ModelType.SMALL, "medium": ModelType.MEDIUM}


# ---------------------------------------------------------------- 引擎
def build_engine(model_type: str = "medium") -> RapidOCR:
    """按设计文档 §2.2：显式锁定 PP-OCRv6 + 指定规格；log_level=error 保证 stdout 干净。"""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"未知 model_type: {model_type}，可选: {sorted(MODEL_TYPES)}")
    mt = MODEL_TYPES[model_type]
    return RapidOCR(params={
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Det.model_type": mt,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_type": mt,
        # §2.3：text_score=0.0 关闭内置过滤（低置信由我方以 0.5 为界打标记），
        # 随引擎单例固化，避免后续 engine(img) 调用静默回到默认 0.5 过滤。
        "Global.text_score": 0.0,
        "Global.log_level": "error",
        # §2.2 确定性：显式禁用 CoreML（rapidocr 3.9.2 默认即 false，锁定防默认翻转）。
        "EngineConfig.onnxruntime.use_coreml": False,
    })


def warmup(engine: RapidOCR) -> None:
    """内置含文字小图跑完整管线（det→cls→rec），确保三个模型全部加载（§2.4）。"""
    from PIL import ImageDraw
    img = Image.new("RGB", (128, 64), "white")
    ImageDraw.Draw(img).text((8, 16), "OCR test", fill="black")
    engine(np.asarray(img), text_score=0.0)


# ---------------------------------------------------------------- 纯函数
def box_to_xyxy(box: np.ndarray) -> List[float]:
    """检测框 → [x1, y1, x2, y2] 外接矩形（兼容四顶点与四点一维两种格式）。"""
    a = np.asarray(box, dtype=float)
    if a.shape == (4, 2):
        xs, ys = a[:, 0], a[:, 1]
    elif a.shape == (4,):
        xs, ys = a[[0, 2]], a[[1, 3]]
    else:
        raise ValueError(f"unexpected box shape: {a.shape}")
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def cluster_1d(values: List[float], gap: float = 12.0) -> List[float]:
    """一维聚类：排序后相邻差 > gap 分簇，返回各簇均值（簇中心）。"""
    if not values:
        return []
    vals = sorted(values)
    clusters: List[List[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - clusters[-1][-1] > gap:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return [sum(c) / len(c) for c in clusters]


def group_into_rows(items: List[Dict]) -> List[List[Dict]]:
    """按 y 中心聚类成行：行内 y 中心差 <= max(min(行高, 候选框高)*0.6, 8) 归一行。"""
    rows: List[List[Dict]] = []
    for it in sorted(items, key=lambda i: (i["box"][1] + i["box"][3]) / 2):
        cy = (it["box"][1] + it["box"][3]) / 2
        it_h = it["box"][3] - it["box"][1]
        for row in rows:
            rys = [(r["box"][1] + r["box"][3]) / 2 for r in row]
            row_h = max(r["box"][3] - r["box"][1] for r in row)
            if abs(cy - sum(rys) / len(rys)) <= max(min(row_h, it_h) * 0.6, 8.0):
                row.append(it)
                break
        else:
            rows.append([it])
    return rows


def sort_lines_reading_order(items: List[Dict]) -> List[Dict]:
    """阅读顺序（§4.2）：行内按 x 排序，行间按 y 排序。"""
    rows = group_into_rows(items)
    for row in rows:
        row.sort(key=lambda i: i["box"][0])
    rows.sort(key=lambda r: sum(i["box"][1] for i in r) / len(r))
    return [it for row in rows for it in row]


# ---------------------------------------------------------------- 三模式输出
def render_text(items: List[Dict]) -> str:
    """模式一：阅读顺序纯文本（默认）。低置信文本保留并加 ⟦低置信⟧ 前缀。"""
    if not items:
        return "未检测到文字"
    ordered = sort_lines_reading_order(items)
    lines: List[str] = []
    prev_y2 = None
    for it in ordered:
        prefix = "⟦低置信⟧ " if it["conf"] < LOW_CONF_THRESHOLD else ""
        if prev_y2 is not None and it["box"][1] - prev_y2 > PARAGRAPH_GAP_PX:
            lines.append("")          # 段落/块分隔
        lines.append(prefix + it["text"])
        prev_y2 = it["box"][3]
    return "\n".join(lines)


def render_json(results: List[Dict]) -> str:
    """模式二：JSON（UTF-8，中文不转义）。"""
    return json.dumps(results, ensure_ascii=False, indent=2)


def build_tsv(items: List[Dict]) -> str:
    """模式三：表格 TSV。列边界 = 全部 x1 的一维聚类（gap=COLUMN_GAP_PX）→ 列中心；
    单元格 = 行带 × 列带，文本按框左缘 x1 就近落入列带（距离相等取左侧列）。"""
    if not items:
        return ""
    x1s = sorted({round(i["box"][0], 1) for i in items})
    col_centers = cluster_1d(x1s, gap=COLUMN_GAP_PX)
    out: List[str] = []
    for row in group_into_rows(items):
        row = sorted(row, key=lambda i: i["box"][0])   # 同格多文本按 x 排序（左→右），顺序确定
        cells = [""] * len(col_centers)
        for it in row:
            # 按框左缘 x1 就近落入列带；min 取首个最小下标 → 距离相等时取左侧列
            idx = min(range(len(col_centers)), key=lambda j: abs(col_centers[j] - it["box"][0]))
            cells[idx] = (cells[idx] + " " + it["text"]).strip()
        out.append("\t".join(cells))
    return "\n".join(out)


# ---------------------------------------------------------------- 输入
def load_images(path: str) -> Iterator[Tuple[Image.Image, int, int]]:
    """图片或 PDF → (PIL Image, 页号, 总页数)。PDF 按像素量 ~2MP 上限渲染（§2.4）。"""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        import pymupdf as fitz  # PyMuPDF（新包名导入，避免弃用别名 fitz 导入时 print 警告污染 stdout）
        doc = fitz.open(p)
        if doc.needs_pass:
            raise RuntimeError("PDF 已加密，请先解密")
        total = doc.page_count
        for i, page in enumerate(doc, start=1):
            r = page.rect
            scale = min(1.0, (MAX_PDF_PIXELS / (r.width * r.height)) ** 0.5)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples), i, total
        doc.close()
    else:
        yield Image.open(p).convert("RGB"), 1, 1


def ocr_file(engine: RapidOCR, path: str) -> List[Dict]:
    """单个文件 → JSON 结构列表（一文件可能多页）。text_score=0.0 关闭内置过滤（§2.3）。"""
    results: List[Dict] = []
    for img, page, _total in load_images(path):
        arr = np.asarray(img)
        res = engine(arr, text_score=0.0)
        items: List[Dict] = []
        if res.boxes is not None:
            for box, txt, score in zip(res.boxes, res.txts, res.scores):
                b = box_to_xyxy(box)
                items.append({
                    "text": txt,
                    "conf": round(float(score), 4),
                    "font_size": round(b[3] - b[1], 1),
                    "box": [round(v, 1) for v in b],
                    "low_conf": bool(score < LOW_CONF_THRESHOLD),
                })
        h, w = arr.shape[:2]
        results.append({
            "file": path, "page": page,
            "width": int(w), "height": int(h),
            "lines": items,
        })
    return results


# ---------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="ocr", description="图片/PDF OCR：阅读顺序文本 / JSON / 表格 TSV")
    ap.add_argument("files", nargs="*", help="图片或 PDF 路径（本地文件）")
    ap.add_argument("--json", action="store_true", help="JSON 输出（含坐标/字号/置信度）")
    ap.add_argument("--table", action="store_true", help="表格 TSV 输出")
    ap.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="medium")
    ap.add_argument("--fast", action="store_true", help="等价 --model-type small")
    ap.add_argument("--smoke", action="store_true", help="模型预热烟雾测试（下载模型/SHA256 校验）")
    args = ap.parse_args(argv)

    model_type = "small" if args.fast else args.model_type
    engine = build_engine(model_type)
    warmup(engine)

    if args.smoke:
        print("SMOKE OK")
        return 0
    if not args.files:
        ap.error("至少需要一个文件")

    results_all: List[Dict] = []
    for f in args.files:
        if not Path(f).exists():
            print(f"错误: 文件不存在 {f}", file=sys.stderr)
            return 1
        try:
            results_all.extend(ocr_file(engine, f))
        except Exception as e:  # noqa: BLE001
            print(f"错误: {f}: {e}", file=sys.stderr)
            return 1

    if args.json:
        print(render_json(results_all))
        return 0

    total_files = len(args.files)
    for i, f in enumerate(args.files, start=1):
        file_results = [r for r in results_all if r["file"] == f]
        for r in file_results:
            header = f"[文件{i}/共{total_files}] 文件: {f}（页 {r['page']}/共{len(file_results)}）"
            if args.table:
                print(header)
                print(build_tsv(r["lines"]))
                print("（提示：列边界取自 x1，右对齐数值列可能碎列；如需精确请用 --json 按 box 推理）")
            else:
                print(header)
                print(render_text(r["lines"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
