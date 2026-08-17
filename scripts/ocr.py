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
