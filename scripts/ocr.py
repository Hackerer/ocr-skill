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
