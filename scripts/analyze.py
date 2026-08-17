#!/usr/bin/env python
"""设计审查 CLI：对比度/字号/对齐/配色 → 事实清单 JSON。

自包含设计（spec §4.3）：内部复用 rapidocr 拿文本块，不与 ocr.py 耦合。
用法: analyze.py <图片...> [--model-type tiny|small|medium]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from rapidocr import ModelType, OCRVersion, RapidOCR

MODEL_TYPES = {"tiny": ModelType.TINY, "small": ModelType.SMALL, "medium": ModelType.MEDIUM}


def build_engine(model_type: str = "medium") -> RapidOCR:
    mt = MODEL_TYPES[model_type]
    return RapidOCR(params={
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Det.model_type": mt,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_type": mt,
        "Global.log_level": "error",
    })


def warmup(engine: RapidOCR) -> None:
    from PIL import ImageDraw
    img = Image.new("RGB", (128, 64), "white")
    ImageDraw.Draw(img).text((8, 16), "OCR test", fill="black")
    engine(np.asarray(img), text_score=0.0)


def box_to_xyxy(box) -> List[float]:
    a = np.asarray(box, dtype=float)
    if a.shape == (4, 2):
        xs, ys = a[:, 0], a[:, 1]
    elif a.shape == (4,):
        xs, ys = a[[0, 2]], a[[1, 3]]
    else:
        raise ValueError(f"unexpected box shape: {a.shape}")
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


# ---------------------------------------------------------------- 颜色与对比度
def srgb_to_linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = (srgb_to_linear(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> float:
    l1, l2 = luminance(fg), luminance(bg)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)


def kmeans_colors(pixels: np.ndarray, k: int = 5, iters: int = 8, seed: int = 0
                  ) -> List[Tuple[Tuple[int, int, int], int]]:
    """确定性 k-means（numpy 手写，无 sklearn）。返回 [(rgb, 像素数)] 按数量降序。"""
    n = len(pixels)
    if n == 0:
        return []
    idx = np.linspace(0, n - 1, k).astype(int)
    centers = pixels[idx].copy().astype(float)
    # 初始中心若重复（纯色区域/采样到同色），加确定性扰动保证 k 簇可分
    for j in range(1, k):
        if np.array_equal(centers[j], centers[j - 1]):
            centers[j] = (centers[j] + np.array([j * 7.0, j * 13.0, j * 29.0])) % 256.0
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((pixels[:, None, :].astype(float) - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for j in range(k):
            m = pixels[labels == j]
            if len(m):
                centers[j] = m.mean(axis=0)
    res = []
    for j in range(k):
        m = pixels[labels == j]
        if len(m):
            res.append((tuple(int(v) for v in centers[j]), int(len(m))))
    res.sort(key=lambda t: -t[1])
    return res


def split_fg_bg(box: List[float], img: np.ndarray
                ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """文本块内颜色聚类：面积大的簇=背景，面积小的簇=文字笔画（前景）。"""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    region = img[y1:y2, x1:x2].reshape(-1, 3)
    if len(region) == 0:
        return (0, 0, 0), (255, 255, 255)
    clusters = kmeans_colors(region, k=2)
    if len(clusters) < 2:
        c = clusters[0][0]
        return c, c
    bg = clusters[0][0]   # 面积最大
    fg = clusters[1][0]   # 面积次之（文字笔画）
    return fg, bg
