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
    if model_type not in MODEL_TYPES:
        raise ValueError(f"未知 model_type: {model_type}，可选: {sorted(MODEL_TYPES)}")
    mt = MODEL_TYPES[model_type]
    return RapidOCR(params={
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Det.model_type": mt,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_type": mt,
        "Global.text_score": 0.0,
        "Global.log_level": "error",
        "EngineConfig.onnxruntime.use_coreml": False,
    })


def warmup(engine: RapidOCR) -> None:
    from PIL import ImageDraw
    img = Image.new("RGB", (128, 64), "white")
    ImageDraw.Draw(img).text((8, 16), "OCR test", fill="black")
    engine(np.asarray(img), text_score=0.0)


def box_to_xyxy(box: np.ndarray) -> List[float]:
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


def kmeans_colors(pixels: np.ndarray, k: int = 5, iters: int = 8,
                  seed: int = 0  # 预留兼容位：确定性实现，无需 RNG
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
    """文本块内颜色聚类：面积大的簇=背景，面积小的簇=文字笔画（前景）。

    输入契约：img 为 RGB ndarray (H, W, 3)；RGBA 需先 convert("RGB")。
    初始种子取块内亮度极值（min/max），避免均匀采样在纯色/浅色区域塌缩成单簇。
    纯色块（无文字）时 min/max 种子相同 → 单簇 → fg==bg 返回原色；
    调用方（Task 9 contrast_check）以 fg == bg 判定"无法分离"并跳过，不输出伪造 ratio。
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    region = img[y1:y2, x1:x2].reshape(-1, 3)
    if len(region) == 0:
        return (0, 0, 0), (255, 255, 255)
    lum = 0.2126 * region[:, 0] + 0.7152 * region[:, 1] + 0.0722 * region[:, 2]
    centers = np.stack([region[int(lum.argmin())], region[int(lum.argmax())]]).astype(float)
    labels = np.zeros(len(region), dtype=int)
    for _ in range(8):
        d = ((region[:, None, :].astype(float) - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for j in range(2):
            m = region[labels == j]
            if len(m):
                centers[j] = m.mean(axis=0)
    counts = [int((labels == 0).sum()), int((labels == 1).sum())]
    order = sorted(range(2), key=lambda j: -counts[j])
    bg = tuple(int(v) for v in centers[order[0]])
    fg = tuple(int(v) for v in centers[order[1]]) if counts[order[1]] > 0 else bg
    return fg, bg


# ---------------------------------------------------------------- 字号/对齐/配色
def cluster_groups(values: List[float], gap: float) -> List[Tuple[float, List[float]]]:
    """一维聚类，返回 [(簇中心, 成员值列表)]；与 cluster_centers 同一划分。"""
    if not values:
        return []
    vals = sorted(values)
    groups: List[List[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] > gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [(sum(g) / len(g), g) for g in groups]


def cluster_centers(values: List[float], gap: float) -> List[float]:
    """一维聚类簇中心（analyze 自包含实现，与 ocr.py 解耦）。"""
    return [c for c, _ in cluster_groups(values, gap)]


def font_size_clusters(items: List[Dict], gap: float = 6.0) -> List[Dict]:
    """字号聚类；孤立值（count==1）标 consistent: false。

    成员与 cluster_groups 同一划分（等值字号天然同组），不做距离窗口推断。
    """
    if not items:
        return []
    out = []
    for center, group in cluster_groups([it["font_size"] for it in items], gap):
        members = [it for it in items if it["font_size"] in group]
        out.append({
            "size": round(center, 1),
            "count": len(members),
            "texts": [m["text"] for m in members],
            "consistent": len(members) >= 2,
        })
    out.sort(key=lambda d: -d["count"])
    return out


def alignment_notes(items: List[Dict], min_group: int = 3) -> List[Dict]:
    """左（x1）/右（x2）/居中（中心 x）三向对齐聚类；≥min_group 成组。

    注意：等宽元素组会在左/右/居中三个方向同时成组（三向独立聚类，属预期）。
    box 须为归一化 [x1, y1, x2, y2]（真实流程由 box_to_xyxy 保证）。
    """
    notes: List[Dict] = []
    for key, axis in (("left_align_group", 0), ("right_align_group", 2), ("center_align_group", None)):
        if axis is None:
            vals = [(it["box"][0] + it["box"][2]) / 2 for it in items]
        else:
            vals = [it["box"][axis] for it in items]
        for center, group in cluster_groups(vals, gap=6.0):
            if len(group) >= min_group:
                notes.append({"type": key, "x": round(center, 1), "elements": len(group)})
    return notes


def dominant_colors(img: Image.Image, n: int = 5) -> List[Dict]:
    """降采样 64×64 后 k-means 主色；占比最大者标 role="bg"（启发式）。

    img 须为 RGB（真实流程由调用方 convert），NEAREST 不发明新颜色。
    """
    px = np.asarray(img.resize((64, 64), Image.Resampling.NEAREST)).reshape(-1, 3)
    clusters = kmeans_colors(px, k=n)
    total = sum(c[1] for c in clusters) or 1
    out = []
    for i, (rgb, count) in enumerate(clusters):
        out.append({
            "hex": "#%02X%02X%02X" % rgb,
            "pct": round(100.0 * count / total, 1),
            "role": "bg" if i == 0 else None,
        })
    return out


# ---------------------------------------------------------------- 审查主流程
def contrast_check(box: List[float], text: str, img: np.ndarray) -> Optional[Dict]:
    """单个文本块对比度事实；wcag 为 None 表示通过 AA。

    若 fg == bg（无法分离前景/背景，如纯色块），返回 None 由调用方跳过——
    不输出伪造 ratio（T7 审查修正：灰字白底种子塌缩曾产出合成色 + ratio 恒 1.0）。
    """
    fg, bg = split_fg_bg(box, img)
    if fg == bg:
        return None
    ratio = contrast_ratio(fg, bg)
    return {
        "text": text,
        "box": [round(v, 1) for v in box],
        "fg": "#%02X%02X%02X" % fg,
        "bg": "#%02X%02X%02X" % bg,
        "ratio": round(ratio, 2),
        "wcag": None if ratio >= 4.5 else "AA 未达标(需≥4.5)",
    }


def analyze_image(path: str, model_type: str = "medium",
                  engine: Optional[RapidOCR] = None) -> Dict:
    """单张图片完整审查（spec §4.3：事实清单，结论由 LLM 下）。

    engine 由调用方传入并复用（§2.4.1 引擎单例）；不传时内部自建。
    """
    if engine is None:
        engine = build_engine(model_type)
        warmup(engine)
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    res = engine(arr, text_score=0.0)
    items: List[Dict] = []
    if res.boxes is not None:
        for box, txt, score in zip(res.boxes, res.txts, res.scores):
            b = box_to_xyxy(box)
            items.append({
                "text": txt,
                "conf": float(score),
                "font_size": round(b[3] - b[1], 1),
                "box": b,
            })
    contrast_issues = []
    for it in items:
        issue = contrast_check(it["box"], it["text"], arr)
        if issue is not None and issue["wcag"] is not None:
            contrast_issues.append(issue)
    return {
        "file": path,
        "palette": dominant_colors(img),
        "contrast_issues": contrast_issues,
        "font_size_clusters": font_size_clusters(items),
        "alignment_notes": alignment_notes(items),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="analyze", description="设计审查：对比度/字号/对齐/配色 → 事实清单 JSON（仅图片）")
    ap.add_argument("files", nargs="+", help="图片路径（不支持 PDF）")
    ap.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="medium")
    args = ap.parse_args(argv)
    engine = build_engine(args.model_type)
    warmup(engine)
    out: List[Dict] = []
    for f in args.files:
        if not Path(f).exists():
            print(f"错误: 文件不存在 {f}", file=sys.stderr)
            return 1
        try:
            out.append(analyze_image(f, args.model_type, engine))
        except Exception as e:  # noqa: BLE001
            print(f"错误: {f}: {e}", file=sys.stderr)
            return 1
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
