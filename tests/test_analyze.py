"""analyze.py 纯函数单元测试（不触发模型推理）"""
import numpy as np

from analyze import contrast_ratio, kmeans_colors, luminance, split_fg_bg, srgb_to_linear


def test_srgb_to_linear_black_white():
    assert srgb_to_linear(0) == 0.0
    assert srgb_to_linear(255) == 1.0


def test_luminance_black_white():
    assert luminance((0, 0, 0)) == 0.0
    assert luminance((255, 255, 255)) == 1.0


def test_contrast_ratio_black_white():
    assert contrast_ratio((0, 0, 0), (255, 255, 255)) == 21.0


def test_contrast_ratio_same():
    assert contrast_ratio((10, 20, 30), (10, 20, 30)) == 1.0


def test_contrast_ratio_gray_on_dark_fails_aa():
    # #6B7280 on #1F2937 ≈ 3.06 < 4.5（WCAG AA 未达标）
    ratio = contrast_ratio((107, 114, 128), (31, 41, 55))
    assert 2.5 < ratio < 4.5


def test_kmeans_two_flat_colors():
    red = np.array([[255, 0, 0]] * 100)
    blue = np.array([[0, 0, 255]] * 300)
    pixels = np.vstack([red, blue]).astype(np.uint8)
    res = kmeans_colors(pixels, k=2, iters=8)
    assert len(res) == 2
    biggest = res[0]
    assert biggest[0] == (0, 0, 255)          # 面积大的排前
    assert biggest[1] == 300


def test_split_fg_bg_dark_text_on_white():
    img = np.full((60, 100, 3), 255, dtype=np.uint8)
    img[27:33, 30:70] = (0, 0, 0)            # 中间黑色横条=文字笔画（面积 < 背景）
    fg, bg = split_fg_bg([30, 20, 70, 40], img)
    assert bg == (255, 255, 255)
    assert fg == (0, 0, 0)
