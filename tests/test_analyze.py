"""analyze.py 纯函数单元测试（不触发模型推理）"""
import numpy as np
import pytest

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
    # #6B7280 on #1F2937 = 3.0363 < 4.5（WCAG AA 未达标）
    ratio = contrast_ratio((107, 114, 128), (31, 41, 55))
    assert ratio == pytest.approx(3.0363, abs=0.01)


def test_contrast_ratio_symmetric():
    fg, bg = (107, 114, 128), (31, 41, 55)
    assert contrast_ratio(fg, bg) == contrast_ratio(bg, fg)


def test_luminance_red_is_red_coefficient():
    assert luminance((255, 0, 0)) == pytest.approx(0.2126, abs=1e-4)


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


def test_split_fg_bg_single_color_returns_self():
    img = np.full((40, 40, 3), (31, 41, 55), dtype=np.uint8)   # 纯 #1F2937
    fg, bg = split_fg_bg([10, 10, 30, 30], img)
    assert fg == bg == (31, 41, 55)


def test_split_fg_bg_light_text_dark_bg():
    img = np.full((60, 60, 3), (31, 41, 55), dtype=np.uint8)   # 深底 #1F2937
    img[27:33, 20:40] = (200, 200, 200)                        # 浅色文字条（面积小）
    fg, bg = split_fg_bg([20, 20, 40, 40], img)
    assert bg == (31, 41, 55)
    assert fg == (200, 200, 200)


def test_engine_params_match_ocr():
    """自包含副本的参数必须与 ocr.py 一致，防止漂移（低置信策略/CoreML 锁定）。"""
    from analyze import build_engine as analyze_engine
    from ocr import build_engine as ocr_engine
    ae = analyze_engine("small")
    oe = ocr_engine("small")
    assert ae.cfg.Global.text_score == oe.cfg.Global.text_score == 0.0
    assert ae.cfg.EngineConfig.onnxruntime.use_coreml is False
    assert oe.cfg.EngineConfig.onnxruntime.use_coreml is False
    assert ae.cfg.Rec.ocr_version == oe.cfg.Rec.ocr_version
