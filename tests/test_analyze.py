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


from PIL import Image
from analyze import alignment_notes, cluster_centers, dominant_colors, font_size_clusters


def _item(text, box):
    return {"text": text, "conf": 0.99, "font_size": box[3] - box[1], "box": box}


def test_font_size_clusters():
    items = [
        _item("标题", [10, 10, 100, 50]),      # 40px ×1 → consistent False
        _item("按钮1", [10, 60, 100, 84]),     # 24px ×2
        _item("按钮2", [10, 90, 100, 114]),    # 24px ×2
    ]
    clusters = font_size_clusters(items)
    assert len(clusters) >= 2
    c40 = next(c for c in clusters if c["size"] > 30)
    assert c40["count"] == 1 and c40["consistent"] is False
    c24 = next(c for c in clusters if 20 < c["size"] < 30)
    assert c24["count"] == 2 and c24["consistent"] is True


def test_alignment_left_group():
    items = [
        _item("A", [40, 10, 100, 30]),
        _item("B", [40, 50, 90, 70]),
        _item("C", [40, 90, 120, 110]),
        _item("D", [300, 10, 400, 30]),        # 离群
    ]
    notes = alignment_notes(items)
    left = [n for n in notes if n["type"] == "left_align_group"]
    assert any(n["elements"] >= 3 and abs(n["x"] - 40) < 5 for n in left)


def test_dominant_colors_role_bg():
    img = Image.new("RGB", (100, 100), (31, 41, 55))   # 全 #1F2937
    palette = dominant_colors(img, n=3)
    assert palette[0]["hex"] == "#1F2937"
    assert palette[0]["role"] == "bg"
    assert palette[0]["pct"] == 100.0


def test_font_size_clusters_chained_no_member_loss():
    # 链式簇 {12,16,20}：相邻差 4,4 ≤ gap 6 → 单簇，三个成员全部保留
    items = [_item("a", [10, 10, 30, 22]),   # 12px
             _item("b", [10, 30, 30, 46]),   # 16px
             _item("c", [10, 50, 30, 70])]   # 20px
    clusters = font_size_clusters(items)
    assert len(clusters) == 1
    assert clusters[0]["count"] == 3
    assert clusters[0]["consistent"] is True
    assert sorted(clusters[0]["texts"]) == ["a", "b", "c"]


def test_alignment_right_and_center():
    items = [
        _item("A", [40, 10, 200, 30]),    # 右缘 200，中心 120
        _item("B", [60, 50, 200, 70]),    # 右缘 200，中心 130
        _item("C", [80, 90, 200, 110]),   # 右缘 200，中心 140
        _item("D", [300, 10, 400, 30]),   # 离群
        _item("E", [20, 130, 220, 150]),  # 中心 120
        _item("F", [0, 170, 240, 190]),   # 中心 120
    ]
    notes = alignment_notes(items)
    right = [n for n in notes if n["type"] == "right_align_group"]
    assert any(n["elements"] >= 3 and abs(n["x"] - 200) < 5 for n in right)
    center = [n for n in notes if n["type"] == "center_align_group"]
    assert any(n["elements"] >= 3 and abs(n["x"] - 120) < 5 for n in center)


def test_alignment_min_group_boundary():
    items = [_item("A", [40, 10, 100, 30]), _item("B", [40, 50, 100, 70])]
    assert alignment_notes(items) == []      # 2 元素 < min_group 3


def test_dominant_colors_two_colors_5050():
    import numpy as _np
    from PIL import Image as _Image
    arr = _np.zeros((100, 100, 3), dtype=_np.uint8)
    arr[:, :50] = (255, 0, 0)
    arr[:, 50:] = (0, 0, 255)
    palette = dominant_colors(_Image.fromarray(arr), n=5)
    assert len(palette) == 2
    assert palette[0]["pct"] == 50.0 and palette[1]["pct"] == 50.0
    assert palette[0]["hex"] in ("#FF0000", "#0000FF")


def test_cluster_centers_matches_ocr_cluster_1d():
    # 防漂移：analyze 的聚类行为必须与 ocr.py 一致（自包含设计的代价）
    from ocr import cluster_1d
    from analyze import cluster_centers
    samples = [[1.0, 2.0, 50.0, 51.0], [42.0], [], [10.0, 20.0], [51.0, 1.0, 50.0, -5.0, 2.0]]
    for s in samples:
        assert cluster_centers(s, gap=10.0) == cluster_1d(s, gap=10.0)


import pytest

from analyze import analyze_image


@pytest.mark.integration
def test_ui_review_full(tmp_path):
    from image_factory import make_ui_image
    p = make_ui_image(tmp_path / "ui.png")
    result = analyze_image(str(p))
    # 低对比度标题被检出
    assert any(i["text"] == "商品详情" or "商品" in i["text"]
               for i in result["contrast_issues"]), result["contrast_issues"]
    # 深色背景为主色
    assert result["palette"][0]["role"] == "bg"
    # 字号至少两簇（标题 40px vs 按钮 24px）
    assert len(result["font_size_clusters"]) >= 2
    # 三个文本 x1 相同 → 左对齐组
    assert any(n["type"] == "left_align_group" and n["elements"] >= 3
               for n in result["alignment_notes"])


@pytest.mark.integration
def test_ui_review_no_text_only_palette(tmp_path):
    from PIL import Image
    p = tmp_path / "blank.png"
    Image.new("RGB", (200, 100), (255, 255, 255)).save(p)
    result = analyze_image(str(p))
    assert result["contrast_issues"] == []
    assert result["font_size_clusters"] == []
    assert result["palette"][0]["hex"] == "#FFFFFF"
