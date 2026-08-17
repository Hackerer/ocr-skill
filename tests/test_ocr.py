"""ocr.py 纯函数单元测试（不触发模型推理）"""
import numpy as np
import pytest

from ocr import box_to_xyxy, cluster_1d


def test_box_to_xyxy_quad_points():
    box = np.array([[10.0, 20.0], [110.0, 20.0], [110.0, 60.0], [10.0, 60.0]])
    assert box_to_xyxy(box) == [10.0, 20.0, 110.0, 60.0]


def test_box_to_xyxy_flat():
    box = np.array([12.0, 20.0, 108.0, 52.0])
    assert box_to_xyxy(box) == [12.0, 20.0, 108.0, 52.0]


def test_box_to_xyxy_rotated_quad():
    # 倾斜框：外接矩形应取 min/max
    box = np.array([[50.0, 10.0], [90.0, 30.0], [70.0, 50.0], [30.0, 30.0]])
    assert box_to_xyxy(box) == [30.0, 10.0, 90.0, 50.0]


def test_box_to_xyxy_bad_shape():
    with pytest.raises(ValueError):
        box_to_xyxy(np.zeros((3, 3)))


def test_cluster_1d_basic():
    assert cluster_1d([1.0, 2.0, 50.0, 51.0], gap=10.0) == [1.5, 50.5]


def test_cluster_1d_empty():
    assert cluster_1d([], gap=10.0) == []


def test_box_to_xyxy_flat_int():
    box = np.array([12, 20, 108, 52])            # int 输入
    assert box_to_xyxy(box) == [12.0, 20.0, 108.0, 52.0]


def test_box_to_xyxy_two_by_four_rejected():
    with pytest.raises(ValueError):
        box_to_xyxy(np.zeros((2, 4)))


def test_cluster_1d_single():
    assert cluster_1d([42.0], gap=10.0) == [42.0]


def test_cluster_1d_exact_gap_no_split():
    # 相邻差恰好等于 gap：> gap 才分簇，等于不分
    assert cluster_1d([10.0, 20.0], gap=10.0) == [15.0]


def test_cluster_1d_unsorted_and_negative():
    # gap=5.0：1.0-(-5.0)=6.0 > 5.0 拆出单点簇 -5.0；2.0-1.0=1.0 并入 → [-5.0, 1.5, 50.5]
    assert cluster_1d([51.0, 1.0, 50.0, -5.0, 2.0], gap=5.0) == [-5.0, 1.5, 50.5]


from ocr import group_into_rows, sort_lines_reading_order


def _item(text, box):
    return {"text": text, "conf": 0.99, "font_size": box[3] - box[1], "box": box, "low_conf": False}


def test_group_rows_two_visual_rows():
    items = [
        _item("R1", [200, 10, 300, 30]),   # 同行右侧
        _item("L1", [10, 10, 100, 30]),    # 同行左侧
        _item("L2", [10, 50, 100, 70]),    # 下一行左侧
        _item("R2", [200, 50, 300, 70]),   # 下一行右侧
    ]
    rows = group_into_rows(items)
    assert len(rows) == 2
    assert sorted(r["text"] for r in rows[0]) == ["L1", "R1"]
    assert sorted(r["text"] for r in rows[1]) == ["L2", "R2"]


def test_reading_order_xy():
    items = [
        _item("R1", [200, 10, 300, 30]),
        _item("L1", [10, 10, 100, 30]),
        _item("L2", [10, 50, 100, 70]),
    ]
    ordered = sort_lines_reading_order(items)
    assert [i["text"] for i in ordered] == ["L1", "R1", "L2"]


def test_group_rows_tall_box_does_not_absorb_short_line():
    # 高框（标题）与紧贴其下的矮行必须分行（回归 I-1）
    items = [
        _item("标题", [10, 20, 300, 100]),     # h=80, cy=60
        _item("说明", [10, 100, 80, 108]),     # h=8, cy=104（紧贴）
        _item("另一行", [10, 150, 100, 170]),  # h=20, cy=160
    ]
    rows = group_into_rows(items)
    assert len(rows) == 3
    assert [r["text"] for r in rows[0]] == ["标题"]
    assert [r["text"] for r in rows[1]] == ["说明"]


def test_group_rows_boundary_equal_threshold_same_row():
    # 边界语义：|Δcy| 恰好等于阈值 → 归一行（与 cluster_1d 的"相等不分簇"一致）
    items = [
        _item("上", [10, 10, 100, 30]),    # h=20, cy=20
        _item("下", [10, 22, 100, 42]),    # h=20, cy=32, |Δ|=12 == max(min(20,20)*0.6,8)=12
    ]
    rows = group_into_rows(items)
    assert len(rows) == 1
    assert sorted(r["text"] for r in rows[0]) == ["上", "下"]


from ocr import build_tsv, render_json, render_text


def test_render_text_low_conf_marked():
    items = [_item("清楚", [10, 10, 80, 30]), _item("模糊", [10, 40, 80, 60])]
    items[1]["conf"] = 0.3
    out = render_text(items)
    assert "清楚" in out
    assert "⟦低置信⟧ 模糊" in out


def test_render_text_empty():
    assert render_text([]) == "未检测到文字"


def test_build_tsv_2x2():
    items = [
        _item("A1", [10, 10, 100, 30]),
        _item("B1", [200, 10, 300, 30]),
        _item("A2", [10, 50, 100, 70]),
        _item("B2", [200, 50, 300, 70]),
    ]
    assert build_tsv(items) == "A1\tB1\nA2\tB2"


def test_build_tsv_empty_cell():
    items = [
        _item("A1", [10, 10, 100, 30]),
        _item("A2", [10, 50, 100, 70]),
        _item("B2", [200, 50, 300, 70]),   # B1 缺失 → 第一行第二列留空
    ]
    assert build_tsv(items) == "A1\t\nA2\tB2"


def test_render_json_utf8():
    results = [{"file": "a.png", "page": 1, "width": 10, "height": 10, "lines": [_item("中文", [0, 0, 5, 5])]}]
    out = render_json(results)
    assert "中文" in out          # ensure_ascii=False：中文不转义
    assert "\\u" not in out


def test_render_text_paragraph_gap():
    items = [
        _item("段一", [10, 10, 100, 30]),
        _item("段二", [10, 70, 100, 90]),     # y1 间距 70-30=40 > 24 → 空行分隔
        _item("同段", [10, 80, 100, 100]),    # 与段二同行（cy 差 10 < 12）→ 不插空行
    ]
    out = render_text(items)
    assert out == "段一\n\n段二\n同段"


def test_render_text_conf_boundary():
    a = _item("恰好", [10, 10, 80, 30]); a["conf"] = 0.5
    b = _item("差一点", [10, 40, 80, 60]); b["conf"] = 0.499
    out = render_text([a, b])
    assert "恰好" in out and "⟦低置信⟧" not in out.split("恰好")[0]
    assert "⟦低置信⟧ 差一点" in out


def test_build_tsv_wide_cell_stays_in_own_column():
    items = [
        _item("宽内容", [10, 10, 260, 30]),   # 右缘越过第二列左缘，但左缘 x1=10 → 第一列
        _item("右列", [250, 10, 300, 30]),
    ]
    assert build_tsv(items) == "宽内容\t右列"


def test_build_tsv_same_cell_order_by_x():
    items = [
        _item("后", [150, 10, 200, 30]),
        _item("先", [140, 10, 160, 30]),      # x1 与"后"差 10 < COLUMN_GAP_PX → 同列；按 x 排序后"先"在前
    ]
    assert build_tsv(items) == "先 后"


def test_render_json_roundtrip():
    import json as _json
    results = [{"file": "a.png", "page": 1, "width": 10, "height": 10,
                "lines": [_item("中文", [0, 0, 5, 5])]}]
    parsed = _json.loads(render_json(results))
    assert isinstance(parsed, list)
    assert parsed[0]["lines"][0]["box"] == [0, 0, 5, 5]


from PIL import Image
from ocr import load_images


def test_load_images_single_image(tmp_path):
    p = tmp_path / "t.png"
    Image.new("RGB", (64, 32), "white").save(p)
    pages = list(load_images(str(p)))
    assert len(pages) == 1
    img, page, total = pages[0]
    assert img.size == (64, 32)
    assert (page, total) == (1, 1)
