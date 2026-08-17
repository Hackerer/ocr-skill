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
