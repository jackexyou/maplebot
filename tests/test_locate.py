"""小地圖自動定位：合成畫面上貼角落圖案，驗證找得回正確 ROI。"""
import numpy as np
import pytest

from maplebot.vision.locate import find_minimap


def _corner(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (12, 12), dtype=np.uint8)


@pytest.fixture
def scene():
    rng = np.random.default_rng(3)
    frame = rng.integers(90, 130, (400, 600, 3), dtype=np.uint8)
    tl = _corner(1)
    br = _corner(2)
    # 小地圖外框：左上角在 (50, 40)，右下角圖案的右下端在 (250, 150)
    frame[40:52, 50:62] = tl[:, :, None]
    frame[138:150, 238:250] = br[:, :, None]
    return frame, tl, br


def test_find_minimap(scene):
    frame, tl, br = scene
    region = find_minimap(frame, tl, br, border=4)
    assert region == (54, 44, 192, 102)  # (50+4, 40+4, 246-54, 146-44)


def test_missing_corners_returns_none(scene):
    frame, tl, _ = scene
    stranger = _corner(9)  # 畫面上不存在的圖案
    assert find_minimap(frame, tl, stranger, border=4) is None


def test_degenerate_region_rejected():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tl = _corner(1)
    br = _corner(2)
    # 兩個角貼在幾乎同一個位置 -> 區域太小要拒絕
    frame[10:22, 10:22] = tl[:, :, None]
    frame[14:26, 18:30] = br[:, :, None]
    assert find_minimap(frame, tl, br, border=4) is None


def test_search_window_ignores_a_decoy_corner_far_away(scene):
    """右下角的模板在畫面別處有個更像的複本（實機：快捷鍵面板的白框轉角）。
    限制只在左上角往右下的窗裡找，就不會把整個畫面當成小地圖。"""
    frame, tl, br = scene
    frame[300:312, 500:512] = br[:, :, None]          # 誘餌：跟真的一模一樣
    assert find_minimap(frame, tl, br, border=4) in ((54, 44, 192, 102), (54, 44, 454, 264))
    assert find_minimap(frame, tl, br, border=4, search_px=250) == (54, 44, 192, 102)


def test_search_window_too_small_returns_none(scene):
    frame, tl, br = scene
    assert find_minimap(frame, tl, br, border=4, search_px=8) is None


def test_min_size_rejects_a_collapsed_minimap(scene):
    frame, tl, br = scene
    assert find_minimap(frame, tl, br, border=4, min_size=(300, 50)) is None
    assert find_minimap(frame, tl, br, border=4, min_size=(150, 80)) == (54, 44, 192, 102)


def test_anchor_box_restricts_where_the_top_left_corner_may_be(scene):
    """左上角模板在畫面別處有個更像的複本：限制只在左上區域找就不受影響，
    而把範圍框到沒有小地圖的地方則找不到。"""
    frame, tl, br = scene
    frame[300:312, 400:412] = tl[:, :, None]
    assert find_minimap(frame, tl, br, border=4, anchor_box=(0, 0, 200, 200)) == (54, 44, 192, 102)
    assert find_minimap(frame, tl, br, border=4, anchor_box=(300, 0, 300, 200)) is None


def test_several_templates_per_corner_take_the_best(scene):
    frame, tl, br = scene
    other = _corner(7)
    assert find_minimap(frame, [other, tl], [other, br], border=4) == (54, 44, 192, 102)
    assert find_minimap(frame, [other], [br], border=4) is None


def test_max_size_rejects_an_oversized_region(scene):
    frame, tl, br = scene
    assert find_minimap(frame, tl, br, border=4, max_size=(100, 100)) is None
    assert find_minimap(frame, tl, br, border=4, max_size=(300, 200)) == (54, 44, 192, 102)


def test_the_second_best_corner_wins_when_the_best_makes_a_silly_region(scene):
    """右下角的第一名配出一塊太小的區域（實機：燃燒之地 45x476 排第一），
    就往下看第二名，而不是直接放棄。"""
    frame, tl, br = scene
    frame[60:72, 70:82] = br[:, :, None]          # 緊貼左上角的誘餌：區域只有 16x20
    assert find_minimap(frame, tl, br, border=4, min_size=(100, 50)) == (54, 44, 192, 102)


def test_load_ui_templates_collects_variants(tmp_path):
    import cv2
    from maplebot.vision.locate import load_ui_templates
    for name in ("minimap_br.png", "minimap_br_2.png", "minimap_br_burnt.png", "minimap_brx.png", "minimap_tl.png"):
        cv2.imwrite(str(tmp_path / name), np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(load_ui_templates(str(tmp_path), "minimap_br.png")) == 3
    assert load_ui_templates(str(tmp_path), "minimap_player.png") is None
