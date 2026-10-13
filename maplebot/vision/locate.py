"""小地圖自動定位（參考 auto-maple 的 corner-template 校正法）。

使用者截一次小地圖「左上角」與「右下角」的框角模板後，
每次啟動用模板匹配自動找出小地圖 ROI——小地圖被拖動、
展開收合、換解析度都不用重新手動校正。
"""
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import Region
from . import match

TL_NAME = "minimap_tl.png"
BR_NAME = "minimap_br.png"
PLAYER_NAME = "minimap_player.png"

MATCH_THRESHOLD = 0.6


def load_ui_template(ui_dir: str, name: str) -> Optional[np.ndarray]:
    """載入使用者截的 UI 模板。**讀彩色**——比對要用到顏色（見 vision/match.py）。
    以前這裡讀灰階，於是小地圖上黃點/紅點/藍點被壓成差不多一樣亮，模板比對
    跟背景的差距只有 +0.055；改成彩色之後是 +0.36。
    """
    path = os.path.join(ui_dir, name)
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)


def load_ui_templates(ui_dir: str, name: str) -> Optional[List[np.ndarray]]:
    """`name` 本身加上同名加後綴的變體：minimap_br.png、minimap_br_2.png、minimap_br_burnt.png……
    一張都沒有回 None（呼叫端照舊用 `is None` 判斷）。

    為什麼要好幾張：角落模板裡總有幾 px 是畫布內容和面板外的背景，而那兩樣
    每張地圖都不一樣——螞蟻洞的畫布是黑的、面板外也是黑的；燃燒之地的畫布是
    沙色、面板外是白雲。一張模板在另一種地圖上只剩 0.68，輸給畫面裡隨便一個
    白框轉角（0.8）。各截一張，find_minimap 兩張都比、取分數最高的。
    """
    stem, ext = os.path.splitext(name)
    out = []
    for fn in sorted(os.listdir(ui_dir)) if os.path.isdir(ui_dir) else []:
        s, e = os.path.splitext(fn)
        if e.lower() == ext.lower() and (s == stem or s.startswith(stem + "_")):
            img = cv2.imread(os.path.join(ui_dir, fn), cv2.IMREAD_COLOR)
            if img is not None:
                out.append(img)
    return out or None


def _as_list(tpl) -> List[np.ndarray]:
    return list(tpl) if isinstance(tpl, (list, tuple)) else [tpl]


def _best_match(img: np.ndarray, tpl: np.ndarray, roi=None) -> Tuple[float, Tuple[int, int]]:
    hit = match.match(img, tpl, roi=roi)
    return hit if hit is not None else (-1.0, (0, 0))


# 右下角每張模板最多看幾個候選點。真正的角通常就是第一名；留幾個名額是給
# 「第一名尺寸不合理」的情況（實測燃燒之地一張 45x476 的假小地圖以 0.80 排第一）
BR_CANDIDATES = 6


def _br_candidates(win: np.ndarray, tpl: np.ndarray) -> List[Tuple[float, int, int]]:
    """窗裡這張模板分數 >= 門檻的候選點 [(score, x, y)]，從高到低，彼此不重疊。"""
    out = []
    img = win.copy()
    th, tw = tpl.shape[:2]
    for _ in range(BR_CANDIDATES):
        hit = match.match(img, tpl)
        if hit is None or hit[0] < MATCH_THRESHOLD:
            break
        score, (x, y) = hit
        out.append((score, x, y))
        # 把這個峰塗掉再找下一個（連同周圍幾 px，模板平移一兩格分數差不了多少）
        img[max(0, y - 6):y + th + 6, max(0, x - 6):x + tw + 6] = 0
    return out


def find_minimap(frame_bgr: np.ndarray, tl_tpl, br_tpl,
                 border: int = 6, search_px: int = 0,
                 min_size: Tuple[int, int] = (20, 15),
                 anchor_box: Optional[Region] = None,
                 max_size: Optional[Tuple[int, int]] = None) -> Optional[Region]:
    """用左上/右下角落模板找出小地圖內部區域；找不到（分數過低）回 None。

    tl_tpl、br_tpl 各可以是一張模板或一串模板（見 load_ui_templates）。

    左上角：anchor_box 給了就只在那塊 (x, y, w, h) 裡找。小地圖面板永遠在畫面左上，
    而左上角模板在淺色地圖上的分數會掉到 0.7（模板裡那幾 px 的畫布內容是黑的），
    這時畫面別處的白框轉角（0.8）就會贏過它。幾張模板裡取分數最高的那個位置。

    右下角：search_px > 0 時只在左上角往右下 search_px px 的窗裡找。右下角模板
    就是一截白框轉角，畫面上長得一樣的地方不少：實測 2560x1440 的 Classic 客戶端，
    快捷鍵面板的轉角以 0.89 贏過真正的小地圖角（0.86），定位出來的「小地圖」
    是整個畫面。每張模板各取幾個候選點，全部依分數排序，取第一個配出來的區域
    尺寸落在 min_size ~ max_size 之間的。

    min_size / max_size 是配對出來的區域尺寸 (寬, 高) 的下限與上限：小地圖收合時
    畫面上只剩標題列，兩個模板各自在別的面板上比到一個角，配出一塊 100x40 或
    570x412 的「小地圖」，尺寸是唯一擋得住它們的東西。
    """
    tls, brs = _as_list(tl_tpl), _as_list(br_tpl)
    th = max(t.shape[0] for t in tls + brs)
    tw = max(t.shape[1] for t in tls + brs)
    if frame_bgr.shape[0] < th or frame_bgr.shape[1] < tw:
        return None
    tl_score, tl = max((_best_match(frame_bgr, t, roi=anchor_box) for t in tls),
                       key=lambda h: h[0])
    if tl_score < MATCH_THRESHOLD:
        return None
    if search_px > 0:
        win = frame_bgr[tl[1]:tl[1] + search_px, tl[0]:tl[0] + search_px]
        ox, oy = tl
    else:
        win, (ox, oy) = frame_bgr, (0, 0)
    if win.shape[0] < min(b.shape[0] for b in brs) or win.shape[1] < min(b.shape[1] for b in brs):
        return None
    cands = []
    for b in brs:
        if win.shape[0] < b.shape[0] or win.shape[1] < b.shape[1]:
            continue
        for score, x, y in _br_candidates(win, b):
            cands.append((score, ox + x + b.shape[1], oy + y + b.shape[0]))
    cands.sort(key=lambda c: -c[0])
    x1 = tl[0] + border
    y1 = tl[1] + border
    min_w, min_h = max(min_size[0], 20), max(min_size[1], 15)
    for _, bx2, by2 in cands:
        x2, y2 = bx2 - border, by2 - border
        w, h = x2 - x1, y2 - y1
        if w < min_w or h < min_h:
            continue
        if max_size and (w > max_size[0] or h > max_size[1]):
            continue
        return (x1, y1, w, h)
    return None
