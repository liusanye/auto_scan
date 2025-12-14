"""页面去透视/去卷曲。"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional, Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

def _order_points(pts: np.ndarray) -> np.ndarray:
    """将四边形点排序为 tl,tr,br,bl。"""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def _warp_from_quad(image: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    quad = _order_points(quad.reshape(4, 2))
    (tl, tr, br, bl) = quad
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    if maxW == 0 or maxH == 0:
        return image, None, {"method": "fallback_invalid_size"}
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped, M, {"method": "trapezoid_warp", "quad": quad.tolist(), "matrix": M.tolist()}


def _fallback_perspective(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, str]]:
    """
    简单透视回退：优先使用分割 mask 拟合矩形，避免重新阈值导致粉框比绿框更小。
    """
    img_h, img_w = image.shape[:2]
    cnts = []
    source = "mask" if mask is not None else "gray"
    if mask is not None:
        mask_bin = (mask > 127).astype("uint8") * 255
        cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # 无有效 mask 时退回灰度阈值
        source = "gray"
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return image, mask, {"method": "fallback_none", "matrix": None}

    c = max(cnts, key=cv2.contourArea)
    # 优先尝试梯形校正：四边形拟合
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        warped_img, M, info = _warp_from_quad(image, approx)
        warped_mask = None
        if mask is not None and M is not None:
            warped_mask = cv2.warpPerspective(mask, M, (warped_img.shape[1], warped_img.shape[0]))
        # 覆盖率检查，过小则回退矩形方案
        coverage = cv2.contourArea(c) / float(max(1.0, img_h * img_w))
        if coverage >= 0.03:
            info["coverage"] = coverage
            info["source"] = source
            if M is not None:
                info["matrix"] = M.tolist()
            return warped_img, warped_mask, info

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    w = int(rect[1][0])
    h = int(rect[1][1])
    coverage = cv2.contourArea(c) / float(max(1.0, img_h * img_w))
    if w == 0 or h == 0:
        return image, mask, {"method": "fallback_invalid_size", "matrix": None, "coverage": coverage, "source": source}
    # 若 coverage 过小，直接返回原图，避免粉框收缩过度
    if coverage < 0.03:
        log.warning("fallback_perspective: coverage too small (%.4f), skip warp", coverage)
        return image, mask, {"method": "fallback_skip_small", "coverage": coverage, "matrix": None, "source": source}

    dst = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(box.astype("float32"), dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    warped_mask = None
    if mask is not None:
        warped_mask = cv2.warpPerspective(mask, M, (w, h))
    return warped, warped_mask, {"method": "fallback_perspective", "source": source, "coverage": coverage, "matrix": M.tolist()}

def gentle_curve_adjust(
    page_image: np.ndarray,
    page_mask: Optional[np.ndarray],
    max_shift_px: int = 6,
    min_span_ratio: float = 0.6,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    """
    轻量曲率微调：基于 mask 上下边缘拟合二次曲线，小幅矫正弯曲。
    无 mask 或位移过小时跳过。
    """
    if page_mask is None or page_mask.size == 0:
        return page_image, page_mask, {"method": "curve_skip", "reason": "no_mask"}
    mask_bin = (page_mask > 0).astype(np.uint8)
    H, W = mask_bin.shape[:2]
    xs = np.arange(W)
    y_min = np.full(W, -1, dtype=np.int32)
    y_max = np.full(W, -1, dtype=np.int32)
    for x in xs:
        ys = np.where(mask_bin[:, x] > 0)[0]
        if ys.size > 0:
            y_min[x] = ys.min()
            y_max[x] = ys.max()
    valid = np.where(y_min >= 0)[0]
    if valid.size < int(W * min_span_ratio):
        return page_image, page_mask, {"method": "curve_skip", "reason": "insufficient_columns"}
    x_use = xs[valid]
    top = y_min[valid].astype(np.float32)
    bottom = y_max[valid].astype(np.float32)
    # 基线（直线）采用两端点插值
    def _baseline(y_arr: np.ndarray) -> np.ndarray:
        return np.interp(xs, [x_use[0], x_use[-1]], [y_arr[0], y_arr[-1]])
    top_base = _baseline(top)
    bottom_base = _baseline(bottom)
    top_poly = np.poly1d(np.polyfit(x_use, top, deg=2))(xs)
    bottom_poly = np.poly1d(np.polyfit(x_use, bottom, deg=2))(xs)
    delta_top = np.clip(top_poly - top_base, -max_shift_px, max_shift_px)
    delta_bottom = np.clip(bottom_poly - bottom_base, -max_shift_px, max_shift_px)
    max_shift = float(np.max(np.abs([delta_top, delta_bottom])))
    if max_shift < 1.0:
        return page_image, page_mask, {"method": "curve_skip", "reason": "tiny_shift"}

    # 构建映射，按 y 在上下边之间的比例插值位移
    grid_x, grid_y = np.meshgrid(xs, np.arange(H))
    top_line = np.clip(top_base, 0, H - 1)
    bottom_line = np.clip(bottom_base, 0, H - 1)
    span = np.maximum(bottom_line - top_line, 1.0)
    t = (grid_y - top_line) / span
    t = np.clip(t, 0.0, 1.0)
    delta = delta_top * (1.0 - t) + delta_bottom * t
    map_x = grid_x.astype(np.float32)
    map_y = (grid_y - delta).astype(np.float32)
    map_y = np.clip(map_y, 0, H - 1).astype(np.float32)
    warped_img = cv2.remap(page_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_mask = cv2.remap(mask_bin, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_img, warped_mask, {
        "method": "curve_polywarp",
        "max_shift": max_shift,
        "applied": True,
        "columns_used": int(valid.size),
        "width": W,
    }


def dewarp_page(page_image: np.ndarray, page_mask: Optional[np.ndarray] = None, use_polyline: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, str]]:
    """
    直接使用透视矫正，必要时结合曲率微调。
    返回: (图像, mask, 信息字典: method 等)
    """
    return _fallback_perspective(page_image, page_mask)
