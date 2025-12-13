"""页面去透视/去卷曲。"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional

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
    return warped, M, {"method": "trapezoid_warp", "quad": quad.tolist()}


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
        return image, mask, {"method": "fallback_none"}

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
        if coverage >= 0.01:
            info["coverage"] = coverage
            info["source"] = source
            return warped_img, warped_mask, info

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    w = int(rect[1][0])
    h = int(rect[1][1])
    coverage = cv2.contourArea(c) / float(max(1.0, img_h * img_w))
    if w == 0 or h == 0:
        return image, mask, {"method": "fallback_invalid_size"}
    # 若 coverage 过小，直接返回原图，避免粉框收缩过度
    if coverage < 0.01:
        log.warning("fallback_perspective: coverage too small (%.4f), skip warp", coverage)
        return image, mask, {"method": "fallback_skip_small", "coverage": coverage}

    dst = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(box.astype("float32"), dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    warped_mask = None
    if mask is not None:
        warped_mask = cv2.warpPerspective(mask, M, (w, h))
    return warped, warped_mask, {"method": "fallback_perspective", "source": source, "coverage": coverage}


def dewarp_page(page_image: np.ndarray, page_mask: Optional[np.ndarray] = None, use_polyline: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, str]]:
    """
    尝试使用专用 dewarp 库，失败则回退透视矫正。
    返回: (图像, mask, 信息字典: method 等)
    """
    # 预留 page-dewarp 接入点，可选依赖，失败自动回退
    try:
        import importlib

        dewarp_mod = importlib.import_module("page_dewarp")
        if hasattr(dewarp_mod, "dewarp"):
            result = dewarp_mod.dewarp(page_image, use_polyline=use_polyline)  # type: ignore[attr-defined]
            if isinstance(result, tuple) and len(result) >= 1:
                dewarped = result[0]
                dewarp_mask = result[1] if len(result) > 1 else page_mask
                return dewarped, dewarp_mask, {"method": "page_dewarp"}
    except Exception as exc:  # noqa: BLE001
        log.warning("dewarp: page-dewarp 不可用，使用透视回退: %s", exc)
    return _fallback_perspective(page_image, page_mask)
