"""OpenCV 几何精修：基于分割 mask 做四边形拟合与透视，并做形状质量过滤。"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _find_quad(mask: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box


def _order_points(pts: np.ndarray) -> np.ndarray:
    # 点顺序：tl, tr, br, bl
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def _warp_quad(image: np.ndarray, quad: np.ndarray, border_px: int = 12, target_ratio: float = 0.0, ratio_tolerance: float = 0.0) -> np.ndarray:
    """透视矫正并可选向目标比例靠拢（如 A4）。"""
    quad = _order_points(quad)
    (tl, tr, br, bl) = quad
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    ratio = maxH / float(max(maxW, 1))
    if target_ratio > 0 and ratio_tolerance > 0:
        if ratio < target_ratio * (1 - ratio_tolerance):
            maxH = int(maxW * target_ratio)
            dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
        elif ratio > target_ratio * (1 + ratio_tolerance):
            maxW = int(maxH / target_ratio)
            dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    if border_px > 0:
        warped = cv2.copyMakeBorder(warped, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return warped


def _shape_ok(rect: np.ndarray, mask: np.ndarray, shape_filter: Dict[str, float]) -> bool:
    (cx, cy), (w, h), _ = cv2.minAreaRect(rect.astype("float32"))
    if w == 0 or h == 0:
        return False
    ratio = float(w) / float(h)
    fill = float(cv2.countNonZero(mask)) / max(1.0, w * h)
    # 框中心偏移
    img_h, img_w = mask.shape[:2]
    off = np.linalg.norm(np.array([cx - img_w / 2, cy - img_h / 2]))
    off_norm = off / max(1.0, min(img_w, img_h))
    if ratio < shape_filter["min_ratio"] or ratio > shape_filter["max_ratio"]:
        return False
    if fill < shape_filter["min_fill"]:
        return False
    if off_norm > shape_filter["center_tolerance"]:
        return False
    return True


def _quad_from_rect(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32")


def _deskew(image: np.ndarray, mask: np.ndarray, max_angle: float) -> tuple[np.ndarray, np.ndarray, float]:
    """基于最小外接矩形做轻量 deskew，限制旋转角度。"""
    if mask is None or mask.size == 0:
        return image, mask, 0.0
    mask_bin = (mask > 0).astype("uint8")
    pts = cv2.findNonZero(mask_bin)
    if pts is None or len(pts) == 0:
        return image, mask, 0.0
    rect = cv2.minAreaRect(pts)
    angle = rect[2]
    if angle < -45:
        angle += 90
    if abs(angle) < 1e-3 or abs(angle) > max_angle:
        return image, mask, 0.0
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rotated_mask = cv2.warpAffine(mask_bin, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_img, rotated_mask, angle


def refine_geometry_with_opencv(
    page_image: np.ndarray,
    page_mask: Optional[np.ndarray] = None,
    border_px: int = 12,
    shape_filter: Optional[Dict[str, float]] = None,
    min_coverage: float = 0.9,
    deskew_max_angle: float = 5.0,
    a4_ratio: float = 1.414,
    a4_tolerance: float = 0.10,
) -> tuple[np.ndarray, Dict[str, object]]:
    """
    基于已有页面区域进行轮廓拟合、A4 比例微调、deskew、留白。
    返回 (图像, 信息字典)。
    """
    mask = page_mask
    if mask is None:
        gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        mask = 255 - mask
    deskew_angle = 0.0
    if deskew_max_angle > 0:
        page_image, mask, deskew_angle = _deskew(page_image, mask, deskew_max_angle)
    quad = _find_quad(mask)
    if quad is None:
        log.warning("geom_refine: 未找到四边形，直接返回原图")
        return page_image, {"method": "refine_skip", "reason": "no_quad", "quad": None, "deskew": deskew_angle}

    # 形状过滤：防止吸附到手掌/表格或偏离中心
    if shape_filter is None:
        shape_filter = {"min_fill": 0.0, "min_ratio": 0.0, "max_ratio": 999.0, "center_tolerance": 1.0}
    if not _shape_ok(quad, mask, shape_filter):
        log.warning("geom_refine: quad 未通过形状过滤，跳过透视")
        return page_image, {"method": "refine_skip", "reason": "shape_filter", "quad": quad.tolist(), "deskew": deskew_angle}

    # 覆盖率检查：若四边形覆盖率偏低，回退到裁剪框矩形
    mask_h, mask_w = mask.shape[:2]
    x0, y0, w_rect, h_rect = cv2.boundingRect(mask)
    rect_box = (x0, y0, x0 + w_rect, y0 + h_rect)
    rect_area = w_rect * h_rect
    quad_area = cv2.contourArea(quad.astype("float32"))
    coverage = quad_area / max(1.0, rect_area)
    if coverage < min_coverage:
        log.warning("geom_refine: quad 覆盖率低于阈值，回退用矩形透视 coverage=%.3f", coverage)
        quad = _quad_from_rect(rect_box)
        # 若矩形本身也过小，直接返回原图，避免粉框比绿框还小
        rect_fill = rect_area / float(mask_w * mask_h)
        if rect_fill < 0.2:
            log.warning("geom_refine: rect 覆盖率过低(%.3f)，跳过透视", rect_fill)
            return page_image, {"method": "refine_skip", "reason": "low_coverage", "quad": quad.tolist(), "coverage": coverage, "rect_fill": rect_fill, "deskew": deskew_angle}

    refined = _warp_quad(page_image, quad, border_px=border_px, target_ratio=a4_ratio, ratio_tolerance=a4_tolerance)
    return refined, {"method": "refine_warp", "reason": "ok", "quad": quad.tolist(), "coverage": coverage, "deskew": deskew_angle}
