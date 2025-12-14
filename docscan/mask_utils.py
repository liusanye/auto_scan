"""基于内容的兜底 mask 与 bbox 生成工具。"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def content_mask_with_coverage(
    image: np.ndarray,
    margin_ratio: float = 0.02,
    use_adaptive: bool = True,
    adaptive_block: int = 31,
    adaptive_c: int = 10,
) -> tuple[np.ndarray, float, tuple[int, int, int, int] | None]:
    """生成内容驱动的 mask，返回覆盖率与 bbox。"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if use_adaptive:
        block = adaptive_block + (adaptive_block % 2 == 0)
        bin_inv = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            adaptive_c,
        )
    else:
        _, bin_inv = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    H, W = mask.shape[:2]
    margin = int(min(H, W) * margin_ratio)
    if margin > 0:
        mask[:margin, :] = 0
        mask[H - margin :, :] = 0
        mask[:, :margin] = 0
        mask[:, W - margin :] = 0
    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros_like(mask, dtype=np.uint8), 0.0, None
    x, y, w, h = cv2.boundingRect(coords)
    coverage = (w * h) / float(max(1, H * W))
    mask_full = np.zeros_like(mask, dtype=np.uint8)
    cv2.rectangle(mask_full, (x, y), (x + w, y + h), 255, -1)
    return mask_full, coverage, (x, y, x + w, y + h)


def _tight_content_mask(image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int, int, int] | None]:
    """自适应阈值为主，若覆盖过大则尝试高阈值收紧。"""
    mask0, cov0, bbox0 = content_mask_with_coverage(image, margin_ratio=0.02, use_adaptive=True)
    if cov0 >= 0.92:
        mask1, cov1, bbox1 = content_mask_with_coverage(image, margin_ratio=0.03, use_adaptive=False)
        if bbox1 is not None and 0.03 < cov1 < cov0:
            return mask1, cov1, bbox1
    return mask0, cov0, bbox0


def best_content_mask(
    image: np.ndarray,
    target_cov: float = 0.7,
    cov_range: Tuple[float, float] = (0.2, 0.92),
) -> tuple[np.ndarray, tuple[int, int, int, int] | None, float]:
    """
    原图与 90° 旋转各取一次兜底 mask，选择覆盖率更合适者，避免整图化。
    """
    mask0, cov0, bbox0 = _tight_content_mask(image)
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    mask_r, cov_r, bbox_r = _tight_content_mask(rotated)
    candidates = []
    if bbox0 is not None and cov0 > 0.02:
        candidates.append(("origin", mask0, cov0, bbox0))
    if bbox_r is not None and cov_r > 0.02:
        candidates.append(("rot90", cv2.rotate(mask_r, cv2.ROTATE_90_COUNTERCLOCKWISE), cov_r, bbox_r))
    if not candidates:
        return mask0, bbox0, cov0

    def score(cov: float) -> float:
        return abs(cov - target_cov)

    lower, upper = cov_range
    filtered = [c for c in candidates if lower <= c[2] <= upper]
    chosen = min(filtered, key=lambda x: score(x[2])) if filtered else max(candidates, key=lambda x: x[2])
    tag, mask_sel, cov_sel, bbox_sel = chosen
    if tag == "rot90":
        coords = cv2.findNonZero(mask_sel)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            bbox_sel = (x, y, x + w, y + h)
    return mask_sel, bbox_sel, cov_sel


def content_mask_local(
    image: np.ndarray,
    expand_px: int,
    margin_ratio: float = 0.02,
) -> tuple[np.ndarray, float, tuple[int, int, int, int] | None]:
    """在局部区域生成内容 mask，便于针对分割偏移做补救。"""
    H, W = image.shape[:2]
    x0 = max(0, expand_px)
    y0 = max(0, expand_px)
    x1 = min(W, W - expand_px)
    y1 = min(H, H - expand_px)
    region = image[y0:y1, x0:x1]
    mask_local, cov_local, bbox_local = content_mask_with_coverage(region, margin_ratio=margin_ratio, use_adaptive=True)
    if bbox_local is not None:
        bx, by, bx1, by1 = bbox_local
        bbox_local = (bx + x0, by + y0, bx1 + x0, by1 + y0)
    return mask_local, cov_local, bbox_local


def attenuate_mask_edges(
    image: np.ndarray,
    mask: np.ndarray,
    band_ratio: float = 0.02,
    grad_ratio: float = 0.8,
    min_area_ratio: float = 0.6,
) -> tuple[np.ndarray, dict]:
    """
    若 mask 紧贴图像边缘且边缘梯度弱，则衰减边缘区域，避免吸附背景。
    返回 (新 mask, 信息)。
    """
    if mask is None or mask.size == 0:
        return mask, {"applied": False, "reason": "empty_mask"}
    H, W = mask.shape[:2]
    band = max(3, int(min(H, W) * band_ratio))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_med = float(np.median(grad))
    thresh = max(10.0, grad_med * grad_ratio)

    def _touch(side: str) -> bool:
        if side == "left":
            return bool(np.any(mask[:, :1] > 0))
        if side == "right":
            return bool(np.any(mask[:, -1:] > 0))
        if side == "top":
            return bool(np.any(mask[:1, :] > 0))
        if side == "bottom":
            return bool(np.any(mask[-1:, :] > 0))
        return False

    touched = {s: _touch(s) for s in ("left", "right", "top", "bottom")}
    removed = []
    new_mask = mask.copy()
    for side, is_touch in touched.items():
        if not is_touch:
            continue
        if side == "left":
            band_grad = grad[:, :band]
            if float(np.mean(band_grad)) < thresh:
                new_mask[:, :band] = 0
                removed.append("left")
        elif side == "right":
            band_grad = grad[:, -band:]
            if float(np.mean(band_grad)) < thresh:
                new_mask[:, -band:] = 0
                removed.append("right")
        elif side == "top":
            band_grad = grad[:band, :]
            if float(np.mean(band_grad)) < thresh:
                new_mask[:band, :] = 0
                removed.append("top")
        elif side == "bottom":
            band_grad = grad[-band:, :]
            if float(np.mean(band_grad)) < thresh:
                new_mask[-band:, :] = 0
                removed.append("bottom")
    # 若面积缩减过多，回退原 mask，避免误伤
    orig_area = float(np.count_nonzero(mask))
    new_area = float(np.count_nonzero(new_mask))
    if orig_area > 0 and new_area / orig_area < min_area_ratio:
        return mask, {"applied": False, "reason": "area_drop", "removed": removed, "orig_area": orig_area, "new_area": new_area}
    return new_mask, {
        "applied": bool(removed),
        "removed": removed,
        "band": band,
        "grad_thresh": thresh,
        "orig_area": orig_area,
        "new_area": new_area,
    }
