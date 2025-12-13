"""页面分割：rembg + 连通域分析。"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np
from rembg import remove

PageRegion = Tuple[np.ndarray, Tuple[int, int, int, int]]
log = logging.getLogger(__name__)


def _preprocess_image(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def _post_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    mask = (mask > 127).astype("uint8") * 255
    if kernel_size > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.medianBlur(mask, kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def _find_regions(mask: np.ndarray, min_area_ratio: float) -> List[PageRegion]:
    h, w = mask.shape[:2]
    total = h * w
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[PageRegion] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < total * min_area_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        bbox = (x, y, x + bw, y + bh)
        mask_region = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(mask_region, [c], -1, 255, -1)
        regions.append((mask_region, bbox))
    regions.sort(key=lambda r: cv2.countNonZero(r[0]), reverse=True)
    return regions


def _merge_small_regions(regions: List[PageRegion], small_merge_ratio: float) -> List[PageRegion]:
    if not regions:
        return regions
    merged = [regions[0]]
    base_area = cv2.countNonZero(regions[0][0])
    base_mask, base_bbox = regions[0]
    base_mask = base_mask.copy()
    bx0, by0, bx1, by1 = base_bbox
    for mask_region, bbox in regions[1:]:
        area = cv2.countNonZero(mask_region)
        if area < base_area * small_merge_ratio:
            # 合并到主区域
            base_mask = cv2.bitwise_or(base_mask, mask_region)
            x0, y0, x1, y1 = bbox
            bx0, by0 = min(bx0, x0), min(by0, y0)
            bx1, by1 = max(bx1, x1), max(by1, y1)
        else:
            merged.append((mask_region, bbox))
    merged[0] = (base_mask, (bx0, by0, bx1, by1))
    return merged


def segment_pages(image: np.ndarray, max_side: int = 1600, min_area_ratio: float = 0.01, morph_kernel: int = 5, small_merge_ratio: float = 0.25) -> List[PageRegion]:
    """
    使用预训练分割模型获取页面区域，并返回按面积排序的区域列表。
    """
    log.info("segment: start, image shape=%s", image.shape)
    small, scale = _preprocess_image(image, max_side)
    try:
        rgba = remove(small)
        alpha = rgba[:, :, 3]
        mask_small = _post_mask(alpha, morph_kernel)
    except Exception:  # noqa: BLE001
        log.exception("segment: rembg 失败，使用整图兜底")
        mask_small = np.ones(small.shape[:2], dtype=np.uint8) * 255
    # 还原到原尺寸
    if scale < 1.0:
        mask = cv2.resize(mask_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_small
    regions = _find_regions(mask, min_area_ratio)
    regions = _merge_small_regions(regions, small_merge_ratio)
    log.info("segment: regions=%d", len(regions))
    return regions
