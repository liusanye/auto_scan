"""页面分割：rembg + 连通域分析。"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np
from rembg import remove, new_session

PageRegion = Tuple[np.ndarray, Tuple[int, int, int, int]]
log = logging.getLogger(__name__)


def _preprocess_image(image: np.ndarray, max_side: int, preview_side: int | None = None) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    side_limit = max_side
    if preview_side is not None and preview_side > 0:
        side_limit = min(side_limit, preview_side)
    scale = min(1.0, side_limit / max(h, w))
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


def _keep_only_main_region(mask: np.ndarray, min_fill_ratio: float) -> np.ndarray:
    """保留最大连通域；若主体占比过低则返回原 mask 以避免误伤。"""
    if mask is None or mask.size == 0:
        return mask
    mask_bin = (mask > 0).astype("uint8")
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, 4]
    main_idx_rel = int(np.argmax(areas))
    main_idx = main_idx_rel + 1
    main_area = float(areas[main_idx_rel])
    h, w = mask.shape[:2]
    area_total = float(h * w)
    if main_area / max(1.0, area_total) < min_fill_ratio:
        return mask
    out = np.zeros_like(mask_bin)
    out[labels == main_idx] = 255
    return out.astype("uint8")


def _strip_edge_small(
    mask: np.ndarray,
    direction: str,
    width_frac: float,
    max_height_frac: float,
    max_area_frac: float,
    min_run: int,
) -> np.ndarray:
    """
    按方向（top/bottom）裁掉宽度明显变窄且面积占比小的贴边段，用于翻页/折角清理。
    """
    if mask is None or mask.size == 0:
        return mask
    proj_rows = np.sum(mask > 0, axis=1)
    if proj_rows.size == 0:
        return mask
    h = mask.shape[0]
    max_w = float(np.max(proj_rows))
    if max_w <= 0:
        return mask
    thresh = max_w * width_frac
    run_len = 0
    main_start = None
    indices = enumerate(proj_rows if direction == "top" else proj_rows[::-1])
    for idx, val in indices:
        if val >= thresh:
            run_len += 1
            if run_len >= min_run:
                main_start = idx - min_run + 1
                break
        else:
            run_len = 0
    if main_start is None:
        return mask
    cut_len = main_start + min_run
    if cut_len <= 0 or cut_len > max_height_frac * h:
        return mask
    area_total = float(np.count_nonzero(mask))
    if area_total <= 0:
        return mask
    if direction == "top":
        area_edge = float(np.count_nonzero(mask[:cut_len, :]))
    else:
        area_edge = float(np.count_nonzero(mask[h - cut_len :, :]))
    if area_edge / area_total > max_area_frac:
        return mask
    mask_out = mask.copy()
    if direction == "top":
        mask_out[:cut_len, :] = 0
    else:
        mask_out[h - cut_len :, :] = 0
    return mask_out


def _clean_mask(mask: np.ndarray, clean_cfg: dict | None) -> np.ndarray:
    """统一的 mask 清理：主体保留 + 上下贴边小块裁剪。"""
    if mask is None:
        return mask
    cfg = clean_cfg or {}
    min_fill_ratio = float(cfg.get("min_fill_ratio", 0.25))
    # 主连通域
    mask = _keep_only_main_region(mask, min_fill_ratio=min_fill_ratio)
    # 顶部/底部贴边小块
    mask = _strip_edge_small(
        mask,
        direction="top",
        width_frac=float(cfg.get("top_width_frac", 0.9)),
        max_height_frac=float(cfg.get("top_max_height_frac", 0.2)),
        max_area_frac=float(cfg.get("top_max_area_frac", 0.12)),
        min_run=int(cfg.get("top_min_run", 3)),
    )
    mask = _strip_edge_small(
        mask,
        direction="bottom",
        width_frac=float(cfg.get("bottom_width_frac", 0.9)),
        max_height_frac=float(cfg.get("bottom_max_height_frac", 0.2)),
        max_area_frac=float(cfg.get("bottom_max_area_frac", 0.12)),
        min_run=int(cfg.get("bottom_min_run", 3)),
    )
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


def _merge_small_regions(regions: List[PageRegion], small_merge_ratio: float, max_merge_gap_ratio: float) -> List[PageRegion]:
    if not regions:
        return regions
    # 按面积降序，最大区域作为主体，其余按距离和面积决定是否合并
    merged = [regions[0]]
    base_mask, base_bbox = regions[0]
    base_mask = base_mask.copy()
    bx0, by0, bx1, by1 = base_bbox
    h_img, w_img = base_mask.shape[:2]
    max_side = float(max(h_img, w_img))
    base_cx, base_cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    base_area = cv2.countNonZero(base_mask)

    for mask_region, bbox in regions[1:]:
        area = cv2.countNonZero(mask_region)
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dist = np.hypot(cx - base_cx, cy - base_cy)
        dist_ratio = dist / max(1.0, max_side)
        # 非主体小块，且面积占比低于阈值时，只有“足够近”才合并，否则丢弃
        if area < base_area * small_merge_ratio:
            if dist_ratio <= max_merge_gap_ratio:
                base_mask = cv2.bitwise_or(base_mask, mask_region)
                bx0, by0 = min(bx0, x0), min(by0, y0)
                bx1, by1 = max(bx1, x1), max(by1, y1)
            # 远的小块直接丢弃，不进入 merged
            continue
        merged.append((mask_region, bbox))
    merged[0] = (base_mask, (bx0, by0, bx1, by1))
    return merged


def segment_pages(
    image: np.ndarray,
    max_side: int = 1600,
    min_area_ratio: float = 0.01,
    morph_kernel: int = 5,
    small_merge_ratio: float = 0.25,
    max_merge_gap_ratio: float = 0.12,
    preview_side: int | None = None,
    clean_cfg: dict | None = None,
    rembg_model: str = "u2net",
    rembg_session=None,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    alpha_matting_base_size: int = 1000,
) -> List[PageRegion]:
    """
    使用预训练分割模型获取页面区域，并返回按面积排序的区域列表。
    """
    log.info("segment: start, image shape=%s", image.shape)
    small, scale = _preprocess_image(image, max_side, preview_side=preview_side)
    try:
        session = rembg_session
        if session is None and rembg_model:
            session = new_session(rembg_model)
        mask_small = remove(
            small,
            session=session,
            only_mask=True,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            alpha_matting_base_size=alpha_matting_base_size,
        )
        if mask_small.ndim == 3 and mask_small.shape[2] == 4:
            alpha = mask_small[:, :, 3]
            mask_small = _post_mask(alpha, morph_kernel)
        elif mask_small.ndim == 3:
            # 如果返回三通道，取第一通道
            mask_small = _post_mask(mask_small[:, :, 0], morph_kernel)
        else:
            mask_small = _post_mask(mask_small, morph_kernel)
    except Exception:  # noqa: BLE001
        log.exception("segment: rembg 失败，使用整图兜底")
        mask_small = np.ones(small.shape[:2], dtype=np.uint8) * 255
    # 还原到原尺寸
    if scale < 1.0:
        mask = cv2.resize(mask_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_small
    mask = _clean_mask(mask, clean_cfg)
    regions = _find_regions(mask, min_area_ratio)
    regions = _merge_small_regions(regions, small_merge_ratio, max_merge_gap_ratio)
    log.info("segment: regions=%d", len(regions))
    return regions
