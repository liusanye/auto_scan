"""单/双页判定与拆分，保留每页的 mask 与 bbox。"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from docscan import mask_utils
from docscan.segment import PageRegion

log = logging.getLogger(__name__)


def _crop_region(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def _expand_bbox(bbox: tuple[int, int, int, int], img_shape: tuple[int, int], expand_ratio: float) -> tuple[int, int, int, int]:
    """按比例扩展 bbox，防止裁剪过紧。"""
    x0, y0, x1, y1 = bbox
    h, w = img_shape
    bw, bh = x1 - x0, y1 - y0
    dx = int(bw * expand_ratio)
    dy = int(bh * expand_ratio)
    return (
        max(0, x0 - dx),
        max(0, y0 - dy),
        min(w, x1 + dx),
        min(h, y1 + dy),
    )


def _expand_bbox_directional(
    bbox: tuple[int, int, int, int],
    img_shape: tuple[int, int],
    base_ratio: float,
    top_ratio: float,
    bottom_ratio: float,
    lr_ratio: float,
) -> tuple[int, int, int, int]:
    """按方向单独扩展 bbox，缓解截顶/漏标题。"""
    x0, y0, x1, y1 = bbox
    h, w = img_shape
    bw, bh = x1 - x0, y1 - y0
    dx_base = int(bw * base_ratio)
    dy_base = int(bh * base_ratio)
    dx_lr = int(bw * lr_ratio)
    dy_top = int(bh * top_ratio)
    dy_bottom = int(bh * bottom_ratio)
    return (
        max(0, x0 - dx_base - dx_lr),
        max(0, y0 - dy_base - dy_top),
        min(w, x1 + dx_base + dx_lr),
        min(h, y1 + dy_base + dy_bottom),
    )


def _expand_bbox_with_secondary(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    img_shape: tuple[int, int],
    area_range: tuple[float, float] = (0.02, 0.25),
    aspect_thresh: float = 3.0,
    gap_ratio: float = 0.25,
    margin_ratio: float = 0.01,
    max_side_ratio: float = 0.10,
    min_side_ratio: float = 0.005,
    clamp_ratio_high_cover: float = 0.02,
) -> tuple[tuple[int, int, int, int], bool]:
    """副块驱动的定向扩边：主体用最大连通域，细长副块驱动单侧扩展。"""
    h, w = img_shape
    if mask is None or mask.size == 0:
        return bbox, False
    bin_mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num <= 1:
        return bbox, False
    # 主体 = 最大面积
    main_idx = 1 + int(np.argmax(stats[1:, 4]))
    mx, my, mw, mh, main_area = stats[main_idx]
    if mw == 0 or mh == 0:
        return bbox, False
    mx1, my1 = mx + mw, my + mh
    short_side = min(mw, mh)
    long_side = max(mw, mh)
    if short_side == 0:
        return bbox, False
    area_img = float(max(1, h * w))
    margin_px = max(1, int(short_side * margin_ratio))
    gap_limit = gap_ratio * short_side
    side_max = max(1, int(long_side * max_side_ratio))
    side_min = max(1, int(long_side * min_side_ratio))
    ext = {"left": 0, "right": 0, "top": 0, "bottom": 0}

    for lbl in range(1, num):
        if lbl == main_idx:
            continue
        x, y, bw, bh, area = stats[lbl]
        area_ratio = area / area_img
        if area_ratio < area_range[0] or area_ratio > area_range[1]:
            continue
        if min(bw, bh) == 0:
            continue
        aspect = max(bw, bh) / float(min(bw, bh))
        if aspect <= aspect_thresh:
            continue
        sx0, sy0, sx1, sy1 = x, y, x + bw, y + bh
        overlap_x = min(mx1, sx1) - max(mx, sx0)
        overlap_y = min(my1, sy1) - max(my, sy0)
        direction = None
        gap = 0.0
        if overlap_y > 0 and sx0 >= mx1:
            direction = "right"
            gap = float(sx0 - mx1)
        elif overlap_y > 0 and sx1 <= mx:
            direction = "left"
            gap = float(mx - sx1)
        elif overlap_x > 0 and sy0 >= my1:
            direction = "bottom"
            gap = float(sy0 - my1)
        elif overlap_x > 0 and sy1 <= my:
            direction = "top"
            gap = float(my - sy1)
        if direction is None:
            continue
        if gap > gap_limit:
            continue
        desired = gap + margin_px
        desired = max(desired, side_min)
        desired = min(desired, side_max)
        ext[direction] = max(ext[direction], int(desired))

    if all(v == 0 for v in ext.values()):
        return bbox, False

    x0, y0, x1, y1 = bbox
    margin_left = x0
    margin_right = w - x1
    margin_top = y0
    margin_bottom = h - y1
    box_area = (x1 - x0) * (y1 - y0)
    cover_ratio = box_area / area_img
    min_margin = min(margin_left, margin_right, margin_top, margin_bottom)
    if cover_ratio > 0.85 or min_margin < 0.08 * min(w, h):
        clamp_max = max(1, int(max(w, h) * clamp_ratio_high_cover))
        for k in ext:
            ext[k] = min(ext[k], clamp_max)

    new_bbox = (
        max(0, x0 - ext["left"]),
        max(0, y0 - ext["top"]),
        min(w, x1 + ext["right"]),
        min(h, y1 + ext["bottom"]),
    )
    return new_bbox, True


def _content_bbox(image: np.ndarray, thresh: int | None = None) -> tuple[int, int, int, int] | None:
    """基于内容的兜底 bbox：自适应阈值反转后取最大连通域外接矩形。"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if thresh is None:
        t, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if t == 0:
            _, bin_inv = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bin_inv = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    pad = max(1, int(0.02 * min(image.shape[:2])))
    H, W = image.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return (x0, y0, x1, y1)


def _find_valley(mask: np.ndarray) -> Tuple[int, float]:
    """在水平投影中寻找最小谷值位置及显著性（0-1）。"""
    proj = np.sum(mask > 0, axis=0)
    if proj.size == 0:
        return mask.shape[1] // 2, 0.0
    # 平滑
    proj = cv2.blur(proj.astype("float32"), (9, 1)).ravel()
    min_val = float(np.min(proj))
    max_val = float(np.max(proj) + 1e-6)
    prominence = (max_val - min_val) / max_val
    min_idx = int(np.argmin(proj))
    return min_idx, float(prominence)


def split_single_and_double_pages(
    image: np.ndarray,
    page_regions: List[PageRegion],
    enable_double: bool = True,
    double_ratio_threshold: float = 1.6,
    valley_prominence: float = 0.08,
    symmetry_tolerance: float = 0.25,
    cut_range: Tuple[float, float] = (0.4, 0.6),
    expand_ratio: float = 0.05,
    expand_extra: Tuple[float, float, float] = (0.12, 0.02, 0.05),
    single_min_height_ratio: float = 0.9,
    force_top_padding_ratio: float = 0.02,
    right_min_expand_ratio: float = 0.10,
    force_whole_page: bool = False,
    content_thresh: int | None = None,
    max_expand_ratio: float | None = None,
) -> List[dict]:
    """
    基于分割区域判断单/双页并拆分，返回带 mask/bbox 的页面列表。
    每个元素包含：image, mask, bbox(全图坐标), info。
    """
    pages: List[dict] = []
    H, W = image.shape[:2]
    for mask_region, bbox in page_regions:
        x0, y0, x1, y1 = bbox
        region = _crop_region(image, bbox)
        mask_crop = _crop_region(mask_region, bbox)
        h, w = region.shape[:2]
        ratio = w / max(h, 1)

        # 分割失败兜底（收紧触发条件）：仅在“高度明显不足且整体下沉”为带状时启用
        fallback_mask_applied = False
        height_ratio_full = (y1 - y0) / H
        if height_ratio_full < 0.65 and y0 > 0.10 * H:
            # 先在当前外接框附近做内容兜底，避免全图误伤
            local_bbox = (max(0, x0 - int(0.05 * W)), max(0, y0 - int(0.05 * H)), min(W, x1 + int(0.05 * W)), min(H, y1 + int(0.05 * H)))
            region_local = _crop_region(image, local_bbox)
            mask_local, cov_local, bbox_local = mask_utils.content_mask_with_coverage(region_local)
            if bbox_local is not None and cov_local > 0.35:
                lx, ly, lx1, ly1 = bbox_local
                lx += local_bbox[0]
                lx1 += local_bbox[0]
                ly += local_bbox[1]
                ly1 += local_bbox[1]
                mask_region_full = np.zeros_like(mask_region)
                mask_region_full[ly:ly1, lx:lx1] = 255
                mask_region = mask_region_full
                x0, y0, x1, y1 = lx, ly, lx1, ly1
                region = _crop_region(image, (x0, y0, x1, y1))
                mask_crop = _crop_region(mask_region, (x0, y0, x1, y1))
                h, w = region.shape[:2]
                ratio = w / max(h, 1)
                fallback_mask_applied = True
            # 若仍然过窄，再尝试全图 + 90° 旋转取较优覆盖
            if not fallback_mask_applied and height_ratio_full < 0.5:
                alt_mask, cov_alt, alt_bbox = mask_utils.best_content_mask(image)
                if alt_bbox is not None and cov_alt > height_ratio_full:
                    mask_region = alt_mask
                    x0, y0, x1, y1 = alt_bbox
                    region = _crop_region(image, (x0, y0, x1, y1))
                    mask_crop = _crop_region(mask_region, (x0, y0, x1, y1))
                    h, w = region.shape[:2]
                    ratio = w / max(h, 1)
                    fallback_mask_applied = True

        if force_whole_page:
            pad_top = int(force_top_padding_ratio * H)
            bbox_full = (0, pad_top, W, H)
            bbox_exp = _expand_bbox_directional(
                bbox_full,
                (H, W),
                base_ratio=expand_ratio,
                top_ratio=expand_extra[0],
                bottom_ratio=expand_extra[1],
                lr_ratio=expand_extra[2],
            )
            pages.append(
                {
                    "image": _crop_region(image, bbox_exp),
                    "mask": _crop_region(mask_region, bbox_exp),
                    "bbox": bbox_exp,
                    "info": {"type": "single", "maybe_double": ratio > double_ratio_threshold, "reason": "force_whole_page"},
                }
            )
            log.info("split: force whole page, ratio=%.2f", ratio)
            continue

        # 双页检测：宽高比 + 谷值显著性 + 左右面积对称度
        valley, prominence = 0, 0.0
        symmetry = 0.0
        cut_low = int(w * cut_range[0])
        cut_high = int(w * cut_range[1])
        is_double = False
        if enable_double:
            valley, prominence = _find_valley(mask_crop)
            area_left = float(np.sum(mask_crop[:, :valley] > 0))
            area_right = float(np.sum(mask_crop[:, valley:] > 0))
            symmetry = 1.0 if max(area_left, area_right) == 0 else abs(area_left - area_right) / max(area_left, area_right)
            is_double = (
                ratio > double_ratio_threshold
                and prominence >= valley_prominence
                and symmetry <= symmetry_tolerance
                and cut_low <= valley <= cut_high
            )

        if is_double and not force_whole_page:
            valley = max(cut_low, min(valley, cut_high))
            # 左右 bbox（全图坐标）
            left_bbox = (x0, y0, x0 + valley, y1)
            right_bbox = (x0 + valley, y0, x1, y1)
            left_bbox_base = left_bbox
            right_bbox_base = right_bbox
            left_bbox = _expand_bbox(left_bbox, (H, W), expand_ratio)
            right_bbox = _expand_bbox(right_bbox, (H, W), expand_ratio)
            if max_expand_ratio is not None:
                limit = int(max(H, W) * max_expand_ratio)
                lx0, ly0, lx1, ly1 = left_bbox
                lbx0, lby0, lbx1, lby1 = left_bbox_base
                rx0, ry0, rx1, ry1 = right_bbox
                rbx0, rby0, rbx1, rby1 = right_bbox_base
                lx0 = max(lx0, lbx0 - limit)
                ly0 = max(ly0, lby0 - limit)
                lx1 = min(lx1, lbx1 + limit)
                ly1 = min(ly1, lby1 + limit)
                rx0 = max(rx0, rbx0 - limit)
                ry0 = max(ry0, rby0 - limit)
                rx1 = min(rx1, rbx1 + limit)
                ry1 = min(ry1, rby1 + limit)
                left_bbox = (max(0, lx0), max(0, ly0), min(W, lx1), min(H, ly1))
                right_bbox = (max(0, rx0), max(0, ry0), min(W, rx1), min(H, ry1))
            pages.append(
                {
                    "image": _crop_region(image, left_bbox),
                    "mask": _crop_region(mask_region, left_bbox),
                    "bbox": left_bbox,
                    "info": {"type": "double", "side": "left", "valley": valley, "prominence": prominence, "symmetry": symmetry},
                }
            )
            pages.append(
                {
                    "image": _crop_region(image, right_bbox),
                    "mask": _crop_region(mask_region, right_bbox),
                    "bbox": right_bbox,
                    "info": {"type": "double", "side": "right", "valley": valley, "prominence": prominence, "symmetry": symmetry},
                }
            )
            log.info("split: double page ratio=%.2f valley=%d prom=%.3f sym=%.3f", ratio, valley, prominence, symmetry)
        else:
            # 若分割区域明显偏小，择机兜底；否则尊重分割框，避免直接撑满整图
            mask_area = float(cv2.countNonZero(mask_crop))
            area_ratio = mask_area / float(max(1, H * W))
            fallback = False
            height_ratio = (y1 - y0) / H
            right_gap = W - x1
            # 兜底条件A：高度不足 + 面积偏小且整体下沉（典型如 006/008/009）
            if height_ratio < 0.75 and area_ratio < 0.35 and y0 > 0.05 * H:
                fallback = True
            # 兜底条件B：高度不足且整体下移（但仅在面积不高时触发，避免正常页被整图化）
            if not fallback and height_ratio < single_min_height_ratio and y0 > force_top_padding_ratio * H and area_ratio < 0.5:
                fallback = True
            # 默认保留分割框，避免整图；仅在 fallback 时用 mask 外接框且不再扩边，防止膨胀
            if fallback:
                # fallback：直接用全局 mask 外接框，不扩边
                coords = cv2.findNonZero(mask_region)
                if coords is not None:
                    mx, my, mw, mh = cv2.boundingRect(coords)
                    bbox = (mx, my, mx + mw, my + mh)
                else:
                    bbox = (x0, y0, x1, y1)
                expand_ratio = 0.0
                expand_extra = (0.0, 0.0, 0.0)
                max_expand_ratio = 0.0
                right_min_expand_ratio_eff = 0.0
            else:
                bbox = (x0, y0, x1, y1)
                right_min_expand_ratio_eff = right_min_expand_ratio
            # 内容兜底：若分割框仍偏小，融合内容外接矩形
            content_box = _content_bbox(region, thresh=content_thresh)
            if content_box is not None:
                cx0, cy0, cx1, cy1 = content_box
                # 转换到全图坐标
                cx0 += x0
                cx1 += x0
                cy0 += y0
                cy1 += y0
                # 若内容框高度覆盖比例足够，则与当前 bbox 合并
                if (cy1 - cy0) / H > 0.5:
                    x0 = min(x0, cx0)
                    y0 = min(y0, cy0)
                    x1 = max(x1, cx1)
                    y1 = max(y1, cy1)
                    bbox = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))
            # 右侧最小扩边，防止竖排标题漏出
            x0, y0, x1, y1 = bbox
            if right_min_expand_ratio_eff > 0 and x1 < W:
                extra_right = int((x1 - x0) * right_min_expand_ratio_eff)
                if extra_right > 0:
                    x1 = min(W, x1 + extra_right)
                    bbox = (x0, y0, x1, y1)
            base_bbox = bbox

            # 定向扩边：如检测到细长副块（标题/装订条）则按方向扩展
            secondary_used = False
            if not fallback:
                bbox, secondary_used = _expand_bbox_with_secondary(mask_region, bbox, (H, W))

            top_extra, bottom_extra, lr_extra = expand_extra
            x0, y0, x1, y1 = bbox
            margin_left = x0
            margin_right = W - x1
            margin_top = y0
            margin_bottom = H - y1
            box_area = (x1 - x0) * (y1 - y0)
            cover_ratio = box_area / float(max(1, W * H))
            # 基础扩边与额外扩边均由配置传入，避免写死
            base_ratio_eff = max(0.0, float(expand_ratio))
            top_extra = max(0.0, float(top_extra))
            bottom_extra = max(0.0, float(bottom_extra))
            lr_extra = max(0.0, float(lr_extra))
            # 如果当前 bbox 已覆盖大部分区域，直接取消扩边，避免膨胀
            if cover_ratio >= 0.75:
                base_ratio_eff = 0.0
                top_extra = 0.0
                bottom_extra = 0.0
                lr_extra = 0.0
                if max_expand_ratio is not None:
                    max_expand_ratio = 0.0
            bbox_exp = bbox if base_ratio_eff == 0.0 and top_extra == 0.0 and bottom_extra == 0.0 and lr_extra == 0.0 else _expand_bbox_directional(
                bbox,
                (H, W),
                base_ratio=base_ratio_eff,
                top_ratio=top_extra,
                bottom_ratio=bottom_extra,
                lr_ratio=lr_extra,
            )
            if max_expand_ratio is not None:
                # 限制扩边不超过长边比例，防止整图化
                limit = int(max(W, H) * max_expand_ratio)
                bx0, by0, bx1, by1 = bbox_exp
                basex0, basey0, basex1, basey1 = base_bbox
                bx0 = max(bx0, basex0 - limit)
                by0 = max(by0, basey0 - limit)
                bx1 = min(bx1, basex1 + limit)
                by1 = min(by1, basey1 + limit)
                bbox_exp = (max(0, bx0), max(0, by0), min(W, bx1), min(H, by1))
            page_mask_out = None if fallback else _crop_region(mask_region, bbox_exp)
            pages.append(
                {
                    "image": _crop_region(image, bbox_exp),
                    "mask": page_mask_out,
                    "bbox": bbox_exp,
                    "info": {
                        "type": "single",
                        "maybe_double": ratio > double_ratio_threshold,
                        "valley": valley,
                        "prominence": prominence,
                        "symmetry": symmetry,
                        "secondary_expand": secondary_used,
                    },
                }
            )
            log.info("split: single page ratio=%.2f prom=%.3f sym=%.3f", ratio, prominence, symmetry)
    return pages
