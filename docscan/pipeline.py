"""主流程：串联各模块完成单张图片处理。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from dataclasses import dataclass

from docscan import config as cfg
from docscan import io_utils
from docscan import mask_utils
from docscan import preproc
from docscan.segment import segment_pages
from docscan.page_split import split_single_and_double_pages
from docscan.dewarp import dewarp_page, gentle_curve_adjust
from docscan.geom_refine import refine_geometry_with_opencv
from docscan.enhance import enhance_scan_style

log = logging.getLogger(__name__)


@dataclass
class PageContext:
    """统一管理页面级元数据，避免散落的 dict 误用。"""

    bbox: tuple[int, int, int, int]
    split_info: dict | None
    segment_fallback: bool
    segment_fallback_reason: str | None
    refine_quad_global: list | None = None
    refine_quad_backprojected: bool = False
    dewarp_info: dict | None = None
    curve_info: dict | None = None
    refine_info: dict | None = None
    enhanced_gray_path: str | None = None
    enhanced_bw_path: str | None = None
    error: str | None = None
    stage_times: dict | None = None
    edge_mask_info: dict | None = None


def _save_debug_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.save_image(arr, str(path))


def _mask_bbox_with_margin(
    mask: np.ndarray, margin_ratio: float = 0.01, min_margin: int = 6
) -> tuple[int, int, int, int] | None:
    """根据 mask 计算外接矩形并外扩，供调试/兜底区域使用。"""
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = mask.shape[:2]
    margin = max(min_margin, int(max(h, w) * margin_ratio))
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(w - 1, x1 + margin),
        min(h - 1, y1 + margin),
    )


def _clamp_quad_to_bbox(
    quad: list[list[float]] | None, bbox: tuple[int, int, int, int] | None
) -> list[list[int]] | None:
    """将四边形坐标限制在 bbox 范围内，避免越界。"""
    if quad is None or bbox is None:
        return quad
    bx0, by0, bx1, by1 = bbox
    clamped: list[list[int]] = []
    for x, y in quad:
        cx = min(max(int(round(x)), bx0), bx1)
        cy = min(max(int(round(y)), by0), by1)
        clamped.append([cx, cy])
    return clamped


def _mask_quality(mask: np.ndarray, shape: tuple[int, int], qa_cfg: dict) -> tuple[bool, dict]:
    """评估掩码是否“像一张纸”，返回 (合格?, 详情)。"""
    h, w = shape[:2]
    total = float(h * w)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return False, {"reason": ["empty"], "area_ratio": 0.0}
    x, y, bw, bh = cv2.boundingRect(coords)
    area = float(cv2.countNonZero(mask))
    area_ratio = area / total
    rect_ratio = area / max(1.0, bw * bh)
    aspect = max(bw, bh) / max(1.0, min(bw, bh))
    cx, cy = x + bw / 2.0, y + bh / 2.0
    cx_img, cy_img = w / 2.0, h / 2.0
    center_dist = np.hypot(cx - cx_img, cy - cy_img) / max(1.0, min(h, w))
    reasons = []
    cfg_q = qa_cfg or {}
    min_area = cfg_q.get("min_area_ratio", 0.15)
    min_rect = cfg_q.get("min_rect_ratio", 0.5)
    min_size_ratio = cfg_q.get("min_size_ratio", 0.3)
    ratio_range = cfg_q.get("ratio_range", [0.6, 1.8])
    center_max = cfg_q.get("center_dist_max", 0.4)
    if area_ratio < min_area:
        reasons.append("area_small")
    if bw < w * min_size_ratio or bh < h * min_size_ratio:
        reasons.append("bbox_small")
    if rect_ratio < min_rect:
        reasons.append("rect_ratio_low")
    if not (ratio_range[0] <= aspect <= ratio_range[1]):
        reasons.append("aspect_out")
    if center_dist > center_max:
        reasons.append("center_off")
    ok = len(reasons) == 0
    return ok, {
        "reason": reasons,
        "area_ratio": area_ratio,
        "rect_ratio": rect_ratio,
        "aspect": aspect,
        "center_dist": center_dist,
        "bbox": (x, y, bw, bh),
    }


def _draw_overlay(
    overlay: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    refine_quad_global: list | None,
    segment_fallback: bool,
    segment_fallback_msg: str | None,
    refine_quad_backprojected: bool,
) -> None:
    """在原图上绘制裁剪框/四边形与提示信息。"""
    line_thickness = 5
    if bbox is not None:
        bx0, by0, bx1, by1 = bbox
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 200, 0), line_thickness)
    if refine_quad_global:
        quad_np = np.array(refine_quad_global, dtype=int)
        cv2.polylines(overlay, [quad_np], isClosed=True, color=(255, 0, 255), thickness=line_thickness)
        for idx_c, (cx, cy) in enumerate(quad_np, start=1):
            cv2.circle(overlay, (int(cx), int(cy)), 10, (50, 50, 255), -1)
            cv2.putText(
                overlay,
                f"P1-{idx_c}",
                (int(cx) + 12, int(cy) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"P1-{idx_c}",
                (int(cx) + 12, int(cy) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
    # 图例
    cv2.putText(
        overlay,
        "Green=segment area (mask hull)  Pink=refine quad (actual warp)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "Green=segment area (mask hull)  Pink=refine quad (actual warp)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if segment_fallback:
        cv2.putText(
            overlay,
            f"segment fallback ({segment_fallback_msg or 'auto'})",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"segment fallback ({segment_fallback_msg or 'auto'})",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if refine_quad_global and not refine_quad_backprojected:
        cv2.putText(
            overlay,
            "refine quad 未反投影(仅供参考)",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "refine quad 未反投影(仅供参考)",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def warmup_models(config_path: str | None, mode: str, profile: str | None) -> None:
    """预热分割，减少首帧等待。"""
    conf = cfg.load_config(config_path, mode=mode, profile=profile)
    dummy = np.ones((256, 256, 3), dtype=np.uint8) * 255
    try:
        segment_pages(
            dummy,
            max_side=conf["limits"]["rembg_max_side"],
            min_area_ratio=conf["segment"]["min_area_ratio"],
            morph_kernel=conf["segment"]["morph_kernel"],
        )
        log.info("分割模型预热完成")
    except Exception:  # noqa: BLE001
        log.exception("分割预热失败")


def process_image_file(
    image_path: str,
    output_root: str,
    mode: str = "quality",
    profile: str | None = None,
    config_path: str | None = None,
    debug: bool = False,
    debug_level: str | None = None,
    dry_run: bool = False,
    max_pages: int | None = None,
) -> List[Dict[str, Any]]:
    """
    读图 → segment → split → dewarp → geom_refine → enhance。
    """
    conf = cfg.load_config(config_path, mode=mode, profile=profile)
    # debug_level：优先 CLI debug_level，其次 debug 开关（debug=True 默认 full）
    if debug_level:
        conf["run"]["debug_level"] = debug_level
        conf["run"]["debug"] = debug_level != "none"
    elif debug:
        conf["run"]["debug"] = True
        conf["run"]["debug_level"] = conf["run"].get("debug_level") or "full"
    else:
        conf["run"]["debug"] = conf["run"].get("debug", False)
        conf["run"]["debug_level"] = conf["run"].get("debug_level", "none")
    conf["run"]["max_pages"] = max_pages
    conf["run"]["dry_run"] = dry_run
    debug_level = (conf["run"].get("debug_level") or "none").lower()
    debug_enabled = debug_level != "none"

    image = io_utils.load_image(image_path)
    # 入口尺寸保护：限制最大边并保证最小边不过小
    h_raw, w_raw = image.shape[:2]
    h, w = h_raw, w_raw
    max_side = conf["limits"]["max_side"]
    min_side = conf["limits"]["min_side"]
    scale_down = min(1.0, max_side / max(h, w))
    if scale_down < 1.0:
        image = cv2.resize(image, (int(w * scale_down), int(h * scale_down)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
    min_dim = min(h, w)
    if min_dim < min_side:
        scale_up = min_side / float(min_dim)
        cap = max_side / float(max(h, w))
        scale_up = min(scale_up, cap)
        if scale_up > 1.0:
            image = cv2.resize(image, (int(w * scale_up), int(h * scale_up)), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

    effective_mode = conf["mode"]
    if conf.get("mode") == "auto":
        longest_for_mode = max(max(h_raw, w_raw), max(h, w))
        if longest_for_mode <= 1800:
            conf["dewarp"]["enabled"] = False
            conf["enhance"]["profile"] = "fast"
            effective_mode = "auto-fast"
        elif longest_for_mode <= 2600:
            conf["dewarp"]["enabled"] = True
            conf["enhance"]["profile"] = "quality"
            effective_mode = "auto-quality"
        else:
            conf["dewarp"]["enabled"] = True
            conf["enhance"]["profile"] = "quality"
            effective_mode = "auto-quality-highres"

    image_stem = Path(image_path).stem
    out_dir = Path(output_root) / image_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    t0 = time.time()
    stage_times: Dict[str, float] = {}
    # 分割前预处理 + 失败重试：仅供 mask/框识别，主图保持不变
    t_pre = time.time()
    preproc_info: Dict[str, Any] = {"enabled": conf["preproc"].get("enable", True), "applied": False}
    segment_info: Dict[str, Any] = {}
    retry_conf = conf.get("segment_retry", {}) or {}
    qa_cfg = (retry_conf.get("quality") or {}).copy()
    max_trials = int(retry_conf.get("max_trials", 3))
    retry_enabled = retry_conf.get("enable", True) and max_trials > 1
    trial_profiles = ["base", "boost", "edge"]
    page_regions = []
    combined_mask = None
    used_profile = "base"
    quality_detail: Dict[str, Any] | None = None

    for trial_idx in range(max_trials):
        profile = trial_profiles[min(trial_idx, len(trial_profiles) - 1)]
        used_profile = profile
        preproc_cfg = conf["preproc"].copy()
        preproc_cfg["profile"] = profile
        t_pre_trial = time.time()
        if preproc_cfg.get("enable", True):
            seg_image, preproc_info = preproc.preprocess_for_segmentation(image, preproc_cfg)
        else:
            seg_image = image
            preproc_info = {"enabled": False, "applied": False, "reason": "disabled", "profile": profile}
        stage_times["preproc"] = stage_times.get("preproc", 0.0) + (time.time() - t_pre_trial)

        t_seg = time.time()
        page_regions = segment_pages(
            seg_image,
            max_side=conf["limits"]["rembg_max_side"],
            min_area_ratio=conf["segment"]["min_area_ratio"],
            morph_kernel=conf["segment"]["morph_kernel"],
            small_merge_ratio=conf["segment"].get("small_merge_ratio", 0.25),
            max_merge_gap_ratio=conf["segment"].get("max_merge_gap_ratio", 0.12),
            preview_side=conf["run"].get("segment_preview_side"),
            clean_cfg=conf["segment"].get("clean"),
        )
        stage_times["segment"] = stage_times.get("segment", 0.0) + (time.time() - t_seg)

        # 主体筛选：保留最像纸张的连通域
        if page_regions:
            combined_mask = np.zeros_like(page_regions[0][0])
            for m, _bbox in page_regions:
                combined_mask = cv2.bitwise_or(combined_mask, m)
            main_mask, main_info = mask_utils.select_main_region(combined_mask)
            filtered_regions = []
            for m, _bbox in page_regions:
                intersect = cv2.bitwise_and(m, main_mask)
                if cv2.countNonZero(intersect) > 0:
                    filtered_regions.append((intersect, _bbox))
            if filtered_regions:
                page_regions = filtered_regions
            segment_info["main_select"] = main_info
            ok, quality_detail = _mask_quality(main_mask, image.shape[:2], qa_cfg)
            segment_info["quality"] = quality_detail
            if ok or not retry_enabled:
                break
            else:
                log.warning("pipeline: segment 质量不足，尝试重试 trial=%d reasons=%s", trial_idx + 1, quality_detail.get("reason"))
        else:
            combined_mask = np.zeros(seg_image.shape[:2], dtype=np.uint8)
            segment_info["quality"] = {"reason": ["empty"]}
            if not retry_enabled:
                break
    preproc_info["profile_used"] = used_profile
    stage_times["preproc"] = stage_times.get("preproc", 0.0)

    segment_fallback = False
    segment_fallback_msg: str | None = None
    if not page_regions:
        mask_fb, bbox_fb, cov_fb = mask_utils.best_content_mask(image)
        if bbox_fb is not None and cov_fb > 0.05:
            page_regions = [(mask_fb, bbox_fb)]
            log.warning("pipeline: segment 无结果，使用内容兜底 mask 覆盖率=%.3f bbox=%s", cov_fb, bbox_fb)
            segment_fallback = True
            segment_fallback_msg = "content_mask"
        else:
            # 分割失败兜底：整张图作为单页
            h2, w2 = image.shape[:2]
            full_mask = np.ones((h2, w2), dtype=np.uint8) * 255
            page_regions = [(full_mask, (0, 0, w2, h2))]
            log.warning("pipeline: segment 无结果，回退使用整张图")
            segment_fallback = True
            segment_fallback_msg = "full_image"
    t_split = time.time()
    pages = split_single_and_double_pages(
        image,
        page_regions,
        enable_double=conf["split"].get("enable_double", False),
        double_ratio_threshold=conf["split"]["double_ratio_threshold"],
        valley_prominence=conf["split"].get("valley_prominence", 0.08),
        symmetry_tolerance=conf["split"].get("symmetry_tolerance", 0.25),
        cut_range=tuple(conf["split"].get("cut_range", (0.4, 0.6))),
        expand_ratio=conf["geom"].get("crop_expand_ratio", 0.05),
        expand_extra=(
            conf["geom"].get("crop_expand_extra", {}).get("top", 0.12),
            conf["geom"].get("crop_expand_extra", {}).get("bottom", 0.02),
            conf["geom"].get("crop_expand_extra", {}).get("lr", 0.05),
        ),
        single_min_height_ratio=conf["split"].get("single_min_height_ratio", 0.9),
        force_top_padding_ratio=conf["split"].get("force_top_padding_ratio", 0.02),
        right_min_expand_ratio=conf["split"].get("right_min_expand_ratio", 0.10),
        force_whole_page=conf["split"].get("force_whole_page", False),
        max_expand_ratio=conf["geom"].get("max_expand_ratio"),
    )
    stage_times["split"] = time.time() - t_split
    log.info("pipeline: segmented pages=%d", len(pages))

    combined_mask = None
    if page_regions:
        combined_mask = np.zeros_like(page_regions[0][0])
        for mask_region, _bbox in page_regions:
            combined_mask = cv2.bitwise_or(combined_mask, mask_region)
    else:
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    segment_coverage = float(cv2.countNonZero(combined_mask)) / float(max(1, h * w))
    segment_info = {
        "regions": len(page_regions),
        "fallback": segment_fallback,
        "fallback_reason": segment_fallback_msg,
        "coverage": segment_coverage,
    }
    if debug_enabled:
        _save_debug_image(combined_mask, out_dir / "01_debug_mask.png")
        overlay = image.copy()
    else:
        overlay = None

    for idx, page in enumerate(pages):
        if max_pages is not None and idx >= max_pages:
            break
        page_id = f"page_{idx+1:03d}"
        file_prefix = f"{image_stem}_{page_id}"
        page_ctx = PageContext(
            bbox=page.get("bbox"),
            split_info=page.get("info"),
            segment_fallback=segment_fallback,
            segment_fallback_reason=segment_fallback_msg,
            stage_times={},
        )
        # 边缘置信度衰减，避免贴边背景
        edge_info = None
        if page.get("mask") is not None:
            new_mask, edge_info = mask_utils.attenuate_mask_edges(page["image"], page["mask"])
            page["mask"] = new_mask
        page_ctx.edge_mask_info = edge_info

        # 绿框绘制区域：优先使用分割 mask 的外扩 hull，若覆盖率过大则隐藏
        overlay_bbox: tuple[int, int, int, int] | None = None
        overlay_mask_cov: float | None = None
        if overlay is not None and page.get("mask") is not None:
            mask_bin = page["mask"]
            overlay_mask_cov = float(cv2.countNonZero(mask_bin)) / float(max(1, mask_bin.size))
            if overlay_mask_cov < 0.95:
                local_bbox = _mask_bbox_with_margin(mask_bin)
                if local_bbox is not None:
                    bx0, by0, _, _ = page["bbox"]
                    lx0, ly0, lx1, ly1 = local_bbox
                    overlay_bbox = (bx0 + lx0, by0 + ly0, bx0 + lx1, by0 + ly1)
        if overlay is not None and overlay_bbox is None:
            # 非全图时兜底用 split 后的 bbox
            if overlay_mask_cov is None or overlay_mask_cov < 0.95:
                overlay_bbox = page_ctx.bbox

        t_page = time.time()
        try:
            if dry_run:
                # 只做分割/裁剪调试，不执行后续重处理
                page_ctx.dewarp_info = {"method": "dry_run_skip"}
                page_ctx.refine_info = {"method": "dry_run_skip"}
                page_ctx.stage_times = {}
                results.append(
                    {
                        "page_index": idx,
                        "page_id": page_id,
                        "split": page_ctx.split_info,
                        "bbox": page_ctx.bbox,
                        "segment_fallback": page_ctx.segment_fallback,
                        "segment_fallback_reason": page_ctx.segment_fallback_reason,
                        "edge_mask": page_ctx.edge_mask_info,
                        "dewarp": page_ctx.dewarp_info,
                        "refine": page_ctx.refine_info,
                        "dry_run": True,
                        "stage_times": page_ctx.stage_times,
                    }
                )
                continue

            stage_local: Dict[str, float] = {}
            refine_quad_local = None
            refine_quad_global = None
            if debug_level == "full":
                _save_debug_image(page["image"], out_dir / f"10_{file_prefix}_raw.png")

            dewarp_enabled = conf["dewarp"].get("enabled", True) and not conf["split"].get("force_whole_page", False)
            refine_enabled = conf["geom"].get("enable_refine", True)
            t_dewarp = time.time()
            if dewarp_enabled:
                dewarped, dewarped_mask, dewarp_info = dewarp_page(
                    page["image"],
                    page_mask=page.get("mask"),
                    use_polyline=conf["dewarp"]["enable_polyline"],
                )
            else:
                dewarped = page["image"]
                dewarped_mask = page.get("mask")
                dewarp_info = {"method": "skip_disabled"}
            stage_local["dewarp"] = time.time() - t_dewarp
            # 轻量曲率微调
            curve_info = None
            t_curve = time.time()
            if conf["dewarp"].get("enable_curve_adjust", True) and dewarped_mask is not None:
                dewarped, dewarped_mask, curve_info = gentle_curve_adjust(
                    dewarped,
                    dewarped_mask,
                    max_shift_px=int(conf["dewarp"].get("curve_max_shift_px", 6)),
                )
            stage_local["curve_adjust"] = time.time() - t_curve
            t_refine = time.time()
            if refine_enabled:
                page_refined, refine_info = refine_geometry_with_opencv(
                    dewarped,
                    page_mask=dewarped_mask,
                    border_px=conf["geom"]["border_px"],
                    shape_filter=conf["geom"].get("shape_filter"),
                    deskew_max_angle=conf["geom"].get("deskew_max_angle", 5.0),
                    a4_ratio=conf["geom"].get("a4_ratio", 1.414),
                    a4_tolerance=conf["geom"].get("a4_tolerance", 0.10),
                )
            else:
                page_refined = dewarped
                refine_info = {"method": "refine_skip", "reason": "disabled", "quad": None}
            stage_local["refine"] = time.time() - t_refine
            t_enh = time.time()

            gray, bw = enhance_scan_style(page_refined, enhance_cfg=conf.get("enhance"))
            gray_path = out_dir / f"20_{file_prefix}_scan_gray.png"
            bw_path = out_dir / f"21_{file_prefix}_scan_bw.png"
            io_utils.save_image(gray, gray_path)
            io_utils.save_image(bw, bw_path)
            stage_local["enhance"] = time.time() - t_enh

            if debug_level == "full":
                _save_debug_image(dewarped, out_dir / f"11_{file_prefix}_dewarp.png")
                _save_debug_image(page_refined, out_dir / f"12_{file_prefix}_refine.png")
                _save_debug_image(gray, out_dir / f"13_{file_prefix}_gray.png")
                _save_debug_image(bw, out_dir / f"14_{file_prefix}_bw.png")

            refine_quad_local = refine_info.get("quad") if refine_enabled else None
            refine_quad_backprojected = False
            if refine_quad_local and isinstance(refine_quad_local, list):
                bx0, by0, _, _ = page["bbox"]
                quad_local_np = np.array(refine_quad_local, dtype=np.float32)
                # 尝试用去透视矩阵把粉框从“拉直后的坐标”反投影回裁剪页，再加上 bbox 偏移落到原图
                dewarp_matrix = dewarp_info.get("matrix")
                if dewarp_matrix is not None:
                    try:
                        M = np.array(dewarp_matrix, dtype=np.float32)
                        M_inv = np.linalg.inv(M)
                        quad_orig = cv2.perspectiveTransform(quad_local_np.reshape(1, -1, 2), M_inv)[0]
                        refine_quad_global = [[int(pt[0] + bx0), int(pt[1] + by0)] for pt in quad_orig]
                        refine_quad_backprojected = True
                    except Exception:  # noqa: BLE001
                        log.exception("refine quad 反投影失败，退回 dewarp quad 显示")
                if not refine_quad_backprojected:
                    quad_src = dewarp_info.get("quad") or refine_quad_local
                    quad_src_np = np.array(quad_src, dtype=np.float32)
                    refine_quad_global = [[int(pt[0] + bx0), int(pt[1] + by0)] for pt in quad_src_np]
            page_ctx.dewarp_info = dewarp_info
            page_ctx.curve_info = curve_info
            page_ctx.refine_info = refine_info
            clamp_bbox = overlay_bbox or page_ctx.bbox
            page_ctx.refine_quad_global = _clamp_quad_to_bbox(refine_quad_global, clamp_bbox)
            page_ctx.refine_quad_backprojected = refine_quad_backprojected
            page_ctx.enhanced_gray_path = str(gray_path)
            page_ctx.enhanced_bw_path = str(bw_path)
            page_ctx.stage_times = stage_local

            page_result = {
                "page_index": idx,
                "page_id": page_id,
                "split": page_ctx.split_info,
                "bbox": page_ctx.bbox,
                "segment_fallback": page_ctx.segment_fallback,
                "segment_fallback_reason": page_ctx.segment_fallback_reason,
                "edge_mask": page_ctx.edge_mask_info,
                "enhanced_gray_path": page_ctx.enhanced_gray_path,
                "enhanced_bw_path": page_ctx.enhanced_bw_path,
                "dewarp": page_ctx.dewarp_info,
                "curve_adjust": page_ctx.curve_info,
                "refine": page_ctx.refine_info,
                "refine_quad_global": page_ctx.refine_quad_global,
                "refine_quad_backprojected": page_ctx.refine_quad_backprojected,
                "elapsed": time.time() - t_page,
                "stage_times": page_ctx.stage_times,
            }
            results.append(page_result)
        except Exception as e:  # noqa: BLE001
            log.exception("处理页面失败 %s #%s", image_path, page_id)
            page_ctx.error = str(e)
            page_ctx.stage_times = page_ctx.stage_times or stage_local
            page_result = {
                "page_index": idx,
                "page_id": page_id,
                "split": page_ctx.split_info,
                "bbox": page_ctx.bbox,
                "segment_fallback": page_ctx.segment_fallback,
                "segment_fallback_reason": page_ctx.segment_fallback_reason,
                "edge_mask": page_ctx.edge_mask_info,
                "dewarp": page_ctx.dewarp_info,
                "curve_adjust": page_ctx.curve_info,
                "refine": page_ctx.refine_info,
                "refine_quad_global": page_ctx.refine_quad_global,
                "refine_quad_backprojected": page_ctx.refine_quad_backprojected,
                "error": page_ctx.error,
                "elapsed": time.time() - t_page,
                "stage_times": page_ctx.stage_times,
            }
            results.append(page_result)

        # 绘制调试 overlay：裁剪 bbox + refine 四边形
        if debug_enabled and overlay is not None:
            _draw_overlay(
                overlay,
                overlay_bbox,
                page_ctx.refine_quad_global,
                page_ctx.segment_fallback,
                page_ctx.segment_fallback_reason,
                page_ctx.refine_quad_backprojected,
            )

    if debug_enabled and overlay is not None:
        _save_debug_image(overlay, out_dir / "02_debug_bbox.png")

    summary = {
        "file": image_path,
        "mode": mode,
        "mode_effective": effective_mode,
        "profile": profile,
        "debug_level": debug_level,
        "dry_run": dry_run,
        "image_shape_raw": [h_raw, w_raw],
        "image_shape_processed": [h, w],
        "preproc": preproc_info,
        "segment": segment_info,
        "pages": results,
        "elapsed_total": time.time() - t0,
        "stage_times": stage_times,
    }
    if debug_enabled:
        summary["debug_overlay"] = str(out_dir / "02_debug_bbox.png")
    summary_path = out_dir / "run_summary.json"
    try:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        log.exception("写入 run_summary 失败 %s", summary_path)

    return results
