"""主流程：串联各模块完成单张图片处理。"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from docscan import config as cfg
from docscan import debug_utils
from docscan import io_utils
from docscan import mask_utils
from docscan import summary_utils
from docscan.context import PageResult, PipelineContext
from docscan.dewarp import dewarp_page, gentle_curve_adjust
from docscan.enhance import enhance_scan_style
from docscan.geom_refine import refine_geometry_with_opencv
from docscan.page_split import split_single_and_double_pages
from docscan.mask_utils import score_paper_mask
from rembg import new_session
from docscan.segment_strategy import Strategy, run_strategies

_REMBG_SESSIONS: dict[str, Any] = {}


def _get_rembg_session(model: str):
    """按模型缓存 rembg session，避免重复初始化。"""
    if not model:
        return None
    sess = _REMBG_SESSIONS.get(model)
    if sess is not None:
        return sess
    sess = new_session(model)
    _REMBG_SESSIONS[model] = sess
    return sess


_json_default = summary_utils.json_default


def _save_debug_image(arr: np.ndarray, path: Path) -> None:
    """包装 debug 保存，保证目录存在。"""
    debug_utils.save_debug_image(arr, path)


def _build_strategies_from_conf(conf: dict) -> list[Strategy]:
    """根据配置构建分割策略列表，默认保持原有三路策略顺序。"""
    strat_conf = (conf.get("segment_strategy") or {}).get("strategies") or []
    if not strat_conf:
        strat_conf = [
            {"name": "u2net", "model": "u2net", "alpha_matting": False, "preproc": "none"},
            {"name": "u2netp_mat", "model": "u2netp", "alpha_matting": True, "preproc": "none"},
            {"name": "light_u2netp_mat", "model": "u2netp", "alpha_matting": True, "preproc": "light"},
        ]
    max_side_default = conf["limits"].get("rembg_max_side", 1600)
    strategies = [
        Strategy(
            name=cfg_item.get("name", cfg_item.get("model", "u2net")),
            model=cfg_item.get("model", "u2net"),
            alpha_matting=cfg_item.get("alpha_matting", False),
            max_side=cfg_item.get("max_side", max_side_default),
            preproc=cfg_item.get("preproc", "none"),
        )
        for cfg_item in strat_conf
    ]
    max_trials = (conf.get("segment_retry") or {}).get("max_trials")
    if max_trials and max_trials > 0:
        strategies = strategies[:max_trials]
    return strategies


def _get_retry_condition(conf: dict):
    """从配置生成分割重试判定函数。"""
    retry_cfg = conf.get("segment_retry") or {}
    cond_cfg = retry_cfg.get("condition") or {}
    score_min = cond_cfg.get("score_min", 6.0)
    area_min = cond_cfg.get("area_ratio_min", 0.4)
    rect_min = cond_cfg.get("rect_ratio_min", 0.7)
    enabled = retry_cfg.get("enable", True)

    def need_retry(score: float, detail: dict) -> bool:
        if not enabled:
            return False
        return (score < score_min) and (
            (detail.get("area_ratio", 0.0) < area_min) or (detail.get("rect_ratio", 1.0) < rect_min)
        )

    return need_retry


def _prepare_image_and_mode(image_path: str, conf: dict) -> tuple:
    """
    读取图片并按配置约束尺寸，返回 (image, h_raw, w_raw, h, w, effective_mode)。
    保持原有尺度与 auto 模式逻辑不变。
    """
    image = io_utils.load_image(image_path)
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
    return image, h_raw, w_raw, h, w, effective_mode


def _build_runtime_config(
    mode: str,
    profile: str | None,
    config_path: str | None,
    debug: bool,
    debug_level: str | None,
    dry_run: bool,
    max_pages: int | None,
) -> tuple[dict, str, bool]:
    """
    加载配置并应用 debug/dry_run/max_pages 开关，返回 (conf, debug_level, debug_enabled)。
    行为保持与原逻辑一致。
    """
    conf = cfg.load_config(config_path, mode=mode, profile=profile)
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
    debug_level_eff = (conf["run"].get("debug_level") or "none").lower()
    debug_enabled = debug_level_eff != "none"
    return conf, debug_level_eff, debug_enabled


def _run_segmentation(image: np.ndarray, conf: dict, qa_cfg: dict, need_retry=None) -> tuple:
    """执行多策略分割并兜底，返回分割区域与元信息。策略与阈值均来源配置。"""
    strategies = _build_strategies_from_conf(conf)
    retry_cond = need_retry or _get_retry_condition(conf)

    best = run_strategies(image, image.shape[:2], strategies, qa_cfg, retry_cond, session_provider=_get_rembg_session)
    page_regions = best.get("regions") or []
    combined_mask = best.get("combined_mask")
    quality_detail = best.get("detail")
    attempts = best.get("attempts", [])
    segment_time = sum(a.get("time", 0.0) for a in attempts)
    segment_fallback = best.get("name") != "u2net"
    segment_fallback_msg = best.get("name") if segment_fallback else None

    def _apply_content_fallback(reason: str):
        nonlocal page_regions, combined_mask, segment_fallback, segment_fallback_msg, quality_detail, best
        mask_fb, bbox_fb, cov_fb = mask_utils.best_content_mask(image)
        if bbox_fb is not None and cov_fb > 0.05:
            page_regions = [(mask_fb, bbox_fb)]
            combined_mask = mask_fb
            score_fb, detail_fb = score_paper_mask(mask_fb, image.shape[:2], qa_cfg)
            detail_fb = detail_fb or {}
            detail_fb.setdefault("reason", []).insert(0, "content_fallback")
            detail_fb["area_ratio"] = detail_fb.get("area_ratio", cov_fb)
            detail_fb["rect_ratio"] = detail_fb.get("rect_ratio", 1.0)
            quality_detail = detail_fb
            segment_fallback = True
            segment_fallback_msg = f"content_fallback_{reason}"
            best = {
                "score": score_fb,
                "model": best.get("model"),
                "alpha_matting": best.get("alpha_matting", False),
                "preproc": best.get("preproc", "none"),
                "name": best.get("name"),
                "detail": quality_detail,
            }
        else:
            h2, w2 = image.shape[:2]
            full_mask = np.ones((h2, w2), dtype=np.uint8) * 255
            page_regions = [(full_mask, (0, 0, w2, h2))]
            combined_mask = full_mask
            score_fb, detail_fb = score_paper_mask(full_mask, image.shape[:2], qa_cfg)
            detail_fb = detail_fb or {}
            detail_fb.setdefault("reason", []).insert(0, "full_image_fallback")
            detail_fb["area_ratio"] = detail_fb.get("area_ratio", 1.0)
            detail_fb["rect_ratio"] = detail_fb.get("rect_ratio", 1.0)
            quality_detail = detail_fb
            segment_fallback = True
            segment_fallback_msg = f"full_image_{reason}"
            best = {
                "score": score_fb,
                "model": best.get("model"),
                "alpha_matting": best.get("alpha_matting", False),
                "preproc": best.get("preproc", "none"),
                "name": best.get("name"),
                "detail": quality_detail,
            }

    area_ratio = quality_detail.get("area_ratio", 0.0) if quality_detail else 0.0
    rect_ratio = quality_detail.get("rect_ratio", 0.0) if quality_detail else 0.0
    if area_ratio < 0.20 or rect_ratio < 0.60:
        _apply_content_fallback("low_area_or_rect")

    if not page_regions:
        mask_fb, bbox_fb, cov_fb = mask_utils.best_content_mask(image)
        if bbox_fb is not None and cov_fb > 0.05:
            page_regions = [(mask_fb, bbox_fb)]
            combined_mask = mask_fb
            log.warning("pipeline: segment 无结果，使用内容兜底 mask 覆盖率=%.3f bbox=%s", cov_fb, bbox_fb)
            segment_fallback = True
            segment_fallback_msg = "content_mask"
        else:
            h2, w2 = image.shape[:2]
            full_mask = np.ones((h2, w2), dtype=np.uint8) * 255
            page_regions = [(full_mask, (0, 0, w2, h2))]
            combined_mask = full_mask
            log.warning("pipeline: segment 无结果，回退使用整张图")
            segment_fallback = True
            segment_fallback_msg = "full_image"

    return (
        page_regions,
        combined_mask,
        quality_detail,
        attempts,
        segment_time,
        segment_fallback,
        segment_fallback_msg,
        best,
    )


def _split_pages_with_stats(image: np.ndarray, page_regions, conf: dict, stage_times: dict) -> list:
    """拆分页面并记录耗时，保持原参数与行为。"""
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
    return pages


def _process_single_page_entry(
    idx: int,
    page: dict,
    image_stem: str,
    out_dir: Path,
    conf: dict,
    debug_level: str,
    debug_enabled: bool,
    segment_fallback: bool,
    segment_fallback_msg: str | None,
    overlay_base: np.ndarray | None,
    dry_run: bool,
    max_pages: int | None,
) -> tuple[PageResult | None, np.ndarray | None]:
    """处理单页（含 dry_run），保持原有行为，返回 (PageResult 或 None, overlay_bbox)。"""
    if max_pages is not None and idx >= max_pages:
        return None, None
    page_id = f"page_{idx+1:03d}"
    file_prefix = f"{image_stem}_{page_id}"
    bbox = page.get("bbox")
    split_info = page.get("info")
    overlay = overlay_base.copy() if overlay_base is not None else None

    edge_info = None
    if page.get("mask") is not None:
        new_mask, edge_info = mask_utils.attenuate_mask_edges(page["image"], page["mask"])
        page["mask"] = new_mask

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
        if overlay_mask_cov is None or overlay_mask_cov < 0.95:
            overlay_bbox = bbox

    t_page = time.time()
    try:
        if dry_run:
            return (
                PageResult(
                    page_index=idx,
                    page_id=page_id,
                    split_info=split_info,
                    bbox=bbox,
                    segment_fallback=segment_fallback,
                    segment_fallback_reason=segment_fallback_msg,
                    edge_mask=edge_info,
                    dewarp={"method": "dry_run_skip"},
                    refine={"method": "dry_run_skip"},
                    dry_run=True,
                    elapsed=0.0,
                    stage_times={},
                ),
                overlay_bbox,
            )

        stage_local: Dict[str, float] = {}
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
        refine_quad_global = None
        refine_quad_backprojected = False
        if refine_quad_local and isinstance(refine_quad_local, list):
            bx0, by0, _, _ = page["bbox"]
            quad_local_np = np.array(refine_quad_local, dtype=np.float32)
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

        page_result = PageResult(
            page_index=idx,
            page_id=page_id,
            split_info=split_info,
            bbox=bbox,
            segment_fallback=segment_fallback,
            segment_fallback_reason=segment_fallback_msg,
            edge_mask=edge_info,
            enhanced_gray_path=str(gray_path),
            enhanced_bw_path=str(bw_path),
            dewarp=dewarp_info,
            curve_adjust=curve_info,
            refine=refine_info,
            refine_quad_global=_clamp_quad_to_bbox(refine_quad_global, overlay_bbox or bbox),
            refine_quad_backprojected=refine_quad_backprojected,
            elapsed=time.time() - t_page,
            stage_times=stage_local,
        )
        return page_result, overlay_bbox
    except Exception as e:  # noqa: BLE001
        log.exception("处理页面失败 %s", file_prefix)
        page_result = PageResult(
            page_index=idx,
            page_id=page_id,
            split_info=split_info,
            bbox=bbox,
            segment_fallback=segment_fallback,
            segment_fallback_reason=segment_fallback_msg,
            edge_mask=edge_info,
            dewarp=dewarp_info if "dewarp_info" in locals() else None,
            curve_adjust=curve_info if "curve_info" in locals() else None,
            refine=refine_info if "refine_info" in locals() else None,
            refine_quad_global=_clamp_quad_to_bbox(locals().get("refine_quad_global"), overlay_bbox or bbox),
            refine_quad_backprojected=locals().get("refine_quad_backprojected", False),
            error=str(e),
            elapsed=time.time() - t_page,
            stage_times=stage_local if "stage_local" in locals() else {},
        )
        return page_result, overlay_bbox


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

log = logging.getLogger(__name__)


def _mask_bbox_with_margin(
    mask: np.ndarray, margin_ratio: float = 0.01, min_margin: int = 6
) -> tuple[int, int, int, int] | None:
    """根据 mask 计算外接矩形并外扩，供调试/兜底区域使用。"""
    return debug_utils.mask_bbox_with_margin(mask, margin_ratio=margin_ratio, min_margin=min_margin)


def _clamp_quad_to_bbox(
    quad: list[list[float]] | None, bbox: tuple[int, int, int, int] | None
) -> list[list[int]] | None:
    """将四边形坐标限制在 bbox 范围内，避免越界。"""
    return debug_utils.clamp_quad_to_bbox(quad, bbox)


def _page_result_to_dict(page_result: PageResult) -> Dict[str, Any]:
    """PageResult 转为与历史一致的字典字段。"""
    data = asdict(page_result)
    data["split"] = data.pop("split_info", None)
    if not data.get("dry_run"):
        data.pop("dry_run", None)
    return data


def _draw_overlay_if_needed(overlay: np.ndarray | None, page_result: dict, overlay_bbox, debug_enabled: bool) -> None:
    """统一 overlay 绘制逻辑。"""
    if not debug_enabled or overlay is None:
        return
    debug_utils.draw_overlay(
        overlay,
        overlay_bbox,
        page_result.get("refine_quad_global"),
        page_result.get("segment_fallback"),
        page_result.get("segment_fallback_reason"),
        page_result.get("refine_quad_backprojected"),
    )


def warmup_models(config_path: str | None, mode: str, profile: str | None) -> None:
    """预热分割，减少首帧等待。"""
    conf = cfg.load_config(config_path, mode=mode, profile=profile)
    dummy = np.ones((256, 256, 3), dtype=np.uint8) * 255
    qa_cfg = (conf.get("segment_retry", {}) or {}).get("quality") or {}
    try:
        _run_segmentation(dummy, conf, qa_cfg, _get_retry_condition(conf))
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

    conf, debug_level, debug_enabled = _build_runtime_config(
        mode=mode,
        profile=profile,
        config_path=config_path,
        debug=debug,
        debug_level=debug_level,
        dry_run=dry_run,
        max_pages=max_pages,
    )

    image, h_raw, w_raw, h, w, effective_mode = _prepare_image_and_mode(image_path, conf)

    stage_times: Dict[str, float] = {}
    ctx = PipelineContext(
        image_path=image_path,
        output_root=Path(output_root),
        mode=mode,
        profile=profile,
        config=conf,
        debug_level=debug_level,
        debug_enabled=debug_enabled,
        max_pages=max_pages,
        dry_run=dry_run,
        effective_mode=effective_mode,
        stage_times=stage_times,
    )

    image_stem = Path(image_path).stem
    out_dir = ctx.output_root / image_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    t0 = time.time()
    # 分割前预处理（可选 rembg）：默认不开启，不改变原流程
    preproc_info: Dict[str, Any] = {"enabled": False, "applied": False, "reason": "removed"}
    segment_info: Dict[str, Any] = {}
    page_regions = []
    combined_mask = None
    quality_detail: Dict[str, Any] | None = None

    seg_image = image
    ctx.stage_times["preproc"] = 0.0

    retry_conf = conf.get("segment_retry", {}) or {}
    qa_cfg = (retry_conf.get("quality") or {}).copy()
    need_retry = _get_retry_condition(conf)

    (
        page_regions,
        combined_mask,
        quality_detail,
        attempts,
        seg_time,
        segment_fallback,
        segment_fallback_msg,
        best,
    ) = _run_segmentation(seg_image, conf, qa_cfg, need_retry)
    ctx.stage_times["segment"] = seg_time
    pages = _split_pages_with_stats(image, page_regions, conf, ctx.stage_times)

    segment_info, overlay, combined_mask = _aggregate_segment(
        image,
        page_regions,
        combined_mask,
        quality_detail,
        attempts,
        h,
        w,
        segment_fallback,
        segment_fallback_msg,
        best,
        debug_enabled,
        out_dir,
    )

    for idx, page in enumerate(pages):
        page_result, overlay_bbox = _process_single_page_entry(
            idx=idx,
            page=page,
            image_stem=image_stem,
            out_dir=out_dir,
            conf=conf,
            debug_level=debug_level,
            debug_enabled=debug_enabled,
            segment_fallback=segment_fallback,
            segment_fallback_msg=segment_fallback_msg,
            overlay_base=overlay,
            dry_run=dry_run,
            max_pages=max_pages,
        )
        if page_result is None:
            continue
        page_result_dict = _page_result_to_dict(page_result)
        results.append(page_result_dict)
        _draw_overlay_if_needed(overlay, page_result_dict, overlay_bbox, debug_enabled)

    debug_overlay_path = None
    if debug_enabled and overlay is not None:
        debug_overlay_path = str(out_dir / "02_debug_bbox.png")
        debug_utils.save_overlay(overlay, Path(debug_overlay_path))

    summary = summary_utils.build_summary(
        file_path=image_path,
        mode=mode,
        effective_mode=ctx.effective_mode,
        profile=profile,
        debug_level=debug_level,
        dry_run=dry_run,
        image_shape_raw=(h_raw, w_raw),
        image_shape_processed=(h, w),
        preproc=preproc_info,
        segment=segment_info,
        pages=results,
        elapsed_total=time.time() - t0,
        stage_times=ctx.stage_times,
        debug_overlay_path=debug_overlay_path,
    )
    summary_path = out_dir / "run_summary.json"
    try:
        summary_utils.save_summary(summary, summary_path)
    except Exception:  # noqa: BLE001
        log.exception("写入 run_summary 失败 %s", summary_path)

    return results


def _aggregate_segment(
    image: np.ndarray,
    page_regions,
    combined_mask,
    quality_detail,
    attempts,
    h: int,
    w: int,
    segment_fallback: bool,
    segment_fallback_msg: str | None,
    best: dict,
    debug_enabled: bool,
    out_dir: Path,
    ) -> tuple[dict, np.ndarray | None, np.ndarray]:
    """整理分割信息与调试 overlay，保持原字段与行为。"""
    if page_regions:
        combined_mask = np.zeros_like(page_regions[0][0])
        for mask_region, _bbox in page_regions:
            combined_mask = cv2.bitwise_or(combined_mask, mask_region)
    else:
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    segment_coverage = float(cv2.countNonZero(combined_mask)) / float(max(1, h * w))
    attempt_summaries = [
        {
            "name": a.get("name"),
            "model": a["model"],
            "alpha_matting": a["alpha_matting"],
            "preproc": a.get("preproc", "none"),
            "score": a["score"],
            "area_ratio": a["detail"].get("area_ratio"),
            "rect_ratio": a["detail"].get("rect_ratio"),
            "need_retry": a["need_retry"],
            "time": a["time"],
        }
        for a in attempts
    ]
    segment_info = {
        "regions": len(page_regions),
        "fallback": segment_fallback,
        "fallback_reason": segment_fallback_msg,
        "coverage": segment_coverage,
        "quality": quality_detail or {"reason": ["empty"]},
        "score": best.get("score", 0.0),
        "model": best.get("model"),
        "alpha_matting": best.get("alpha_matting", False),
        "preproc": best.get("preproc", "none"),
        "name": best.get("name"),
        "attempts": attempt_summaries,
    }
    overlay = None
    if debug_enabled:
        debug_utils.save_debug_image(combined_mask, out_dir / "01_debug_mask.png")
        overlay = debug_utils.prepare_overlay(image, combined_mask)
    return segment_info, overlay, combined_mask
