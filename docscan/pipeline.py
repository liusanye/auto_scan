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
from docscan import runtime_utils
from docscan import segment_report
from docscan import split_module
from docscan.dewarp import dewarp_page, gentle_curve_adjust
from docscan.enhance import enhance_scan_style
from docscan.geom_refine import refine_geometry_with_opencv
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
    """拆分页面并记录耗时，可按开关跳过拆分页。"""
    t_split = time.time()
    pages = split_module.split_pages(image, page_regions, conf, stage_times)
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
    overlay_bbox = None

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
                    edge_mask=None,
                    dewarp={"method": "dry_run_skip"},
                    refine={"method": "dry_run_skip"},
                    dry_run=True,
                    elapsed=0.0,
                    stage_times={},
                ),
                overlay_bbox,
            )

        from docscan import postproc
        from docscan import output_utils

        outputs = postproc.process_page_image(page, conf, debug_level)
        page_result = output_utils.save_page_outputs(
            outputs=outputs,
            out_dir=out_dir,
            file_prefix=file_prefix,
            debug_level=debug_level,
            bbox=bbox,
            split_info=split_info,
            segment_fallback=segment_fallback,
            segment_fallback_msg=segment_fallback_msg,
            page_index=idx,
            page_id=page_id,
            dry_run=dry_run,
            output_cfg=conf.get("output"),
        )
        overlay_bbox = outputs.overlay_bbox
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
            edge_mask=None,
            dewarp=None,
            curve_adjust=None,
            refine=None,
            refine_quad_global=None,
            refine_quad_backprojected=False,
            error=str(e),
            elapsed=time.time() - t_page,
            stage_times={},
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
    output_mode: str | None = None,
    tone: str | None = None,
) -> List[Dict[str, Any]]:
    """
    读图 → segment → split → dewarp → geom_refine → enhance。
    """

    conf, debug_level, debug_enabled = runtime_utils.build_runtime_config(
        mode=mode,
        profile=profile,
        config_path=config_path,
        debug=debug,
        debug_level=debug_level,
        dry_run=dry_run,
        max_pages=max_pages,
        output_mode=output_mode,
        tone=tone,
    )

    image, h_raw, w_raw, h, w, effective_mode = runtime_utils.prepare_image_and_mode(image_path, conf)

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

    segment_info, overlay, combined_mask = segment_report.build_segment_report(
        image=image,
        page_regions=page_regions,
        quality_detail=quality_detail,
        attempts=attempts,
        h=h,
        w=w,
        segment_fallback=segment_fallback,
        segment_fallback_msg=segment_fallback_msg,
        best=best,
        debug_enabled=debug_enabled,
        out_dir=out_dir,
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
