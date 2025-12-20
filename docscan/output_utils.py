"""输出落盘与 run_summary 写入。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from PIL import Image

from docscan import debug_utils
from docscan import io_utils
from docscan import summary_utils
from docscan.postproc import ProcessOutputs
from docscan.context import PageResult


def save_page_outputs(
    outputs: ProcessOutputs,
    out_dir: Path,
    file_prefix: str,
    debug_level: str,
    bbox,
    split_info,
    segment_fallback: bool,
    segment_fallback_msg: str | None,
    page_index: int,
    page_id: str,
    dry_run: bool,
    output_cfg: Dict[str, Any] | None = None,
) -> PageResult:
    """保存单页输出与调试图，返回 PageResult。"""
    extras: Dict[str, Any] = {}
    out_cfg = output_cfg or {}
    save_jpeg = out_cfg.get("save_jpeg", False)
    jpeg_quality = int(out_cfg.get("jpeg_quality", 95) or 95)
    preview_max_side = out_cfg.get("preview_max_side")
    orientation_cfg = out_cfg.get("orientation", {}) if output_cfg else {}
    orient_adjust = orientation_cfg.get("adjust_output", False)
    orient_save_raw = orientation_cfg.get("save_raw_bw", True)
    orient_score_th = float(orientation_cfg.get("score_threshold", 1.2))

    def _save_jpeg(arr: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(arr)
        img.save(path, format="JPEG", quality=jpeg_quality, optimize=True)

    def _resize_to_max_side(arr: np.ndarray, max_side: int) -> np.ndarray:
        h, w = arr.shape[:2]
        if max(h, w) <= max_side:
            return arr
        scale = float(max_side) / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    orient_angle = outputs.orientation_angle if orient_adjust else 0
    orient_score = outputs.orientation_score if orient_adjust else 0.0
    gray_to_save = outputs.gray
    bw_to_save = outputs.bw
    if orient_adjust and orient_angle in (90, 180, 270) and orient_score > orient_score_th:
        if orient_save_raw:
            raw_bw_path = out_dir / f"21_{file_prefix}_scan_bw_raw.png"
            io_utils.save_image(outputs.bw, raw_bw_path)
            extras["bw_raw"] = str(raw_bw_path)
        gray_to_save = orientation.rotate_image(outputs.gray, orient_angle)
        bw_to_save = orientation.rotate_image(outputs.bw, orient_angle)
        extras["orientation_angle"] = orient_angle
        extras["orientation_score"] = orient_score

    gray_path = out_dir / f"20_{file_prefix}_scan_gray.png"
    bw_path = out_dir / f"21_{file_prefix}_scan_bw.png"
    io_utils.save_image(gray_to_save, gray_path)
    io_utils.save_image(bw_to_save, bw_path)

    if save_jpeg:
        jpeg_gray = out_dir / f"20_{file_prefix}_scan_gray.jpg"
        _save_jpeg(gray_to_save, jpeg_gray)
        extras["jpeg_gray"] = str(jpeg_gray)
        jpeg_bw = out_dir / f"21_{file_prefix}_scan_bw.jpg"
        _save_jpeg(bw_to_save, jpeg_bw)
        extras["jpeg_bw"] = str(jpeg_bw)

    if preview_max_side:
        try:
            max_side_int = int(preview_max_side)
            preview = _resize_to_max_side(gray_to_save, max_side_int)
            preview_path = out_dir / f"preview_{file_prefix}.jpg"
            _save_jpeg(preview, preview_path)
            extras["preview"] = str(preview_path)
        except Exception:
            pass
    if debug_level == "full":
        if "raw" in outputs.debug_images:
            debug_utils.save_debug_image(outputs.debug_images["raw"], out_dir / f"10_{file_prefix}_raw.png")
            if "dewarp" in outputs.debug_images:
                debug_utils.save_debug_image(outputs.debug_images["dewarp"], out_dir / f"11_{file_prefix}_dewarp.png")
        if "refine" in outputs.debug_images:
            debug_utils.save_debug_image(outputs.debug_images["refine"], out_dir / f"12_{file_prefix}_refine.png")
        if "gray" in outputs.debug_images:
            debug_utils.save_debug_image(outputs.debug_images["gray"], out_dir / f"13_{file_prefix}_gray.png")
        if "bw" in outputs.debug_images:
            debug_utils.save_debug_image(outputs.debug_images["bw"], out_dir / f"14_{file_prefix}_bw.png")

    return PageResult(
        page_index=page_index,
        page_id=page_id,
        split_info=split_info,
        bbox=bbox,
        segment_fallback=segment_fallback,
        segment_fallback_reason=segment_fallback_msg,
        edge_mask=outputs.edge_info,
        enhanced_gray_path=str(gray_path),
        enhanced_bw_path=str(bw_path),
        dewarp=outputs.dewarp_info,
        curve_adjust=outputs.curve_info,
        refine=outputs.refine_info,
        refine_quad_global=outputs.refine_quad_global,
        refine_quad_backprojected=outputs.refine_quad_backprojected,
        elapsed=outputs.elapsed,
        stage_times=outputs.stage_times,
        dry_run=dry_run,
        extras=extras or None,
    )


def save_overlay_image(overlay, out_path: Path) -> None:
    if overlay is None:
        return
    debug_utils.save_overlay(overlay, out_path)


def build_and_save_summary(summary: Dict[str, Any], out_path: Path) -> None:
    summary_utils.save_summary(summary, out_path)
