"""分割结果整理与调试输出。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from docscan import debug_utils


def build_segment_report(
    image: np.ndarray,
    page_regions,
    quality_detail,
    attempts,
    h: int,
    w: int,
    segment_fallback: bool,
    segment_fallback_msg: str | None,
    best: dict,
    debug_enabled: bool,
    out_dir: Path,
) -> Tuple[Dict[str, Any], np.ndarray | None, np.ndarray]:
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
