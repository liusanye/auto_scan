"""分割结果整理与调试输出。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from docscan import debug_utils
from docscan import summary_utils


def _sanitize_label(value: str | None, fallback: str) -> str:
    """将标签清洗为文件名可用格式，避免空值与特殊字符。"""
    if not value:
        return fallback
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value)).strip("_")
    return cleaned or fallback


def _save_attempts_manifest(attempts: list[dict], out_dir: Path) -> Path | None:
    """保存分割策略的尝试结果与对应 mask 图片（仅供调试）。"""
    if not attempts:
        return None
    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    best_attempt_idx = None
    best_score = None
    for idx, att in enumerate(attempts, start=1):
        score = att.get("score", 0.0)
        if best_score is None or score > best_score:
            best_score = score
            best_attempt_idx = idx

    manifest: Dict[str, Any] = {"total": len(attempts), "attempts": []}
    for idx, att in enumerate(attempts, start=1):
        detail = att.get("detail") or {}
        name = _sanitize_label(att.get("name"), f"attempt_{idx:02d}")
        model = _sanitize_label(att.get("model"), "model")
        preproc = _sanitize_label(att.get("preproc", "none"), "none")
        tag_parts = [f"{idx:02d}", name, model]
        if preproc != "none":
            tag_parts.append(preproc)
        if att.get("alpha_matting"):
            tag_parts.append("matting")
        prefix = "_".join(tag_parts)

        combined_path = None
        combined_mask = att.get("combined_mask")
        if combined_mask is not None:
            combined_path = attempts_dir / f"{prefix}_combined.png"
            debug_utils.save_debug_image(combined_mask, combined_path)
        main_path = None
        main_mask = att.get("main_mask")
        if main_mask is not None:
            main_path = attempts_dir / f"{prefix}_main.png"
            debug_utils.save_debug_image(main_mask, main_path)

        manifest["attempts"].append(
            {
                "index": idx,
                "name": att.get("name"),
                "model": att.get("model"),
                "alpha_matting": bool(att.get("alpha_matting", False)),
                "preproc": att.get("preproc", "none"),
                "score": att.get("score", 0.0),
                "need_retry": bool(att.get("need_retry", False)),
                "time": att.get("time", 0.0),
                "area_ratio": detail.get("area_ratio"),
                "rect_ratio": detail.get("rect_ratio"),
                "reason": detail.get("reason"),
                "is_best_attempt": idx == best_attempt_idx,
                "combined_mask": str(combined_path) if combined_path else None,
                "main_mask": str(main_path) if main_path else None,
            }
        )
    manifest_path = attempts_dir / "attempts.json"
    summary_utils.save_summary(manifest, manifest_path)
    return manifest_path


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
    debug_level: str,
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

    if debug_level == "full":
        manifest_path = _save_attempts_manifest(attempts, out_dir)
        if manifest_path:
            segment_info["attempts_manifest"] = str(manifest_path)
            segment_info["attempts_dir"] = str(manifest_path.parent)
    overlay = None
    if debug_enabled:
        debug_utils.save_debug_image(combined_mask, out_dir / "01_debug_mask.png")
        overlay = debug_utils.prepare_overlay(image, combined_mask)
    return segment_info, overlay, combined_mask
