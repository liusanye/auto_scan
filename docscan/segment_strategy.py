"""分割策略执行器：按多路策略尝试分割并择优。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from docscan.mask_utils import score_paper_mask
from docscan.segment import segment_pages


@dataclass
class Strategy:
    name: str
    model: str
    alpha_matting: bool = False
    max_side: int = 1600
    preproc: str = "none"  # none | light


def _apply_light(img: np.ndarray) -> np.ndarray:
    """轻量增强：CLAHE + 适度平滑，避免过度改变颜色分布。"""
    if img is None or img.size == 0:
        return img
    img_bgr = img
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # 轻微平滑，降低噪声
    out = cv2.medianBlur(out, 3)
    return out


def run_strategies(
    seg_image: np.ndarray,
    image_shape: Tuple[int, int],
    strategies: List[Strategy],
    qa_cfg: dict,
    retry_cond,
    session_provider=None,
) -> dict:
    """
    依次执行策略，记录得分，择优返回。
    返回 dict: {regions, combined_mask, main_mask, detail, score, model, alpha_matting, name, attempts}
    """
    attempts = []
    best = None
    h, w = image_shape
    for strat in strategies:
        img_in = seg_image
        if strat.preproc == "light":
            img_in = _apply_light(seg_image)

        t0 = time.time()
        session = session_provider(strat.model) if session_provider else None
        regions = segment_pages(
            img_in,
            max_side=strat.max_side,
            rembg_model=strat.model,
            rembg_session=session,
            alpha_matting=strat.alpha_matting,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            alpha_matting_base_size=1000,
        )
        elapsed = time.time() - t0

        result = {
            "name": strat.name,
            "model": strat.model,
            "alpha_matting": strat.alpha_matting,
            "regions": regions,
            "combined_mask": np.zeros((h, w), dtype=np.uint8),
            "main_mask": np.zeros((h, w), dtype=np.uint8),
            "detail": {"reason": ["empty"], "area_ratio": 0.0, "rect_ratio": 0.0},
            "score": 0.0,
            "need_retry": True,
            "time": elapsed,
            "preproc": strat.preproc,
        }

        if regions:
            combined = np.zeros_like(regions[0][0])
            for m, _bbox in regions:
                combined = cv2.bitwise_or(combined, m)
            from docscan import mask_utils

            main_mask, main_info = mask_utils.select_main_region(combined)
            filtered = []
            for m, bbox in regions:
                inter = cv2.bitwise_and(m, main_mask)
                if cv2.countNonZero(inter) > 0:
                    filtered.append((inter, bbox))
            if filtered:
                regions = filtered
            score, detail = score_paper_mask(main_mask, (h, w), qa_cfg)
            result.update(
                {
                    "regions": regions,
                    "combined_mask": combined,
                    "main_mask": main_mask,
                    "detail": detail,
                    "score": score,
                    "need_retry": retry_cond(score, detail),
                    "main_info": main_info,
                }
            )
        attempts.append(result)
        if best is None or result["score"] > best["score"]:
            best = result
        # 提前停止：达到基本合格就不再试
        if best and not best.get("need_retry", False):
            break
    if best is None:
        best = attempts[-1] if attempts else {}
    best["attempts"] = attempts
    return best
