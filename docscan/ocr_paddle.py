"""PaddleOCR 表格识别封装。"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import cv2
import numpy as np
from paddleocr import PPStructure, save_structure_res

log = logging.getLogger(__name__)
_ocr_instance: Optional[PPStructure] = None
_ocr_lock = Lock()


def _prepare_image(image_gray_or_bw: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    """限制尺寸并转换为 BGR，保证符合 PPStructure 输入要求。"""
    if image_gray_or_bw.ndim == 2:
        image_bgr = cv2.cvtColor(image_gray_or_bw, cv2.COLOR_GRAY2BGR)
    else:
        # 输入默认为 RGB，转为 BGR
        image_bgr = cv2.cvtColor(image_gray_or_bw, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        nw, nh = int(w * scale), int(h * scale)
        image_bgr = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return image_bgr


def _get_ocr(use_angle_cls: bool = True) -> PPStructure:
    global _ocr_instance
    with _ocr_lock:
        if _ocr_instance is None:
            _ocr_instance = PPStructure(show_log=False, use_angle_cls=use_angle_cls)
    return _ocr_instance


def warmup_ocr(use_angle_cls: bool = True) -> None:
    """预热 OCR，提前加载模型。"""
    try:
        _ = _get_ocr(use_angle_cls=use_angle_cls)
        log.info("OCR 预热完成")
    except Exception:  # noqa: BLE001
        log.exception("OCR 预热失败")


def run_table_ocr(
    image_gray_or_bw: np.ndarray,
    output_dir: str,
    page_id: str,
    use_angle_cls: bool = True,
    max_side: Optional[int] = None,
    max_retry: int = 1,
) -> Dict[str, Any]:
    """
    调用 PP-Structure / 表格识别，返回结构化结果。
    """
    attempt = 0
    last_exc: Exception | None = None
    while attempt <= max_retry:
        try:
            ocr = _get_ocr(use_angle_cls=use_angle_cls if attempt == 0 else False)
            img_bgr = _prepare_image(image_gray_or_bw, max_side=max_side)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            result = ocr(img_bgr)
            img_name = page_id if attempt == 0 else f"{page_id}_retry{attempt}"
            save_structure_res(result, output_dir, img_name=img_name)
            return {"raw": result, "output_dir": output_dir, "page_id": page_id, "attempt": attempt}
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.exception("run_table_ocr 失败，attempt=%d", attempt)
            attempt += 1
    return {"error": str(last_exc) if last_exc else "unknown"}
