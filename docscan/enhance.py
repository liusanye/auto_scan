"""扫描风格增强。"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from skimage import filters


def _division_normalization(gray: np.ndarray, blur_ksize: int = 31) -> np.ndarray:
    """背景归一化：大尺度模糊后做除法，再归一化到 0-255。"""
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    blur = blur.astype("float32") + 1.0
    norm = gray.astype("float32") / blur * 128.0
    norm = np.clip(norm, 0, 255).astype("uint8")
    return norm


def _unsharp_mask(gray: np.ndarray, amount: float = 0.6, ksize: int = 5) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype("uint8")


def enhance_scan_style(page_image: np.ndarray, enhance_cfg: Dict[str, float] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    division normalization + CLAHE + Sauvola + 轻量锐化，输出灰度与二值图。
    参数从配置读取，不同模式可调整强度。
    """
    cfg = enhance_cfg or {}
    profile = cfg.get("profile", "quality")
    blur_ksize = int(cfg.get("division_blur", 31))
    clahe_clip = float(cfg.get("clahe_clip", 2.0))
    clahe_grid = int(cfg.get("clahe_grid", 8))
    sauvola_window = int(cfg.get("sauvola_window", 31))
    sauvola_k = float(cfg.get("sauvola_k", 0.2))
    unsharp_amount = float(cfg.get("unsharp_amount", 0.6))
    unsharp_ksize = int(cfg.get("unsharp_ksize", 5))

    if profile == "fast":
        blur_ksize = max(15, blur_ksize // 2)
        clahe_clip = min(clahe_clip, 1.6)
        unsharp_amount = min(unsharp_amount, 0.4)
    elif profile == "auto":
        # 介于 fast 与 quality 之间
        clahe_clip = min(clahe_clip, 1.8)
        unsharp_amount = min(unsharp_amount, 0.5)

    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if sauvola_window % 2 == 0:
        sauvola_window += 1
    if unsharp_ksize % 2 == 0:
        unsharp_ksize += 1

    gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
    norm = _division_normalization(gray, blur_ksize=blur_ksize)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    norm = clahe.apply(norm)
    thresh = filters.threshold_sauvola(norm, window_size=sauvola_window, k=sauvola_k)
    bw = (norm > thresh).astype("uint8") * 255
    sharp = _unsharp_mask(norm, amount=unsharp_amount, ksize=unsharp_ksize)
    return sharp, bw
