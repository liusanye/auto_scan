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


def _threshold_wolf(image: np.ndarray, window_size: int, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """Wolf-Jolion 阈值，参考 nlbin 思路。"""
    if window_size % 2 == 0:
        window_size += 1
    img_f = image.astype("float32")
    mean = cv2.boxFilter(img_f, ddepth=-1, ksize=(window_size, window_size), normalize=True)
    mean_sq = cv2.boxFilter(img_f * img_f, ddepth=-1, ksize=(window_size, window_size), normalize=True)
    std = cv2.sqrt(cv2.max(mean_sq - mean * mean, 0))
    min_img = cv2.erode(img_f, np.ones((window_size, window_size), dtype=np.uint8))
    R_eff = max(R, float(np.max(std)) + 1e-6)
    thresh = mean + k * ((std / R_eff) - 1.0) * (mean - min_img)
    return thresh


def enhance_scan_style(page_image: np.ndarray, enhance_cfg: Dict[str, float] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    division normalization + CLAHE + 自适应阈值 + 轻量锐化，输出灰度与二值图。
    参数从配置读取，不同模式可调整强度。
    """
    cfg = enhance_cfg or {}
    profile = cfg.get("profile", "quality")
    profile_overrides = (cfg.get("profiles") or {}).get(profile, {})

    bw_method = (profile_overrides.get("bw_method") or cfg.get("bw_method") or "sauvola").lower()
    blur_ksize = int(cfg.get("division_blur", 31))
    clahe_clip = float(cfg.get("clahe_clip", 2.0))
    clahe_grid = int(cfg.get("clahe_grid", 8))
    sauvola_window = int(profile_overrides.get("sauvola_window", cfg.get("sauvola_window", 31)))
    sauvola_k = float(profile_overrides.get("sauvola_k", cfg.get("sauvola_k", 0.2)))
    wolf_k = float(cfg.get("wolf_k", 0.2))
    bw_pre_smooth_ksize = int(profile_overrides.get("bw_pre_smooth_ksize", cfg.get("bw_pre_smooth_ksize", 0)) or 0)
    unsharp_amount = float(profile_overrides.get("unsharp_amount", cfg.get("unsharp_amount", 0.6)))
    unsharp_ksize = int(cfg.get("unsharp_ksize", 5))
    denoise_cfg = cfg.get("denoise", {}) or {}
    denoise_enable = denoise_cfg.get("enable", False)
    bilateral_d = int(denoise_cfg.get("bilateral_d", 0) or 0)
    bilateral_sigma_color = float(denoise_cfg.get("bilateral_sigma_color", 0) or 0)
    bilateral_sigma_space = float(denoise_cfg.get("bilateral_sigma_space", 0) or 0)
    post_morph_open = denoise_cfg.get("post_morph_open", False)
    post_morph_ksize = int(denoise_cfg.get("post_morph_ksize", 3) or 3)
    post_min_area = int(denoise_cfg.get("post_min_area", 0) or 0)

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

    # 逻辑修复开始：清晰的线性流
    norm_for_bw = norm.copy()
    if bw_pre_smooth_ksize and bw_pre_smooth_ksize > 1:
        if bw_pre_smooth_ksize % 2 == 0:
            bw_pre_smooth_ksize += 1
        norm_for_bw = cv2.medianBlur(norm_for_bw, bw_pre_smooth_ksize)
    
    if denoise_enable and bilateral_d and bilateral_sigma_color and bilateral_sigma_space:
        try:
            norm_for_bw = cv2.bilateralFilter(
                norm_for_bw, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space
            )
        except Exception:
            pass
    # 逻辑修复结束：移除了重复赋值和重复平滑

    if bw_method == "wolf":
        thresh = _threshold_wolf(norm_for_bw, window_size=sauvola_window, k=wolf_k)
    else:
        thresh = filters.threshold_sauvola(norm_for_bw, window_size=sauvola_window, k=sauvola_k)
    bw = (norm_for_bw > thresh).astype("uint8") * 255
    if denoise_enable:
        if post_morph_open and post_morph_ksize > 1:
            if post_morph_ksize % 2 == 0:
                post_morph_ksize += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (post_morph_ksize, post_morph_ksize))
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        if post_min_area and post_min_area > 0:
            try:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
                cleaned = np.zeros_like(bw)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= post_min_area:
                        cleaned[labels == i] = 255
                bw = cleaned
            except Exception:
                pass
    sharp = _unsharp_mask(norm, amount=unsharp_amount, ksize=unsharp_ksize)
    return sharp, bw
