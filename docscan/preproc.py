"""分割前的轻量预处理：只作用于分割通路，提升纸张边缘对比，不改后续输出。"""

from __future__ import annotations

import cv2
import numpy as np


def _white_balance(img: np.ndarray, pct: float = 1.0) -> np.ndarray:
    """简单分位截断的白平衡，平衡通道防止色偏。"""
    if pct <= 0:
        return img
    img_f = img.astype(np.float32)
    balanced = np.zeros_like(img_f)
    for c in range(3):
        lo, hi = np.percentile(img_f[:, :, c], [pct, 100 - pct])
        if hi <= lo + 1e-3:
            balanced[:, :, c] = img_f[:, :, c]
            continue
        channel = (img_f[:, :, c] - lo) * (255.0 / (hi - lo))
        balanced[:, :, c] = np.clip(channel, 0, 255)
    return balanced.astype("uint8")


def _contrast_stretch(gray: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """按分位数拉伸对比度，避免极端值干扰。"""
    lo, hi = np.percentile(gray, [low_pct, high_pct])
    if hi <= lo + 1e-3:
        return gray
    stretched = np.clip((gray - lo) * (255.0 / (hi - lo)), 0, 255)
    return stretched.astype("uint8")


def _unsharp(image: np.ndarray, radius: int = 3, amount: float = 1.0) -> np.ndarray:
    if radius <= 0 or amount <= 0:
        return image
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=radius)
    sharp = cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype("uint8")


def _edge_boost(image: np.ndarray, alpha: float = 0.08, dilate: int = 2) -> np.ndarray:
    """通过梯度生成边缘蒙版并轻度叠加，增强边缘可见性。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(sobelx), 1.0, cv2.convertScaleAbs(sobely), 1.0, 0))
    thr = np.percentile(grad, 75.0)
    _, mask = cv2.threshold(grad, max(10.0, thr), 255, cv2.THRESH_BINARY)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
        mask = cv2.dilate(mask, k, iterations=1)
    mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    boosted = cv2.addWeighted(image, 1.0, mask_col, alpha, 0)
    return boosted


def preprocess_for_segmentation(image: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """
    仅用于分割的预处理：生成一张调色后的副本，提升边缘对比，返回处理后的图与信息。
    """
    info = {
        "enabled": bool(cfg.get("enable", True)),
        "applied": False,
        "reason": None,
        "gray_mean": None,
        "gray_std": None,
        "edge_boost": False,
        "profile": cfg.get("profile", "base"),
    }
    if not info["enabled"]:
        return image, info

    img = image.copy()
    profile = info["profile"]

    # 0) 白平衡
    pct = float(cfg.get("white_balance_pct", 1.0))
    img = _white_balance(img, pct=pct)

    # 1) 对比度拉伸 + 轻 gamma（基于灰度）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mean = float(gray.mean())
    gray_std = float(gray.std())
    info["gray_mean"] = gray_mean
    info["gray_std"] = gray_std
    stretched = _contrast_stretch(gray)
    gamma_min, gamma_max = cfg.get("gamma_range", (0.9, 1.1))
    if profile == "boost":
        gamma_min, gamma_max = 0.75, 0.95
    gamma = np.clip((gamma_min + gamma_max) * 0.5, 0.2, 3.0)
    stretched = np.clip((stretched / 255.0) ** (1.0 / gamma) * 255.0, 0, 255).astype("uint8")

    # 2) CLAHE 在 L 通道进行（用拉伸后的亮度）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)
    clahe_clip = float(cfg.get("clahe_clip", 2.0))
    clahe_grid = int(cfg.get("clahe_grid", 8))
    if profile == "boost":
        clahe_clip = min(2.5, max(1.2, clahe_clip * 0.8))
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(clahe_grid, clahe_grid),
    )
    l = clahe.apply(stretched)
    lab = cv2.merge([l, a, b])
    proc = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3) 去噪 + unsharp
    ksize = int(cfg.get("median_ksize", 3))
    if ksize > 1:
        proc = cv2.medianBlur(proc, ksize)
    amount_cfg = float(cfg.get("unsharp_amount", 1.1))
    if profile == "boost":
        amount_cfg = max(0.4, min(1.2, amount_cfg + 0.2))
    proc = _unsharp(proc, radius=int(cfg.get("unsharp_radius", 3)), amount=amount_cfg)
    # 3.1) 轻量平滑抑制撕裂
    k_morph = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
    proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4) 可选边缘引导
    edge_cfg = cfg.get("edge_boost", {}) or {}
    if edge_cfg.get("enable", False) or profile == "edge":
        proc = _edge_boost(proc, alpha=float(edge_cfg.get("alpha", 0.08)), dilate=int(edge_cfg.get("dilate", 2)))
        info["edge_boost"] = True

    info["applied"] = True
    info["reason"] = "forced"
    return proc, info
