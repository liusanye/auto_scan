"""页面方向检测与矫正（轻量 0/90/180/270 判定）。"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _score_orientation(img: np.ndarray) -> float:
    """对二值图做简单打分：横向文本密度越高分越高。"""
    if img is None or img.size == 0:
        return 0.0
    # 计算水平/垂直投影的方差比值
    proj_h = np.sum(img > 0, axis=1).astype(np.float32)
    proj_v = np.sum(img > 0, axis=0).astype(np.float32)
    var_h = float(np.var(proj_h)) + 1e-6
    var_v = float(np.var(proj_v)) + 1e-6
    return var_h / var_v


def detect_orientation(page_rgb: np.ndarray) -> Tuple[int, float]:
    """
    简易方向判定，返回 (angle, score)，angle ∈ {0,90,180,270}。
    基于多角度打分，非高置信度不建议旋转。
    """
    if page_rgb is None or page_rgb.size == 0:
        return 0, 0.0
    h, w = page_rgb.shape[:2]
    scale = 800.0 / max(h, w)
    if scale < 1.0:
        page_rgb = cv2.resize(page_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
    # 轻量增强：CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    angles = [0, 90, 180, 270]
    scores = []
    for ang in angles:
        if ang == 0:
            bw_rot = bw
        elif ang == 90:
            bw_rot = cv2.rotate(bw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif ang == 180:
            bw_rot = cv2.rotate(bw, cv2.ROTATE_180)
        else:
            bw_rot = cv2.rotate(bw, cv2.ROTATE_90_CLOCKWISE)
        scores.append(_score_orientation(bw_rot))
    best_idx = int(np.argmax(scores))
    return angles[best_idx], float(scores[best_idx])


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """按 0/90/180/270 旋转 RGB/灰度图。"""
    if img is None or img.size == 0 or angle % 360 == 0:
        return img
    ang = angle % 360
    if ang == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if ang == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if ang == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img
