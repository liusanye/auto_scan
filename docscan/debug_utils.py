"""调试与可视化工具：统一绘制与保存调试图。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from docscan import io_utils


def save_debug_image(arr: np.ndarray, path: Path) -> None:
    """保存调试图，保证目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.save_image(arr, str(path))


def mask_bbox_with_margin(
    mask: np.ndarray, margin_ratio: float = 0.01, min_margin: int = 6
) -> Tuple[int, int, int, int] | None:
    """根据 mask 计算外接矩形并外扩，供调试/兜底区域使用。"""
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = mask.shape[:2]
    margin = max(min_margin, int(max(h, w) * margin_ratio))
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(w - 1, x1 + margin),
        min(h - 1, y1 + margin),
    )


def clamp_quad_to_bbox(
    quad: List[List[float]] | None, bbox: Tuple[int, int, int, int] | None
) -> List[List[int]] | None:
    """将四边形坐标限制在 bbox 范围内，避免越界。"""
    if quad is None or bbox is None:
        return quad
    bx0, by0, bx1, by1 = bbox
    clamped: List[List[int]] = []
    for x, y in quad:
        cx = min(max(int(round(x)), bx0), bx1)
        cy = min(max(int(round(y)), by0), by1)
        clamped.append([cx, cy])
    return clamped


def draw_overlay(
    overlay: np.ndarray,
    bbox: Tuple[int, int, int, int] | None,
    refine_quad_global: List[List[int]] | None,
    segment_fallback: bool,
    segment_fallback_msg: str | None,
    refine_quad_backprojected: bool,
) -> None:
    """在原图上绘制裁剪框/四边形与提示信息。"""
    line_thickness = 5
    if bbox is not None:
        bx0, by0, bx1, by1 = bbox
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 200, 0), line_thickness)
    if refine_quad_global:
        quad_np = np.array(refine_quad_global, dtype=int)
        cv2.polylines(overlay, [quad_np], isClosed=True, color=(255, 0, 255), thickness=line_thickness)
        for idx_c, (cx, cy) in enumerate(quad_np, start=1):
            cv2.circle(overlay, (int(cx), int(cy)), 10, (50, 50, 255), -1)
            cv2.putText(
                overlay,
                f"P1-{idx_c}",
                (int(cx) + 12, int(cy) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"P1-{idx_c}",
                (int(cx) + 12, int(cy) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
    # 图例
    cv2.putText(
        overlay,
        "Green=segment area (mask hull)  Pink=refine quad (actual warp)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "Green=segment area (mask hull)  Pink=refine quad (actual warp)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if segment_fallback:
        cv2.putText(
            overlay,
            f"segment fallback ({segment_fallback_msg or 'auto'})",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"segment fallback ({segment_fallback_msg or 'auto'})",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if refine_quad_global and not refine_quad_backprojected:
        cv2.putText(
            overlay,
            "refine quad 未反投影(仅供参考)",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "refine quad 未反投影(仅供参考)",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def prepare_overlay(image: np.ndarray, combined_mask: np.ndarray | None) -> np.ndarray | None:
    """生成用于绘制的 overlay 图，debug 关闭时返回 None。"""
    if image is None or combined_mask is None:
        return None
    overlay = image.copy()
    if combined_mask is not None:
        hull = cv2.convexHull(cv2.findNonZero(combined_mask)) if cv2.countNonZero(combined_mask) > 0 else None
        if hull is not None:
            # 使用蓝色描边 mask 凸包，避免与裁剪框绿色混淆
            cv2.polylines(overlay, [hull], isClosed=True, color=(70, 130, 180), thickness=5)
    return overlay


def save_overlay(overlay: np.ndarray | None, out_path: Path) -> None:
    """保存 overlay 调试图。"""
    if overlay is None:
        return
    save_debug_image(overlay, out_path)
