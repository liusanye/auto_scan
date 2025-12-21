"""分割后的单页处理：曲率/精修/增强，不负责落盘。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from docscan.dewarp import dewarp_page, gentle_curve_adjust
from docscan.geom_refine import refine_geometry_with_opencv
from docscan.enhance import enhance_scan_style
from docscan import mask_utils
from docscan.debug_utils import clamp_quad_to_bbox, mask_bbox_with_margin
from docscan import orientation


@dataclass
class ProcessOutputs:
    gray: np.ndarray
    bw: np.ndarray
    dewarp_info: Dict[str, Any] | None
    curve_info: Dict[str, Any] | None
    refine_info: Dict[str, Any] | None
    refine_quad_global: List[List[int]] | None
    refine_quad_backprojected: bool
    edge_info: Dict[str, Any] | None
    elapsed: float
    stage_times: Dict[str, float] = field(default_factory=dict)
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
    overlay_bbox: tuple[int, int, int, int] | None = None
    orientation_angle: int = 0
    orientation_score: float = 0.0


def _backproject_quad(refine_quad_local, dewarp_info, bbox):
    if refine_quad_local and isinstance(refine_quad_local, list):
        bx0, by0, _, _ = bbox
        quad_local_np = np.array(refine_quad_local, dtype=np.float32)
        dewarp_matrix = dewarp_info.get("matrix") if dewarp_info else None
        if dewarp_matrix is not None:
            try:
                M = np.array(dewarp_matrix, dtype=np.float32)
                M_inv = np.linalg.inv(M)
                quad_orig = cv2.perspectiveTransform(quad_local_np.reshape(1, -1, 2), M_inv)[0]
                refine_quad_global = [[int(pt[0] + bx0), int(pt[1] + by0)] for pt in quad_orig]
                return refine_quad_global, True
            except Exception:  # noqa: BLE001
                pass
        quad_src = dewarp_info.get("quad") if dewarp_info else None
        quad_src = quad_src or refine_quad_local
        quad_src_np = np.array(quad_src, dtype=np.float32)
        refine_quad_global = [[int(pt[0] + bx0), int(pt[1] + by0)] for pt in quad_src_np]
        return refine_quad_global, False
    return None, False


def process_page_image(page: dict, conf: dict, debug_level: str) -> ProcessOutputs:
    """
    处理单页：曲率、精修、增强。
    输入 page 包含 image/mask/bbox/info。
    返回处理结果（未落盘）。
    """
    page_img = page["image"]
    page_mask = page.get("mask")
    bbox = page.get("bbox")
    debug_images: Dict[str, np.ndarray] = {}

    edge_info = None
    if page_mask is not None:
        new_mask, edge_info = mask_utils.attenuate_mask_edges(page_img, page_mask)
        page_mask = new_mask

    overlay_bbox: tuple[int, int, int, int] | None = None
    overlay_mask_cov: float | None = None
    if page_mask is not None:
        overlay_mask_cov = float(cv2.countNonZero(page_mask)) / float(max(1, page_mask.size))
        if overlay_mask_cov < 0.95:
            local_bbox = mask_bbox_with_margin(page_mask)
            if local_bbox is not None and bbox is not None:
                bx0, by0, _, _ = bbox
                lx0, ly0, lx1, ly1 = local_bbox
                overlay_bbox = (bx0 + lx0, by0 + ly0, bx0 + lx1, by0 + ly1)
    if overlay_bbox is None and bbox is not None:
        if overlay_mask_cov is None or overlay_mask_cov < 0.95:
            overlay_bbox = bbox

    stage_local: Dict[str, float] = {}
    t_page = time.time()
    if debug_level == "full":
        debug_images["raw"] = page_img

    dewarp_enabled = conf["dewarp"].get("enabled", True) and not conf["split"].get("force_whole_page", False)
    refine_enabled = conf["geom"].get("enable_refine", True)

    t_dewarp = time.time()
    if dewarp_enabled:
        dewarped, dewarped_mask, dewarp_info = dewarp_page(
            page_img,
            page_mask=page_mask,
            use_polyline=conf["dewarp"]["enable_polyline"],
        )
    else:
        dewarped = page_img
        dewarped_mask = page_mask
        dewarp_info = {"method": "skip_disabled"}
    stage_local["dewarp"] = time.time() - t_dewarp

    curve_info = None
    t_curve = time.time()
    if conf["dewarp"].get("enable_curve_adjust", True) and dewarped_mask is not None:
        dewarped, dewarped_mask, curve_info = gentle_curve_adjust(
            dewarped,
            dewarped_mask,
            max_shift_px=int(conf["dewarp"].get("curve_max_shift_px", 6)),
        )
    stage_local["curve_adjust"] = time.time() - t_curve

    t_refine = time.time()
    if refine_enabled:
        page_refined, refine_info = refine_geometry_with_opencv(
            dewarped,
            page_mask=dewarped_mask,
            border_px=conf["geom"]["border_px"],
            shape_filter=conf["geom"].get("shape_filter"),
            deskew_max_angle=conf["geom"].get("deskew_max_angle", 5.0),
            a4_ratio=conf["geom"].get("a4_ratio", 1.414),
            a4_tolerance=conf["geom"].get("a4_tolerance", 0.10),
        )
    else:
        page_refined = dewarped
        refine_info = {"method": "refine_skip", "reason": "disabled", "quad": None}
    stage_local["refine"] = time.time() - t_refine

    t_enh = time.time()
    gray, bw = enhance_scan_style(page_refined, enhance_cfg=conf.get("enhance"))
    stage_local["enhance"] = time.time() - t_enh

    # 方向检测与矫正：基于裁剪页进行 0/90/180/270 判定，高置信度才旋转
    orient_angle = 0
    orient_score = 0.0
    if conf.get("run", {}).get("enable_orientation", True):
        orient_angle, orient_score = orientation.detect_orientation(page_refined)
        # 简单阈值：分数足够且非 0 才旋转
        if orient_angle in (90, 180, 270) and orient_score > 1.2:
            gray = orientation.rotate_image(gray, orient_angle)
            bw = orientation.rotate_image(bw, orient_angle)
            if debug_level == "full":
                if "dewarp" in debug_images:
                    debug_images["dewarp"] = orientation.rotate_image(debug_images["dewarp"], orient_angle)
                if "refine" in debug_images:
                    debug_images["refine"] = orientation.rotate_image(debug_images["refine"], orient_angle)
                if "gray" in debug_images:
                    debug_images["gray"] = orientation.rotate_image(debug_images["gray"], orient_angle)
                if "bw" in debug_images:
                    debug_images["bw"] = orientation.rotate_image(debug_images["bw"], orient_angle)
                if "raw" in debug_images:
                    debug_images["raw"] = orientation.rotate_image(debug_images["raw"], orient_angle)

    if debug_level == "full":
        debug_images["dewarp"] = dewarped
        debug_images["refine"] = page_refined
        debug_images["gray"] = gray
        debug_images["bw"] = bw

    refine_quad_local = refine_info.get("quad") if refine_enabled else None
    refine_quad_global, refine_quad_backprojected = _backproject_quad(refine_quad_local, dewarp_info, bbox)
    refine_quad_global = clamp_quad_to_bbox(refine_quad_global, overlay_bbox or bbox)

    return ProcessOutputs(
        gray=gray,
        bw=bw,
        dewarp_info=dewarp_info,
        curve_info=curve_info,
        refine_info=refine_info,
        refine_quad_global=refine_quad_global,
        refine_quad_backprojected=refine_quad_backprojected,
        edge_info=edge_info,
        elapsed=time.time() - t_page,
        stage_times=stage_local,
        debug_images=debug_images,
        overlay_bbox=overlay_bbox,
        orientation_angle=orient_angle,
        orientation_score=orient_score,
    )
