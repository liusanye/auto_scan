"""主流程：串联各模块完成单张图片处理。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from docscan import config as cfg
from docscan import io_utils
from docscan.segment import segment_pages
from docscan.page_split import split_single_and_double_pages
from docscan.dewarp import dewarp_page
from docscan.geom_refine import refine_geometry_with_opencv
from docscan.enhance import enhance_scan_style

log = logging.getLogger(__name__)


def _save_debug_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.save_image(arr, str(path))


def _content_mask_with_coverage(image: np.ndarray, margin_ratio: float = 0.02, use_adaptive: bool = True) -> tuple[np.ndarray, float, tuple[int, int, int, int] | None]:
    """基于内容生成兜底 mask，适度抑制纸边，返回覆盖率与 bbox。"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if use_adaptive:
        bin_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    else:
        _, bin_inv = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    H, W = mask.shape[:2]
    margin = int(min(H, W) * margin_ratio)
    if margin > 0:
        mask[:margin, :] = 0
        mask[H - margin :, :] = 0
        mask[:, :margin] = 0
        mask[:, W - margin :] = 0
    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros_like(mask, dtype=np.uint8), 0.0, None
    x, y, w, h = cv2.boundingRect(coords)
    coverage = (w * h) / float(max(1, H * W))
    mask_full = np.zeros_like(mask, dtype=np.uint8)
    cv2.rectangle(mask_full, (x, y), (x + w, y + h), 255, -1)
    return mask_full, coverage, (x, y, x + w, y + h)


def _tight_content_mask(image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int, int, int] | None]:
    """内容兜底：优先自适应，过大则尝试高阈值收紧，避免整页占满。"""
    mask0, cov0, bbox0 = _content_mask_with_coverage(image, margin_ratio=0.02, use_adaptive=True)
    # 若覆盖过大，尝试更高阈值收紧
    if cov0 >= 0.92:
        mask1, cov1, bbox1 = _content_mask_with_coverage(image, margin_ratio=0.03, use_adaptive=False)
        if bbox1 is not None and 0.03 < cov1 < cov0:
            return mask1, cov1, bbox1
    return mask0, cov0, bbox0


def _best_content_mask(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None, float]:
    """尝试原图与顺时针 90°，择优覆盖且避免撑满整页。"""
    mask0, cov0, bbox0 = _tight_content_mask(image)
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    mask_r, cov_r, bbox_r = _tight_content_mask(rotated)
    candidates = []
    if bbox0 is not None and cov0 > 0.02:
        candidates.append(("origin", mask0, cov0, bbox0))
    if bbox_r is not None and cov_r > 0.02:
        candidates.append(("rot90", cv2.rotate(mask_r, cv2.ROTATE_90_COUNTERCLOCKWISE), cov_r, bbox_r))
    if not candidates:
        return mask0, bbox0, cov0
    # 选择覆盖率 0.2-0.92 区间内最接近 0.7 的候选，避免全页
    def score(cov: float) -> float:
        return abs(cov - 0.7)
    filtered = [c for c in candidates if 0.2 <= c[2] <= 0.92]
    chosen = min(filtered, key=lambda x: score(x[2])) if filtered else max(candidates, key=lambda x: x[2])
    tag, mask_sel, cov_sel, bbox_sel = chosen
    if tag == "rot90":
        coords = cv2.findNonZero(mask_sel)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            bbox_sel = (x, y, x + w, y + h)
    return mask_sel, bbox_sel, cov_sel


def warmup_models(config_path: str | None, mode: str, profile: str | None) -> None:
    """预热分割，减少首帧等待。"""
    conf = cfg.load_config(config_path, mode=mode, profile=profile)
    dummy = np.ones((256, 256, 3), dtype=np.uint8) * 255
    try:
        segment_pages(
            dummy,
            max_side=conf["limits"]["rembg_max_side"],
            min_area_ratio=conf["segment"]["min_area_ratio"],
            morph_kernel=conf["segment"]["morph_kernel"],
        )
        log.info("分割模型预热完成")
    except Exception:  # noqa: BLE001
        log.exception("分割预热失败")


def process_image_file(
    image_path: str,
    output_root: str,
    mode: str = "quality",
    profile: str | None = None,
    config_path: str | None = None,
    debug: bool = False,
    max_pages: int | None = None,
) -> List[Dict[str, Any]]:
    """
    读图 → segment → split → dewarp → geom_refine → enhance。
    """
    conf = cfg.load_config(config_path, mode=mode, profile=profile)
    conf["run"]["debug"] = debug
    conf["run"]["max_pages"] = max_pages

    image = io_utils.load_image(image_path)
    # 入口尺寸保护：限制最大边并保证最小边不过小
    h_raw, w_raw = image.shape[:2]
    h, w = h_raw, w_raw
    max_side = conf["limits"]["max_side"]
    min_side = conf["limits"]["min_side"]
    scale_down = min(1.0, max_side / max(h, w))
    if scale_down < 1.0:
        image = cv2.resize(image, (int(w * scale_down), int(h * scale_down)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
    min_dim = min(h, w)
    if min_dim < min_side:
        scale_up = min_side / float(min_dim)
        cap = max_side / float(max(h, w))
        scale_up = min(scale_up, cap)
        if scale_up > 1.0:
            image = cv2.resize(image, (int(w * scale_up), int(h * scale_up)), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

    effective_mode = conf["mode"]
    if conf.get("mode") == "auto":
        longest_for_mode = max(max(h_raw, w_raw), max(h, w))
        if longest_for_mode <= 1800:
            conf["dewarp"]["enabled"] = False
            conf["enhance"]["profile"] = "fast"
            effective_mode = "auto-fast"
        elif longest_for_mode <= 2600:
            conf["dewarp"]["enabled"] = True
            conf["enhance"]["profile"] = "quality"
            effective_mode = "auto-quality"
        else:
            conf["dewarp"]["enabled"] = True
            conf["enhance"]["profile"] = "quality"
            effective_mode = "auto-quality-highres"

    image_stem = Path(image_path).stem
    out_dir = Path(output_root) / image_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    t0 = time.time()
    stage_times: Dict[str, float] = {}
    t_seg = time.time()
    page_regions = segment_pages(
        image,
        max_side=conf["limits"]["rembg_max_side"],
        min_area_ratio=conf["segment"]["min_area_ratio"],
        morph_kernel=conf["segment"]["morph_kernel"],
    )
    stage_times["segment"] = time.time() - t_seg
    segment_fallback = False
    segment_fallback_msg: str | None = None
    if not page_regions:
        mask_fb, bbox_fb, cov_fb = _best_content_mask(image)
        if bbox_fb is not None and cov_fb > 0.05:
            page_regions = [(mask_fb, bbox_fb)]
            log.warning("pipeline: segment 无结果，使用内容兜底 mask 覆盖率=%.3f bbox=%s", cov_fb, bbox_fb)
            segment_fallback = True
            segment_fallback_msg = "content_mask"
        else:
            # 分割失败兜底：整张图作为单页
            h2, w2 = image.shape[:2]
            full_mask = np.ones((h2, w2), dtype=np.uint8) * 255
            page_regions = [(full_mask, (0, 0, w2, h2))]
            log.warning("pipeline: segment 无结果，回退使用整张图")
            segment_fallback = True
            segment_fallback_msg = "full_image"
    t_split = time.time()
    pages = split_single_and_double_pages(
        image,
        page_regions,
        enable_double=conf["split"].get("enable_double", False),
        double_ratio_threshold=conf["split"]["double_ratio_threshold"],
        valley_prominence=conf["split"].get("valley_prominence", 0.08),
        symmetry_tolerance=conf["split"].get("symmetry_tolerance", 0.25),
        cut_range=tuple(conf["split"].get("cut_range", (0.4, 0.6))),
        expand_ratio=conf["geom"].get("crop_expand_ratio", 0.05),
        expand_extra=(
            conf["geom"].get("crop_expand_extra", {}).get("top", 0.12),
            conf["geom"].get("crop_expand_extra", {}).get("bottom", 0.02),
            conf["geom"].get("crop_expand_extra", {}).get("lr", 0.05),
        ),
        single_min_height_ratio=conf["split"].get("single_min_height_ratio", 0.9),
        force_top_padding_ratio=conf["split"].get("force_top_padding_ratio", 0.02),
        right_min_expand_ratio=conf["split"].get("right_min_expand_ratio", 0.10),
        force_whole_page=conf["split"].get("force_whole_page", False),
        max_expand_ratio=conf["geom"].get("max_expand_ratio"),
    )
    stage_times["split"] = time.time() - t_split
    log.info("pipeline: segmented pages=%d", len(pages))

    if conf["run"]["debug"]:
        # 覆盖保存分割 mask，避免旧文件残留
        if page_regions:
            combined_mask = np.zeros_like(page_regions[0][0])
            for mask_region, _bbox in page_regions:
                combined_mask = cv2.bitwise_or(combined_mask, mask_region)
        else:
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        _save_debug_image(combined_mask, out_dir / "01_debug_mask.png")
        overlay = image.copy()
    else:
        overlay = None

    for idx, page in enumerate(pages):
        if max_pages is not None and idx >= max_pages:
            break
        page_id = f"page_{idx+1:03d}"
        file_prefix = f"{image_stem}_{page_id}"
        stage_local: Dict[str, float] = {}
        refine_quad_local = None
        refine_quad_global = None
        page_result: Dict[str, Any] = {
            "page_index": idx,
            "page_id": page_id,
            "split": page.get("info"),
            "bbox": page.get("bbox"),
            "segment_fallback": segment_fallback,
            "segment_fallback_reason": segment_fallback_msg,
        }
        t_page = time.time()
        try:
            if conf["run"]["debug"]:
                _save_debug_image(page["image"], out_dir / f"10_{file_prefix}_raw.png")

            dewarp_enabled = conf["dewarp"].get("enabled", True) and not conf["split"].get("force_whole_page", False)
            refine_enabled = conf["geom"].get("enable_refine", True)
            t_dewarp = time.time()
            if dewarp_enabled:
                dewarped, dewarped_mask, dewarp_info = dewarp_page(
                    page["image"],
                    page_mask=page.get("mask"),
                    use_polyline=conf["dewarp"]["enable_polyline"],
                )
            else:
                dewarped = page["image"]
                dewarped_mask = page.get("mask")
                dewarp_info = {"method": "skip_disabled"}
            stage_local["dewarp"] = time.time() - t_dewarp
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
            gray_path = out_dir / f"20_{file_prefix}_scan_gray.png"
            bw_path = out_dir / f"21_{file_prefix}_scan_bw.png"
            io_utils.save_image(gray, gray_path)
            io_utils.save_image(bw, bw_path)
            stage_local["enhance"] = time.time() - t_enh

            if conf["run"]["debug"]:
                _save_debug_image(dewarped, out_dir / f"11_{file_prefix}_dewarp.png")
                _save_debug_image(page_refined, out_dir / f"12_{file_prefix}_refine.png")
                _save_debug_image(gray, out_dir / f"13_{file_prefix}_gray.png")
                _save_debug_image(bw, out_dir / f"14_{file_prefix}_bw.png")

            refine_quad_local = refine_info.get("quad") if refine_enabled else None
            if refine_quad_local and isinstance(refine_quad_local, list):
                bx0, by0, _, _ = page["bbox"]
                refine_quad_global = [[int(pt[0] + bx0), int(pt[1] + by0)] for pt in refine_quad_local]
            page_result.update(
                {
                    "enhanced_gray_path": str(gray_path),
                    "enhanced_bw_path": str(bw_path),
                    "dewarp": dewarp_info,
                    "refine": refine_info,
                    "refine_quad": refine_quad_local,
                    "refine_quad_global": refine_quad_global,
                }
            )

            page_result["elapsed"] = time.time() - t_page
            page_result["stage_times"] = stage_local
            results.append(page_result)
        except Exception as e:  # noqa: BLE001
            log.exception("处理页面失败 %s #%s", image_path, page_id)
            page_result["error"] = str(e)
            page_result["stage_times"] = stage_local
            results.append(page_result)

        # 绘制调试 overlay：裁剪 bbox + refine 四边形
        if conf["run"]["debug"] and overlay is not None:
            line_thickness = 5
            bx0, by0, bx1, by1 = page["bbox"]
            cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 200, 0), line_thickness)
            if refine_quad_global:
                quad_np = np.array(refine_quad_global, dtype=int)
                cv2.polylines(overlay, [quad_np], isClosed=True, color=(255, 0, 255), thickness=line_thickness)
                for idx_c, (cx, cy) in enumerate(quad_np, start=1):
                    cv2.circle(overlay, (int(cx), int(cy)), 10, (50, 50, 255), -1)
                    cv2.putText(
                        overlay,
                        f"P{idx+1}-{idx_c}",
                        (int(cx) + 12, int(cy) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        overlay,
                        f"P{idx+1}-{idx_c}",
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
                "Green=crop area  Pink=refine quad (actual warp)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                "Green=crop area  Pink=refine quad (actual warp)",
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
            if refine_quad_global:
                quad_np = np.array(refine_quad_global, dtype=int)
                cv2.polylines(overlay, [quad_np], isClosed=True, color=(255, 0, 255), thickness=line_thickness)

    if conf["run"]["debug"] and overlay is not None:
        _save_debug_image(overlay, out_dir / "02_debug_bbox.png")

    summary = {
        "file": image_path,
        "mode": mode,
        "mode_effective": effective_mode,
        "profile": profile,
        "pages": results,
        "elapsed_total": time.time() - t0,
        "stage_times": stage_times,
    }
    summary_path = out_dir / "run_summary.json"
    try:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        log.exception("写入 run_summary 失败 %s", summary_path)

    return results
