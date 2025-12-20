"""拆分页模块封装：单/双页判定与裁剪，可按开关选择执行或跳过。"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from docscan.page_split import split_single_and_double_pages


def split_pages(
    image: np.ndarray,
    page_regions,
    conf: Dict[str, Any],
    stage_times: Dict[str, float],
) -> List[dict]:
    """封装拆分页逻辑，便于开关控制。"""
    if not conf.get("split", {}).get("enable_split", True):
        # 跳过拆分：直接输出整页，避免误裁剪
        h, w = image.shape[:2]
        mask_region = page_regions[0][0] if page_regions else None
        pages = []
        pages.append(
            {
                "image": image,
                "mask": mask_region,
                "bbox": (0, 0, w - 1, h - 1),
                "info": {"skipped_split": True},
            }
        )
        return pages

    return split_single_and_double_pages(
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
