"""配置集中管理模块：默认参数 + 模式 + 场景 profile 合并。"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "mode": "quality",  # fast | quality | auto
    "profile": None,  # 业务场景，例如 invoice/purchase/inventory
    "paths": {
        "model_cache": ".cache/models",
        "debug_dir": "debug",
    },
    "limits": {
        "max_side": 2400,  # 输入最长边
        "min_side": 480,  # 输入最小边
        "rembg_max_side": 1600,
        "ocr_max_side": 2400,
    },
    "threads": {
        "omp_num_threads": 1,
    },
    "segment": {
        "min_area_ratio": 0.01,
        "morph_kernel": 5,
        "small_merge_ratio": 0.25,  # 小片段相对最大面积比例，低于则合并
        "max_merge_gap_ratio": 0.12,  # 小块合并距离阈值（相对长边），超出则不并入主体
        "clean": {  # 分割后清理参数（清理翻页/贴边突起）
            "min_fill_ratio": 0.25,  # 主连通域占图比例低于该值则不做清理以避免误伤
            "top_width_frac": 0.9,
            "top_max_height_frac": 0.2,
            "top_max_area_frac": 0.12,
            "top_min_run": 3,
            "bottom_width_frac": 0.9,
            "bottom_max_height_frac": 0.2,
            "bottom_max_area_frac": 0.12,
            "bottom_min_run": 3,
        },
    },
    "split": {
        "enable_split": False,  # 默认跳过拆分页，按单页整页处理
        "enable_double": True,  # 如启用拆分，则可按双页判定
        "double_ratio_threshold": 1.6,
        "valley_prominence": 0.08,
        "symmetry_tolerance": 0.25,
        "cut_range": [0.4, 0.6],  # 双页切割线位置下限/上限（相对宽度）
        "force_whole_page": False,  # 单页裁剪按 mask 外接框，必要时可强制整图
        "single_min_height_ratio": 0.9,
        "force_top_padding_ratio": 0.02,
        "right_min_expand_ratio": 0.12,
    },
    "dewarp": {
        "enable_polyline": True,
        "enabled": True,
        "enable_curve_adjust": True,  # 轻量曲率微调
        "curve_max_shift_px": 6,
    },
    "geom": {
        "a4_ratio": 1.414,
        "a4_tolerance": 0.10,
        "deskew_max_angle": 5.0,
        "border_px": 12,
        "enable_refine": True,  # 开启粉框透视（绿框裁剪基础上做拟合）
        "crop_expand_ratio": 0.07,  # 基础扩展比例（加大留白）
        "crop_expand_extra": {
            "top": 0.15,     # 额外向上扩展比例（适中保护标题）
            "bottom": 0.05,  # 额外向下扩展比例
            "lr": 0.10,      # 额外左右扩展比例（保护竖排标题）
        },
        "max_expand_ratio": 0.10,  # 扩边上限（相对长边），防止撑满整幅
        "shape_filter": {
            "min_fill": 0.10,  # mask 填充率下限（放宽，避免过度收紧）
            "min_ratio": 0.50,  # 长宽比下限（适度放宽兼容窄幅长表）
            "max_ratio": 2.0,  # 长宽比上限（放宽，兼容轻微扭曲）
            "center_tolerance": 0.35,  # 框中心偏移容忍（相对短边）
        },
    },
    "enhance": {
        "profile": "quality",  # fast/quality/auto
        "bw_method": "sauvola",  # sauvola | wolf
        "clahe_clip": 2.0,
        "clahe_grid": 8,
        "sauvola_window": 31,
        "sauvola_k": 0.2,
        "wolf_k": 0.2,
        "bw_pre_smooth_ksize": 0,  # 回到默认：阈值前不平滑
        "unsharp_amount": 0.6,
        "unsharp_ksize": 5,
        "division_blur": 31,
        "ocr_alt_profile": None,  # 默认关闭额外 OCR 档
        "profiles": {},
        "denoise": {
            "enable": False,
            "bilateral_d": 0,
            "bilateral_sigma_color": 0,
            "bilateral_sigma_space": 0,
            "post_morph_open": False,
            "post_morph_ksize": 3,
            "post_min_area": 0,
        },
        "orientation": {
            "adjust_output": True,  # 是否对最终输出做方向旋正（生成副本）
            "save_raw_bw": True,    # 是否保留未旋转的 bw 原图
            "score_threshold": 1.2, # 简易打分阈值，低于不旋转
        },
    },
    "output": {
        "save_jpeg": True,
        "jpeg_quality": 95,
        "preview_max_side": 800,  # 预览图最长边
    },
    "run": {
        "debug": False,
        "debug_level": "none",  # none | bbox | full
        "max_pages": None,
        "segment_preview_side": 2000,  # 分割用的预览分辨率上限（长边），减小耗时
    },
    "segment_strategy": {
        # 默认三路策略：u2net → u2netp+matting → light+u2netp+matting
        "strategies": [
            {"name": "u2net", "model": "u2net", "alpha_matting": False, "preproc": "none"},
            {"name": "u2netp_mat", "model": "u2netp", "alpha_matting": True, "preproc": "none"},
            {"name": "light_u2netp_mat", "model": "u2netp", "alpha_matting": True, "preproc": "light"},
        ],
    },
    "segment_retry": {
        "enable": True,  # 当分割结果过小/形状不符纸张时自动重试
        "max_trials": 3,  # 总尝试次数（含第一次）
        "condition": {  # 判定需重试的阈值
            "score_min": 6.0,
            "area_ratio_min": 0.4,
            "rect_ratio_min": 0.7,
        },
        "quality": {
            "min_area_ratio": 0.15,  # 掩码面积占比阈值
            "min_rect_ratio": 0.5,  # 矩形度下限
            "min_size_ratio": 0.3,  # 外接框宽/高需各≥图像 30%
            "center_dist_max": 0.4,  # 重心距中心（相对短边）上限
            "ratio_range": [0.6, 1.8],  # 长宽比允许范围
        },
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str | None = None, mode: str = "quality", profile: str | None = None) -> Dict[str, Any]:
    """加载配置，合并默认 + YAML + 模式 + 场景 profile。"""
    cfg = copy.deepcopy(DEFAULTS)
    if config_path:
        path = Path(config_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f) or {}
                cfg = _deep_update(cfg, yaml_cfg)
    cfg["mode"] = mode or cfg.get("mode", "quality")
    cfg["profile"] = profile or cfg.get("profile")
    # 模式驱动的轻量参数映射
    mode_lower = (cfg.get("mode") or "quality").lower()
    if mode_lower == "fast":
        cfg["dewarp"]["enabled"] = False
        cfg["enhance"]["profile"] = "fast"
    elif mode_lower == "quality":
        cfg["dewarp"]["enabled"] = True
        cfg["enhance"]["profile"] = "quality"
    else:  # auto 默认启用 dewarp，后续可按尺寸自适应
        cfg["dewarp"]["enabled"] = True
        cfg["enhance"]["profile"] = "auto"
    # 环境线程数：仅在未预先设置时写入，避免覆盖外部配置
    omp_threads = cfg.get("threads", {}).get("omp_num_threads")
    if omp_threads and not os.environ.get("OMP_NUM_THREADS"):
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    return cfg
