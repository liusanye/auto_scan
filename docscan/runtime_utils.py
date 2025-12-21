"""运行时辅助工具：配置加载、模式判定与图片缩放。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from docscan import config as cfg
from docscan import io_utils


def _apply_output_mode(conf: dict, output_mode: str | None) -> None:
    """根据输出模式调整 debug 与输出选项。"""
    if output_mode:
        conf.setdefault("output", {})["mode"] = output_mode
    mode = (conf.get("output") or {}).get("mode")
    if not mode:
        return
    mode = str(mode).lower()
    out_cfg = conf.setdefault("output", {})
    run_cfg = conf.setdefault("run", {})
    if mode == "debug":
        run_cfg["debug"] = True
        run_cfg["debug_level"] = "full"
        out_cfg.setdefault("tone", "both")
        out_cfg["save_jpeg"] = True
        out_cfg["preview_max_side"] = int(out_cfg.get("preview_max_side") or 800)
    elif mode == "review":
        run_cfg["debug"] = True
        run_cfg["debug_level"] = "bbox"
        out_cfg.setdefault("tone", "bw")
        out_cfg["save_jpeg"] = False
        out_cfg["preview_max_side"] = 0
    elif mode == "result":
        run_cfg["debug"] = False
        run_cfg["debug_level"] = "none"
        out_cfg.setdefault("tone", "bw")
        out_cfg["save_jpeg"] = False
        out_cfg["preview_max_side"] = 0


def build_runtime_config(
    mode: str,
    profile: str | None,
    config_path: str | None,
    debug: bool,
    debug_level: str | None,
    dry_run: bool,
    max_pages: int | None,
    output_mode: str | None = None,
    tone: str | None = None,
) -> tuple[dict, str, bool]:
    """
    加载配置并应用 output/debug/dry_run/max_pages 开关，返回 (conf, debug_level, debug_enabled)。
    """
    conf = cfg.load_config(config_path, mode=mode, profile=profile)
    _apply_output_mode(conf, output_mode)
    if debug_level:
        conf["run"]["debug_level"] = debug_level
        conf["run"]["debug"] = debug_level != "none"
    elif debug:
        conf["run"]["debug"] = True
        conf["run"]["debug_level"] = conf["run"].get("debug_level") or "full"
    else:
        conf["run"]["debug"] = conf["run"].get("debug", False)
        conf["run"]["debug_level"] = conf["run"].get("debug_level", "none")
    conf["run"]["max_pages"] = max_pages
    conf["run"]["dry_run"] = dry_run
    if tone:
        conf.setdefault("output", {})["tone"] = tone.lower()
    debug_level_eff = (conf["run"].get("debug_level") or "none").lower()
    debug_enabled = debug_level_eff != "none"
    return conf, debug_level_eff, debug_enabled


def prepare_image_and_mode(image_path: str, conf: dict) -> Tuple[np.ndarray, int, int, int, int, str]:
    """
    读取图片并按配置约束尺寸，返回 (image, h_raw, w_raw, h, w, effective_mode)。
    保持原有尺度与 auto 模式逻辑不变。
    """
    image = io_utils.load_image(image_path)
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
    return image, h_raw, w_raw, h, w, effective_mode
