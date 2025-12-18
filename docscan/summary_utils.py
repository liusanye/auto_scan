"""summary 构建与 JSON 序列化工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def json_default(obj):
    """兼容 numpy 标量的 JSON 序列化。"""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def save_summary(summary: Dict[str, Any], path: Path) -> None:
    """保存 run_summary.json，保证目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def build_summary(
    file_path: str,
    mode: str,
    effective_mode: str,
    profile: str | None,
    debug_level: str,
    dry_run: bool,
    image_shape_raw: tuple[int, int],
    image_shape_processed: tuple[int, int],
    preproc: Dict[str, Any],
    segment: Dict[str, Any],
    pages: list[Dict[str, Any]],
    elapsed_total: float,
    stage_times: Dict[str, Any],
    debug_overlay_path: str | None = None,
) -> Dict[str, Any]:
    """构建 run_summary 字典，集中管理字段。"""
    summary = {
        "file": file_path,
        "mode": mode,
        "mode_effective": effective_mode,
        "profile": profile,
        "debug_level": debug_level,
        "dry_run": dry_run,
        "image_shape_raw": list(image_shape_raw),
        "image_shape_processed": list(image_shape_processed),
        "preproc": preproc,
        "segment": segment,
        "pages": pages,
        "elapsed_total": elapsed_total,
        "stage_times": stage_times,
    }
    if debug_overlay_path:
        summary["debug_overlay"] = debug_overlay_path
    return summary
