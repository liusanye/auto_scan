"""流程上下文与结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PipelineContext:
    """处理级上下文：模式/配置/调试/输出目录与耗时。"""

    image_path: str
    output_root: Path
    mode: str
    profile: Optional[str]
    config: Dict[str, Any]
    debug_level: str
    debug_enabled: bool
    max_pages: Optional[int] = None
    dry_run: bool = False
    effective_mode: Optional[str] = None
    stage_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class PageResult:
    """单页结果：包含坐标、输出路径、耗时与降级信息。"""

    page_index: int
    page_id: str
    bbox: Tuple[int, int, int, int]
    split_info: Dict[str, Any] | None
    segment_fallback: bool
    segment_fallback_reason: str | None
    edge_mask: Dict[str, Any] | None = None
    enhanced_gray_path: Optional[str] = None
    enhanced_bw_path: Optional[str] = None
    dewarp: Optional[Dict[str, Any]] = None
    curve_adjust: Optional[Dict[str, Any]] = None
    refine: Optional[Dict[str, Any]] = None
    refine_quad_global: Optional[List[List[int]]] = None
    refine_quad_backprojected: bool = False
    error: Optional[str] = None
    elapsed: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    dry_run: bool = False
