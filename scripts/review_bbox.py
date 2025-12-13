"""自动评审 bbox（基于 run_summary.json 与原始图片）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image


def _load_image_size(path: Path) -> tuple[int, int]:
    img = Image.open(path)
    return img.height, img.width


def _calc_metrics(img_h: int, img_w: int, bbox: List[int]) -> Dict[str, float]:
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    area_ratio = (bw * bh) / float(img_h * img_w)
    height_ratio = bh / float(img_h)
    width_ratio = bw / float(img_w)
    top_margin = y0 / float(img_h)
    bottom_margin = (img_h - y1) / float(img_h)
    left_margin = x0 / float(img_w)
    right_margin = (img_w - x1) / float(img_w)
    cx, cy = x0 + bw / 2.0, y0 + bh / 2.0
    off = ((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2) ** 0.5
    center_offset_norm = off / float(min(img_h, img_w))
    aspect_ratio = max(bw, bh) / float(min(bw, bh))
    return {
        "area_ratio": area_ratio,
        "height_ratio": height_ratio,
        "width_ratio": width_ratio,
        "top_margin": top_margin,
        "bottom_margin": bottom_margin,
        "left_margin": left_margin,
        "right_margin": right_margin,
        "center_offset_norm": center_offset_norm,
        "aspect_ratio": aspect_ratio,
    }


def _flag(metrics: Dict[str, float]) -> List[str]:
    flags: List[str] = []
    if metrics["height_ratio"] < 0.9:
        flags.append("short_height")
    if metrics["area_ratio"] < 0.8:
        flags.append("small_area")
    if metrics["top_margin"] > 0.05:
        flags.append("top_gap")
    if metrics["right_margin"] > 0.05:
        flags.append("right_gap")
    if metrics["bottom_margin"] > 0.05:
        flags.append("bottom_gap")
    if metrics["center_offset_norm"] > 0.2:
        flags.append("off_center")
    if metrics["aspect_ratio"] > 2.0 or metrics["aspect_ratio"] < 0.5:
        flags.append("shape_weird")
    return flags


def _bbox_from_quad(quad: List[List[float]] | None) -> List[int] | None:
    if not quad:
        return None
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def review_outputs(outputs_root: Path, report_path: Path) -> Dict[str, int]:
    records: List[Dict[str, object]] = []
    counts = {"pages": 0, "flagged": 0}
    for summary_file in outputs_root.glob("*/run_summary.json"):
        data = json.loads(summary_file.read_text(encoding="utf-8"))
        # 使用 debug_bbox.png 的尺寸作为评审基准（与 bbox 坐标一致），若不存在则回退原图
        overlay_path = summary_file.parent / "debug_bbox.png"
        if overlay_path.exists():
            img_h, img_w = _load_image_size(overlay_path)
        else:
            img_path = Path(data["file"])
            if not img_path.is_absolute():
                img_path = Path.cwd() / img_path
            if not img_path.exists():
                continue
            img_h, img_w = _load_image_size(img_path)
        for page in data.get("pages", []):
            quad = page.get("refine_quad_global") or page.get("refine_quad")
            bbox = _bbox_from_quad(quad) or page.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            metrics = _calc_metrics(img_h, img_w, bbox)
            flags = _flag(metrics)
            counts["pages"] += 1
            if flags:
                counts["flagged"] += 1
            records.append(
                {
                    "file": data.get("file"),
                    "page_id": page.get("page_id"),
                    "bbox": bbox,
                    "metrics": metrics,
                    "flags": flags,
                }
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"counts": counts, "records": records}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="评审 bbox，标记可能的截顶/偏移问题")
    parser.add_argument("--outputs", default="outputs", help="输出根目录（含各文件夹/run_summary.json）")
    parser.add_argument("--report", default="outputs/bbox_auto_review.json", help="评审报告保存路径")
    args = parser.parse_args()

    counts = review_outputs(Path(args.outputs), Path(args.report))
    print(f"评审完成：页面数={counts['pages']}, 异常页={counts['flagged']}，报告写入 {args.report}")


if __name__ == "__main__":
    main()
