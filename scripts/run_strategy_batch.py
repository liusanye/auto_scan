"""批量跑分割策略：随机抽样、保存调试图并生成 CSV。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys
import shutil

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docscan import io_utils
from docscan.pipeline import process_image_file


def rename_debug_files(out_dir: Path, best_name: str) -> None:
    """
    只保留三个文件并重命名：
    01_<stem>_<strategy>_mask.png
    02_<stem>_<strategy>_bbox.png
    03_<stem>_<strategy>_scan_bw.png
    同时复制到 exports 便于总览。
    """
    stem = out_dir.name
    mask_src = out_dir / "01_debug_mask.png"
    bbox_src = out_dir / "02_debug_bbox.png"
    # 黑白稿文件名可能是 14_<stem>_bw.png（pipeline 生成），尝试匹配任意 *_bw.png
    bw_candidates = list(out_dir.glob("*_bw.png"))
    scan_bw_src = bw_candidates[0] if bw_candidates else None

    mask_dst = out_dir / f"01_{stem}_{best_name}_mask.png"
    bbox_dst = out_dir / f"02_{stem}_{best_name}_bbox.png"
    scan_bw_dst = out_dir / f"03_{stem}_{best_name}_scan_bw.png"

    export_dir = out_dir.parent / "exports"
    export_dir.mkdir(exist_ok=True)

    kept = []
    if mask_src.exists():
        mask_src.rename(mask_dst)
        shutil.copy(mask_dst, export_dir / mask_dst.name)
        kept.append(mask_dst.name)
    if bbox_src.exists():
        bbox_src.rename(bbox_dst)
        shutil.copy(bbox_dst, export_dir / bbox_dst.name)
        kept.append(bbox_dst.name)
    if scan_bw_src and scan_bw_src.exists():
        scan_bw_src.rename(scan_bw_dst)
        shutil.copy(scan_bw_dst, export_dir / scan_bw_dst.name)
        kept.append(scan_bw_dst.name)

    # 清理目录中其他文件（保留 summary.csv 由上层生成，不在子目录）
    for p in out_dir.iterdir():
        if p.name in kept:
            continue
        if p.is_file():
            p.unlink()


def run_batch(inputs, output_root: Path, mode: str, debug_level: str) -> list[dict]:
    rows: list[dict] = []
    for img_path in inputs:
        process_image_file(
            image_path=str(img_path),
            output_root=str(output_root),
            mode=mode,
            debug=True,
            debug_level=debug_level,
        )
        stem = Path(img_path).stem
        run_dir = output_root / stem
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as f:
            summary = json.load(f)
        seg = summary.get("segment", {})
        best_name = seg.get("name") or seg.get("model") or "unknown"
        rename_debug_files(run_dir, best_name)
        rows.append(
            {
                "file": stem,
                "best_name": best_name,
                "model": seg.get("model"),
                "preproc": seg.get("preproc"),
                "alpha_matting": seg.get("alpha_matting"),
                "score": seg.get("score"),
                "area_ratio": seg.get("quality", {}).get("area_ratio"),
                "rect_ratio": seg.get("quality", {}).get("rect_ratio"),
                "coverage": seg.get("coverage"),
                "fallback": seg.get("fallback"),
                "attempts": len(seg.get("attempts", [])),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="批量跑分割策略并生成 CSV")
    parser.add_argument("--input", required=True, help="输入文件或目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--num", type=int, default=20, help="随机抽样数量")
    parser.add_argument("--seed", type=int, default=17, help="随机种子")
    parser.add_argument("--mode", default="quality", help="pipeline 模式")
    parser.add_argument("--debug-level", default="bbox", choices=["none", "bbox", "full"], help="调试输出粒度")
    args = parser.parse_args()

    inputs = list(io_utils.list_images(args.input))
    if not inputs:
        print("未找到图片")
        return
    random.seed(args.seed)
    random.shuffle(inputs)
    inputs = inputs[: args.num]

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = run_batch(inputs, out_root, args.mode, args.debug_level)

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "best_name",
                "model",
                "preproc",
                "alpha_matting",
                "score",
                "area_ratio",
                "rect_ratio",
                "coverage",
                "fallback",
                "attempts",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"完成 {len(rows)} 张，结果写入 {csv_path}")


if __name__ == "__main__":
    main()
