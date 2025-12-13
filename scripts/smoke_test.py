"""简单烟囱测试：验证裁剪/几何/增强流程是否跑通（当前阶段无 OCR）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docscan import io_utils
from docscan.pipeline import process_image_file


def _check_outputs(pages: list[dict]) -> int:
    """检查每页的关键输出是否存在，返回失败页数。"""
    failed = 0
    for page in pages:
        if page.get("error"):
            failed += 1
            continue
        for key in ("enhanced_gray_path", "enhanced_bw_path"):
            path_str = page.get(key)
            if not path_str or not Path(path_str).exists():
                failed += 1
                break
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="烟囱测试（跳过 OCR）")
    parser.add_argument("--input", default="source_images", help="输入目录（默认使用现有样例源）")
    parser.add_argument("--output", default="outputs_smoke", help="输出目录")
    parser.add_argument("--mode", default="fast", choices=["fast", "quality", "auto"], help="处理模式（无 OCR）")
    parser.add_argument("--max-files", type=int, default=3, help="最多处理的图片数量")
    parser.add_argument("--debug", action="store_true", help="是否输出调试图")
    args = parser.parse_args()

    inputs = list(io_utils.list_images(args.input))
    if args.max_files:
        inputs = inputs[: args.max_files]
    if not inputs:
        print(f"未找到可处理的图片：{args.input}", file=sys.stderr)
        sys.exit(1)

    total_pages = 0
    failed_pages = 0
    for path in inputs:
        pages = process_image_file(
            image_path=path,
            output_root=args.output,
            mode=args.mode,
            profile=None,
            config_path=None,
            debug=args.debug,
            max_pages=None,
        )
        total_pages += len(pages)
        failed_pages += _check_outputs(pages)
        print(f"[OK] {path} -> {len(pages)} 页")

    if failed_pages > 0:
        print(f"烟囱测试完成，但 {failed_pages}/{total_pages} 页输出缺失或报错", file=sys.stderr)
        sys.exit(1)
    print(f"烟囱测试完成：处理 {len(inputs)} 个文件，共 {total_pages} 页，输出目录 {args.output}")


if __name__ == "__main__":
    main()
