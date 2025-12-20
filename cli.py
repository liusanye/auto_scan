"""CLI 入口：处理单个文件或目录。"""

import argparse
import logging

from docscan import io_utils
from docscan.pipeline import process_image_file, warmup_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    """解析参数并调用 pipeline。"""
    parser = argparse.ArgumentParser(description="文档拍照自动整理 CLI")
    parser.add_argument("--input", required=True, help="输入文件或目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--mode", default="quality", choices=["fast", "quality", "auto"], help="处理模式")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--profile", default=None, help="业务场景 profile")
    parser.add_argument("--debug", action="store_true", help="开启调试输出")
    parser.add_argument("--debug-level", choices=["none", "bbox", "full"], default=None, help="调试输出粒度，none/bbox/full")
    parser.add_argument("--dry-run", action="store_true", help="仅分割/裁剪调试，不执行去透视与增强")
    parser.add_argument("--max-pages", type=int, default=None, help="最多处理的页数")
    parser.add_argument("--warmup", action="store_true", help="预热模型，避免首帧过慢")
    parser.add_argument("--output-mode", choices=["review", "result", "debug"], default=None, help="输出模式：review/result/debug（覆盖配置）")
    parser.add_argument("--tone", choices=["bw", "gray", "both"], default=None, help="输出色调：bw/gray/both（默认读取配置）")
    args = parser.parse_args()

    inputs = io_utils.list_images(args.input)
    if not inputs:
        log.error("未找到可处理的图片：%s", args.input)
        return

    if args.warmup:
        warmup_models(config_path=args.config, mode=args.mode, profile=args.profile)

    for path in inputs:
        process_image_file(
            image_path=path,
            output_root=args.output,
            mode=args.mode,
            profile=args.profile,
            config_path=args.config,
            debug=args.debug,
            debug_level=args.debug_level,
            dry_run=args.dry_run,
            max_pages=args.max_pages,
            output_mode=args.output_mode,
            tone=args.tone,
        )


if __name__ == "__main__":
    main()
