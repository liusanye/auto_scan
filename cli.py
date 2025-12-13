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
    parser.add_argument("--max-pages", type=int, default=None, help="最多处理的页数")
    parser.add_argument("--warmup", action="store_true", help="预热模型，避免首帧过慢")
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
            max_pages=args.max_pages,
        )


if __name__ == "__main__":
    main()
