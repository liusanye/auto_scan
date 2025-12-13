"""批量处理入口，控制并发与失败重试。"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from docscan import io_utils
from docscan.pipeline import process_image_file, warmup_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    """解析参数并批量调用 pipeline。"""
    parser = argparse.ArgumentParser(description="批量处理文档照片")
    parser.add_argument("--input", required=True, help="输入目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--mode", default="auto", choices=["fast", "quality", "auto"], help="处理模式")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--profile", default=None, help="业务场景 profile")
    parser.add_argument("--concurrency", type=int, default=1, help="并发度（默认保守串行）")
    parser.add_argument("--debug", action="store_true", help="开启调试输出")
    parser.add_argument("--warmup", action="store_true", help="预热模型")
    args = parser.parse_args()

    inputs = io_utils.list_images(args.input)
    if not inputs:
        log.error("未找到可处理的图片：%s", args.input)
        return
    if args.warmup:
        warmup_models(config_path=args.config, mode=args.mode, profile=args.profile)

    concurrency = max(1, args.concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(
                process_image_file,
                image_path=path,
                output_root=args.output,
                mode=args.mode,
                profile=args.profile,
                config_path=args.config,
                debug=args.debug,
                max_pages=None,
            ): path
            for path in inputs
        }
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                fut.result()
                log.info("完成：%s", path)
            except Exception:  # noqa: BLE001
                log.exception("处理失败：%s", path)


if __name__ == "__main__":
    main()
