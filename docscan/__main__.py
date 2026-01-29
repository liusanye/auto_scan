"""Entry point for auto-scan package."""

from __future__ import annotations

import argparse
import logging
import sys


def run_cli():
    """Run the CLI mode."""
    from docscan import io_utils
    from docscan.pipeline import process_image_file, warmup_models
    
    parser = argparse.ArgumentParser(description="Document photo automatic scanning CLI")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mode", default="quality", choices=["fast", "quality", "auto"], help="Processing mode")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--profile", default=None, help="Business profile")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--debug-level", choices=["none", "bbox", "full"], default=None, help="Debug level")
    parser.add_argument("--dry-run", action="store_true", help="Segmentation preview only")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to process")
    parser.add_argument("--warmup", action="store_true", help="Warmup models")
    parser.add_argument("--output-mode", choices=["review", "result", "debug"], default=None, help="Output mode")
    parser.add_argument("--tone", choices=["bw", "gray", "both"], default=None, help="Output tone")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    
    inputs = io_utils.list_images(args.input)
    if not inputs:
        logging.error("No images found: %s", args.input)
        sys.exit(1)
    
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


def run_mcp():
    """Run the MCP server mode."""
    from docscan.mcp_server import main as mcp_main
    mcp_main()


def main():
    """Main entry point that dispatches to CLI or MCP."""
    if len(sys.argv) > 1 and sys.argv[1] == "--mcp":
        sys.argv.pop(1)  # Remove --mcp flag
        run_mcp()
    else:
        run_cli()


if __name__ == "__main__":
    main()
