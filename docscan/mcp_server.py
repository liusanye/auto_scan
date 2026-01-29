"""MCP Server for auto-scan document processing."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from docscan import config as cfg
from docscan.pipeline import process_image_file, warmup_models
from docscan import io_utils

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "fastmcp is required for MCP server mode. "
        "Install with: pip install fastmcp"
    )

# Initialize MCP server
mcp = FastMCP("auto-scan")
log = logging.getLogger(__name__)

# Lazy loading flag
_MODELS_LOADED = False

def _ensure_models_loaded():
    """延迟加载模型，避免启动时长时间阻塞。"""
    global _MODELS_LOADED
    if not _MODELS_LOADED:
        log.info("正在加载 AI 模型（首次使用）...")
        try:
            warmup_models(config_path=None, mode="quality", profile=None)
            _MODELS_LOADED = True
            log.info("模型加载完成")
        except Exception as e:
            log.warning(f"模型预热失败（非致命）: {e}")


@mcp.tool()
def scan_document(
    input_path: str,
    output_path: str | None = None,
    mode: str = "quality",
    tone: str = "bw",
    debug: bool = False,
) -> dict[str, Any]:
    """
    Scan a document photo and convert it to a clean scanned image.

    Args:
        input_path: Path to the input image file
        output_path: Path for the output image (optional, defaults to docscan_outputs/)
        mode: Processing mode - "fast", "quality", or "auto"
        tone: Output tone - "bw" (black & white), "gray", or "both"
        debug: If True, output debug files (mask, bbox overlay) for inspection

    Returns:
        Dictionary with success status, output paths, and processing info
    """
    # 延迟加载模型（首次调用时才加载）
    _ensure_models_loaded()

    input_file = Path(input_path)
    if not input_file.exists():
        return {
            "success": False,
            "error": f"Input file not found: {input_path}",
        }

    # Determine output directory
    if output_path:
        out_path = Path(output_path)
        # If output_path is a directory, use it
        if out_path.suffix == "":
            out_path.mkdir(parents=True, exist_ok=True)
            output_root = str(out_path)
        else:
            output_root = str(out_path.parent)
    else:
        # Default: create docscan_outputs in current working directory
        output_root = str(Path.cwd() / "docscan_outputs")
        Path(output_root).mkdir(parents=True, exist_ok=True)

    try:
        results = process_image_file(
            image_path=str(input_file),
            output_root=output_root,
            mode=mode,
            tone=tone,
            output_mode="debug" if debug else "result",
        )
        
        # Extract output file paths from results
        output_files = []
        for result in results:
            if "outputs" in result:
                for output in result["outputs"]:
                    if "path" in output:
                        output_files.append(output["path"])
        
        return {
            "success": True,
            "input": str(input_file),
            "outputs": output_files,
            "pages_processed": len(results),
        }
        
    except Exception as e:
        log.exception("Failed to process document")
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def scan_directory(
    input_dir: str,
    output_dir: str | None = None,
    mode: str = "quality",
    tone: str = "bw",
    debug: bool = False,
) -> dict[str, Any]:
    """
    Scan all document photos in a directory.

    Args:
        input_dir: Path to directory containing images
        output_dir: Path for output images (optional, defaults to docscan_outputs/)
        mode: Processing mode - "fast", "quality", or "auto"
        tone: Output tone - "bw" (black & white), "gray", or "both"
        debug: If True, output debug files (mask, bbox overlay) for inspection

    Returns:
        Dictionary with success status and processing summary
    """
    # 延迟加载模型
    _ensure_models_loaded()

    input_path = Path(input_dir)
    if not input_path.exists():
        return {
            "success": False,
            "error": f"Input directory not found: {input_dir}",
        }

    if output_dir:
        out_path = Path(output_dir)
    else:
        # Default: create docscan_outputs in current working directory
        out_path = Path.cwd() / "docscan_outputs"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    images = io_utils.list_images(str(input_path))
    if not images:
        return {
            "success": False,
            "error": f"No images found in {input_dir}",
        }
    
    results_summary = []
    errors = []
    
    for img_path in images:
        try:
            result = scan_document(
                input_path=img_path,
                output_path=str(out_path),
                mode=mode,
                tone=tone,
                debug=debug,
            )
            results_summary.append({
                "file": Path(img_path).name,
                "success": result["success"],
            })
            if not result["success"]:
                errors.append(f"{Path(img_path).name}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            errors.append(f"{Path(img_path).name}: {str(e)}")
            results_summary.append({
                "file": Path(img_path).name,
                "success": False,
            })
    
    success_count = sum(1 for r in results_summary if r["success"])
    
    return {
        "success": len(errors) == 0,
        "total": len(images),
        "successful": success_count,
        "failed": len(errors),
        "output_dir": str(out_path),
        "results": results_summary,
        "errors": errors if errors else None,
    }


@mcp.tool()
def convert_to_pdf(
    input_paths: list[str],
    output_path: str | None = None,
    combine: bool = False,
) -> dict[str, Any]:
    """
    Convert scanned images to PDF format.

    Args:
        input_paths: List of image file paths to convert
        output_path: Output PDF path (optional, defaults to docscan_outputs/)
        combine: If True, combine all images into a single PDF; if False, create separate PDFs

    Returns:
        Dictionary with success status and output PDF paths
    """
    if not _PIL_AVAILABLE:
        return {
            "success": False,
            "error": "PIL (Pillow) is required for PDF conversion. Install with: pip install Pillow",
        }

    # Validate input files
    valid_inputs = []
    for path in input_paths:
        p = Path(path)
        if p.exists():
            valid_inputs.append(p)
        else:
            return {
                "success": False,
                "error": f"Input file not found: {path}",
            }

    if not valid_inputs:
        return {
            "success": False,
            "error": "No valid input files provided",
        }

    # Determine output directory
    if output_path:
        out_path = Path(output_path)
        if out_path.suffix == "":
            out_path.mkdir(parents=True, exist_ok=True)
            output_dir = out_path
            output_file = None
        else:
            output_dir = out_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_path
    else:
        output_dir = Path.cwd() / "docscan_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = None

    output_pdfs = []

    try:
        if combine:
            # Combine all images into one PDF
            if output_file:
                pdf_path = output_file
            else:
                pdf_path = output_dir / "combined.pdf"

            images = []
            for img_path in valid_inputs:
                img = Image.open(img_path)
                # Convert to RGB if necessary (for PNG with transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                images.append(img)

            # Save first image, append others
            images[0].save(
                pdf_path,
                "PDF",
                resolution=100.0,
                save_all=True,
                append_images=images[1:]
            )
            output_pdfs.append(str(pdf_path))

        else:
            # Create separate PDFs for each image
            for img_path in valid_inputs:
                img = Image.open(img_path)
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')

                pdf_name = img_path.stem + ".pdf"
                pdf_path = output_dir / pdf_name
                img.save(pdf_path, "PDF", resolution=100.0)
                output_pdfs.append(str(pdf_path))

        return {
            "success": True,
            "input_count": len(valid_inputs),
            "output_pdfs": output_pdfs,
            "combined": combine,
        }

    except Exception as e:
        log.exception("Failed to convert to PDF")
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def warmup() -> dict[str, Any]:
    """
    Warm up the AI models to reduce first-run latency.
    Call this after starting the server for better performance.
    """
    try:
        warmup_models(config_path=None, mode="quality", profile=None)
        return {"success": True, "message": "Models warmed up successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    """Start the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
