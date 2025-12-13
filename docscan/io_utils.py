"""文件与图像读写工具。"""

from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
from PIL import ExifTags


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """根据 EXIF 方向信息旋转图片，避免后续角度偏差。"""
    try:
        exif = img._getexif()  # noqa: SLF001
        if not exif:
            return img
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break
        if orientation_key is None or orientation_key not in exif:
            return img
        orientation = exif.get(orientation_key)
        if orientation == 3:
            return img.rotate(180, expand=True)
        if orientation == 6:
            return img.rotate(270, expand=True)
        if orientation == 8:
            return img.rotate(90, expand=True)
    except Exception:  # noqa: BLE001
        # 若读取失败则直接返回原图，保持稳健
        return img
    return img


def load_image(path: str) -> np.ndarray:
    """读取图像为 RGB numpy 数组。"""
    with Image.open(path) as img:
        img = _apply_exif_orientation(img).convert("RGB")
        return np.array(img)


def save_image(array: np.ndarray, path: str) -> None:
    """将 numpy 数组保存为图片文件，创建父目录。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def list_images(input_path: str) -> Tuple[str, ...]:
    """列出输入路径下的所有支持图片文件（简单按后缀过滤）。"""
    p = Path(input_path)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if p.is_file() and p.suffix.lower() in exts:
        return (str(p),)
    if p.is_dir():
        return tuple(str(f) for f in sorted(p.iterdir()) if f.suffix.lower() in exts)
    return ()
