"""批量评测不同“前置增强”方案对 rembg 分割 mask 的影响。

不修改主流程，只输出：源图副本（一次）、各 preset 的 mask 及评分 JSON、总览 CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from rembg import remove, new_session

from docscan import io_utils
from docscan.mask_utils import score_paper_mask


# --------- 预设增强函数（保持轻量、易对比） ---------


def _resize_max_side(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    if max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 1.0:
        return img, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _remove_shadow(img: np.ndarray, dilate_kernel: int, median_ksize: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel, dilate_kernel))
    out = []
    for ch in cv2.split(img):
        dilated = cv2.dilate(ch, k)
        bg = cv2.medianBlur(dilated, median_ksize)
        diff = 255 - cv2.absdiff(ch, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        out.append(norm.astype("uint8"))
    return cv2.merge(out)


def preset_none(img: np.ndarray) -> np.ndarray:
    return img


def preset_light(img: np.ndarray) -> np.ndarray:
    # 轻度去阴影 + 轻度 CLAHE
    x = _remove_shadow(img, dilate_kernel=7, median_ksize=21)
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preset_strong(img: np.ndarray) -> np.ndarray:
    # 略强：更大核去阴影 + CLAHE
    x = _remove_shadow(img, dilate_kernel=11, median_ksize=33)
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(10, 10))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


PRESETS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": preset_none,
    "light": preset_light,
    "strong": preset_strong,
}


# --------- mask 后处理：开闭运算 + 最大连通域 + 可选凸包 ---------


def postprocess_mask(mask: np.ndarray, use_hull: bool = False) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    mask = (mask > 127).astype("uint8") * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        areas = stats[1:, 4]
        main_idx = 1 + int(np.argmax(areas))
        mask = np.zeros_like(mask, dtype="uint8")
        mask[labels == main_idx] = 255
    if use_hull:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            hull = cv2.convexHull(np.vstack(cnts))
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [hull], -1, 255, -1)
    return mask


# --------- 运行单张，遍历预设 × 模型 × matting 组合 ---------


def _run_single_combo(
    img: np.ndarray,
    h0: int,
    w0: int,
    preset: str,
    model: str,
    matting: bool,
    base_size: int,
    session,
    max_side: int,
    out_dir: Path,
    image_stem: str,
    image_suffix: str,
) -> Dict[str, object]:
    fn = PRESETS[preset]
    img_proc = fn(img.copy())
    resized, scale = _resize_max_side(img_proc, max_side)
    mask = remove(
        resized,
        session=session,
        only_mask=True,
        alpha_matting=matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
        alpha_matting_base_size=base_size,
    )
    mask_u8 = (mask * 255).astype("uint8") if mask.dtype != np.uint8 else mask
    if scale < 1.0:
        mask_u8 = cv2.resize(mask_u8, (w0, h0), interpolation=cv2.INTER_NEAREST)
    mask_u8 = postprocess_mask(mask_u8, use_hull=False)
    score, detail = score_paper_mask(mask_u8, (h0, w0), qa_cfg={})
    info = {
        "image": image_stem + image_suffix,
        "preset": preset,
        "model": model,
        "matting": matting,
        "base_size": base_size,
        "score": score,
        "area_ratio": detail.get("area_ratio"),
        "rect_ratio": detail.get("rect_ratio"),
        "aspect": detail.get("aspect"),
        "center_dist": detail.get("center_dist"),
        "bbox": detail.get("bbox"),
        "scale": scale,
        "mask_shape": list(mask_u8.shape),
    }
    combo_name = f"{preset}_{model}_{'mat' if matting else 'nomat'}_bs{base_size}"
    out_mask = out_dir
    out_mask.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_mask / f"{image_stem}_{combo_name}_mask.png"), mask_u8)
    out_json = out_mask / f"{image_stem}_{combo_name}_info.json"
    out_json.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def run_one(image_path: Path, out_dir: Path, max_side: int, sessions: Dict[str, object]) -> List[Dict[str, object]]:
    """按照降级策略跑：none+u2net -> none+u2netp+matting -> light+u2netp+matting，只保留各阶段输出，最终以最高分为准。"""
    img = io_utils.load_image(str(image_path))
    h0, w0 = img.shape[:2]
    src_out = out_dir / "source" / (image_path.stem + image_path.suffix)
    src_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(src_out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    results: List[Dict[str, object]] = []
    # 阈值：score < 6 且 (面积 < 0.4 或 矩形度 < 0.7) 才降级
    def need_retry(info: Dict[str, object]) -> bool:
        return (info.get("score", 0) < 6) and (
            info.get("area_ratio", 0) < 0.4 or info.get("rect_ratio", 1) < 0.7
        )

    # 1) none + u2net
    info1 = _run_single_combo(
        img, h0, w0, "none", "u2net", False, 1000, sessions.get("u2net"), max_side, out_dir, image_path.stem, image_path.suffix
    )
    info1["stage"] = "primary"
    results.append(info1)
    if not need_retry(info1):
        return results

    # 2) none + u2netp + matting
    info2 = _run_single_combo(
        img, h0, w0, "none", "u2netp", True, 1000, sessions.get("u2netp"), max_side, out_dir, image_path.stem, image_path.suffix
    )
    info2["stage"] = "fallback1"
    results.append(info2)
    if not need_retry(info2):
        return results

    # 3) light + u2netp + matting
    info3 = _run_single_combo(
        img, h0, w0, "light", "u2netp", True, 1000, sessions.get("u2netp"), max_side, out_dir, image_path.stem, image_path.suffix
    )
    info3["stage"] = "fallback2"
    results.append(info3)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="按降级策略评测 rembg mask（none+u2net -> none+u2netp+matting -> light+u2netp+matting）")
    parser.add_argument("--input", required=True, help="输入文件或目录")
    parser.add_argument("--output", default="outputs_mask_eval", help="输出目录")
    parser.add_argument("--max-files", type=int, default=10, help="最多处理文件数")
    parser.add_argument("--max-side", type=int, default=2400, help="分割前最长边缩放")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    args = parser.parse_args()

    imgs = list(io_utils.list_images(args.input))
    if args.max_files and len(imgs) > args.max_files:
        random.seed(args.seed)
        imgs = random.sample(imgs, args.max_files)

    out_dir = Path(args.output)

    # 准备 rembg session，避免重复加载模型
    sessions = {}
    for m in ["u2net", "u2netp"]:
        try:
            sessions[m] = new_session(m)
        except Exception:
            sessions[m] = None

    all_rows: List[Dict[str, object]] = []
    for p in imgs:
        rows = run_one(Path(p), out_dir, args.max_side, sessions=sessions)
        all_rows.extend(rows)

    # 汇总 CSV（记录各阶段，便于溯源；文件名中已含 preset/model/matting，目录不再分层）
    csv_path = out_dir / "scores.csv"
    fieldnames = [
        "image",
        "stage",
        "preset",
        "model",
        "matting",
        "base_size",
        "score",
        "area_ratio",
        "rect_ratio",
        "aspect",
        "center_dist",
        "scale",
        "mask_shape",
        "bbox",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


if __name__ == "__main__":
    main()
