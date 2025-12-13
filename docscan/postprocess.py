"""OCR 后处理与字段映射。"""

from __future__ import annotations

from typing import Any, Dict


def _digit_fix(text: str) -> str:
    rep = {"O": "0", "o": "0", "I": "1", "l": "1", "B": "8"}
    return "".join(rep.get(ch, ch) for ch in text)


def normalize_table(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    简单表头归一化与数字纠错。
    """
    if not ocr_result or "raw" not in ocr_result:
        return {}
    raw = ocr_result["raw"]
    # 占位：直接返回，附带数字纠错示例
    for item in raw:
        if "res" in item:
            for cell in item["res"]:
                text = cell.get("text", "")
                cell["text"] = _digit_fix(text)
    return {"normalized": raw}
