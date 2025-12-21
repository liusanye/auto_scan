# 可视化调试与成果使用教程

> 目标：把分割与裁剪从“黑箱”变成“可验证的过程”，让你能直观看到每一步的成败、原因与选择。

## 1. 什么时候用它
- **质量审阅（默认）**：只需要确认 mask 与 bbox 是否合理，以及最终黑白图是否可用。
- **问题定位**：出现“裁剪偏、漏页、背景残留、二值发灰/断字”等问题时，必须打开全量调试。
- **成果交付**：只需要最终 `bw` 图，不关心中间过程时用成果模式。

## 2. 如何开启
- **质量审阅模式（默认）**：
  ```bash
  PYTHONPATH=. .venv311/bin/python cli.py --input <path> --output outputs --mode quality --output-mode review
  ```
- **可视化调试（全量）**：
  ```bash
  PYTHONPATH=. .venv311/bin/python cli.py --input <path> --output outputs --mode quality --output-mode debug
  # 或者：--debug-level full
  ```
- **成果模式（只要最终图）**：
  ```bash
  PYTHONPATH=. .venv311/bin/python cli.py --input <path> --output outputs --mode quality --output-mode result
  ```

## 3. 你会看到哪些输出（按价值排序）
以下以 `outputs/<stem>/` 为例：

- **成果图**
  - `21_*_scan_bw.png`：最终黑白二值图（默认成果）
  - `20_*_scan_gray.png`：灰度成果图（仅 `--tone gray`/`both`）
- **质量审阅（核心）**
  - `01_debug_mask.png`：分割出来的整页 mask
  - `02_debug_bbox.png`：绿框=裁剪 bbox，粉框=透视精修四边形，蓝色=mask 凸包
- **可视化调试（全量）**
  - `attempts/attempts.json`：每次策略尝试的评分与关键指标
  - `attempts/*_combined.png`：每次策略输出的合并 mask
  - `attempts/*_main.png`：每次策略选中的主体 mask
  - `10/11/12/13/14_*`：原图/透视/精修/灰度/黑白中间图（仅 `debug-level=full`）
- **元数据**
  - `run_summary.json`：记录模式、降级、分割评分、每页输出路径与耗时

## 4. 看什么、怎么判断（最关心的部分）

### 4.1 分割是否可靠（看 `01_debug_mask.png`）
- **合格**：mask 覆盖完整纸张，四边连续，背景基本不被吃进来。
- **常见问题**：
  - **漏边/漏角**：纸张边缘没被罩住，后续裁剪会缺角。
  - **吃背景**：桌面/地面被罩住，导致裁剪过大。

### 4.2 裁剪与精修是否准确（看 `02_debug_bbox.png`）
- **合格**：绿框贴合纸张外缘；粉框（精修四边形）与纸张边缘一致。
- **常见问题**：
  - 绿框偏离：分割质量不足或回退到内容兜底。
  - 粉框扭曲：透视精修识别失败，可能只剩矩形兜底。

### 4.3 策略为什么成功或失败（看 `attempts/attempts.json`）
关键字段：
- `score`：评分越高越好；主流程用它选“最佳策略”。
- `area_ratio`：mask 面积占比，低于 0.4 很可能触发重试。
- `rect_ratio`：矩形度，低于 0.7 说明 mask 很不规则。
- `need_retry`：是否判定为失败并需要尝试下一策略。
- `is_best_attempt`：本次尝试在策略列表中得分最高。

**如何读这些数据：**
- 若所有尝试 `area_ratio` 都偏低 → 原图对比度差或纸张边缘不清晰。
- 若 `rect_ratio` 低 → mask 形状破碎，可能有遮挡/阴影。
- 若只有带 `matting` 的策略成功 → 说明边缘软/背景复杂，matting 有价值。

### 4.4 最终成果是否可用（看 `21_*_scan_bw.png`）
- **合格**：文字清晰、黑白分明、背景干净。
- **常见问题**：
  - 字体断裂：阈值过严或局部阴影。
  - 背景发灰：阈值偏松或纸面有纹理。

## 5. 如何反馈与调整（从“结果”反推“原因”）
- **mask 完整但 bbox 偏**：优先检查 `split`/`geom_refine` 的参数或回退逻辑。
- **mask 破碎**：优先检查分割策略与预处理（`light` 是否帮助）。
- **重复触发兜底**：看 `run_summary.json` 的 `segment.fallback` 原因，说明原策略不稳定。
- **黑白图不清晰**：考虑改 `enhance` 参数或切换灰度输出再人工阈值化。

## 6. 建议的人工评审流程（质量审阅模式）
1. 看 `01_debug_mask.png`：确认是否完整覆盖纸张。
2. 看 `02_debug_bbox.png`：确认裁剪框/透视四边形是否贴合纸张。
3. 看 `21_*_scan_bw.png`：确认文字是否清晰、背景是否干净。
4. 如有异常，再切换 `output-mode=debug` 查看 `attempts` 明细。

## 7. 典型问题定位速查
- **“漏角”**：mask 不完整 → 分割失败 → 检查 `attempts` 哪次成功。
- **“切歪/裁窄”**：bbox 贴不住 → 可能是内容兜底或精修失败。
- **“字断裂/黑块”**：增强或二值化过度 → 可尝试灰度输出对比。

## 8. 给成果数据的最低交付标准
- `21_*_scan_bw.png`：清晰可读、无大面积背景。
- `run_summary.json`：必须保留，便于回溯模式/阈值/耗时。

> 如果只需要最简单输出，请使用成果模式：`--output-mode result --tone bw`。
