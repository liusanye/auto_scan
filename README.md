# 文档拍照自动整理系统

> 目标：将手机拍摄的文档照片转换为“扫描风格灰度/二值图”，仅使用预训练模型（无训练），在 macOS/Apple Silicon CPU 环境即可运行。  
> 入口请先阅读根目录 `DEVLOG.md` 的“接手速览”，按其中步骤激活 `.venv` 并设置 `PYTHONPATH=.`。

## 快速开始
> 当前重构阶段 1-4 已完成：分割策略/重试阈值配置化、pipeline 拆分与调试/summary 封装收敛；阶段5（贴边弱梯度专项回归）暂缓。
> 最新补充：enhance 支持可选 Wolf-Jolion 阈值（`enhance.bw_method=wolf`）；输出配置新增 JPEG/预览开关（`output.save_jpeg`/`output.preview_max_side`，默认关闭保持 PNG 高保真）。

### 环境准备
- Python 3.10+（推荐 3.11）。
- 建议使用虚拟环境：
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  export PYTHONPATH=.
  ```
- 安装依赖（Apple Silicon 建议 silicon 轮子）：
  ```bash
  pip install -r requirements.txt
  # 如遇 numpy/onnxruntime 版本冲突，可显式：
  pip install 'onnxruntime-silicon>=1.16,<1.19' 'numpy<2.0'
  ```
- 环境变量建议：`OMP_NUM_THREADS=1`，避免 onnxruntime 占满所有核。

### 运行命令
- 单文件/目录：
  ```bash
  PYTHONPATH=. .venv/bin/python cli.py --input <文件或目录> --output outputs --mode quality --debug-level bbox
  # 仅分割/裁剪预览：--dry-run
  # 关闭调试：--debug-level none
  # 预热分割模型：--warmup
  ```
- 批量（默认线程池，进程池权限受限会自动回退）：
  ```bash
  PYTHONPATH=. .venv/bin/python scripts/run_batch.py --input source_images --output outputs --mode auto --concurrency 2 --debug-level bbox
  ```
- 分割策略对比与导出（仅保留 01/02/03 调试图 + summary.csv）：
  ```bash
  PYTHONPATH=. .venv/bin/python scripts/run_strategy_batch.py --input source_images --output outputs_strategy --num 20 --mode quality --debug-level bbox
  ```
- 烟囱测试（当前不含 OCR）：
  ```bash
  PYTHONPATH=. .venv/bin/python scripts/smoke_test.py --input source_images --output outputs_smoke --mode fast --max-files 2 --debug
  ```
- 分割预设评测（对比不同预处理/模型组合的 mask 评分，仅落盘 mask/JSON/CSV）：
  ```bash
  PYTHONPATH=. .venv/bin/python scripts/eval_mask_presets.py --input source_images --output outputs_mask_eval --max-files 10
  ```

## 项目结构
```
docscan/
  cli.py                    # CLI 入口
  docscan/
    config.py               # 配置默认值与模式映射
    segment.py              # rembg 分割 + 清理 + 小块合并
    segment_strategy.py     # 多路分割策略执行与评分择优
    page_split.py           # 单/双页判定、裁剪与扩边
    dewarp.py               # 透视回退 + 轻量曲率微调
    geom_refine.py          # 几何精修（粉框透视/A4 微调/deskew）
    enhance.py              # 扫描风格增强（division + CLAHE + Sauvola + 锐化）
    mask_utils.py           # 内容兜底、纸张评分、边缘削弱
    io_utils.py             # 读写与 EXIF 方向矫正
    ocr_paddle.py           # OCR 封装（当前流程未接入）
    postprocess.py          # OCR 后处理占位
    pipeline.py             # 主流程串联
  scripts/
    run_batch.py            # 批量/并发入口
    run_strategy_batch.py   # 分割策略回归与导出
    smoke_test.py           # 烟囱测试（无 OCR）
    review_bbox.py          # bbox 自动评审
  examples/README.md        # 样例说明（无真实图片）
  requirements.txt
  planCodex版.md            # 方案与改进思路
  DEVLOG.md                 # 接手速览与变更记录
```

## 流程与技术路线
- **输入保护**：入口最长边限制（默认 2400），最小边不足时适度放大，避免分割失败。
- **分割**（`segment_strategy` → `segment`）：按 u2net → u2netp+matting → light+u2netp+matting 顺序尝试，基于 `score_paper_mask` 评分择优；若面积 <20% 或矩形度 <0.6 自动内容兜底（再不行用整图）。
- **单/双页拆分**（`page_split`）：宽高比 + 水平投影谷值 + 对称度判定双页；单页按 mask 外接框并配置化扩边，贴边弱梯度可削边；分割偏小时可局部/全局内容兜底。
- **去透视/曲率**（`dewarp` + `gentle_curve_adjust`）：默认透视回退（四边形失败则矩形），可选轻量曲率二次拟合 remap。
- **几何精修**（`geom_refine`）：粉框拟合 + A4 比例微调 + deskew；覆盖率不足时退矩形或跳过透视。
- **增强**（`enhance`）：division normalization → CLAHE → Sauvola → 轻量锐化，输出灰度与二值版。
- **输出与调试**：按前缀 01/02/10/11/12/13/14/20/21 命名；`run_summary.json` 记录模式、耗时、降级、分割尝试细节、输出路径。`debug-level=bbox` 仅输出 01/02/20/21，`full` 额外输出中间件。
- **OCR**：`ocr_paddle.py` 可选封装，但当前 pipeline 未调用；后续集成需要在 pipeline 增加阶段。

### 模式与自适应
- `mode=fast`：跳过 dewarp，增强减弱。
- `mode=quality`：全流程。
- `mode=auto`：按最长边判定：≤1800 走 fast（关闭 dewarp、轻量增强）；1800–2600 走 quality；>2600 仍走 quality 但保持高分辨率限制。

## 功能与限制
- **已实现**：多策略分割兜底、单/双页裁剪、透视回退+曲率微调、粉框精修、扫描风格增强、批处理与调试导出。
- **未实现/待集成**：OCR 与字段归一化未在主流程启用；前置预处理模块已移除，当前仅使用 rembg 内置预处理。
- **已知风险**：贴边弱梯度场景可能残留背景；曲率矫正为轻量级；高分辨率大图需确保最长边限制以控时。

## 输出文件说明（单页示例）
- `01_debug_mask.png`：分割/兜底整体 mask。
- `02_debug_bbox.png`：原图叠加绿框（分割外扩 hull）+ 粉框（透视四边形）+ 图例/降级提示。
- `10_*_raw.png`：绿框裁剪页（debug full）。
- `11_*_dewarp.png`：去透视/曲率后（debug full）。
- `12_*_refine.png`：粉框透视结果（debug full）。
- `13_*_gray.png` / `14_*_bw.png`：调试灰度/二值（与正式输出对齐，debug full）。
- `20_*_scan_gray.png` / `21_*_scan_bw.png`：正式输出。
- `run_summary.json`：处理元数据（模式/耗时/降级/分割尝试/输出路径等）。

## 性能与调试建议（M3 CPU 参考）
- 模式：`fast` 跳过 dewarp，适合拍得较正；`quality` 全流程；`auto` 按分辨率自适应。
- 分割预览：`segment_preview_side=2000`（默认）可明显降低耗时。
- 并发：`run_batch` 推荐线程池，并发≈核数/2；process 若受权限限制会自动回退。
- 预热：首帧较慢时使用 `--warmup`（分割空跑一次）。
- 输出体积：`debug-level full` 会生成大量中间图，批量时可改用 `bbox`。
- 输出压缩（可选）：`config.output` 支持开启 JPEG 输出（`save_jpeg`/`jpeg_quality`）与预览图（`preview_max_side`），默认关闭；需要最小化体积时再启用。

## 下一步方向
- 集成 OCR（调用 `ocr_paddle.py`），补充表格/结构化输出与后处理（建议独立分支推进）。
- 迭代贴边弱梯度场景的 mask 改善（内容补全或边界削弱）。
- 探索前置预处理自动打分/多路择优（当前已移除，需新方案验证）。

## 维护与协作规则
- 分支策略：功能/实验统一用 `feature/<name>` 或 `exp/<name>`，高风险/大改动务必独立分支验证后再合入；热修可用 `hotfix/<name>`。
- 提交与记录：任何影响运行、配置、输出的改动，需同步更新 `DEVLOG.md`（接手速览/变更记录）与本文件的相关段落；提交信息用中文简洁描述“做了什么+为什么”。
- 测试约定：最少执行一次烟囱测试（`scripts/smoke_test.py`）验证裁剪/几何/增强链路；涉及分割策略调整时跑 `run_strategy_batch.py` 抽样输出 `summary.csv`；批量并发改动需注明耗时与并发设置。
- 配置与阈值：新增/修改阈值应优先配置化（`config.py`），避免硬编码；默认参数需兼顾质量与性能。
- 调试输出：日常调试使用 `--debug-level bbox`，全量中间件仅在排查或回归时启用；批量运行请避免 `full` 以控制 I/O。
- OCR 开发：当前主流程不含 OCR，后续应在独立分支接入并补充输出/日志/配置，再评估合入主干。
