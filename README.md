# 文档拍照自动整理系统（设计与落地说明）

> 接手前请先阅读根目录 `DEVLOG.md` 的“接手速览”，再按其中说明激活 `.venv` 与 `PYTHONPATH=.` 后运行。

本项目目标：在 macOS（MacBook Air M3，CPU）上，把手机拍摄的文档照片自动转成“扫描风格图片 + Excel/结构化数据”，仅使用预训练模型，不训练新模型。当前完成了方案设计，代码实现与脚本将在此方案基础上展开。

## 环境与依赖

- Python 3.10+（推荐 3.11）。
- 建议开启虚拟环境：
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- 推荐安装（后续可按 requirements.txt）：
  ```bash
  # Apple Silicon 用 onnxruntime-silicon 提升兼容性
  pip install onnxruntime-silicon
  pip install rembg opencv-python-headless scikit-image tqdm pyyaml Pillow 'numpy<2.0' 'onnxruntime-silicon>=1.16,<1.19'
  # page-dewarp 如安装失败可暂时跳过，流程有回退
  pip install page-dewarp || true
  ```
- 如后续需要启用 OCR，再额外安装 `paddlepaddle` 与 `paddleocr`；当前阶段无需。
- 环境变量建议：`OMP_NUM_THREADS=1`，避免 onnxruntime 占满核。

## 目录结构（规划）

```
docscan/
  docscan/               # 核心模块（segment/dewarp/geom_refine/enhance/pipeline）
  scripts/run_batch.py   # 批量处理与并发控制
  cli.py                 # CLI 入口
  examples/              # 样例输入
  requirements.txt
  README.md
  planCodex版.md         # 详细实现与改进方案（主文档）
```

## 模式与关键策略

- 模式：`fast`（跳过 dewarp，轻量增强），`quality`（全流程），`auto`（按分辨率/核数自适应）。
- 分辨率保护：入口最长边建议 2048–2560，超限等比压缩。
- 回退链：dewarp→透视→原图；segment 失败→全图裁剪+留白。
- 调试与观测：debug 输出 mask/投影曲线/四边形/deskew 角度/矫正前后对比；run-summary 记录耗时、置信度、降级、输出路径。
- 预热：CLI 提供 `--warmup` 对分割空跑，避免首帧极慢。
- 实现现状与裁剪逻辑：绿框来自分割 mask 的外接框，扩边比例由配置 `geom.crop_expand_ratio/crop_expand_extra` 决定；粉框由几何精修拟合四边形，最终生成的 scan_gray/scan_bw 以粉框（或矩形回退）透视结果为准，绿框仅作为初裁范围。当前 dewarp 为透视回退方案，page-dewarp/曲率拟合尚未集成。

## 运行示例（待代码落地后）

```bash
python cli.py --input input_dir_or_file --output output_dir --mode quality --config config.yaml --debug
```

批量：
```bash
python scripts/run_batch.py --input input_dir --output output_dir --mode auto --concurrency 2
```

调试：
- `--debug` 时，每个源文件的输出目录会包含调试图：`debug_mask.png`/`debug_bbox.png`（整体分割掩膜与 bbox）、每页的 `page_xxx_raw/dewarp/refine/gray/bw`，便于对照流程与结果。

快速烟囱测试（当前阶段不含 OCR，验证裁剪/几何/增强链路）：
```bash
python scripts/smoke_test.py --input source_images --output outputs_smoke --mode fast --max-files 2
```

## Smoke Test（规划）

- 在 `examples/` 放 3–5 张样例（单页、双页、强光差、旋转 ~10°）。
- 提供 smoke test 脚本：运行 pipeline 处理样例，检查输出文件存在、日志无 ERROR；验证 dewarp 回退路径。

## 性能预估（M3 CPU 参考）

- fast 模式：单张几秒级（取决于分辨率），内存 <1GB。
- quality 模式：dewarp 占主要时间；建议保持输入最长边 ≤2560。

## 当前状态与下一步

- 已完成方案整合与落地策略：见 `planCodex版.md`。
- 当前阶段仅输出扫描风格图，OCR 功能暂不启用，后续作为扩展再开启。
- 下一步：按方案搭建代码框架、配置文件、CLI/run_batch、smoke test，并锁定已验证依赖版本。
