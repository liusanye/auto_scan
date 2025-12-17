# 开发记录 / 记忆总览

> 所有人接手前必须先读本文件与 `README.md`。确保一屏掌握环境、当前能力、风险与下一步。

## 接手速览（实时维护）
- 环境：根目录已有 `.venv`，进入后执行 `source .venv/bin/activate` 且 `export PYTHONPATH=.`。依赖按 `requirements.txt` 安装，Apple Silicon 建议显式安装 `onnxruntime-silicon` 与 `numpy<2.0`。建议 `OMP_NUM_THREADS=1`。
- 版本与分支：当前分支 `feature/preproc-module`，HEAD `548efb7`，工作区干净。
- 运行常用命令：  
  - 单文件/目录：`PYTHONPATH=. .venv/bin/python cli.py --input <path> --output outputs --mode quality --debug-level bbox`  
  - 批量：`PYTHONPATH=. .venv/bin/python scripts/run_batch.py --input source_images --output outputs --mode auto --concurrency 2 --debug-level bbox`  
  - 烟囱测试（无 OCR）：`PYTHONPATH=. .venv/bin/python scripts/smoke_test.py --input source_images --output outputs_smoke --mode fast --max-files 2 --debug`
- 当前能力：仅输出扫描风格灰度/二值图（无 OCR）。分割多策略兜底，单/双页拆分，透视回退+轻量曲率微调，粉框精修，扫描风格增强。`debug-level=bbox` 仅输出关键调试图，`full` 输出全部中间件。
- 输出位置与命名：每个输入对应 `outputs/<stem>/`，调试前缀 01/02，中间件 10-14，正式输出 20/21，元数据 `run_summary.json`。
- 已知风险：贴边弱梯度可能残留背景；曲率矫正为轻量级；高分辨率大图需控制最长边；OCR 未接入主流程。

## 维护铁律
- “接手速览”必须实时更新任何影响环境/运行/能力/输出的信息。
- 阶段性进展或有效调试结果务必用 git 提交；高风险/实验性改动新建分支。
- 任何反馈、风险、异常或恢复操作，先更新本文件再执行。
- 语言与注释全部使用中文；如发现描述与代码不符，先改文档再改代码。
- 质检需要目视（可用多模态），不能只看日志；结论写明检查方法与样本。
- OCR 开发另起分支，未验证前不得混入主流程；分支合并需补全文档与测试记录。

## 当前状态
- 分割：`segment_strategy` 顺序尝试 u2net → u2netp+matting → light+u2netp+matting；面积<20%或矩形度<0.6 自动内容兜底，评分使用 `mask_utils.score_paper_mask`。
- 拆分与裁剪：`page_split` 支持单/双页判定（宽高比+投影谷值+对称度），单页裁剪可扩边，贴边弱梯度可削边；分割不足时可局部/全局内容兜底。
- 去透视与曲率：`dewarp` 透视回退（四边形失败退矩形），`gentle_curve_adjust` 做轻量二次拟合微调。
- 几何精修：`geom_refine` 粉框透视 + A4 比例微调 + deskew，覆盖率不足时退矩形或跳过透视。
- 增强：`enhance` 采用 division + CLAHE + Sauvola + 轻量锐化，输出灰度/二值。
- 预处理：历史前置预处理模块已删除，当前仅使用 rembg 内置预处理；如需新方案需另开分支验证。
- OCR：`ocr_paddle.py` 在代码中可用，但 pipeline 未调用；后续接入需新增阶段。

## TODO（按优先级粗排）
- 贴边弱梯度/背景残留场景：探索内容补全或边界削弱的改进方案，回归高风险样本。
- 集成 OCR 流程：在 pipeline 中增加 OCR 阶段（灰度/二值择优），补充输出与日志。
- 前置预处理新方案：设计可开关的多路预处理+评分择优，验证后再合并主干。
- 运行一轮烟囱测试（scripts/smoke_test.py）并更新结果，作为基准。
- 配置化检查：确认重要阈值是否仍硬编码，必要时下沉到 config。
- 大批量评审：抽样 20+ 张质量模式 + bbox 调试，目视 `02_debug_bbox.png`，记录问题与建议。

## 记录指引
- 时间顺序追加，最新放末尾；每条包含日期、动作/发现、影响、下一步。
- 不另起冗余 TODO 列表，本文件即主记忆。
- 外部依赖/警告/降级路径必须注明。

## 变更记录（重要节点）
- 2025-12-12：重置 DEVLOG，明确绿框配置化、粉框开启、dewarp 透视回退；新增 TODO 列表。
- 2025-12-13：修复裁剪/几何逻辑，开启双页判定，补 EXIF 方向与 deskew，添加 `scripts/smoke_test.py` 与 examples 说明；输出命名加前缀排序，新增 `debug_level` 与 `--dry-run`。
- 2025-12-14：确认 git 可用（提交 `f3ea31a`）；全量跑 source_images（fast+debug），记录分割回退与精修覆盖率；停止压缩包备份，统一用 git。
- 2025-12-15：提交“稳定版本 1.0”（`b076464`）：兜底裁剪用全局 mask boundingRect（不扩边），保留右侧定向扩边，精修覆盖率不足回退矩形；批处理并发与调试参数完善。
- 2025-12-17：接入统一内容兜底工具，增加边缘置信度衰减与轻量曲率微调；调试可视化改为分割 hull 绿框（覆盖率≥95% 隐藏），粉框叠加；彻底放弃 page-dewarp 依赖。
- 2025-12-17：移除 rembg 前置预处理模块，分割改为多策略执行器（u2net → u2netp+matting → light+u2netp+matting）；若面积/矩形度过低自动内容兜底；新增 `scripts/run_strategy_batch.py` 生成 exports 与 summary.csv。
- 2025-12-18：分割质量判定与自动重试落地；`select_main_region` 收紧标题长条合并条件；低覆盖样本回退矩形并跳过透视；回归 15 张样本覆盖率正常。
- 2025-12-21：彻底删除旧预处理文件与配置，策略执行器为唯一分割路径；全量回归 91 张统计策略占比，封面类兜底正常。
