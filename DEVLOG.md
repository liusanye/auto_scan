# 开发记录 / 记忆总览

> 接手前必须先读本文件与 `README.md`。本文件用于记录“当前状态 + 风险 + 变更记录”，以便快速恢复上下文。

## 接手速览（实时维护）
- 环境：主开发/运行环境为 Python 3.11（`.venv311`），进入后执行 `source .venv311/bin/activate` 且 `export PYTHONPATH=.`。依赖按 `requirements.txt` 安装，Apple Silicon 建议显式安装 `onnxruntime-silicon` 与 `numpy<2.0`。建议 `OMP_NUM_THREADS=1`。
- 版本与分支：当前分支 `feature/post-seg-processing`，HEAD `1a1b13e`（pipeline 抽取五个模块功能）。若接手时不一致，以 `git status -sb` / `git log -1` 为准。
- 运行常用命令：
  - 单文件/目录：`PYTHONPATH=. .venv/bin/python cli.py --input <path> --output outputs --mode quality --debug-level bbox`
  - 批量：`PYTHONPATH=. .venv/bin/python scripts/run_batch.py --input source_images --output outputs --mode auto --concurrency 2 --debug-level bbox`
  - 烟囱测试（无 OCR）：`PYTHONPATH=. .venv/bin/python scripts/smoke_test.py --input source_images --output outputs_smoke --mode fast --max-files 2 --debug`
- 当前能力：默认输出模式 `review`（mask + bbox + bw，可选灰度或双输出，无 OCR）；多策略分割兜底；单双页拆分逻辑可用但默认关闭；透视回退+轻量曲率微调；几何精修；扫描风格增强；方向校正（默认开启）。
- 输出位置与命名：每个输入对应 `outputs/<stem>/`，调试前缀 01/02，中间件 10–14（仅 debug-level=full），正式输出默认 21（20 为可选灰度），元数据 `run_summary.json`。
- 已知风险：贴边弱梯度可能残留背景；曲率矫正为轻量级；高分辨率大图需控制最长边；OCR 未接入主流程；部分配置项未接入（见 README“实现备注”）。

## 维护铁律
- “接手速览”必须实时更新任何影响环境/运行/能力/输出的信息。
- 阶段性进展或有效调试结果务必用 git 提交；高风险/实验性改动新建分支。
- 任何影响运行、配置、输出的改动，需同步更新 `DEVLOG.md` 与 `README.md`。
- 语言与注释全部使用中文；如发现描述与代码不符，先改文档再改代码。
- 质检需要目视（可用多模态），不能只看日志；结论写明检查方法与样本。
- OCR 开发另起分支，未验证前不得混入主流程；分支合并需补全文档与测试记录。

## 当前状态
- 分割：`segment_strategy` 顺序尝试 u2net → u2netp+matting → light+u2netp+matting；重试阈值 `score<6 且 (area<0.4 或 rect<0.7)`，最终低于 `area<0.20` 或 `rect<0.60` 触发内容兜底；评分使用 `mask_utils.score_paper_mask`。
- 拆分与裁剪：`split.enable_split=false` 默认跳过拆分页；开启后按宽高比 + 投影谷值 + 对称度判定双页，单页裁剪支持扩边/右侧最小扩边与内容兜底。
- 去透视与曲率：`dewarp` 透视回退（四边形失败退矩形），`gentle_curve_adjust` 做轻量二次拟合微调。
- 几何精修：`geom_refine` 粉框透视 + A4 比例微调 + deskew，覆盖率不足时退矩形或跳过透视。
- 增强：`enhance` 采用 division + CLAHE + Sauvola（可选 Wolf）+ 轻量锐化，输出灰度/二值。
- 方向校正：基于投影方差判定 0/90/180/270，置信度阈值 1.2，默认开启。
- 预处理：历史前置预处理模块已删除，当前仅使用 rembg 内置预处理。
- OCR：`ocr_paddle.py` 与 `postprocess.py` 已存在，但 pipeline 未调用。

## TODO（按优先级粗排）
- 集成 OCR 流程：在 pipeline 增加 OCR 阶段（灰度/二值择优），补充输出与日志。
- 贴边弱梯度/背景残留场景：探索内容补全或边界削弱的改进方案，回归高风险样本。
- 清理与接入未生效配置：`segment` 组细节参数、`run.segment_preview_side`、方向校正配置入口。
- 运行一轮烟囱测试（`scripts/smoke_test.py`）并更新结果，作为基准。
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
- 2025-12-17：提交“策略执行器稳定版 1.1”（`71c97b9`），重写 README/DEVLOG，明确维护规则、运行指令、功能现状与未接入 OCR 的说明，便于后续接手。
- 2025-12-17：重构阶段1（零行为变更）：新增上下文/调试/summary 工具模块（context/debug_utils/summary_utils），pipeline 调试与 summary 写入改用工具函数，保留原有输出与流程不变。
- 2025-12-17：重构阶段2（进行中，行为不变）：抽离图片准备/模式判定与分割步骤为独立函数（_prepare_image_and_mode/_run_segmentation），便于后续拆分 pipeline；未更改算法与阈值，输出保持一致。
- 2025-12-17：重构阶段2 追加：拆分运行配置构建与 split 调用（_build_runtime_config/_split_pages_with_stats），恢复 PageContext/_save_debug_image 定义，烟囱测试 1 张（fast+debug）通过，输出目录 outputs_smoke_tmp。
- 2025-12-17：重构阶段2 进一步：抽出单页处理函数 `_process_single_page_entry`，移除残留 PageContext 依赖，主循环调用该函数（逻辑/阈值不变）；烟囱测试 1 张（fast+debug）通过，run_strategy_batch 抽样 3 张（quality+bbox）通过，输出目录 outputs_smoke_tmp / outputs_strategy_tmp。
- 2025-12-17：当前工作区未提交变更：新增 context/debug_utils/summary_utils，pipeline 拆分部分完成，README/DEVLOG 已更新；.DS_Store 有外部变动。后续阶段3 待做：分割策略接口标准化/阈值配置化（保持行为不变），完成后需再次回归。
- 2025-12-18：重构续作：`pipeline.py` 接入 `PipelineContext/PageResult`，分割统计收敛到 `_aggregate_segment`，warmup 改为 `_run_segmentation`；烟囱测试 2 张（fast+debug）通过，输出目录 outputs_smoke_tmp。
- 2025-12-18：阶段3（进行中）：分割策略与重试阈值配置化（config.segment_strategy/segment_retry.condition），pipeline 使用配置构建策略并套用阈值；烟囱测试 2 张（fast+debug）通过，输出目录 outputs_smoke_tmp。
- 2025-12-18：阶段3回归：`scripts/run_strategy_batch.py --num 3 --mode quality --debug-level bbox` 在 source_images 抽样通过，summary 输出于 outputs_strategy_tmp/summary.csv。
- 2025-12-18：阶段4（封装收敛）：overlay 生成/保存收敛到 debug_utils（prepare_overlay/save_overlay），summary 构建收敛到 summary_utils.build_summary，新增 runtime_utils/segment_report 拆分配置与分割报告；pipeline 侧行为不变。烟囱测试 2 张（fast+debug）通过，输出目录 outputs_smoke_tmp。
- 2025-12-18：里程碑：重构阶段1-4 完成并通过烟囱 + 策略抽样回归，阶段5（贴边弱梯度专项）暂缓。
- 2025-12-18：增强与输出扩展：enhance 支持可选 Wolf-Jolion 阈值（bw_method=wolf）、输出配置新增 jpeg/preview 选项；新增 postproc/output_utils 模块配合 pipeline 调度。烟囱测试 2 张（fast+debug）通过，输出目录 outputs_smoke_tmp。
- 2025-12-18：撤回 Wolf 试验，恢复默认 Sauvola 参数（关闭二值前平滑），保持 PNG+JPEG+预览输出开关。策略抽样 5 张（quality+bbox）通过，输出目录 outputs_strategy_tmp。
- 2025-12-18：新增方向判定模块 orientation（0/90/180/270 简易打分），单页处理时根据裁剪页判断方向并旋转；分割凸包描边改为蓝色调试线。当前默认开启，低于阈值不旋转；抽样 20 张（quality+bbox）输出目录 outputs_strategy_full。
- 2025-12-19：修复 053 空白/框偏移问题：默认关闭拆分页（split.enable_split=False，跳过时回退整页 bbox），拆分封装为 split_module；方向矫正只作用于输出副本并保留原始 bw（21_*_scan_bw_raw）。验证 image_053 正常，框对齐。
- 2025-12-20：提交“pipeline，抽取五个模块功能”（`1a1b13e`）：抽离 `postproc/output_utils/runtime_utils/segment_report/split_module` 等模块，pipeline 进一步瘦身。
- 2025-12-20：文档与代码对齐，重写 README/planCodex/示例说明，完善配置与限制说明。
- 2025-12-20：输出色调支持 `bw/gray/both`（默认 bw），CLI 增加 `--tone`，烟囱测试适配仅输出二值图。
- 2025-12-20：输出模式支持 `review/result/debug`（默认 review），CLI 增加 `--output-mode`，便于成果/评审/调试切换。
- 2025-12-20：新增 Python 3.11 环境（`.venv311`）并完成回归；后续以 3.11 作为主开发/运行版本。
