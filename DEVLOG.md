# 开发记录 / 记忆总览（重置版）

> 这是项目的“记忆系统”。任何人接手前必须先读本文件。仅保留铁律和工作要求，其余历史记录已重置，请按要求持续维护。

## 接手速览（需随进度实时更新）
- 环境：项目根已存在 `.venv`，进入项目后执行 `source .venv/bin/activate`，再 `export PYTHONPATH=.`。如依赖缺失，可用 `.venv/bin/pip install -r requirements.txt`。
- 运行常用命令：`PYTHONPATH=. .venv/bin/python cli.py --input <文件或目录> --output outputs --mode quality --debug`；烟囱测试（无 OCR）`PYTHONPATH=. .venv/bin/python scripts/smoke_test.py --input source_images --output outputs_smoke_perf --mode fast --max-files 2`。
- 当前能力：只输出扫描风格图，OCR 未启用；质量模式推荐，fast 仅在拍得很正时用；dewarp 透视回退，粉框精修开启。
- 输出与调试：结果在 `outputs/`（含 `run_summary.json`）；debug 打开会生成 `01/02` 调试图和 `10-21` 中间件。
- 版本与分支：当前基线提交 `b076464`（稳定版本 1.0），仓库干净；需提交请直接用 git。
- 更多细节：若本速览不够，请继续阅读下文，并查阅根目录 `README.md` 的环境/运行说明。

## 维护铁律（必须遵守）
- “接手速览”必须保持实时更新：任何影响环境/运行/能力/输出的变更，先更新该版块再行动，确保接手者一屏即读。
- 版本管理：当前仓库 git 可用，达到阶段性进展或有效调试结果后务必提交版本，保持可回退与可追踪。
- 分支策略：调试新功能/高风险改动一律新建分支（例如 `feature/<name>` 或 `exp/<name>`），验证通过再合并回主干，避免直接在主分支试验。
- 任何反馈、进展、风险、异常或中断恢复，**先更新本文件，再行动**，确保他人仅凭本文件即可对齐上下文。
- 记录真实、及时、可复现，禁止含糊或遗漏关键信息。
- 重要决策、调参、风险变化都要写明时间、影响、下一步。

## 工作要求
- 语言：全程使用中文，代码注释也用中文。
- 诚实：发现与预期/描述不符，必须如实汇报，不迎合。
- TODO 可视：执行任何任务前需在对话中先列出当次 TODO 清单并按进度更新，保持全程可见；无需在 DEVLOG 单独维护另一份 TODO 列表（DEVLOG 仅记录重要变更与进展）。
- 调试/评审约定：如需人工检查框选，直接查看 `outputs/<编号>/debug_bbox.png` 并在文档中说明结论；若执行评审，需记录样本、发现和建议。

## 当前状态（待补充）
- 当前代码：绿框裁剪基于分割 mask，扩边比例读取配置（基础 + 额外）；副块定向扩边保留；粉框（几何精修）开启，最终 scan_gray/scan_bw 以粉框或矩形回退为准；单页兜底直接使用全局 mask 的 `boundingRect`，兜底不再扩边；dewarp 为透视回退 + 轻量曲率微调（无需 page-dewarp）。
- 输出位置：全量跑的结果在 `outputs/`（质量模式，调试开启）；当前阶段未启用 OCR，仅生成扫描风格图。
- 已知风险：dewarp 能力有限；参数需对照调试样本（006/007/008/009、23-27）验证标题/装订覆盖；部分样本（如 image_022 右缘）仍存在边缘漏检/带背景，需要继续优化 A4 边缘定位。
- 环境与运行：项目根已有 `.venv`，核心依赖（rembg/opencv/Pillow/numpy/scikit-image 等）已安装；运行脚本需设置 `PYTHONPATH=.`（例如 `PYTHONPATH=. .venv/bin/python scripts/smoke_test.py ...`），否则会找不到 `docscan` 模块。
- 模式约定：当前场景（同设备/同时间/同方式拍摄的稳定批次）建议统一使用 `quality` 模式处理，获得一致的透视矫正与增强；如需极致速度且拍得很正，可改用 `fast`，一般无需 `auto`。新增 `--dry-run` 可仅做分割/裁剪预览。
- 版本控制：仓库当前处于提交 `b076464`（稳定版本 1.0），工作区干净；后续仅使用 git 管理版本。若需开发 OCR，请从此提交新开分支推进。
- 输出文件说明（单页示例，若 `--debug` 则会生成带 `_raw/dewarp/refine/gray/bw` 的中间件）：
  - 命名规则按处理顺序加前缀，并带源图+页标识：`<prefix>_<image_stem>_page_XXX_*`。
  - `01_debug_mask.png`：分割/兜底的整体 mask 可视化（图级）。
  - `02_debug_bbox.png`：原图叠加绿框（裁剪窗口）+ 粉框（实际透视四边形）+ 图例/降级提示（图级）。
  - `10_*_raw.png`：绿框裁剪后的子图（无透视、无增强，debug 才有）。
  - `11_*_dewarp.png`：去透视/去卷曲后的图（debug 才有，fast 模式会跳过）。
  - `12_*_refine.png`：粉框透视后的精修图（带留白，debug 才有）。
  - `13_*_gray.png`：refine 图的灰度版（debug 才有）。
  - `14_*_bw.png`：refine 图的二值版（debug 才有，内容与 scan_bw 相同，便于对比）。
  - `20_*_scan_gray.png`：正式输出的灰度增强版。
  - `21_*_scan_bw.png`：正式输出的二值化版。
  - `run_summary.json`：本次处理元数据（模式、耗时、降级、框坐标、输出路径等）。

## TODO 列表（实时维护）
- 聚焦单页边缘漏检问题：示例 image_022 右缘带入背景，评估通过内容/边缘信号补全 mask 或四边拟合以锁定 A4 边。
- 验证最新扩边配置化后的裁剪效果（优先样本 006/007/008/009 与 23-27），确认标题/装订是否保留且无过度整图。
- 同步/核实其他写死阈值是否需要配置化，并在必要时调整。
- 运行一轮烟囱测试（scripts/smoke_test.py），确认裁剪/几何/增强输出完整；若后续启用 OCR，再补带 OCR 的基准。
- 新增性能/调试参数待验证：`--debug-level`（none/bbox/full，默认 none）、`segment_preview_side`（默认 2000，分割预览减耗时）、`--dry-run`（只做分割/裁剪）。
- run_batch 并发：默认线程池（可选 `--pool process`，若因信号量权限失败会自动回退线程池）；建议并发=核数/2。

## 记录指引
- 采用时间顺序追加，最新条目放在末尾。
- 每条记录至少包含：日期、动作/发现、影响、下一步。
- 若有外部依赖/警告/降级路径，也需在记录中注明。

## 变更记录
- 2025-12-12：重置 DEVLOG，保留铁律/工作要求；记录当前状态（绿框扩边配置化、粉框开启、dewarp 为透视回退）；添加待办列表。
- 2025-12-13：梳理并修复裁剪/几何逻辑：默认开启双页判定，修正 max_expand_ratio 过度夹值；补充 EXIF 方向矫正、轻量 deskew 与 A4 比例微调，dewarp 采用透视回退；pipeline 增加 auto 自适应、阶段耗时记录。compileall 因系统缓存目录权限失败未验证，建议运行 smoke 确认裁剪/增强输出正常。
- 2025-12-13：修复配置重复键（geom deskew/a4）、激活 right_min_expand_ratio 右侧扩边、内容兜底 bbox 增加安全留白；修正 auto 模式判定使用原始尺寸；io_utils 加入上下文读图；新增无 OCR 的烟囱测试脚本 `scripts/smoke_test.py` 和 `examples/README.md`；README 更新“当前仅输出扫描图，不含 OCR”，并移除 OCR 依赖。***
- 2025-12-13：输出文件命名加前缀排序并包含源图名+页号（01/02 debug，10-14 中间件，20/21 正式输出），减少歧义；DEVLOG 增补文件说明。质量模式批量跑因超时中断，fast+debug 试跑亦因 120s 限时中断，已确认单张 fast 约 2–3s，批量需分批或去掉 debug；当前 dewarp 默认关闭（fast）或透视回退（quality，无 page-dewarp）。完成 tar 备份 `checkpoint_20251213.tar.gz`（排除 .git/.venv/outputs*），但本地 `.git` 目录不可写（git add/index.lock 报 Operation not permitted），暂无法提交 git 版本。
- 2025-12-14：确认 `.git` 可用且已在提交 `f3ea31a`，仓库干净；约定后续仅用 git 进行版本控制，停止新增压缩包备份；OCR 功能暂不开发，如需请从 `f3ea31a` 新建分支推进。
- 2025-12-14：全量跑 source_images（共 95 张），命令 `PYTHONPATH=. .venv/bin/python scripts/run_batch.py --input source_images --output outputs --mode fast --debug --concurrency 1`，耗时约 4.5 分钟，OCR 关闭。3 张分割无结果回退内容兜底（image_001/003/088），多张几何精修因覆盖率低回退矩形；可用 `outputs/<image>/02_debug_bbox.png` 对照粉框/绿框检查标题/装订覆盖。
- 2025-12-15：提交“稳定版本 1.0”（commit `b076464`）：兜底裁剪直接使用全局 mask boundingRect，兜底扩边关闭；保留右侧定向扩边逻辑，取消覆盖率大时自动收缩；精修回退仅在覆盖率不足时退矩形。当前已知问题：部分样本（如 image_022 右缘）仍会带入背景，需后续改进 A4 边缘定位。
- 2025-12-15：性能/调试增强：新增 `debug_level`（none/bbox/full，默认 none），分割阶段支持预览分辨率 `segment_preview_side=2000` 降耗时；run_batch 改进程池并发，CLI/run_batch 支持 `--dry-run`（仅分割/裁剪预览）；dewarp 回退统一输出 matrix 占位；smoke_test 增加耗时统计；README/说明同步更新。
- 2025-12-15：批处理验证与并发策略：默认线程池，`--pool thread` 并发 4 跑完 `source_images` 全量约 2 分 40 秒（fast+bbox）；进程池在本机因信号量权限受限，已设置失败自动回退线程池；建议并发=核数/2，debug_level=bbox 以减轻 I/O。
- 2025-12-17：接入统一内容兜底工具 `mask_utils`，分割兜底与 page_split 复用；run_summary 增补分割统计。新增边缘置信度衰减（贴边梯度弱时衰减 mask）与轻量曲率微调（基于上下边缘二次拟合，小幅 remap），可在配置 dewarp.enable_curve_adjust/curve_max_shift_px 控制。随机 15 张回归（seed=42，outputs_sample15）：分割回退 0，边缘衰减触发 0，曲率微调触发 5（max_shift=6px），精修跳过 2（image_016/019 形状过滤，按约定保持裁剪框），覆盖率均值≈0.999，无错误。已知问题：image_022 右缘仍有少量背景未扣除，需后续提升边界置信度/窄带梯度微调；已彻底放弃 page-dewarp 方案（不再依赖）。 
