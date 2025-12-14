# 文档拍照自动整理系统实现与改进方案（面向 Codex / 开发人员）

> 目标：在 macOS（MacBook Air M3，CPU）上实现从“手机拍摄的文档照片”到“扫描风格图片 + Excel/结构化数据”的自动处理流水线，仅使用预训练模型。强调鲁棒性、性能可控、可降级与可扩展，避免以传统 OpenCV 检测作为核心识别手段。

---

## 0. 任务概览与改进要点

你需要完成：

1) 按本文档结构搭建可运行的 Python 项目，串起“输入目录 → 扫描风格页面 + OCR 结构化”全流程。  
2) 在 MacBook Air M3（CPU）可安装、可运行，单张耗时允许几秒级。  
3) 不训练模型，仅用预训练模型/开源库；LLM 仅作可选后处理接口。  

核心改进要点（相对初版）：
- 兼容与降级：rembg/dewarp/OCR 全量加失败回退，最差也能裁剪+增强+OCR；dewarp 失败透视矩阵回退，可选轻量曲率拟合；OCR 失败记录并保留输入。
- 性能与分辨率：统一最长边上限（建议 2048–2560）后再分割/表格检测；允许 rembg/onnxruntime 线程数配置（如 `OMP_NUM_THREADS=1`）；OCR 前可对表格局部二次放大。
- 模式：`fast`（跳过 dewarp、轻量增强）、`quality`（全流程）、`auto`（按分辨率/核数自适应）。
- 双页拆分稳健性：宽高比 + mask 水平投影谷值 + 左右对称度，低置信度标记“可能双页”。
- 日志与调试：记录耗时、阈值、降级路径；debug 模式保存 mask、投影曲线、分割线、矫正前后图；输出 run-summary（耗时/置信度/降级/路径）。
- 字段归一化：表头映射 YAML + 编辑距离/同义词、数字纠错规则可配置；LLM 作为可选 hook；按场景分 profile（如发票/采购/库存）。

---

## 1. 项目概述

### 1.1 功能目标

输入：手机拍摄的报表/文档照片，可能有杂乱背景、歪斜、光照不均、单双页混合、分辨率/方向不一。  
输出（每张原始照片）：
1. 按页切分的“扫描风格”图片：自动识别纸张区域，透视/卷曲矫正，几何精修（边界规整、角度标准），黑白/高对比度灰度。  
2. 每页对应的 OCR + 表格结构化结果：Excel/CSV + 结构化 JSON（字段尽量标准化）。

### 1.2 约束

- 环境：MacBook Air M3，Python CPU。单张几秒可接受。  
- 不得训练模型；仅调用预训练模型/库。  
- 允许调用 LLM 做后处理/字段归一化（需留接口，业务侧调用）。  
- 禁止以传统 OpenCV 边缘/阈值/轮廓检测作为“核心找纸”手段；OpenCV 仅在模型输出基础上做几何润色。  

### 1.3 流水线总览

1) 页面分割 & 双页检测（模型主导）：rembg 前景分割 → mask 连通域 → 单/双页块。  
2) 页面矫正（去透视/卷曲）：透视矩阵回退为主，可选轻量曲率拟合（自研/轻量算子），不依赖 page-dewarp。  
3) 几何精修（OpenCV 润色）：轮廓拟合、A4 比例微调、deskew、统一留白。  
4) 图像增强（扫描王风格）：division 归一化 + CLAHE + Sauvola + 轻量锐化；输出灰度增强与二值版。  
5) OCR + 表格识别：PaddleOCR PP-Structure；分辨率上限控制；可灰度/二值对比取优。  
6) 后处理 & 字段映射：表头归一化、数字纠错，可选 LLM；输出标准化 Excel/CSV/JSON。  

---

## 2. 目录结构与模块划分

```text
docscan/
  ├─ docscan/
  │    ├─ __init__.py
  │    ├─ config.py / config.yaml     # 参数模板：阈值、比例、窗口、模式/profile
  │    ├─ io_utils.py                 # 文件读写、图像加载保存
  │    ├─ segment.py                  # rembg + 连通域 + 形态学平滑
  │    ├─ page_split.py               # 单/双页判定与拆分（宽高比+投影谷值+对称性）
  │    ├─ dewarp.py                   # 透视回退 + 轻量曲率微调，无 page-dewarp 依赖
  │    ├─ geom_refine.py              # 轮廓拟合、A4 微调、deskew、留白
  │    ├─ enhance.py                  # division + CLAHE + Sauvola + 锐化/光照自适应
  │    ├─ ocr_paddle.py               # PP-Structure 封装，分辨率上限，灰度/二值对比
  │    ├─ postprocess.py              # 表头映射 + 数字纠错 + 可选 LLM hook
  │    ├─ pipeline.py                 # 串联流程，fast/quality/auto 模式
  ├─ scripts/run_batch.py             # 批量/并行驱动，进度条，失败重试，缓存
  ├─ cli.py                           # --input/--output/--mode/--config/--debug 等
  ├─ requirements.txt                 # 标注已测版本
  ├─ README.md
  └─ examples/                        # 单页、双页、旋转、光照不均样例
```

---

## 3. 依赖与环境

- Python 3.10+。  
- 核心依赖：
  - `rembg`（依赖 onnxruntime，建议锁 silicon 轮子，提供线程/后端配置）。  
  - `opencv-python-headless`（或 opencv-python，如冲突则换 headless）。  
  - `paddlepaddle`（macOS CPU 轮子）+ `paddleocr`。  
- 无需 page-dewarp，默认透视回退；可选轻量曲率微调（纯 OpenCV/Numpy）。  
  - `numpy`、`Pillow`、`scikit-image`、`tqdm`、`pyyaml`。  
- 可选：`pip-tools/poetry` 生成锁文件。  
- README 必列：已验证版本、M 芯片安装命令、首次跑 PaddleOCR 建议空跑预热。  

---

## 4. 模块详细说明与接口约定（含改进）

### 4.1 segment.py —— 页面分割

- rembg 前等比缩放到最长边上限；输出 mask 做中值/开闭运算平滑；支持线程/后端参数（如 OMP_NUM_THREADS）。  
- 连通域按面积过滤、排序；记录面积比例、数量；保留 mask。  

接口：
```python
PageRegion = Tuple[np.ndarray, Tuple[int, int, int, int]]  # (mask_region, bbox)
def segment_pages(image: np.ndarray) -> List[PageRegion]: ...
```

### 4.2 page_split.py —— 单/双页判定与拆分

- 判定：宽高比（如 >1.6）+ mask 水平投影谷值 + 左右面积对称度；谷值深度/面积比不足时标记“可能双页”。  
- 分割线小窗口平滑搜索，失败回退几何中线；输出按左到右排序。  
- 裁剪：单页按分割 mask 外接框裁剪，基础/额外扩边比例从配置读取；检测到细长副块（如竖排标题/装订条）可单侧定向扩展，并设置单侧上限防止整图化。  

接口：
```python
def split_single_and_double_pages(image: np.ndarray, page_regions: List[PageRegion]) -> List[np.ndarray]: ...
```

### 4.3 dewarp.py —— 去透视/去卷曲

- 透视矫正：mask 四边形透视矫正（minAreaRect → getPerspectiveTransform）；可选行偏移多项式小曲率修正（轻量，无外部依赖）。  
- 失败日志，返回原图；参数化网格/缩放/迭代。  

接口：
```python
def dewarp_page(page_image: np.ndarray) -> np.ndarray: ...
```

### 4.4 geom_refine.py —— OpenCV 几何精修

- 基于已有页面图/mask：轮廓 → approxPolyDP/minAreaRect 得四边形。  
- A4 比例微调：目标 1.414，容忍 ±10% 可配置。  
- Deskew：角度限制（3–5°），失败回退原角度；先做 EXIF 方向修正。  
- 统一留白：裁剪后加固定像素/比例留白。  

接口：
```python
def refine_geometry_with_opencv(page_image: np.ndarray, page_mask: Optional[np.ndarray]=None) -> np.ndarray: ...
```

### 4.5 enhance.py —— 扫描风格增强

- 灰度 → 大尺度模糊估计背景，division normalization。  
- CLAHE（clip limit/tiles 可调），Sauvola，自适应 unsharp。  
- 可选光照判定：背景方差高则增强 division 或调整 gamma。  
- 输出灰度增强与二值图。  

接口：
```python
def enhance_scan_style(page_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
```

### 4.6 ocr_paddle.py —— 表格 OCR

- 输入前限制分辨率上限；必要时对表格区域二次放大；可同时跑灰度/二值取置信度高者。  
- 封装 PP-Structure V2，返回 cells/html/excel 路径、raw_json；支持模型目录缓存与预热。  
- 失败重试一次，仍失败返回空占位并记录。  

接口：
```python
def run_table_ocr(image_gray_or_bw: np.ndarray, output_dir: str, page_id: str) -> Dict[str, Any]: ...
```

### 4.7 postprocess.py —— 后处理 & 映射

- 表头归一化：YAML 映射表 + 编辑距离/同义词；数字纠错规则（O→0, I/l→1, B→8 等）配置化；按场景 profile。  
- 可选 LLM hook（配置开启才用），默认规则先行。  

接口：
```python
def normalize_table(ocr_result: Dict[str, Any]) -> Dict[str, Any]: ...
```

### 4.8 pipeline.py —— 总控

- 流程：读图 → segment → split → dewarp → geom_refine → enhance → OCR → normalize。  
- 模式：`fast`（跳 dewarp，轻量增强）、`quality`（全流程）、`auto`（按分辨率/核数选）。  
- 日志：每步耗时/降级标记；出错不中断；输出 run-summary（耗时、降级、置信度、路径）。  

接口：
```python
def process_image_file(image_path: str, output_root: str, mode: str="quality") -> List[Dict[str, Any]]: ...
```

---

## 5. 命令行与批处理

- CLI：`python cli.py --input path --output path --mode {fast,quality,auto} --config config.yaml --debug --max-pages --skip-ocr`  
- 批处理：`scripts/run_batch.py`，支持目录批量，受控并发（OCR 串行或小并发），tqdm 进度，错误汇总，失败重试，已处理缓存（哈希）。  
- 输出示例：`page_001_scan_gray.png`、`page_001_scan_bw.png`、`page_001_ocr.xlsx/json`、`debug/mask.png` 等；生成 summary（csv/json）汇总路径/耗时/置信度/降级。  

---

## 6. 配置与日志

- 所有阈值/比例/窗口/角度/留白/分辨率上限集中 config.py 或 YAML；模式（fast/quality/auto）与场景 profile（发票/采购等）可切换。  
- 日志：默认 INFO；DEBUG 落盘中间结果；记录文件名、页号、耗时、降级路径；支持 JSON 行格式便于统计。  

---

## 7. 验证与风险缓释

- 样例：`examples/` 准备 3–5 张（单页、双页、强光差、旋转 ~10°）。  
- 最小集成测试：跑 `process_image_file`，断言输出存在、日志无 ERROR；验证 dewarp/OCR 失败回退路径。  
- 依赖验证：在目标硬件先跑安装与空跑，README 记录成功命令与版本；无需 page-dewarp，默认透视回退即可。  

---

## 8. 后续扩展

- 替换/新增 dewarp 模型；可插拔 OCR（Surya/docTR）。
- 支持多页 PDF 输入/输出（PNG 聚合 PDF）。
- 几何精修可增加装订册中缝校正策略。
- 业务对接：REST API/数据库写入适配层。

---

## 9. 落地可行性与运行保障（针对 MacBook Air M3）

- 依赖可用性：onnxruntime-silicon、paddlepaddle-macos、opencv-python-headless 可在 M3 安装；不依赖 page-dewarp。  
- 预热与缓存：提供 `--warmup` 选项，对 rembg/OCR 空跑预热；配置模型缓存目录，避免重复下载。
- 分辨率与内存保护：入口强制最长边上限（建议 2048–2560）、最小边下限；rembg、OCR 各自独立的尺寸上限；超限等比压缩。
- 线程与并发：推荐环境变量 `OMP_NUM_THREADS=1`；OCR 默认串行，其余环节可小并发；run_batch 可配置并发度，默认保守。
- 方向与旋转：入口统一 EXIF 方向矫正；若整体倾斜 >10°，先轻量 deskew 再分割，提升双页判定稳定性。
- 回退链显式：dewarp→透视→原图；OCR 灰度→二值→纯文本；segment 失败→全图裁剪+留白；run-summary 记录实际路径。
- 质量选择：OCR 阶段灰度/二值对比取置信度高者；增强支持轻量/标准/强，与 fast/quality/auto 模式联动。
- 调试与可观测：debug 保存 mask、投影曲线、四边形顶点、deskew 角度、矫正前后对比；run-summary 输出耗时、置信度、降级标记、输出路径。
- 性能基准：README 标注在 M3 上的参考耗时（fast/quality），便于发现异常；提供 smoke test 覆盖样例图，自动检查输出存在且日志无 ERROR。
