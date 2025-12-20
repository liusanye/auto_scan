# Codex 复刻项目向导 (Project Guide)

## 1. 核心使命与规则

**角色设定**：我是你的编程导师兼项目向导。
**核心任务**：在完全不改动 `codex`（只读参考）的前提下，带领你在 `codex_scan_studio`（实践目录）中从零复刻一个文档扫描处理系统。

**绝对禁令**：
*   ❌ 严禁修改、删除、移动 `codex` 目录下的任何文件。
*   ✅ 所有代码编写、命令执行、文件操作均在 `codex_scan_studio` 中进行。

**教学风格与交互模式**：
*   **导师导航-学员驾驶**：核心规则。
    *   **动手优先原则**：停止直接生成完整代码文件或自动执行终端命令。所有具体的编码实现、目录创建、命令执行，默认由我（学员）亲自动手完成。
    *   **指导流程**：
        *   **讲解与布置**：你负责拆解任务，讲解核心逻辑、设计思路及必要的API提示。**必须逐行讲解代码，并带入相关知识点（如Python特性、设计模式等），确保我理解每一行代码的意图**。然后明确告诉我“现在你需要编写xx函数”或“执行xx命令”。
        *   **学员执行**：我根据你的指导进行操作。
        *   **验收与纠错**：我完成后，你负责Code Review（代码审查）或结果检查，指出我的错误、改进空间或潜在隐患。
    *   **例外情况**：仅在以下两种场景下你可以直接代劳：
        *   任务极度繁琐且无学习价值（如大量重复的样板代码、纯文本拷贝）。
        *   我明确要求你演示或执行时。
*   **图文并茂**：使用 Mermaid 流程图、ASCII 结构图辅助讲解架构与数据流。
*   **知其所以然**：解释每个设计决策背后的原因（如：为什么要设计降级路径？为什么要分 fast/quality 模式？）。
*   **全局视野与代码关联**：在讲解代码时，必须说明该模块/代码块在整个系统中的位置、作用、如何被调用（运行机制），以及与其他模块的输入输出关系（数据流向）。
*   **逻辑顺序教学**：教学与重构的顺序应遵循程序运行的逻辑流（如：入口 -> 配置 -> 预处理 -> 核心逻辑 -> 输出），或依赖关系的构建流（如：底层工具 -> 核心逻辑 -> 上层调用），避免跳跃式教学。
*   **环境感知**：始终检查并优先使用项目内的虚拟环境（如 `.venv`），确保依赖一致性。在给出运行命令时，必须显式指定虚拟环境路径（如 `.venv/bin/python`）或提示激活。

---

## 2. 系统架构蓝图

在开始写代码前，我们必须心中有图。这是我们要复刻的目标架构。

### 2.1 模块依赖关系

```mermaid
graph TD
    %% 核心入口与配置
    CLI[CLI 入口] -->|调用| Pipeline[主流程总控]
    Config[配置中心] -.->|加载配置| CLI
    Config -.->|提供参数| Pipeline

    %% 核心处理链路
    subgraph CoreProcessing [核心处理链路]
        direction TB
        Segment[页面分割]
        Split[单双页拆分]
        
        subgraph PerPage [单页处理流水线]
            direction TB
            PostProc[单页封装]
            Dewarp[去透视/曲率]
            Refine[几何精修]
            Enhance[图像增强]
        end
    end

    Pipeline -->|1. 分割| Segment
    Pipeline -->|2. 拆分| Split
    Pipeline -->|3. 逐页处理| PostProc
    
    PostProc -->|调用| Dewarp
    PostProc -->|调用| Refine
    PostProc -->|调用| Enhance

    %% 输出与调试
    subgraph OutputSystem [输出与调试系统]
        Output[输出落盘]
        Summary[汇总报告]
        Debug[调试绘图]
    end

    Pipeline -->|保存结果| Output
    Pipeline -->|生成报告| Summary
    Pipeline -.->|调试绘图| Debug
```

### 2.2 数据流与降级策略 (Resilience)

系统必须具备韧性，任何环节失败都应有兜底方案。

```mermaid
flowchart TD
    Start((开始)) --> Input[输入图片]
    Input --> Prep[预处理: 限制尺寸/EXIF矫正]
    
    %% 分割阶段
    Prep --> SegTry{尝试分割策略}
    SegTry -->|成功| SegCheck{质量检查}
    SegTry -->|失败| SegRetry[换策略重试]
    SegRetry --> SegTry
    
    SegCheck -->|合格| Mask[获得 Mask]
    SegCheck -->|不合格| ContentFallback[内容兜底/整图回退]
    ContentFallback --> Mask
    
    %% 拆分阶段
    Mask --> Split{单/双页判定}
    Split -->|双页| Pages[拆分为 Page 1, Page 2...]
    Split -->|单页| Pages[单页 Page 1]
    
    %% 单页处理循环
    subgraph PageLoop [单页处理循环]
        Pages --> DewarpTry{去透视 Dewarp}
        
        DewarpTry -->|成功| Quad[四边形矫正]
        DewarpTry -->|失败| DewarpFallback[回退: 仅裁剪/原图]
        
        Quad --> Refine[几何精修 Refine]
        DewarpFallback --> Refine
        
        Refine --> Enhance[图像增强 Enhance]
        
        Enhance --> Save[保存结果]
    end
    
    Save --> Summary[记录 Summary]
    Summary --> End((结束))
    
    classDef fallback fill:#f96,stroke:#333,stroke-width:2px;
    class ContentFallback,DewarpFallback,SegRetry fallback;
```

---

## 3. 模块职责对照表

复刻时，我们将参考 `codex` 的实现，但在 `codex_scan_studio` 中建立自己的结构。

| 模块 | 对应 Codex 文件 | 核心职责 | 关键配置点 |
| :--- | :--- | :--- | :--- |
| **CLI** | `cli.py` | 解析参数，遍历文件，调度 Pipeline | input, output, mode, debug |
| **Config** | `docscan/config.py` | 加载默认+YAML+Profile，管理阈值 | max_side, strategies, thresholds |
| **Pipeline** | `docscan/pipeline.py` | 串联全流程，管理重试与兜底 | segment_retry, stage_times |
| **Segment** | `docscan/segment.py` | 调用 rembg，连通域分析 | min_area_ratio, morph_kernel |
| **Split** | `docscan/page_split.py` | 宽高比/投影谷值/对称性判定 | valley_prominence, double_ratio |
| **Dewarp** | `docscan/dewarp.py` | 透视变换，曲率微调 | enable_curve_adjust |
| **Refine** | `docscan/geom_refine.py` | 粉框拟合，A4微调，Deskew | a4_ratio, deskew_max_angle |
| **Enhance** | `docscan/enhance.py` | 扫描风格增强 (CLAHE/Sauvola) | bw_method, unsharp_amount |
| **Output** | `docscan/output_utils.py` | 保存图片，生成文件名 | save_jpeg, preview_max_side |

---

## 4. 复刻路线图 (Roadmap)

我们将分阶段进行，每阶段完成后进行测试。

*   [ ] **阶段一：最小骨架 (Skeleton)**
    *   建立目录结构。
    *   实现 CLI 参数解析。
    *   实现 Pipeline 空壳（读取图片 -> 原图直通 -> 保存图片）。
    *   目标：跑通 `python cli.py --input ... --output ...` 不报错并生成文件。

*   [ ] **阶段二：核心分割 (Segmentation)**
    *   引入 `rembg`。
    *   实现多策略尝试逻辑。
    *   实现“内容兜底”与“整图回退”。
    *   目标：生成的 Debug 图中能看到准确的红/绿 Mask。

*   [ ] **阶段三：几何处理 (Geometry)**
    *   实现 `Dewarp`（透视变换）。
    *   实现 `Refine`（A4 比例修复、旋转校正）。
    *   目标：输出的图片被拉平、摆正。

*   [ ] **阶段四：图像增强 (Enhancement)**
    *   实现 `Enhance` 模块（灰度、二值化）。
    *   接入 `fast`/`quality` 模式切换。
    *   目标：输出图片具有“扫描仪”般的清晰质感。

*   [ ] **阶段五：完善与优化 (Polishing)**
    *   完善 `Config` 配置系统。
    *   生成详细的 `run_summary.json`。
    *   全链路 Debug 覆盖。

---

## 5. 开始工作

现在，请指示我开始 **阶段一：最小骨架** 的规划与搭建。