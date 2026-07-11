# Semantic-Motion Engineering Project Report 写作大纲

## 建议项目标题

**Semantic-Motion: An End-to-End Engineering Pipeline for Rendering, Perception, Language Augmentation and Multimodal Data Refinement**

中文参考：

**Semantic-Motion：面向动作可视化、语义生成与多模态数据精炼的端到端工程系统**

## 报告核心主线

> Motion/Simulation Data → Visual Rendering → Semantic Perception → Language Augmentation → Multimodal Refinement → Aligned Dataset

报告不以提出一个新的 VLA 模型为中心，而以构建一套可运行、可观察、可扩展的工程管线为中心。系统将原始动作或仿真数据转换为 VLM 可感知的视频，再生成和增强任务文本，最终联合视频、运动与文本进行质量过滤，形成 `(Video, Motion, Text)` 数据闭环。Mini-VLO 是其中的视觉语义感知组件，而不是整个项目的唯一主体。

## 章节与字数分配

| 章节 | 建议字数 |
| --- | ---: |
| Abstract | 不超过 200 |
| 1. Introduction | 1,400 |
| 2. Technical Background and Related Systems | 1,700 |
| 3. System Requirements and Overall Design | 1,800 |
| 4. Design and Implementation of the Four Modules | 4,400 |
| 5. System Integration, Demonstration and Evaluation | 2,600 |
| 6. Engineering Discussion | 1,300 |
| 7. Conclusion and Future Work | 600 |
| **合计** | **约 15,000** |

附录不计入 15,000 词。参考文献、声明和目录等行政性内容不单独分配正文词数。

---

## 前置部分

### Cover Page

- The University of Hong Kong
- School of Computing and Data Science
- COMP7705 Project Report
- 项目标题
- 学生姓名和学号
- 导师姓名
- 提交日期

### Declaration

- 原创性及学术诚信声明
- 如为小组项目，详细说明每位成员的贡献

### Abstract（不超过 200 词）

建议按照以下顺序撰写：

1. 动作采集平台能够生成视频和三维姿态，但这些数据缺少可供 VLA 或动作生成模型直接使用的语言语义。
2. 手工标注成本高，原始三维动作又不能直接被通用 VLM 理解，因此需要自动化数据转换与对齐管线。
3. 提出 Semantic-Motion 端到端工程系统。
4. 简述四个模块：Rendering、Perception、Augmentation 和 Refinement。
5. 说明最终输出为经过质量验证的 `(Video, Motion, Text)` 样本，并给出端到端演示和主要评价结果。
6. 总结系统贡献、工程价值与当前局限。

### 其他前置内容

- Acknowledgements（可选）
- Table of Contents
- List of Figures
- List of Tables
- List of Symbols（如需要）

---

# 1. Introduction（约 1,400 词）

## 1.1 Project Context: From Motion Data to VLA-ready Multimodal Data

本节不要从 VLA 模型发展史开始，而应从项目已有的数据能力和工程缺口开始。建议按照以下五段展开：

1. 动作采集或仿真系统可以持续产生 RGB/RGBD 视频与三维姿态轨迹，这些数据保留了“如何运动”的信息。
2. VLA、模仿学习和动作生成模型不仅需要运动数据，还需要与视频、轨迹严格对齐的任务语义和语言描述。
3. 现有原始数据存在语义缺口：三维坐标不能直接被通用 VLM 理解，手工标注又难以扩展。
4. 因而需要一条自动化工程管线，把动作数据依次转换为视觉信号、结构化语义、多样化文本和经过验证的训练样本。
5. 引出本项目的最终目标：将 Rendering、Perception、Augmentation 和 Refinement 四个模块连接起来，并通过可视化中间结果和端到端演示展示完整数据流。

VLA 只需在本节末尾用一小段说明应用背景：本项目为 VLA 提供数据基础设施，不负责设计新的 action model。

## 1.2 Engineering Problem and Motivation

围绕工程链路中的四个断点组织，而不是围绕“模型是否足够准确”组织：

1. **Representation gap**：FBX/BVH 或三维坐标不是 VLM 可直接消费的视觉输入。
2. **Semantic gap**：视频缺少结构化的宏观意图与微观动作描述。
3. **Language diversity gap**：单一、模式化描述不足以支持后续模型训练。
4. **Quality gap**：自动生成文本可能产生幻觉，原始运动也可能包含抖动或异常片段。

由此说明为什么必须把四个模块组合成系统，而不是独立运行若干脚本。

## 1.3 Project Aim and Engineering Deliverables

项目总目标：

> Design and implement an end-to-end pipeline that transforms motion or simulation data into aligned and quality-controlled video-motion-text samples.

具体交付物：

1. 可批量执行的多视角动作渲染与三维轨迹导出模块。
2. 将图像或视频转换为结构化任务语义的感知模块。
3. 将单一任务描述扩展为多种可追踪文本表达的增强模块。
4. 联合视频、运动和文本输出 `keep/drop` 决策的精炼模块。
5. 连接四个模块的统一数据接口、配置和执行流程。
6. 展示每个阶段中间结果及最终样本的端到端可视化或演示。

## 1.4 System Requirements

### Functional Requirements

- 支持 FBX/BVH、视频、帧目录和多种轨迹 JSON 输入
- 生成 Fixed View、Ego View 和三维关键骨骼轨迹
- 生成 macro-intent、micro-instructions 和 task segments
- 输出多种文本变体
- 检查语义一致性和运动质量
- 输出可追踪、可解释的最终结果

### Non-functional Requirements

- 模块化与可替换性
- 可配置性
- 批处理能力
- 中间结果可观察性
- 多数据源兼容
- 资源受限环境下可运行
- 失败原因可追踪

## 1.5 Scope and Boundaries

项目包含：

- 动作渲染、视频理解、文本增强和多模态过滤
- 静态 benchmark、视频案例和端到端系统验证
- 与 LIBERO、Module D 和标准轨迹格式的接口

项目不包含：

- 训练新的基础 VLM 或完整 VLA 模型
- 直接预测机器人控制信号
- 真实机器人闭环部署
- 将未来计划中的 LLM 重写器描述为已经实现的功能

## 1.6 Contributions

按照工程产出而不是算法创新来表述：

1. 设计并集成一条从动作数据到 `(Video, Motion, Text)` 样本的端到端管线。
2. 实现多视角渲染、结构化视频感知、文本增强和双通道质量过滤四个可组合模块。
3. 设计统一数据结构，使视频、语义标注和多来源运动轨迹可以在模块间传递。
4. 提供可解释的中间结果、质量分数、语义证据和过滤原因，支持系统调试及数据审查。
5. 通过静态 benchmark、视频案例、模块对比和端到端演示验证系统的实用性。

## 1.7 Report Organisation

按照工程生命周期说明章节：

- Chapter 2 说明构建系统所需的技术背景与现有方案。
- Chapter 3 定义系统需求、总体架构、数据接口和模块关系。
- Chapter 4 按数据流顺序介绍四个模块的设计与实现。
- Chapter 5 展示四模块集成、端到端运行过程、可视化结果和评价。
- Chapter 6 讨论工程权衡、失败模式、局限与可扩展性。
- Chapter 7 总结交付成果并提出后续工程工作。

---

# 2. Technical Background and Related Systems（约 1,700 词）

## 2.1 Vision-Language-Action Models

- VLA 的基本结构
- 视觉、语言和动作模块之间的关系
- VLM/VLO 作为 VLA 感知与理解基础的作用

## 2.2 Robot Learning Benchmarks

- LIBERO
- RoboCasa
- 常见机器人操作任务
- 演示数据和任务指令的组织方式

## 2.3 Vision-Language Models for Robotics

讨论 Qwen-VL 等模型在以下任务中的应用：

- 物体识别
- 空间关系理解
- 目标物体定位
- 动作和任务推断

## 2.4 Video-to-Task Generation

- 视频抽帧
- 帧级场景理解
- 动作边界检测
- 任务分段
- 任务文本生成
- pinned full-pipeline Video2Tasks baseline 与 prompt-only legacy 区分

## 2.5 Demonstration Data Quality

- 文本与视频语义不一致
- VLM 幻觉
- 错误任务边界
- 运动轨迹抖动
- 速度、加速度和 jerk 异常
- 数据过滤对机器人学习的重要性

## 2.6 Engineering Gap

指出现有工具通常只覆盖动作渲染、视频理解、文本生成或数据清洗中的一个环节。项目的工程缺口不是“缺少另一个 VLA 模型”，而是缺少把四类能力连接起来、保留跨模态对应关系并暴露中间结果的完整管线。

---

# 3. System Requirements and Overall Design（约 1,800 词）

## 3.1 Inputs, Outputs and Use Cases

- 输入：FBX/BVH 动作、仿真/真实视频、帧目录、三维轨迹
- 输出：渲染视频、结构化语义、增强文本、质量分数和最终数据样本
- 典型用例：动作数据自动标注、视频任务理解、演示数据清洗、人工结果审查

## 3.2 Design Goals

- 轻量化
- 无需本地 GPU
- 模块化
- 可配置
- 输出可解释
- 支持多种视频和轨迹来源

## 3.3 Overall Architecture

建议在报告中绘制以下架构图：

```text
FBX / BVH / Simulation Data
            ↓
    Rendering Module
       ↙          ↘
Fixed View + Ego View  3D Trajectory
       └──────┬────────┘
              ↓
 Paired ViewBundle + Shared Timebase
              ↓
 Multi-view Perception (Fixed/Ego/Fused)
              ↓
 Fused Macro/Micro Semantics
              ↓
 Fact-preserving LLM Augmentation
              ↓
 Fail-closed Refinement (Sync + Motion + Semantic)
              ↓
Quality-controlled (Video, Motion, Text)
```

## 3.4 Module Responsibilities and Boundaries

- Rendering Module：只负责动作可视化和轨迹导出
- Perception Module：只负责从视觉输入提取结构化语义
- Augmentation Module：只负责从 fused semantics 生成并验证事实不变的文本变体
- Refinement Module：只负责检查质量并作出保留或过滤决定
- 明确每个模块的输入、输出、失败模式和可替换接口

## 3.5 Data Models and Interface Contracts

介绍以下核心数据结构：

- `ViewBundle`：`sample_id`、Fixed/Ego `ViewStream`、shared FPS/frame count、
  duration、frame map、trajectory reference 和 provenance
- `MotionReference`：路径、来源、空间单位、坐标系和 track names
- `Scenario`
- `Prediction`
- `VideoTaskRecord` v2：bundle、multi-view evidence、macro/micro segments 和
  augmentation audit
- Module C sample：views、segment interval、motion metadata、label 和 provenance
- `RefinementResult`

说明所有视角和轨迹共享同一帧时钟；任何 FPS、frame count、duration、motion
coverage 或单位不一致均由 deterministic sync gate 拒绝。单视频字段仅作为
兼容接口，不是正式系统契约。

## 3.6 Observability and Visualisation Design

- 展示 Fixed View 与 Ego View
- 展示抽取的关键帧和 task segment
- 展示 macro-intent、micro-instructions 与增强文本
- 展示运动轨迹曲线、质量分数和异常原因
- 展示语义验证证据、建议文本与 `keep/drop` 结果
- 如果当前没有完整 GUI，应将其表述为“结果可视化与端到端演示”，而不是已完成的交互式平台

## 3.7 End-to-End Workflow

说明完整执行顺序：

1. 导入动作或读取已有视频。
2. 渲染多视角视频并导出三维轨迹。
3. 读取 `ViewBundle` 并验证两路视频、时间轴、轨迹和 provenance。
4. 使用 16 秒窗口、8 秒步长和每窗 16 帧执行重叠宏观推理与边界投票。
5. 在每个 segment 内使用 1–3 秒 dense windows 生成身体、接触、姿态和轨迹引用。
6. 使用 LLM 生成多种任务文本，并通过确定性事实保持门禁。
7. 组合视频、运动和文本样本。
8. 执行同步、语义一致性与运动质量检查；失败即 drop。
9. 输出最终样本、中间结果、provenance 和过滤原因。

---

# 4. Design and Implementation of the Four Modules（约 4,400 词）

本章是报告的核心，按照数据在系统中的实际流动顺序组织为四个模块：

> The Rendering Module → The Perception Module → The Fact-preserving LLM Augmentation Module → The Refinement Module

四个模块共同将原始动作数据转换成可供机器人学习使用的 `(Video, Motion, Text)` 数据。建议字数分配如下：

| 模块 | 建议字数 |
| --- | ---: |
| 4.1 The Rendering Module | 900 |
| 4.2 The Perception Module | 1,400 |
| 4.3 The Fact-preserving LLM Augmentation Module | 800 |
| 4.4 The Refinement Module | 1,300 |
| **合计** | **4,400** |

## 4.1 The Rendering Module（约 900 词）

本模块将 FBX/BVH 动作捕捉数据中的骨骼和三维坐标转换为 VLM 可以理解的视频，同时导出与视频对应的三维运动轨迹。实现主要位于 Module D 的 `render_pipeline.py`。

### 4.1.1 Motion Data Import and Scene Initialisation

- 输入 FBX 或 BVH 动作文件
- 使用 Blender 导入骨架和动画
- 自动读取动画起止帧
- 清理上一次渲染导入的骨架和网格
- 保留基础场景对象，避免批处理时重复创建场景

需要说明为什么原始三维坐标不能直接输入当前 VLM，以及将动作数据视觉化的必要性。

### 4.1.2 Scene Context Generation

渲染模块根据动作文件名和骨骼轨迹自动添加简单场景上下文：

- `vault` 或 `step`：根据臀部骨骼最高点生成障碍物
- `sit`：根据臀部骨骼最低点生成椅子
- `drop` 或 `jump_down`：根据起跳方向和脚部高度生成平台

说明场景上下文如何帮助 VLM 区分外观相似但语义不同的动作，例如普通跳跃与从高台跳下。

### 4.1.3 Multi-view Video Synthesis

生成两种互补视角：

- **Fixed View**：固定第三人称视角，用于观察全身姿态、移动方向和环境关系
- **Ego View**：将摄像机绑定到头部骨骼，用于模拟第一人称视角和观察近距离交互

同时介绍：

- Blender EEVEE 渲染引擎
- 1024 × 1024 输出分辨率
- H.264 MP4 视频格式
- 批量渲染流程

### 4.1.4 3D Motion Trajectory Export

- 逐帧读取骨骼的世界坐标
- 自动匹配不同骨架命名标准
- 提取 `Root`、`Hand_R` 和 `Hand_L` 等关键轨迹
- 将轨迹保存为 `frame_N → bone → {x, y, z}` JSON
- 保留四位小数以控制文件体积

说明渲染视频将提供给 Perception Module，而三维轨迹将提供给 Refinement Module。

### 4.1.5 Rendering Module Outputs

本模块输出：

- `Fixed_View.mp4`
- `Ego_View.mp4`
- `<action_name>_trajectory.json`

当前证据边界必须明确：本轮未修改 Module D；仓库仍没有 tracked `.blend`
参考场景、render manifest、自动双路同步验证或 trajectory 内嵌 FPS/units。
因此本节只能描述已有渲染脚本和目标契约，不能声称可复现渲染验收完成。
后续 manifest 至少应记录 FPS、frame range、duration、camera、units、frame map、
scene/Blender version 和 checksums。

可在本节末尾给出一个输入动作文件及其视频、轨迹输出示例。

## 4.2 The Perception Module（约 1,400 词）

本模块使用 VLM 分析静态图像、视频关键帧或渲染视频，提取场景对象、空间关系、宏观任务意图和微观动作步骤。

### 4.2.1 Video Frame Sampling

- 使用 OpenCV 读取视频
- 按 shared frame index 同步读取 Fixed/Ego 两路视频
- macro：16 秒重叠窗口、8 秒步长、每窗 16 个时间点
- micro：segment 内 1–3 秒 dense window，用于接触和身体动作细化
- 为每个证据保存 view ID、frame index、timestamp 和 image path
- 支持直接读取预先提取的帧目录

讨论 window overlap、双视角成本、边界稳定性和 API 预算之间的权衡。

### 4.2.2 VLM-based Structured Scene Understanding

说明 VLM 的输入由图像和任务提示组成，输出结构化 JSON：

- `objects`
- `spatial_relations`
- `task_type`
- `action_sequence`
- `target_object`
- `destination`
- `domain`
- `instruction`
- `transitions`
- `confidence`
- `action_details`（body part、contact state、posture、evidence interval）

解释选择结构化 JSON 而不是自由文本的原因：

- 便于自动解析和评估
- 便于后续时间聚合
- 便于 Augmentation 和 Refinement Module 使用
- 降低模块之间的接口复杂度

### 4.2.3 Macro-Intent Recognition

宏观意图描述视频中的总体任务目标，包括：

- task type
- target object
- destination
- confidence
- domain（locomotion / manipulation / mixed / unknown）

说明系统如何将 VLM 的 `Prediction` 转换为 `MacroIntent`，以及宏观意图如何表达“正在完成什么任务”。

### 4.2.4 Micro-Instruction Parsing

微观动作描述完成任务所需的具体操作步骤：

- 按顺序读取 `action_sequence`
- 将每个动作拆分为 verb 和 object
- 为动作分配 step ID
- 保留 body part、contact state、posture、evidence frame IDs
- 将动作绑定到 `ObservedTrajectoryRef`
- 保留动作文本和模型/启发式 confidence

不可见接触必须标为 `unknown`，不得从任务名称推断为已经发生。

### 4.2.5 Temporal Task Boundary Detection

每个 macro window 输出本地 transition indices；系统将其映射回 global frame，
使用 Hanning center weights 聚合重叠窗口的候选 cut，再按 2.5 秒邻域聚类。
0.8 秒以下 segment 被拒绝/合并。Hanning weight 是内部投票权重，不是准确率。

### 4.2.6 Segment-level Temporal Aggregation

对同一任务片段中的多帧结果进行聚合：

- 对重叠 macro/micro predictions 的 task type、target、destination 和 domain
  使用 majority voting
- 对 objects 和 spatial relations 去重
- 合并并去重 micro instructions
- 计算片段平均 confidence
- 生成 segment-level `task_instruction`

最终输出包含：

- 起止时间
- frame IDs
- macro intent
- micro instructions
- objects
- spatial relations
- task instruction
- start/end frame 与 `evidence_by_view`

### 4.2.7 Static Perception Benchmark

介绍用于验证 Perception Module 的 Mini-VLO 静态 benchmark：

- 30 个场景
- 5 类机器人任务
- 人工定义的 ground truth
- 物体、空间、任务、动作、目标和目的地信息

本节只说明 benchmark 的构建原则；具体评价指标和实验结果放在 Chapter 5。

## 4.3 The Fact-preserving LLM Augmentation Module（约 800 词）

本模块根据 Perception Module 输出的宏观意图和微观动作，将单一任务描述转换为多种可用于训练的文本表达。

### 4.3.1 Augmentation Input and Interface

输入为 `PerceptionAnnotation`，主要使用：

- source instruction
- macro intent
- micro instructions
- target object
- destination

系统通过 `InstructionRewriter` 协议定义统一重写接口。正式路径使用
`LLMInstructionRewriter`；模板重写器仅用于显式 offline/debug 测试。

### 4.3.2 Instruction Reconstruction

- 按 step ID 组合 micro instructions
- 使用 `then` 和 `finally` 表达动作顺序
- 根据 task type、target object 和 destination 构造意图短语
- 在缺少微观步骤时回退到原始任务描述

### 4.3.3 Text Augmentation Strategies

LLM 根据权威结构化 facts 生成 N 个文本变体：

1. **Imperative / Descriptive**：在命令式和描述式之间变化
2. **Concise / Detailed**：改变语言粒度但不增加事实
3. **Intent Paraphrase**：保留目标和动作次序的同义改写
4. **Goal Conditioned**：只在 destination 已存在时强调最终状态

说明每种文本保留对应的 `source_step_ids`，以维持增强文本与原始动作步骤之间的可追踪关系。

### 4.3.4 Semantic Preservation and Hallucination Control

- 增强文本不得改变目标物体
- 不得改变动作顺序
- 不得添加视频中缺乏证据的动作或环境细节
- destination 仅在感知结果中存在时使用

每个 LLM 变体必须回传结构化 facts；确定性 validator 比较 task、target、
destination、action order、body part、contact state 和 entity set。改变事实、
遗漏关键 slot 或引入新实体的候选被拒绝并写入 `augmentation_audit`。

### 4.3.5 Augmentation Output

输出为 `AugmentedInstruction` 列表，每条记录包含：

- rewritten text
- augmentation strategy
- source step IDs
- model / prompt version
- fact-validation audit

增强结果随后与原始任务片段一起传入 Refinement Module。

## 4.4 The Refinement Module（约 1,300 词）

本模块对 `(Video, Motion, Text)` 样本执行自动清洗，通过运动质量评分和语义一致性验证生成可解释的 `keep/drop` 决策。

### 4.4.1 Sample Construction

- 将 `VideoTaskRecord` 转换为统一 sample JSONL
- 支持 `segment` 和 `video` 两种样本粒度
- 支持从 `task_instruction` 或 `augmented_first` 选择过滤文本
- 将 video path、text、motion 和可选 label 封装为统一样本

说明 segment-level 适合逐动作片段过滤，而 video-level 适合判断整段任务描述是否可信。

### 4.4.2 Motion Data Normalisation

将不同来源的轨迹统一转换为 `motion.tracks`：

- LIBERO：读取 `steps[].ee_pos`，映射为 `eef`
- Rendering Module：读取 `frame_N → bone → {x, y, z}`
- 标准单轨迹：映射为 `default`
- 标准多轨迹：保留原始 track 名称

说明空间单位先归一化到 meters，时间戳必须递增并覆盖 segment。缺失轨迹
直接 fail-closed；dummy 只能由 debug flag 生成、带明确 provenance，且正式评测拒绝。

### 4.4.3 Motion Quality Scoring

对每条轨迹计算：

- velocity spike ratio
- acceleration spike ratio
- jerk spike ratio
- 3D velocity-direction jitter ratio
- interval CV
- drop-frame / time-shift gap ratio

质量分数由四类异常比例的平均惩罚得到：

```text
motion quality score = 1 - mean(normalised penalties)
```

多轨迹聚合支持：

- `min`：任一关键轨迹质量较差都会降低整体分数
- `mean`：综合所有轨迹的平均质量

同时说明过短轨迹、非有限数值和非递增时间戳等无效输入的处理方法。

### 4.4.4 Semantic Consistency Verification

语义验证器同时接收视频和待检查文本，并检查：

- actor
- object
- action
- temporal order
- goal
- hallucination
- visual evidence

输出标签：

- `consistent`
- `uncertain`
- `inconsistent`

同时输出：

- confidence
- reason
- visual evidence
- suggested text

### 4.4.5 Dual-channel Refinement Decision

### 4.4.5 Deterministic Cross-modal Alignment Checks

- Fixed/Ego 文件均存在且可读
- FPS、frame count 和 duration 在容差内一致
- manifest timebase 与实际视频一致
- segment 完全落在 motion timestamp coverage 内
- spatial unit、coordinate frame 和 trajectory references 合法
- 所有失败输出稳定 reason code

### 4.4.6 Fail-closed Refinement Decision

最终决策同时依赖同步、真实运动和独立多视角语义：

```text
sync gate = pass
AND
real motion quality score >= threshold
AND
all view semantic labels = consistent
AND
semantic confidence >= threshold
→ keep

otherwise
→ drop
```

motion 缺失、dummy、mock、API/解析 failure、`uncertain` 或 `inconsistent`
均输出 `drop`。不存在生产 semantic-only fallback。

### 4.4.7 Explainable Outputs

最终 `RefinementResult` 包含：

- motion quality score
- semantic label
- semantic confidence
- `keep/drop` decision
- reason codes
- per-track motion details
- semantic evidence
- suggested text

原因码示例：

- `semantic_mismatch`
- `semantic_uncertain`
- `low_motion_score`
- `Root:high_jerk_spikes`
- `Hand_R:high_jitter`

本节最后说明四个模块如何形成完整闭环：

```text
Motion Capture / Simulation
          ↓
Rendering Module
          ↓ video                  ↘ 3D trajectory
Perception Module                    ↘
          ↓ structured annotation    Refinement Module
Augmentation Module                 ↗
          ↓ text variants          ↗
        Video + Motion + Text
                  ↓
             Keep / Drop
```

---

# 5. System Integration, Demonstration and Evaluation（约 2,600 词）

本章区分“代码路径可运行”“自动测试通过”和“由独立 gold 验证”三个证据等级，
不得把 keep/drop 数量或内部投票权重当作准确率。

## 5.1 Research Questions

- RQ1：Fixed、Ego 与 Fused 在 macro/micro semantics 上有何差异？
- RQ2：重叠窗口能否提高 Boundary F1（±0.5s）与 segmental F1@IoU？
- RQ3：LLM augmentation 的 slot preservation、hallucination 和多样性如何？
- RQ4：fail-closed sync + motion + semantic refinement 的 false-keep rate 如何？
- RQ5：完整上游 Video2Tasks 与 Semantic-Motion 在同预算下如何比较？

## 5.2 Integration Environment and Technology Stack

- Python、OpenCV、Pydantic、Blender、YAML 和 JSON/JSONL
- Qwen-VL API 配置与调用方式
- 模块入口脚本和目录组织
- 配置管理、异常处理和批处理

## 5.3 Module Integration and Interface Verification

- Rendering 输出如何被 Perception 和 Refinement 读取
- `Prediction` 如何转换为 `VideoTaskRecord`
- 增强文本如何保留 source step IDs
- LIBERO、Rendering Module 和标准 motion JSON 如何统一为 `motion.tracks`
- 通过 schema validation、pretty JSON 和 reason codes 验证接口正确性

## 5.4 End-to-End Demonstration and Visualisation

选择一个完整案例展示：

1. 原始动作或输入视频
2. Fixed View 与 Ego View
3. 关键帧与时间片段
4. macro-intent 与 micro-instructions
5. 多种增强文本
6. 三维轨迹与运动质量指标
7. 语义一致性证据
8. 最终 `keep/drop` 决策

本节是工程叙事的中心，应通过一张总流程图和一组连续截图证明四个模块不是孤立组件。

## 5.5 Evaluation Setup

说明：

- 使用的 Qwen-VL 模型版本
- API 和参数配置
- benchmark 和视频数据
- 运行环境
- baseline
- 评价指标
- gold annotation status、双人独立标注和仲裁规则
- 排除 API failure、dummy motion、mock 和 pending-human 的规则

## 5.6 Perception Module Evaluation

报告当前主要结果：

- Composite score：约 0.807
- Task Accuracy：1.000
- Spatial Accuracy：约 0.628

重点分析：

- 模型能够较好识别任务类别
- 模型的空间关系理解相对较弱
- 不同任务类别之间的表现差异

## 5.7 Video-to-Task System Comparison

比较：

- pinned upstream Video2Tasks revision `8d405a1`
- Semantic-Motion 方法

正式比较直接调用上游 `build_windows` 与 `build_segments_via_cuts`：16 秒窗口、
8 秒步长、每窗 16 帧和 Hanning 聚合。两种方法使用同一模型、frames、token
budget 和 failure policy，并删除 filename hint 与仅 proposed 可见的 task list。

历史 0.195 / 0.655 仅标为 `prompt_only_legacy`，不能用于完整 pipeline 优越性结论。

分别分析：

- task label F1
- target object F1
- action F1
- Boundary F1（±0.5s）
- segmental F1@IoU / mean temporal IoU

## 5.8 Refinement and Alignment Evaluation

- sync：双路 FPS/frame count/duration、trajectory coverage、缺视角率
- keep/drop precision、recall、F1、AUROC 与 false-keep rate
- jitter/spike/drop-frame/time-shift corruption detection AUROC
- Brier score、ECE 与 coverage–accuracy
- 只使用 `adjudicated` 的双人 gold；pending packets 不产生正式指标

## 5.9 Cross-module Case Studies

至少选择三个案例：

1. Franka 成功案例
2. UR5 目标物体不明确案例
3. LIBERO drawer 错误分段案例

每个案例应展示：

- 输入视频或关键帧
- ground truth
- 生成结果
- 正确与错误部分
- 失败原因


## 5.10 Component and Configuration Analysis

建议补充以下实验：

### 5.10.1 Frame Sampling

- 4 帧
- 8 帧
- 12 帧
- 16 帧

### 5.10.2 Refinement Components

- 仅语义验证
- 仅运动质量
- 语义与运动结合

### 5.10.3 Motion Aggregation

- `min`
- `mean`

### 5.10.4 Threshold Sensitivity

- 不同 motion quality threshold
- 不同 jitter threshold
- 不同 jerk threshold

### 5.10.5 Sample Granularity

- segment-level
- video-level

## 5.11 Engineering Acceptance Summary

使用需求追踪表总结 Chapter 1 中每项 functional/non-functional requirement 是否已经实现、使用什么证据验证，以及仍有哪些限制。

---

# 6. Engineering Discussion（约 1,300 词）

## 6.1 Interpretation of System Results

- VLM 能够识别宏观机器人任务
- 空间关系和精细动作理解仍然较弱
- 帧级错误可能传播到后续任务分段
- sync、motion 与 semantic 三层门禁在设计上降低 false-keep；可靠性仍需
  adjudicated gold 定量确认

## 6.2 Engineering Strengths

- 资源需求低
- 无需本地 GPU
- 模块化设计
- 支持多种轨迹来源
- 过滤决策可解释
- 形成可测试的多视角生成与 fail-closed 质量验证代码链路

## 6.3 Failure Mode Analysis

- 背景物体被错误识别为目标物体
- 抽屉、柜门等相似部件混淆
- 关键帧不足导致动作过程缺失
- 单帧幻觉造成错误任务边界
- 模糊视频导致 `uncertain`
- 运动轨迹与视频时间范围不匹配

## 6.4 Limitations

- 静态 benchmark 主要使用合成图
- 视频实验规模较小
- 已生成 23 个 `pending_human` 双视角标注包，但尚无双人仲裁 gold
- dummy motion 已从正式路径移除，历史 placeholder 结果不能作为性能证据
- 云端 API 模型版本变化可能影响复现
- LLM augmentation 依赖 API；事实门禁可能牺牲可接受的语言多样性
- 系统不包含动作生成和真实机器人控制
- Module D 人体骨骼运动与机器人操作存在 embodiment gap

## 6.5 Engineering Trade-offs

- 云端 API 的易用性与可复现性之间的权衡
- overlapping window 的边界稳定性与 API 成本
- 严格事实门禁的安全性与语言多样性
- `min` 聚合的严格性与误过滤风险
- 模块解耦带来的可维护性与数据转换开销

## 6.6 Evaluation Validity

- **Internal validity**：filename/task-list leakage 已移除；generation 和 judge
  应使用不同模型；API failure、mock 和 dummy 必须排除
- **External validity**：30 个 synthetic scenes、3 个 LIBERO demos 和少量视频
  不能代表真实 loco-manipulation 部署
- **Construct validity**：Boundary F1、segmental IoU 和 false-keep 分别测边界与
  清洗；Hanning weight、keep/drop 数量和 task-title F1 不能替代 correctness
- **Data leakage**：weak BDDL title 只能评价 task-level label，不用于 boundary、
  contact 或 keep/drop truth
- **Reproducibility**：记录 git revision、模型、prompt、配置、输入 checksums、
  上游 Video2Tasks revision 和所有 exclusion reason

## 6.7 Implications for VLA Data Pipelines

讨论整个 Semantic-Motion 系统对实际 VLA 数据工程的启示：

- 在动作生成前验证视觉语言理解
- 自动生成的任务标签不应直接用于训练
- 应结合视觉证据和运动证据进行数据筛选

---

# 7. Conclusion and Future Work（约 600 词）

## 7.1 Conclusion

依次总结：

1. 已实现版本化 `ViewBundle`、Fixed/Ego/Fused 下游入口和 shared timebase 检查。
2. 已实现 overlapping macro window、dense micro refinement 和
   loco-manipulation schema；自动测试验证代码行为。
3. 已实现 LLM rewriter 与确定性事实保持门禁；真实语言质量仍依赖 API 实验。
4. 已实现 fail-closed sync + motion + multi-view semantic refinement，以及
   corruption、temporal、classification 和 calibration metrics。
5. 已固定并直接集成 Video2Tasks revision `8d405a1`；正式公平比较结果尚待
   API 重跑和 adjudicated temporal gold。
6. Module D 代码本轮未修改；没有 Blender/可复现场景实测，不声称渲染验收完成。
7. 23 个标注包仍为 `pending_human`，因此不声称 boundary 或 keep/drop accuracy。

## 7.2 Future Work

- 完成 23 个 pending packets 的双人独立标注和仲裁
- 扩展真实机器人 RGB 视频数据
- 在独立生成/评判模型上重跑 LLM augmentation 与多视角消融
- 使用 adjudicated boundary gold 校准时间边界参数
- 引入物体检测、跟踪和视觉 grounding
- 校准语义验证 prompt 和置信度阈值
- 学习运动质量阈值而不是手工设定
- 与真正的 VLA action head 连接
- 开发统一的交互式可视化界面
- 在真实机器人平台上进行闭环验证

---

# References

参考文献按照正文首次引用顺序编号。重点应包括：

- Vision-Language-Action 模型
- Being-H0.5
- Qwen-VL
- LIBERO
- RoboCasa
- Video2Tasks
- 机器人模仿学习与演示数据过滤
- 运动轨迹质量评价

---

# Appendices

附录不计入报告 15,000 词限制，可包含：

## Appendix A：VLM System Prompt

完整展示静态场景理解 prompt。

## Appendix B：Semantic Consistency Prompt

展示 `semantic_consistency_v1.txt` 中的验证规则。

## Appendix C：Data Schemas

- `Prediction`
- `VideoTaskRecord`
- Module C sample
- `RefinementResult`

## Appendix D：Configuration

展示 Module C 默认配置与阈值。

## Appendix E：Additional Results

- 更多视频案例
- 逐场景 benchmark 结果
- 完整消融实验结果
- 失败案例

## Appendix F：User Guide

提供主要命令和运行方法。

---

# 建议图表清单

## Figures

1. Mini-VLO overall system architecture
2. Video-to-task processing pipeline
3. Temporal task segmentation example
4. Module C refinement workflow
5. Semantic consistency verification example
6. Motion trajectory and quality metrics
7. Static benchmark metric comparison
8. Successful and failed video case studies

## Tables

1. Comparison with related work
2. Static benchmark task distribution
3. Core data schemas
4. Supported motion formats
5. Static VLO evaluation results
6. Pinned Video2Tasks full-pipeline fair comparison
7. Module C ablation results
8. Failure mode summary
9. Research question answers

---

# 核心写作原则

不要把报告写成多个独立脚本或模块的功能介绍。全文应始终围绕以下逻辑展开：

> Mini-VLO 首先理解机器人场景并生成结构化任务描述，随后 Module C 利用视频语义证据和运动轨迹质量验证生成结果，最终过滤不可靠的机器人演示数据。

实验章节应同时包含定量结果和定性案例；讨论章节应明确承认合成数据、实验规模、API 依赖和 gold-label 数据不足等局限。
