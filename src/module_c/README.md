# Module C 命令行速查

Module C 用于对 Mini-VLO 生成的视频任务文本做过滤：先把
`VideoTaskRecord` 转换成 samples JSONL，再执行语义一致性校验和运动质量评分，
最后输出 `keep/drop` 决策。

当前版本支持多轨迹 motion：LIBERO 的末端执行器轨迹会映射成 `eef` track，
Module D 的 `Root`、`Hand_R`、`Hand_L` 等骨骼轨迹可以同时参与质量评分。

## 目前实现功能

Module C 目前实现了以下功能：

- 将 Mini-VLO 的 `VideoTaskRecord` 转换成 Module C 统一的 sample JSONL。
- 支持 `segment` 和 `video` 两种样本粒度。
- 支持从 `task_instruction` 或第一条增强文本中选择过滤文本。
- 支持统一的 `--motion-path` 运动轨迹入口，自动识别 LIBERO、Module D 和标准 motion JSON。
- 支持多轨迹运动质量评分，例如同时评估 `Root`、`Hand_R`、`Hand_L`。
- 默认要求真实 motion 与 Fixed/Ego 两路同步视频；缺失任一正式门禁即 `drop`。
- 占位轨迹仅通过 `--debug-dummy-motion` 显式启用，并始终从正式评测排除。
- 支持语义一致性校验，当前可使用 `qwen3-vl-flash`、`qwen3-vl-plus`
  或 `mock` verifier。
- 增加 FPS/frame count/duration、motion coverage、单位和跨模态时间范围检查。
- API/解析失败强制为 `uncertain/drop`，不再按文本长度静默 fallback。
- 输出机器可读 JSONL 和人工可读 pretty JSON。

## 工作流程

Module C 的完整流程分为两步，也可以通过 `run_generate_filter.py` 一键串联。

```text
Paired ViewBundle（Fixed + Ego + shared timebase + trajectory）
  -> Mini-VLO 生成 VideoTaskRecord
  -> prepare_samples 转换成 sample JSONL
  -> deterministic sync/schema/unit gate
  -> 解析并归一化真实 motion
  -> independent per-view semantic judge + 3D motion scoring
  -> 输出 refined JSONL / pretty JSON
```

### 1. 样本准备

`prepare_samples.py` 读取 `VideoTaskRecord`，从 `task_segments` 中提取过滤文本，
并按 `--sample-level` 生成 sample：

- `segment`：每个任务片段生成一条 sample，适合逐段过滤。
- `video`：整段视频聚合为一条 sample，适合判断整体任务文本是否可信。

如果提供 `--motion-path`，该阶段会同步解析运动轨迹并写入 sample 的
`motion.tracks`。如果没有真实轨迹，正式模式会跳过样本或在 refinement
阶段 `drop`；不会自动生成占位轨迹。

### 2. Motion 归一化

不同来源的轨迹会被统一成同一种内部结构：

```json
{
  "motion": {
    "spatial_unit": "meters",
    "coordinate_frame": "world",
    "source": "libero_eef",
    "is_dummy": false,
    "tracks": {
      "track_name": {
        "positions": [[0.0, 0.0, 0.0]],
        "timestamps": [0.0]
      }
    }
  }
}
```

当前支持的来源：

- LIBERO：读取 `steps[].ee_pos`，映射为 `eef` track。
- Module D：读取 `frame_N -> bone -> {x,y,z}`，映射为骨骼名 track。
- 标准单轨迹：`positions/timestamps`，映射为 `default` track。
- 标准多轨迹：`tracks.{name}.positions/timestamps`，直接保留 track 名。

### 3. 运动质量评分

`motion_quality.py` 对每条 track 计算：

- 速度尖峰比例：`velocity_ratio`
- 加速度尖峰比例：`acceleration_ratio`
- jerk 尖峰比例：`jerk_ratio`
- 3D 速度方向反转比例：`jitter_ratio`
- 时间间隔变异：`interval_cv`
- drop-frame / time-shift 比例：`time_gap_ratio`

多条 track 会按配置聚合成一个 `motion_quality_score`。默认 `aggregation: min`，
表示任意关键轨迹质量差都会拉低总体分数。

### 4. 语义一致性校验

`semantic_consistency.py` 会把视频和 sample 文本交给 verifier，判断文本是否和视频内容一致。
默认真实运行使用 `qwen3-vl-flash`。设置 `SEMANTIC_JUDGE_MODEL` 或
`--judge-model` 可与生成模型解耦。离线联调可用 mock，但 mock 默认不能
产生 `keep`。

### 5. 决策输出

`refinement.py` 综合动作质量分数和语义一致性标签：

- 没有 motion、motion 为 dummy、双视角/时间轴不同步时输出 `drop`。
- 动作质量分数或语义置信度低于阈值时输出 `drop`。
- 只有 sync、真实 motion 和所有视角语义均通过时输出 `keep`。
- 语义标签为 `uncertain` 或 `inconsistent` 时输出 `drop`。
- API/解析失败强制标为 `uncertain` 并输出 `semantic_verifier_failed`。

输出中的 `reason_codes` 会说明过滤原因，例如：

- `semantic_not_consistent`
- `semantic_mismatch`
- `semantic_uncertain`
- `low_motion_score`
- `Root:high_jerk_spikes`
- `Hand_R:high_jitter`

以下命令都在 `mini-vlo` 项目根目录运行。

## 1. 一键生成并过滤

正式运行使用带真实轨迹的 paired manifest：

```powershell
python run_generate_filter.py `
  --manifest data\libero_goal\processed\manifest.json `
  --sample-id put_the_bowl_on_the_plate_demo_0 `
  --view-mode fused `
  --vlm-model qwen3-vl-flash `
  --rewriter llm `
  --judge-model YOUR_INDEPENDENT_JUDGE_MODEL `
  --refine-config configs\module_c_default.yaml `
  --sample-level segment
```

单视频或提前抽帧只用于兼容/调试，并必须显式允许：

```powershell
python run_generate_filter.py `
  --frame-dir demos\task_frames `
  --fps 2 `
  --instruction "open the drawer" `
  --vlm-model qwen-vl-plus `
  --motion-path path\to\real_trajectory.json `
  --allow-single-view-debug `
  --sample-level segment
```

离线联调可使用 mock；若确实要观察 keep 路径还需显式 debug 开关：

```powershell
python run_generate_filter.py `
  --video demos\task.mp4 `
  --vlm-model qwen-vl-plus `
  --semantic-verifier mock `
  --allow-mock-debug `
  --allow-single-view-debug `
  --motion-path path\to\real_trajectory.json
```

带 LIBERO 轨迹时，使用统一的 `--motion-path`：

```powershell
python run_generate_filter.py `
  --video demos\task.mp4 `
  --vlm-model qwen-vl-plus `
  --motion-path ..\processed_libero_goal\xxx\demo_0_traj.json `
  --sample-level segment
```

带 Module D 轨迹时，`--motion-path` 可以指向轨迹目录。目录中会按视频名自动查找
`<video_stem>.json` 或 `<video_stem>_trajectory.json`：

```powershell
python run_generate_filter.py `
  --video ..\module_d\output\jumping_down.mp4 `
  --vlm-model qwen-vl-plus `
  --motion-path ..\module_d\output `
  --motion-fps 24 `
  --motion-tracks Root,Hand_R,Hand_L `
  --sample-level video
```

输出默认写入 `results/`：

- `video_task_*.json`：Mini-VLO 生成结果。
- `module_c_samples_*.jsonl`：Module C 输入样本。
- `module_c_samples_*.pretty.json`：可读样本。
- `refined_*.jsonl`：过滤结果。
- `refined_*.pretty.json`：可读过滤结果。

## 2. 只转换已有生成结果

如果已经有 `results/video_task_*.json`：

```powershell
python -m src.module_c.prepare_samples `
  --perception-file results\video_task_xxx.json `
  --output results\module_c_samples_xxx.jsonl `
  --pretty-output results\module_c_samples_xxx.pretty.json `
  --sample-level video
```

带 LIBERO 轨迹转换：

```powershell
python -m src.module_c.prepare_samples `
  --perception-file results\video_task_xxx.json `
  --output results\module_c_samples_xxx.jsonl `
  --pretty-output results\module_c_samples_xxx.pretty.json `
  --motion-path ..\processed_libero_goal\xxx\demo_0_traj.json `
  --sample-level segment
```

带 Module D 轨迹转换：

```powershell
python -m src.module_c.prepare_samples `
  --perception-file results\video_task_jumping_down.json `
  --output results\module_c_samples_jumping_down.jsonl `
  --pretty-output results\module_c_samples_jumping_down.pretty.json `
  --motion-path ..\module_d\output `
  --motion-fps 24 `
  --motion-tracks Root,Hand_R,Hand_L `
  --sample-level video
```

批量转换一个目录：

```powershell
python -m src.module_c.prepare_samples `
  --perception-dir results `
  --output results\module_c_samples_batch.jsonl `
  --pretty-output results\module_c_samples_batch.pretty.json `
  --motion-path ..\module_d\output `
  --motion-fps 24 `
  --sample-level video
```

## 3. 只运行过滤

使用默认配置过滤 samples JSONL：

```powershell
python -m src.module_c.run_refinement `
  --config configs\module_c_default.yaml `
  --input results\module_c_samples_xxx.jsonl `
  --output results\refined_xxx.jsonl `
  --pretty-output results\refined_xxx.pretty.json
```

可用 `--motion-aggregation` 覆盖多轨迹聚合方式：

```powershell
python -m src.module_c.run_refinement `
  --config configs\module_c_default.yaml `
  --input results\module_c_samples_xxx.jsonl `
  --output results\refined_xxx.jsonl `
  --pretty-output results\refined_xxx.pretty.json `
  --motion-aggregation min
```

运行真实 Qwen3-VL 语义校验前，设置 API key：

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"
```

## 4. Motion 输入格式

Module C 内部统一使用 `motion.tracks`：

```json
{
  "motion": {
    "tracks": {
      "Root": {
        "positions": [[0.0, 0.0, 1.0]],
        "timestamps": [0.0]
      },
      "Hand_R": {
        "positions": [[0.1, 0.0, 1.2]],
        "timestamps": [0.0]
      }
    }
  }
}
```

兼容的输入来源：

- LIBERO `demo_0_traj.json`：读取 `steps[].ee_pos`，输出为 `eef` track。
- Module D 原始轨迹：读取 `frame_N -> bone -> {x,y,z}`，按 `--motion-fps` 生成时间戳。
- 标准单轨迹：读取 `{positions, timestamps}`，输出为 `default` track。
- 标准多轨迹：读取 `{tracks: {name: {positions, timestamps}}}`。

如果未提供轨迹或轨迹无法匹配，正式模式不会构造假数据。`--allow-missing-motion`
只允许写出诊断样本，refinement 仍会 fail-closed。`--debug-dummy-motion` 生成的
轨迹带有 `is_dummy: true`，始终不能进入正式结果。

## 5. 质量评分输出

运动质量会对每条 track 分别计算速度、加速度、jerk 和 jitter，再聚合成
`motion_quality_score`。默认配置在 `configs/module_c_default.yaml`：

```yaml
motion_quality:
  max_velocity: 2.5
  max_acceleration: 6.0
  max_jerk: 20.0
  max_jitter_ratio: 0.30
  max_interval_cv: 0.25
  max_time_gap_ratio: 2.5
  aggregation: min
```

`aggregation` 支持：

- `min`：任意关键轨迹质量差都会拉低总体分数，适合严格过滤。
- `mean`：对所有 track 取平均，适合综合评价。

结果中的 `aux.motion.tracks` 会记录每条 track 的分数、ratio 和原因码，例如
`Root:high_jerk_spikes`、`Hand_R:high_jitter`。

## 6. 查看过滤结果分布

```powershell
python -m src.module_c.evaluate --input results\refined_xxx.jsonl
```

## 7. 常用参数

- `--sample-level segment|video`：每个 task segment 一条样本，或整段视频聚合成一条样本。
- `--text-source task_instruction|augmented_first`：选择原始任务文本或第一条增强文本作为过滤文本。
- `--motion-path`：统一轨迹入口，可以是 LIBERO 文件、Module D 文件、标准 motion 文件或轨迹目录。
- `--motion-fps`：Module D `frame_N` 轨迹生成时间戳时使用的 FPS，默认 `24`。
- `--motion-tracks`：保留指定轨迹，例如 `Root,Hand_R,Hand_L`；不传则读取所有合法 track。
- `--motion-aggregation min|mean`：覆盖多轨迹质量分数聚合方式。
- `--allow-missing-motion`：允许写出缺 motion 的诊断样本；正式门禁仍 drop。
- `--debug-dummy-motion`：显式生成调试占位轨迹；永不计入正式评测。
- `--allow-single-view-debug`：关闭 paired-view requirement，仅用于兼容诊断。
- `--allow-mock-debug`：允许 mock 走 keep 路径，仅用于单元测试/调试。
- `--motion-dir`：旧参数，等价于轨迹目录，建议改用 `--motion-path`。
- `--libero-traj-file`：旧参数，等价于 LIBERO 文件，建议改用 `--motion-path`。

