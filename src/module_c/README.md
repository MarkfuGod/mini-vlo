# Module C 命令行速查

Module C 用于对 Mini-VLO 生成的视频任务文本做过滤：先把 `VideoTaskRecord`
转换成 samples JSONL，再执行语义一致性和可选运动质量评分，最后输出
`keep/drop` 决策。

以下命令都在 `mini-vlo` 项目根目录运行。

## 1. 一键生成并过滤

无真实轨迹时，使用语义过滤模式：

```powershell
python run_generate_filter.py `
  --video demos\task.mp4 `
  --instruction "open the drawer" `
  --vlm-model qwen-vl-plus `
  --refine-config configs\module_c_default.yaml `
  --sample-level video `
  --allow-missing-motion
```

如果已经提前抽帧：

```powershell
python run_generate_filter.py `
  --frame-dir demos\task_frames `
  --fps 2 `
  --instruction "open the drawer" `
  --vlm-model qwen-vl-plus `
  --sample-level segment `
  --allow-missing-motion
```

离线联调可使用 mock 语义校验器，避免调用 `qwen3-vl-plus`：

```powershell
python run_generate_filter.py `
  --video demos\task.mp4 `
  --vlm-model qwen-vl-plus `
  --semantic-verifier mock `
  --allow-missing-motion
```

带 LIBERO 轨迹时：

```powershell
python run_generate_filter.py `
  --video demos\task.mp4 `
  --vlm-model qwen-vl-plus `
  --libero-traj-file ..\processed_libero_goal\xxx\demo_0_traj.json `
  --sample-level segment
```

输出默认写入 `results/`：

- `video_task_*.json`：Mini-VLO 生成结果
- `module_c_samples_*.jsonl`：Module C 输入样本
- `module_c_samples_*.pretty.json`：可读样本
- `refined_*.jsonl`：过滤结果
- `refined_*.pretty.json`：可读过滤结果

## 2. 只转换已有生成结果

如果已经有 `results/video_task_*.json`：

```powershell
python -m src.module_c.prepare_samples `
  --perception-file results\video_task_xxx.json `
  --output results\module_c_samples_xxx.jsonl `
  --pretty-output results\module_c_samples_xxx.pretty.json `
  --sample-level video `
  --allow-missing-motion
```

批量转换一个目录：

```powershell
python -m src.module_c.prepare_samples `
  --perception-dir results `
  --output results\module_c_samples_batch.jsonl `
  --pretty-output results\module_c_samples_batch.pretty.json `
  --sample-level video `
  --allow-missing-motion
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

运行真实 `qwen3-vl-plus` 语义校验前，设置 API key：

```powershell
$env:QWEN3VL_PLUS_API_KEY="your-api-key"
```

## 4. 查看过滤结果分布

```powershell
python -m src.module_c.evaluate --input results\refined_xxx.jsonl
```

## 5. 常用参数

- `--sample-level segment|video`：每个 task segment 一条样本，或整段视频聚合成一条样本。
- `--text-source task_instruction|augmented_first`：选择原始任务文本或第一条增强文本作为过滤文本。
- `--allow-missing-motion`：没有轨迹时仍写出样本，并退化为仅语义过滤。
- `--allow-dummy-motion`：生成占位轨迹，仅用于流程联调。
- `--motion-dir`：从 `<video_stem>.json` 读取标准 `positions/timestamps` 轨迹。
- `--libero-traj-file`：从 LIBERO `demo_0_traj.json` 读取 `steps[].ee_pos` 和时间戳。
- `--motion-plugin module:function`：接入自定义轨迹读取函数。

