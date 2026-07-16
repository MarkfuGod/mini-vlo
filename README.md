# Mini-VLO: Lightweight Robot Task Understanding Evaluator

A lightweight **Vision-Language Observer (VLO)** system that evaluates whether a VLM (Vision-Language Model) can correctly understand robot manipulation task scenes and natural language instructions. Inspired by [Being-H0.5](https://github.com/BeingBeyond/Being-H), this project tests the perceptual and reasoning foundation that underlies Vision-Language-Action (VLA) models — without requiring GPUs, large datasets, or action generation.

Beyond static scene evaluation, Mini-VLO now provides a versioned multi-view
Semantic-Motion pipeline:

1. A `ViewBundle` pairs Fixed/Ego videos, a shared timebase, motion and provenance.
2. Overlapping 16 s / 8 s macro windows and dense 1–3 s micro windows produce
   fused loco-manipulation semantics.
3. An LLM rewriter generates variants from a configurable prompt. Code-level
   fact validation is deliberately disabled.
4. Module C computes sync, motion and independent semantic diagnostics, but its
   quality gates are deliberately disabled and every prepared sample is retained.

The legacy single-video path remains for compatibility. Diagnostic warnings,
missing motion, semantic mismatch and missing paired views are preserved in
output provenance instead of controlling `keep/drop`. Module D still needs
Blender and was not changed as part of the current repair.

## Recommended Path: Vendored Video2Tasks

The pinned MIT-licensed Video2Tasks source is now included directly under
`video2tasks/`. It is the recommended primary path in this repository. The
original Semantic-Motion and Module C code remains available for historical
experiments, but is no longer required to run the default video segmentation
workflow.

The new entry point talks directly to the configured DashScope
OpenAI-compatible endpoint. It does not require the upstream FastAPI server,
worker process, local 32B model, NVIDIA GPU, or a separately installed
`video2tasks` package:

```bash
python run_video2tasks.py \
  --video demos/video2tasks_compare/hang_towel.mp4 \
  --model qwen3-vl-flash
```

The direct API runner defaults to 8 frames at 384×256 per window to avoid
multi-minute oversized requests. Pass `--frames-per-window 16
--target-width 720 --target-height 480` to reproduce the upstream input budget.

Paired manifests retain the previous camera ablations:

```bash
python run_video2tasks.py \
  --manifest data/libero_goal/processed/manifest.json \
  --sample-id put_the_bowl_on_the_plate_demo_0 \
  --view-mode fused \
  --model qwen3-vl-flash
```

Use `--view-mode fixed`, `ego`, or `fused`. Fused input interleaves synchronized
fixed/ego evidence at each timestamp while keeping Video2Tasks transition
indices on the shared timestamp axis.

Run a dataset manifest with checkpointing:

```bash
python run_video2tasks.py \
  --manifest data/wgo_bench/full/manifest.json \
  --model qwen3-vl-flash \
  --resume \
  --output results/video2tasks_wgo.json
```

This simplifies deployment, but does not turn a generic VLM into a
robot-domain world model. Video2Tasks still depends on the provider's visual
grounding and can miss boundaries or hallucinate labels.

## What is VLO?

In robotics, a **VLA (Vision-Language-Action)** model takes in camera images and language instructions, then outputs motor commands to control a robot. The pipeline looks like:

```
Camera Image + "Pick up the red mug" ──> VLA Model ──> Joint Angles / EEF Commands
```

A **VLO (Vision-Language Observer)** strips away the Action generation and focuses purely on whether the model can **understand** the scene:

```
Camera Image + "Pick up the red mug" ──> VLM ──> Structured Understanding (JSON)
```

Specifically, the VLM must output:

- **Object recognition**: What objects are in the scene?
- **Spatial reasoning**: How are they arranged?
- **Task classification**: What type of task is this?
- **Action planning**: What sequence of actions is needed?
- **Target identification**: Which object to interact with?

If a model cannot correctly see objects and understand instructions, it certainly cannot generate correct motor actions. VLO tests this foundational capability.

## Connection to Being-H

[Being-H0.5](https://github.com/BeingBeyond/Being-H) is a state-of-the-art VLA model from BeingBeyond that uses:

- **InternVL** (Vision Encoder) to extract visual features from robot camera images
- **Qwen LLM** to process language instructions alongside visual tokens
- **Flow-Matching Action Head** to generate 200-dimensional unified action vectors

```
┌─────────────┐    ┌───────────────┐    ┌──────────────┐
│  ViT (InternVL) │──>│  LLM (Qwen)   │──>│  Action Head │──> Robot Actions
│  Image Encoder  │   │  +Instruction  │   │  (200-dim)   │    (joint cmds)
└─────────────┘    └───────────────┘    └──────────────┘
       ▲                    ▲                    ▲
       │                    │                    │
   Vision (V)         Language (L)          Action (A)
```

Mini-VLO replaces the Action Head with **structured text output** and evaluates the V+L portion:

```
┌─────────────┐    ┌───────────────┐    ┌──────────────┐
│  Qwen-VL    │──>│  VLM Analysis  │──>│  JSON Output │──> Evaluation
│  (API)      │   │  +Instruction  │   │  (structured)│    (metrics)
└─────────────┘    └───────────────┘    └──────────────┘
       ▲                    ▲                    ▲
       │                    │                    │
   Vision (V)         Language (L)          Observer (O)
```

### Why Not Run Being-H Directly?

Being-H requires CUDA GPUs, FSDP distributed training, and datasets that are hundreds of GBs (LIBERO, RoboCasa). It cannot run on a MacBook. Mini-VLO provides a way to evaluate the **perceptual understanding** component using only a cloud VLM API.

## Benchmark Design

Since the original LIBERO/RoboCasa datasets are too large (~100s GB), we create a **synthetic benchmark** of 30 scenarios inspired by their task categories:


| Category     | Inspired By                          | Count | Example                                         |
| ------------ | ------------------------------------ | ----- | ----------------------------------------------- |
| Pick & Place | LIBERO spatial/object, RoboCasa PnP* | 8     | "Pick up the red mug and place it on the shelf" |
| Open / Close | RoboCasa OpenDrawer/CloseDoor        | 8     | "Open the top drawer"                           |
| Turn On/Off  | RoboCasa TurnOnStove/TurnOffFaucet   | 6     | "Turn on the sink faucet"                       |
| Spatial      | LIBERO spatial                       | 4     | "Move the blue bowl to the left of the plate"   |
| Multi-step   | LIBERO long-horizon                  | 4     | "Pick mug, put in microwave, close door"        |


Each scenario includes a **generated schematic image** (top-down robot workspace view) and a **ground truth JSON** with objects, spatial relations, task type, action sequence, target object, and destination.

## Evaluation Metrics


| Metric                           | What It Measures                                   | Score Range |
| -------------------------------- | -------------------------------------------------- | ----------- |
| **Object Recognition F1**        | Can the VLM identify all objects in the scene?     | 0 - 1       |
| **Task Classification Accuracy** | Does it correctly identify the task type?          | 0 or 1      |
| **Action Sequence ROUGE-L**      | Does the predicted action plan match ground truth? | 0 - 1       |
| **Semantic Similarity**          | Overall meaning alignment (bag-of-words cosine)    | 0 - 1       |
| **Spatial Reasoning Accuracy**   | Does it understand "left of", "on top of", etc.?   | 0 - 1       |
| **Composite Score**              | Weighted average (equal weights, 0.2 each)         | 0 - 1       |


## Archived Static Diagnostic Result

**Model**: Qwen-VL-Plus via DashScope API | **Scenarios**: 30 | **Date**: 2026-04-01

These scores come from the synthetic single-image benchmark and are retained
for regression context. They are not comparable to temporal segmentation,
multi-view reasoning, VLA policy success, or the Video2Tasks prompt ablation.

### Overall Performance


| Metric                       | Score     |
| ---------------------------- | --------- |
| Object Recognition F1        | **0.760** |
| Task Classification Accuracy | **1.000** |
| Action Sequence ROUGE-L      | **0.718** |
| Semantic Similarity          | **0.927** |
| Spatial Reasoning Accuracy   | **0.628** |
| **Composite Score**          | **0.807** |


### Performance by Category

### Detailed Metric Breakdown

### Analysis

**Strengths**:

- **Task Classification is perfect (1.000)**: Qwen-VL-Plus correctly identified the task type (pick_and_place, open, close, turn_on, turn_off, move) in all 30 scenarios. This is a strong signal that the VL backbone understands task intent well.
- **Semantic Similarity is very high (0.927)**: The overall meaning of predictions closely matches ground truth, indicating good holistic understanding.
- **Pick & Place scores highest (0.89)**: The most common robot task category is also the best understood.
- **Spatial Reasoning scores well (0.89)**: When the model encounters explicit spatial tasks, it handles "left of", "behind", "next to" correctly.

**Weaknesses**:

- **Spatial Reasoning in non-spatial tasks is low (overall 0.628)**: The model often says objects are "ON floor" or "ON counter" instead of "ON table", causing spatial relation mismatches in Open/Close and Turn On/Off categories.
- **Turn On/Off has the lowest composite (0.68)**: The model struggles with appliance-specific actions. For the stove, it outputs generic "interact with stove" instead of "grasp knob, rotate knob to turn on".
- **Object F1 is capped at ~0.80**: The model consistently misses "table" as an object since it considers it background rather than a distinct object.

**Implications for VLA**:

- The static synthetic benchmark shows promising structured recognition for its
  30 scenarios. It does not establish video boundary, multi-view, contact, or
  downstream control performance.
- Appliance interaction (knobs, buttons, faucets) needs more specific visual grounding — the model sees the appliance but struggles to identify sub-components (knob, handle, button).
- This aligns with Being-H's own benchmark results where simpler tasks (PnP) outperform complex manipulation (multi-step sequences).

## Quick Start

### Prerequisites

- Python 3.10+
- A DashScope API key ([get one here](https://dashscope.console.aliyun.com/))

### Installation

```bash
git clone https://github.com/MarkfuGod/mini-vlo.git
cd mini-vlo
pip install -r requirements.txt
cp .env.example .env
```

Set `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`, and `VLM_MODEL` in the local
`.env` file. The file is ignored by Git; do not commit real API keys.

### 1. Generate Benchmark

```bash
python generate_benchmark.py
```

This creates 30 synthetic robot task images, `benchmark/scenarios.json`, and
`benchmark/manifest.json`. The manifest records the code revision, category
counts and the benchmark's non-temporal limitations.

### 2. Run Evaluation

```bash
python run_eval.py --model qwen3-vl-flash --timeout 300
```

Options:

- `--category pick_and_place` to select one category
- `--limit 5` to test with only the first 5 scenarios
- `--output results/my_run.json` to specify output path
- `--fail-fast` to stop instead of recording provider/missing-file errors

The v2 result schema records prompt/input hashes, code revision, provider
configuration, latency, failures and exclusions. Failed calls are not converted
into zero-scored predictions.

To run the static perception + augmentation streams on the same scenarios:

```bash
python run_semantic_motion.py \
  --model qwen3-vl-flash \
  --rewriter llm \
  --timeout 300
```

### 3. Run Paired Semantic-Motion Perception

The primary input is a manifest sample containing synchronized `fixed` and
`ego` views, shared FPS/frame count, trajectory and provenance. Perception uses
overlapping macro windows, Hanning-weighted transition aggregation and dense
micro windows. The schema includes locomotion/manipulation domain, body part,
contact state, posture, evidence frames and trajectory references.

```bash
python run_video_task.py \
  --manifest data/libero_goal/processed/manifest.json \
  --sample-id put_the_bowl_on_the_plate_demo_0 \
  --view-mode fused \
  --model qwen3-vl-flash \
  --rewriter llm \
  --max-frames 16 \
  --variants 3
```

Use `--view-mode fixed`, `ego`, and `fused` for controlled ablations.
`--rewriter template` is debug-only; `--rewriter none` performs no augmentation.
The default LLM rewriter uses
`src/semantic_motion/llm_augmentation_prompt.txt`; generated variants are not
rejected by hardcoded lexical or fact-preservation rules.

The compatibility single-video entry remains available:

```bash
python run_video_task.py \
  --video demos/task.mp4 \
  --model qwen3-vl-flash \
  --rewriter llm \
  --timeout 300
```

If you already extracted frames, use:

```bash
python run_video_task.py --frame-dir demos/task_frames --fps 2 --model qwen3-vl-flash
```

The output is saved to `results/video_task_*.json` as
`semantic-motion-video-task/v2`. It embeds the `ViewBundle`, synchronized
evidence, task segments, augmentation metadata and reproducibility metadata.
The same macro/micro window flags now apply to manifest and single-video input.

### 4. Module C: Generate + Diagnostic Refinement

Module C converts `VideoTaskRecord` outputs into aligned samples and computes
three diagnostic groups: synchronization/schema checks, 3D motion quality, and
per-view semantic consistency. **Quality gates are currently disabled.** The
diagnostics and reason codes are still emitted, but every prepared sample gets
`decision="keep"` with `quality_gates_disabled`.

One-shot generation + diagnostics:

```bash
python run_generate_filter.py \
  --manifest data/libero_goal/processed/manifest.json \
  --sample-id put_the_bowl_on_the_plate_demo_0 \
  --view-mode fused \
  --vlm-model qwen3-vl-flash \
  --rewrite-model qwen3-vl-flash \
  --semantic-verifier qwen3-vl-flash \
  --judge-model qwen3-vl-flash \
  --refine-config configs/module_c_default.yaml \
  --sample-level segment \
  --timeout 300
```

The manifest trajectory is used automatically when available. Missing motion and
single-view input are retained with diagnostics:

```bash
python run_generate_filter.py \
  --video demos/task.mp4 \
  --vlm-model qwen3-vl-flash \
  --motion-path path/to/trajectory_or_dir \
  --sample-level segment
```

`--debug-dummy-motion` explicitly synthesizes placeholder motion.
`--allow-missing-motion` remains only as a deprecated compatibility no-op.
`--allow-single-view-debug` and `--allow-mock-debug` change diagnostic settings,
not the final keep decision.

Outputs land in `results/`:

- `video_task_*.json` — generation result
- `module_c_samples_*.jsonl` / `.pretty.json` — Module C samples
- `refined_*.jsonl` / `.pretty.json` — diagnostic results and retained decisions

See [`src/module_c/README.md`](src/module_c/README.md) for stepwise commands,
motion formats, configuration, and result fields. Do not report keep/drop
accuracy while gates are disabled.

### 5. Module D: Render Motion Capture to Video + Trajectories

Before (or instead of) using an existing robot demo video, Module D can turn
`.fbx` / `.bvh` files into standardized multi-view videos and JSON trajectories:

- **Multi-view synthesis**: egocentric (`摄像机`) and fixed (`Camera`) `.mp4` views
- **Scene context**: simple geometric anchors from action names (obstacle / chair / …)
- **3D trajectories**: per-frame `Root`, `Hand_L`, `Hand_R` world coordinates

```bash
# Requires Blender + a scene with the expected camera/ground object names.
# Sample motions are under module_d/motions/
blender your_scene.blend --background --python module_d/render_pipeline.py
```

Defaults read `module_d/motions/` and write `module_d/output_videos/`. Override with
`MODULE_D_MOTION_DIR` / `MODULE_D_OUTPUT_DIR`. Full setup notes:
[`module_d/README.md`](module_d/README.md).

### 6. Prepare and Evaluate a Small LIBERO Goal Subset

Download selected task-level HDF5 files from the official
[`yifengzhu-hf/LIBERO-datasets`](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/main/libero_goal)
repository into `data/libero_goal/raw/`, then export one paired-view demo per task:

```bash
python tools/prepare_libero_goal_samples.py --demos-per-task 1
```

This writes synchronized `agentview_rgb` / `eye_in_hand_rgb` videos, EEF
trajectories, weak task-title labels, and
`data/libero_goal/processed/manifest.json`. Run the requested
title-as-single-segment experiment with three repeated inferences:

```bash
python tools/evaluate_libero_goal.py \
  --views all \
  --repeats 3 \
  --max-frames 16 \
  --model qwen3-vl-flash \
  --input-cost-per-million 0.05 \
  --output-cost-per-million 0.40 \
  --resume
```

For this experiment, each complete video is deliberately treated as one segment
whose label is the official BDDL/file title. The output includes Boundary F1 at
±0.5/1 s, Segment F1@IoU 0.5/0.75, title label accuracy, slot F1, labeled
end-to-end Segment F1, action Edit, latency, token-estimated cost, and paired
bootstrap 95% CIs for Fused−Fixed and Fused−Ego. These are explicitly marked
weak proxy metrics: they measure agreement with the title-as-one-segment
experimental definition, not human atomic-subtask boundaries.

For human timestamped ground truth, prepare a small WGO-Bench robot subset and
run the same pinned prompt comparison:

```bash
python tools/prepare_wgo_bench.py --offset 25 --limit 3

python compare_video2tasks.py \
  --samples data/wgo_bench/subset/manifest.json \
  --mode prompt-ablation \
  --model qwen3-vl-flash \
  --judge-model qwen3-vl-flash \
  --repeats 3 \
  --input-cost-per-million 0.05 \
  --output-cost-per-million 0.40 \
  --resume \
  --output results/wgo_video2tasks_prompt_ablation.json
```

WGO-Bench provides human timestamped subtask labels. Its subset manifest records
the source row, license, video checksum, boundaries, and segment labels. The
three-row command is an integration experiment; increase `--limit` before
making benchmark-wide claims.

Latest checked artifacts:

- `results/libero_goal_title_as_single_segment.json`: 27/27 valid runs. Fused
  improves title F1 over Fixed by +0.055 (paired bootstrap 95% CI
  [+0.020, +0.089]) but trails Ego by −0.065 [−0.119, −0.003].
- `results/wgo_video2tasks_prompt_ablation.json`: one human-annotated robot
  episode × three repeats. Video2Tasks instruction F1 is 0.200 versus 0.116 for
  the Semantic prompt; this integration subset does not show superiority.

### 7. Gold Evaluation and Motion Corruptions

Generate 20–30 paired-view annotation packets:

```bash
python tools/create_gold_annotation_packets.py --limit 30
```

Packets remain `pending_human` until two independent annotators and an
adjudicator complete them. Formal metrics reject pending packets. Once
adjudicated:

```bash
python tools/evaluate_semantic_motion.py \
  --gold data/gold/annotations/SAMPLE.json \
  --prediction results/video_task_SAMPLE.json

python -m src.module_c.evaluate \
  --input results/refined_SAMPLE.jsonl \
  --output results/refinement_metrics.json

python tools/evaluate_motion_corruptions.py \
  --motion data/libero_goal/processed/TASK/demo_0_traj.json
```

The metric suite includes Boundary F1 (±0.5 s), segmental F1@IoU, mean IoU,
macro/micro slot and order scores, augmentation diversity, and controlled motion
corruption AUROC. Module C classification/calibration utilities remain
available, but keep/drop metrics are diagnostic-only while decisions are forced
to `keep`.

### 8. Video2Tasks Prompt Ablation

The exact upstream revision is vendored under `video2tasks/`. To compare its
prompt with the historical Semantic-Motion prompt while holding model, windows,
and aggregation constant:

```bash
python compare_video2tasks.py \
  --mode prompt-ablation \
  --model qwen3-vl-flash \
  --timeout 300 \
  --output results/video2tasks_prompt_ablation.json
```

Both arms receive the same model, windows, frames/window, token budget and
upstream `build_segments_via_cuts`. Filename and closed task-list priors are
disabled. The script now reports shared lexical metrics without the old
arbitrary composite and marks inline author labels as non-formal.

This is **not** a full Module A comparison: the Semantic-Motion arm is a blind
prompt inside the Video2Tasks harness. Use `--samples` with independently
adjudicated temporal annotations for formal boundary/segment metrics. Actual
Fixed/Ego/Fused Module A behavior must be evaluated with
`tools/evaluate_semantic_motion.py`. Historical 0.655 vs 0.195 files and the
external 0.135 vs 0.148 run remain diagnostic artifacts, not superiority claims.

### 9. Generate Charts

```bash
python generate_charts.py --input results/eval_RUN.json
python generate_charts.py --input results/video2tasks_prompt_ablation.json
```

The script auto-detects static evaluation, prompt comparison, Semantic-Motion
gold evaluation, and Module C metric schemas. Charts and a provenance manifest
are written to `assets/` by default.

## Project Structure

```
mini-vlo/
├── README.md
├── requirements.txt
├── generate_benchmark.py     # Generate diagnostic static images + provenance
├── generate_charts.py        # Multi-schema result visualization
├── run_eval.py               # Static diagnostic evaluation entry point
├── run_semantic_motion.py    # Static perception + augmentation runner
├── run_video_task.py         # Video-to-task stream runner
├── run_video2tasks.py        # Recommended direct vendored Video2Tasks runner
├── run_generate_filter.py    # Video-to-task + Module C filter (one-shot)
├── compare_video2tasks.py    # Pinned-upstream prompt ablation
├── requirements-video2tasks.txt
├── video2tasks/              # Vendored pinned upstream source + MIT license
├── configs/
│   └── module_c_default.yaml # Module C thresholds + verifier settings
├── module_d/
│   ├── README.md
│   ├── render_pipeline.py    # Blender multi-view render + trajectory export
│   └── motions/              # Sample / drop-in .fbx/.bvh files
├── src/
│   ├── vlm_engine.py         # Qwen-VL API client (OpenAI-compatible)
│   ├── evaluator.py          # Metrics engine (F1, ROUGE-L, cosine sim, etc.)
│   ├── prompts.py            # Structured VLM prompt templates
│   ├── scenario.py           # Pydantic data models
│   ├── semantic_motion/      # ViewBundle, fusion, windowing and augmentation
│   ├── evaluation/           # Gold schemas and independent metrics
│   ├── baselines/            # Pinned external baseline adapters
│   └── module_c/             # Fail-closed sync + motion + semantic refinement
├── benchmark/
│   ├── scenarios.json        # 30 scenario definitions with ground truth
│   └── images/               # Generated schematic workspace images
├── demos/                    # Example videos
├── assets/                   # Charts for README
├── data/gold/                # Pending/adjudicated annotation packets
├── tools/                    # Evaluation, corruption and LIBERO utilities
├── tests/
└── results/                  # Evaluation / generation / refinement outputs
```

## Verification Status

- Automatically verified: Python compilation, paired-view contract/fusion,
  window aggregation, ungated augmentation behavior, diagnostic refinement,
  temporal/classification metrics, motion corruptions, and the pinned upstream
  adapter.
- Available but API-dependent: Qwen perception, LLM rewriting and independent
  semantic judging.
- Pending human work: the generated paired-view packets have not yet been
  independently double-annotated and adjudicated, so no formal boundary or
  keep/drop accuracy is claimed.
- Disabled by current policy: augmentation fact gates and Module C quality
  gates. Their thresholds and reason codes are diagnostic only.
- Pending external environment: Module D rendering still requires Blender and
  a reproducible scene; no new Module D acceptance claim is made here.

## Extending

- **Add more scenarios**: Edit `generate_benchmark.py` to add new task types or objects.
- **Swap the VLM**: Change `--model` / `--vlm-model` or `--base-url` to point at any OpenAI-compatible vision API (GPT-4o, Claude, local Ollama, etc.).
- **Custom metrics**: Add new metric functions in `src/evaluator.py` and update the `WEIGHTS` dict.
- **Module C diagnostics**: Tune `configs/module_c_default.yaml` to inspect
  motion/sync/semantic sensitivity; thresholds do not affect `keep` while
  quality gates are disabled.
- **Module D motions**: Drop new `.fbx` / `.bvh` files into `module_d/motions/` and re-run the Blender script.

## References

- [Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization](https://arxiv.org/pdf/2601.12993) (BeingBeyond, 2026)
- [LIBERO: Lifelong Robot Learning Benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [RoboCasa: Large-Scale Simulation for Everyday Tasks](https://github.com/robocasa/robocasa)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) (Alibaba Cloud)

## License

MIT