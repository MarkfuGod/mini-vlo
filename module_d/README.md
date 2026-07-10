# Module D: High-Fidelity Rendering Pipeline

Blender script that turns FBX/BVH motion-capture files into multi-view videos
plus per-frame 3D trajectories for downstream VLM analysis and Module C filtering.

## What it produces

For each motion file under `motions/`:

- `Fixed_View.mp4` — third-person camera (`Camera`)
- `Ego_View.mp4` — first-person camera bound to the head bone (`摄像机`)
- `<name>_trajectory.json` — frame-wise `Root` / `Hand_L` / `Hand_R` world positions

It also injects simple scene geometry from the action name (e.g. obstacle for
`vault`/`step`, chair for `sit`).

## Prerequisites

- Blender 4.x / 5.x with EEVEE
- A `.blend` scene that already contains:
  - ground plane named `平面` (kept across imports)
  - optional reference mesh `苏珊娜`
  - cameras named `Camera` (fixed) and `摄像机` (ego)
- Sample motions are in `motions/`; drop additional `.fbx` / `.bvh` files there

> Note: `project.blend` is not shipped in this repo. Open your own Blender scene
> (or recreate the objects above), then run the script inside that scene.

## Run

From a terminal (paths default to `module_d/motions` and `module_d/output_videos`):

```bash
blender your_scene.blend --background --python module_d/render_pipeline.py
```

Optional overrides:

```bash
export MODULE_D_MOTION_DIR=/path/to/motions
export MODULE_D_OUTPUT_DIR=/path/to/output_videos
blender your_scene.blend --background --python module_d/render_pipeline.py
```

## Hand off to Module C

Point Module C at the rendered video and the trajectory folder:

```bash
python run_generate_filter.py \
  --video module_d/output_videos/jumping_down/Fixed_View.mp4 \
  --vlm-model qwen-vl-plus \
  --motion-path module_d/output_videos/jumping_down \
  --motion-fps 24 \
  --motion-tracks Root,Hand_R,Hand_L \
  --sample-level video
```
