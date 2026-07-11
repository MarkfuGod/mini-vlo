# Data layout

## `libero_goal/`

LIBERO Goal benchmark demos used by this project (three tasks, one demo each).

| Path | Contents |
|------|----------|
| `raw/` | Original LIBERO demonstration HDF5 files (`*_demo.hdf5`). Stored with Git LFS. |
| `processed/` | Per-demo exports: `demo_0_agentview_rgb.mp4`, `demo_0_eye_in_hand_rgb.mp4`, `demo_0_traj.json`, `demo_0_ground_truth.json`, plus `manifest.json`. Videos use Git LFS. |
| `*_preview.png` | Static preview frames at the dataset root (one per task). |

### Tasks

- `put_the_bowl_on_the_plate_demo`
- `put_the_bowl_on_the_stove_demo`
- `turn_on_the_stove_demo`

Other datasets under `data/` (e.g. `gold/`) may follow separate conventions; see project docs.
