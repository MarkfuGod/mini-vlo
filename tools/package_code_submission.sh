#!/usr/bin/env bash
# Package a code-only submission archive for COMP7705 hand-in.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/dist"
STAMP="$(date +%Y%m%d)"
ARCHIVE="${OUT_DIR}/mini-vlo-code-submission-${STAMP}.zip"
STAGING="${OUT_DIR}/.staging-mini-vlo"
INCLUDE_MOTIONS=1

usage() {
  cat <<'EOF'
Usage: tools/package_code_submission.sh [--lite] [--output PATH]

  --lite     Skip module_d/motions/ (~135 MB) and WGO subset videos
  --output   Override output zip path
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lite) INCLUDE_MOTIONS=0; shift ;;
    --output) ARCHIVE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

rm -rf "$STAGING"
mkdir -p "$STAGING" "$OUT_DIR"

copy_tree() {
  local src="$1" dst="$2"
  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    rsync -a "$src/" "$dst/"
  fi
}

copy_file() {
  local src="$1" dst="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
  fi
}

echo "==> Staging code submission at $STAGING"

# Top-level docs and config
for f in SUBMISSION.md README.md requirements.txt requirements-video2tasks.txt .env.example .gitattributes; do
  copy_file "$ROOT/$f" "$STAGING/$f"
done

# Python entry points
for f in "$ROOT"/*.py; do
  [[ -f "$f" ]] && copy_file "$f" "$STAGING/$(basename "$f")"
done

# Core packages
copy_tree "$ROOT/src" "$STAGING/src"
copy_tree "$ROOT/video2tasks" "$STAGING/video2tasks"
copy_tree "$ROOT/tools" "$STAGING/tools"
copy_tree "$ROOT/tests" "$STAGING/tests"
copy_tree "$ROOT/configs" "$STAGING/configs"
copy_tree "$ROOT/benchmark" "$STAGING/benchmark"
copy_tree "$ROOT/demos" "$STAGING/demos"
copy_tree "$ROOT/assets" "$STAGING/assets"

# Module D (code + README; motions optional)
copy_file "$ROOT/module_d/render_pipeline.py" "$STAGING/module_d/render_pipeline.py"
copy_file "$ROOT/module_d/README.md" "$STAGING/module_d/README.md"
if [[ "$INCLUDE_MOTIONS" -eq 1 ]]; then
  copy_tree "$ROOT/module_d/motions" "$STAGING/module_d/motions"
fi

# Data: small reproducible subsets only
copy_tree "$ROOT/data/gold" "$STAGING/data/gold"
copy_tree "$ROOT/data/module_d_wechat" "$STAGING/data/module_d_wechat"
copy_file "$ROOT/data/README.md" "$STAGING/data/README.md"
copy_tree "$ROOT/data/libero_goal/processed" "$STAGING/data/libero_goal/processed"
for preview in "$ROOT"/data/libero_goal/*_preview.png; do
  [[ -f "$preview" ]] && copy_file "$preview" "$STAGING/data/libero_goal/$(basename "$preview")"
done
copy_file "$ROOT/data/wgo_bench/full/manifest.json" "$STAGING/data/wgo_bench/full/manifest.json"
copy_tree "$ROOT/data/wgo_bench/subset" "$STAGING/data/wgo_bench/subset"

if [[ "$INCLUDE_MOTIONS" -eq 1 ]]; then
  copy_tree "$ROOT/data/wgo_bench/galaxea_subset" "$STAGING/data/wgo_bench/galaxea_subset"
  copy_tree "$ROOT/data/wgo_bench/homer_subset" "$STAGING/data/wgo_bench/homer_subset"
else
  copy_file "$ROOT/data/wgo_bench/galaxea_subset/manifest.json" "$STAGING/data/wgo_bench/galaxea_subset/manifest.json"
  copy_file "$ROOT/data/wgo_bench/homer_subset/manifest.json" "$STAGING/data/wgo_bench/homer_subset/manifest.json"
fi

# Representative result examples
mkdir -p "$STAGING/results/examples"
EXAMPLE_RESULTS=(
  smoke_static_eval.json
  libero_goal_title_as_single_segment.json
  libero_goal_refinement_metrics.json
  wgo_video2tasks_prompt_ablation.json
  video2tasks_module_d_wechat_fixed_metrics.json
  motion_corruption_benchmark.json
  offline_multiview_integration.json
)
for name in "${EXAMPLE_RESULTS[@]}"; do
  copy_file "$ROOT/results/$name" "$STAGING/results/examples/$name"
done
copy_file "$ROOT/results/README.md" "$STAGING/results/README.md"

# Strip caches and secrets if present
find "$STAGING" -type d \( -name __pycache__ -o -name .pytest_cache -o -name .venv \) -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGING" -name '*.pyc' -delete 2>/dev/null || true
find "$STAGING" -name '.DS_Store' -delete 2>/dev/null || true

# Build zip (top-level folder name: mini-vlo/)
rm -f "$ARCHIVE"
(
  cd "$STAGING/.."
  mv "$(basename "$STAGING")" mini-vlo
  zip -rq "$ARCHIVE" mini-vlo
  mv mini-vlo "$(basename "$STAGING")"
)

SIZE="$(du -sh "$ARCHIVE" | cut -f1)"
FILE_COUNT="$(find "$STAGING" -type f | wc -l | tr -d ' ')"
rm -rf "$STAGING"

echo "==> Created $ARCHIVE"
echo "    Files: $FILE_COUNT"
echo "    Size:  $SIZE"
if [[ "$INCLUDE_MOTIONS" -eq 1 ]]; then
  echo "    Profile: full (includes module_d/motions + WGO subset videos)"
else
  echo "    Profile: lite (manifests only, no large motion/video assets)"
fi
