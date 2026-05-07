#!/usr/bin/env bash
set -euo pipefail

record_envs="${1:-${NUM_RECORD_ENVS:-8}}"
parallel_envs="${NUM_PARALLEL_ENVS:-$record_envs}"

checkpoint="${CHECKPOINT:-./checkpoints/sac_franka_2m5_seed0_v11_targetapproach_lrmin0p0001_best.pt}"
output_prefix="${OUTPUT_PREFIX:-./logs/v11_best}"

backend="${BACKEND:-isaac}"
agent_type="${AGENT_TYPE:-sac}"
seed="${SEED:-0}"
device="${DEVICE:-cuda:0}"
settle_steps="${SETTLE_STEPS:-550}"
gif_max_steps="${GIF_MAX_STEPS:-230}"
target_overlay="${TARGET_OVERLAY:-text-reticle}"

if ! [[ "$record_envs" =~ ^[0-9]+$ ]] || [ "$record_envs" -le 0 ]; then
  echo "NUM_RECORD_ENVS / first arg must be a positive integer, got: $record_envs" >&2
  exit 2
fi

if ! [[ "$parallel_envs" =~ ^[0-9]+$ ]] || [ "$parallel_envs" -le 0 ]; then
  echo "NUM_PARALLEL_ENVS must be a positive integer, got: $parallel_envs" >&2
  exit 2
fi

if [ "$record_envs" -gt "$parallel_envs" ]; then
  echo "record env count ($record_envs) cannot exceed parallel env count ($parallel_envs)" >&2
  exit 2
fi

for ((i = 0; i < record_envs; i++)); do
  python -m scripts.record_gif_continuous \
    --backend "$backend" \
    --agent-type "$agent_type" \
    --checkpoint "$checkpoint" \
    --save-gif "${output_prefix}_env${i}.gif" \
    --save-mp4 "${output_prefix}_env${i}.mp4" \
    --save-metrics "${output_prefix}_env${i}_visual_metrics.json" \
    --num-envs "$parallel_envs" \
    --env-index "$i" \
    --seed "$seed" \
    --device "$device" \
    --settle-steps "$settle_steps" \
    --gif-max-steps "$gif_max_steps" \
    --target-overlay "$target_overlay" \
    --headless
done
