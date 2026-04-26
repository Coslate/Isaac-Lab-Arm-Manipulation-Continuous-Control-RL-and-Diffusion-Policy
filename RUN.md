
# Live Smoke Test
python -m scripts.train_sac_continuous \
  --backend isaac \
  --env_id Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 4 \
  --total-env-steps 2024 \
  --progress \
  --log-every-train-steps 5 \
  --log-every-env-steps 128 \
  --eval-every-env-steps 0 \
  --eval-num-episodes 1 \
  --eval-max-steps 50 \
  --eval-settle-steps 20  \
  --warmup-steps 128 \
  --batch-size 16 \
  --replay-capacity 5000 \
  --device cuda:0 \
  --ram-budget-gib 4 \
  --reward-probe-steps 32 \
  --lr-scheduler constant \
  --checkpoint-dir ./checkpoints \
  --checkpoint-name sac_smoke \
  --logs-dir ./logs \
  --jsonl-log ./logs/sac_smoke_train.jsonl \
  --tb-log-dir ./logs/tb/sac_smoke \
  --wandb-project isaac-arm \
  --wandb-run-name sac_smoke_test8 \
  --wandb-mode online