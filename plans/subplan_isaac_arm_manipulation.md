# Subplan: Interview Demo Vertical Slice for Isaac Arm Manipulation

## 1. Goal

This subplan is for the short interview-ready demo, not the full research roadmap.

The goal is **not** to demo a fully trained policy. The goal is to demo a complete robotics data/evaluation loop:

```text
policy rollout -> episode-safe dataset -> metrics -> GIF
```

This is aligned with DeepReach because the role emphasizes:

- robotics data pipelines,
- evaluation loops,
- robot learning infrastructure,
- fast iteration,
- visuomotor / VLA readiness,
- clear experiment artifacts.

The full project roadmap still includes PPO, pure GRPO, SAC, TD3, Diffusion Policy BC, and DAgger. For the interview demo, those are future plug-ins, not required before the first demo.

---

## 2. Current Status

| Demo PR | Name | Status | Notes |
|---|---|---|---|
| PR 0 | Project scaffold | Done | Package layout, config, reproducibility utilities, output directory helpers, and scaffold tests are already implemented. |
| PR 1 | Task/action contract | Done | Defines the tested 7D Franka IK-relative action contract, clipping helper, action splitter, and gripper open/close rule. |
| PR 2 | Formal Isaac Lab observation wrapper | Done | Adds a tested formal Isaac Lab adapter for `Isaac-Lift-Cube-Franka-IK-Rel-v0`; local unit tests use an injected Gymnasium test double because Isaac Lab is not installed in this environment. |
| PR 8-lite | Rollout dataset | Pending | Store rollouts by episode so future action chunks never cross episode boundaries. |
| PR 11-lite | Evaluation metrics | Pending | Compute return, success, episode length, and action jerk. |
| PR 12-lite | GIF output | Pending | Save visual rollout GIFs for random and heuristic policies. |
| Demo PR | One-command script | Pending | One command creates dataset, metrics JSON, and GIF. |

PR0 verification command:

```bash
conda run -n isaac_arm python -m pytest tests/test_project_scaffold.py -v
```

Known PR0 result:

```text
6 passed
```

PR1 verification command:

```bash
conda run -n isaac_arm python -m pytest tests/test_task_contract.py -v
```

Known PR1 result:

```text
9 passed
```

PR2 verification command:

```bash
conda run -n isaac_arm python -m pytest tests/test_observation_wrapper.py -v
```

Known PR2 result:

```text
13 passed
```

---

## 3. Demo Scope

The demo vertical slice should include:

```text
env / policy / rollout / dataset / eval / gif
```

Do now:

- PR 0 Project scaffold
- PR 1 Task/action contract
- PR 2 Formal Isaac Lab observation wrapper
- PR 8-lite Rollout dataset
- PR 11-lite Evaluation metrics
- PR 12-lite GIF output
- one-command demo script

Do not do before the first interview demo:

- PPO
- pure GRPO
- SAC
- TD3
- Diffusion Policy
- DAgger

Reason: those are training/research modules. The interview demo should first prove the infrastructure loop works.

---

## 4. Target One-Command Demo

Final target command:

```bash
conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy heuristic \
  --num_episodes 3 \
  --save_dataset ./data/heuristic_rollouts.h5 \
  --save_metrics ./logs/heuristic_eval.json \
  --save_gif ./out/gifs/heuristic_demo.gif
```

Required outputs:

```text
data/heuristic_rollouts.h5
logs/heuristic_eval.json
out/gifs/heuristic_demo.gif
```

Recommended comparison outputs:

```text
data/random_rollouts.h5
data/heuristic_rollouts.h5
data/replay_from_heuristic_rollouts.h5

logs/random_eval.json
logs/heuristic_eval.json
logs/replay_from_heuristic_eval.json

out/gifs/random_demo.gif
out/gifs/heuristic_demo.gif
out/gifs/replay_from_heuristic_demo.gif
```

Replay outputs are separate because `ReplayPolicy` is not a policy that starts from scratch. It first needs an existing rollout dataset, then replays the saved action sequence to prove that collected data can be inspected, replayed, and debugged.

Interview message:

> Before training a large policy, I built the data and evaluation loop first. The trained SAC and Diffusion policies are drop-in replacements for the same policy interface.

---

## 5. PR 0 - Project Scaffold

**Status:** Done

**Goal / Why**

Make the repository look and behave like an executable robotics project.

**Implemented**

- Project package directories.
- Config defaults.
- Seed/reproducibility utilities.
- Output directory helpers.
- Scaffold tests.
- Conda environment name: `isaac_arm`.
- Python packages recorded in `requirement.txt`.

**Important Directories**

```text
configs/
env/
dataset/
eval/
scripts/
tests/
logs/
data/
out/gifs/
```

Some output directories may be created lazily by helpers instead of committed empty.

**Test Command**

```bash
conda run -n isaac_arm python -m pytest tests/test_project_scaffold.py -v
```

**What To Test**

- Package imports work.
- Config defaults are valid.
- Output directories can be created idempotently.
- Seed fixing makes Python and NumPy deterministic.
- Torch seed test runs when torch is installed.

**Suggested Commit**

```bash
git commit -m "chore: scaffold robotics data-loop demo"
```

---

## 6. PR 1 - Task And Action Contract

**Status:** Done

**Goal / Why**

Define the robot interface clearly for the intended Isaac Lab environment.

**Task**

```python
env_id = "Isaac-Lift-Cube-Franka-IK-Rel-v0"
```

**Action Contract**

```python
action_dim = 7
action_names = [
    "dx", "dy", "dz",
    "droll", "dpitch", "dyaw",
    "gripper",
]
```

Action semantics:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

- First 6 dims: relative end-effector pose delta.
- Last dim: gripper command.
- Isaac Lab binary gripper convention: `gripper >= 0` means open, `gripper < 0` means close.
- All policy outputs should be clipped to `[-1, 1]`.

**Suggested Files**

```text
configs/task_config.py
tests/test_task_contract.py
```

**Test Command**

```bash
conda run -n isaac_arm python -m pytest tests/test_task_contract.py -v
```

**What To Test**

- `action_dim == 7`.
- Action names exactly match the expected order.
- Action clipping clamps values into `[-1, 1]`.
- Gripper rule is explicit and tested.
- Env id is `Isaac-Lift-Cube-Franka-IK-Rel-v0`.

**Suggested Commit**

```bash
git commit -m "feat(env): define Franka 7D action contract"
```

---

## 7. PR 2 - Formal Isaac Lab Observation Wrapper

**Status:** Done

**Goal / Why**

Create a stable observation interface around the official Isaac Lab environment:

```text
Isaac-Lift-Cube-Franka-IK-Rel-v0
```

This subplan no longer includes a mock backend. The demo uses the formal Isaac adapter and therefore requires Isaac Sim / Isaac Lab to be installed and launched with camera support.

**Observation Contract**

```python
obs = {
    "image": np.ndarray,    # shape: (num_envs, 3, 84, 84), dtype uint8
    "proprio": np.ndarray,  # shape: (num_envs, 40), dtype float32
}
```

Formal proprio feature order:

```text
[
  arm_joint_pos_rel,
  arm_joint_vel_rel,
  gripper_finger_pos,
  gripper_finger_vel,
  ee_pos_base,
  cube_pos_base,
  target_pos_base,
  ee_to_cube,
  cube_to_target,
  previous_action,
]
```

The wrapper computes:

```text
ee_to_cube = cube_pos_base - ee_pos_base
cube_to_target = target_pos_base - cube_pos_base
```

Local unit tests may inject a Gymnasium-compatible Isaac test double to verify the adapter without launching Isaac Sim. This is not a project backend and must not appear in user-facing demo commands.

**Suggested Files**

```text
env/isaac_env.py
tests/test_observation_wrapper.py
```

**Test Command**

```bash
conda run -n isaac_arm python -m pytest tests/test_observation_wrapper.py -v
```

**What To Test**

- `reset()` returns an observation dict.
- `step(action)` returns `(obs, reward, terminated, truncated, info)`.
- Image shape is `(num_envs, 3, 84, 84)` and dtype is `uint8`.
- Proprio shape is `(num_envs, 40)` and dtype is `float32`.
- Action shape is `(7,)` for `num_envs=1` or `(num_envs, 7)` for batched envs.
- Action is clipped to `[-1, 1]` before passing into Isaac Lab.
- Derived features `ee_to_cube` and `cube_to_target` are computed correctly.
- Done/truncated behavior is correct.
- If Isaac camera mode is requested without camera support, the error message explains `--enable_cameras`.
- If Isaac Lab runtime is not installed, construction fails with a readable error.

**Suggested Commit**

```bash
git commit -m "feat(env): add Isaac Lab image-proprio wrapper"
```

**Env Install Recipe**

The formal Isaac adapter needs Isaac Sim 5.1 + Isaac Lab 2.3.2 + cu126 PyTorch inside the `isaac_arm` conda env. All top-level dependencies are pinned in `requirement.txt`; Isaac Lab ships two in-wheel sub-packages (`isaaclab_assets`, `isaaclab_tasks`) that must be editable-installed afterwards. `scripts/install_isaac.sh` does both steps.

Fresh restore after rebuilding the container:

```bash
conda activate isaac_arm
bash scripts/install_isaac.sh
```

Equivalent manual flow:

```bash
pip install -r ./requirement.txt
ISAACLAB_DIR=$(python -c "import isaaclab, os; print(os.path.dirname(isaaclab.__file__))")
pip install -e "$ISAACLAB_DIR/source/isaaclab_assets"
pip install -e "$ISAACLAB_DIR/source/isaaclab_tasks"
```

Docker run flags that must be present (without them Isaac Sim's Vulkan/RTX renderer has no driver to talk to, even though `torch.cuda.is_available()` returns `True`):

```bash
docker run --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  ...
```

Install verification:

```bash
conda run -n isaac_arm pytest tests/test_isaac_installation.py -v
```

Expected result: `2 passed`.

---

## 8. Policies For The Demo

**Status:** Pending

The demo only needs three lightweight policies.

### RandomPolicy

```python
action = np.random.uniform(-1.0, 1.0, size=(7,))
```

Purpose:

- Baseline.
- Verifies action handling.
- Verifies dataset/eval/GIF pipeline works.

### HeuristicPolicy

Simple rules:

```text
1. Move gripper toward cube.
2. When close, close gripper.
3. Lift upward.
```

Purpose:

- Sanity policy.
- Should be easier to interpret than random in the formal Isaac env.
- Produces a more meaningful GIF for interview discussion.

### ReplayPolicy

Reads actions from a saved dataset and replays them.

Purpose:

- Shows collected data can be replayed/debugged.
- Demonstrates the data loop more convincingly.
- Useful for future offline IL debugging.

Replay inputs and outputs:

```text
input:  data/heuristic_rollouts.h5
output: logs/replay_from_heuristic_eval.json
output: out/gifs/replay_from_heuristic_demo.gif
output: data/replay_from_heuristic_rollouts.h5  # optional, but recommended
```

The replay dataset output is optional because the policy already consumes a dataset. It is recommended for the demo because it proves replay uses the same collector/evaluator path as random and heuristic.

**Suggested Files**

```text
policies/random_policy.py
policies/heuristic_policy.py
policies/replay_policy.py
tests/test_demo_policies.py
```

**Test Command**

```bash
conda run -n isaac_arm python -m pytest tests/test_demo_policies.py -v
```

**What To Test**

- Every policy returns shape `(7,)`.
- Actions are finite.
- Actions are clipped to `[-1, 1]`.
- Heuristic policy eventually commands close when near the cube.
- Replay policy returns actions in saved order and handles end-of-episode cleanly.

**Suggested Commit**

```bash
git commit -m "feat(policy): add random heuristic and replay demo policies"
```

---

## 9. PR 8-lite - Rollout Dataset

**Status:** Pending

**Goal / Why**

Save rollout data in an episode-safe format. This is the most DeepReach-relevant part of the demo because it shows data infrastructure, not only model code.

**Pipeline**

```text
policy interacts with env
  -> collect obs/action/reward/done/info
  -> save by episode
  -> support future action chunk sampling
```

**Preferred Storage**

Use HDF5 for the first demo:

```text
data/heuristic_rollouts.h5
```

The important design is episode-safe storage. For larger scale, the backend can later become zarr or sharded HDF5.

**Dataset Schema**

```text
episode_000/
  images        (T, 3, 84, 84) uint8
  proprios      (T, proprio_dim) float32
  actions       (T, 7) float32
  rewards       (T,) float32
  dones         (T,) bool
  truncateds    (T,) bool
  metadata/
    policy_name
    env_backend
    action_dim
    proprio_dim
    seed
```

**Suggested Files**

```text
dataset/episode_dataset.py
scripts/collect_rollouts.py
tests/test_demo_dataset.py
```

**Test Command**

```bash
conda run -n isaac_arm python -m pytest tests/test_demo_dataset.py -v
```

**What To Test**

- Dataset file is created.
- Each episode has required keys.
- `actions` shape is `(T, 7)`.
- `images` shape is `(T, 3, 84, 84)`.
- `proprios` shape is `(T, proprio_dim)`.
- Metadata includes policy name, backend, seed, action dim, and proprio dim.
- Action chunk sampling never crosses episode boundaries.
- Done/truncated flags terminate sampling windows correctly.

**Suggested Commit**

```bash
git commit -m "feat(data): add episode-safe rollout dataset"
```

---

## 10. PR 11-lite - Evaluation Metrics

**Status:** Pending

**Goal / Why**

Quantify behavior even without a trained policy. The first comparison should be random vs heuristic.

**Metrics**

```text
mean_return
success_rate
mean_episode_length
mean_action_jerk
```

Action jerk for the lightweight demo:

```python
jerk = mean(norm(action[t] - action[t - 1]))
```

This is a simple smoothness proxy. Later the full project can use higher-order jerk based on acceleration changes if needed.

**Suggested Files**

```text
eval/eval_loop.py
tests/test_eval_metrics.py
```

**Test Command**

```bash
conda run -n isaac_arm python -m pytest tests/test_eval_metrics.py -v
```

**What To Test**

- Metrics dict has fixed keys.
- `success_rate` is in `[0, 1]`.
- Constant-action policy has `mean_action_jerk == 0`.
- Random policy has `mean_action_jerk > 0` in a deterministic seeded test.
- Eval result can be saved to JSON.

**Expected JSON**

```json
{
  "policy_name": "heuristic",
  "env_backend": "isaac",
  "num_episodes": 3,
  "mean_return": 0.0,
  "success_rate": 0.0,
  "mean_episode_length": 0.0,
  "mean_action_jerk": 0.0
}
```

Values above are placeholders. Tests should validate keys/types/ranges; live Isaac performance should be measured in an installed Isaac Lab runtime.

**Suggested Commit**

```bash
git commit -m "feat(eval): add rollout metrics for data-loop demo"
```

---

## 11. PR 12-lite - GIF Visual Output

**Status:** Pending

**Goal / Why**

Create the most interview-friendly artifact. Robotics failures are often easier to understand visually than from scalar metrics.

**Pipeline**

```text
rollout frames -> save GIF -> optionally overlay policy/return/success/jerk
```

**Suggested Files**

```text
eval/gif_recorder.py
tests/test_visual_outputs.py
```

**Test Command**

```bash
conda run -n isaac_arm python -m pytest tests/test_visual_outputs.py -v
```

**What To Test**

- GIF file exists.
- GIF file size is greater than zero.
- GIF has more than one frame.
- Missing output directory is created automatically.
- Recorder handles `uint8` RGB frames.

**Suggested Commit**

```bash
git commit -m "feat(viz): add rollout GIF recording"
```

---

## 12. Demo PR - One-Command Data Loop

**Status:** Pending

**Goal / Why**

One command should generate dataset, metrics, and GIF. This is the command to run before the interview and potentially screen-share.

**Command**

```bash
conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy heuristic \
  --num_episodes 3 \
  --save_dataset ./data/heuristic_rollouts.h5 \
  --save_metrics ./logs/heuristic_eval.json \
  --save_gif ./out/gifs/heuristic_demo.gif
```

**Required Policy Options**

```bash
conda run -n isaac_arm python -m scripts.demo_data_loop --backend isaac --policy random
conda run -n isaac_arm python -m scripts.demo_data_loop --backend isaac --policy heuristic
conda run -n isaac_arm python -m scripts.demo_data_loop --backend isaac --policy replay --replay_dataset ./data/heuristic_rollouts.h5
```

Replay command with explicit outputs:

```bash
conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy replay \
  --replay_dataset ./data/heuristic_rollouts.h5 \
  --num_episodes 3 \
  --save_dataset ./data/replay_from_heuristic_rollouts.h5 \
  --save_metrics ./logs/replay_from_heuristic_eval.json \
  --save_gif ./out/gifs/replay_from_heuristic_demo.gif
```

**Suggested File**

```text
scripts/demo_data_loop.py
tests/test_demo_data_loop.py
```

**Test Command**

```bash
conda run -n isaac_arm python -m pytest tests/test_demo_data_loop.py -v
```

**What To Test**

- CLI accepts `--backend`, `--policy`, `--num_episodes`, `--save_dataset`, `--save_metrics`, and `--save_gif`.
- CLI accepts `--replay_dataset` when `--policy replay`.
- Running with `--policy random` creates all requested outputs.
- Running with `--policy heuristic` creates all requested outputs.
- Running with `--policy replay --replay_dataset ...` creates metrics/GIF outputs and optionally a replay rollout dataset.
- Running with `--policy replay` and no `--replay_dataset` fails with a readable error.
- Metrics JSON has the expected keys.
- Dataset has at least one episode.
- GIF exists and has multiple frames.
- Output directories are created if missing.

**Suggested Commit**

```bash
git commit -m "feat(demo): add one-command robotics data loop"
```

---

## 13. Recommended Implementation Order

Finish in this order:

1. One-command script skeleton.
2. Formal Isaac env wrapper.
3. Random and heuristic policies.
4. Metrics JSON.
5. GIF output.
6. Episode dataset.
7. Replay policy.
8. Tests.

If time is tight:

- GIF and metrics are mandatory for the interview.
- Dataset can be simplified, but keep episode-safe design.
- Replay policy is nice to have.
- Isaac backend is required for this no-mock demo.

---

## 14. Acceptance Criteria For Interview Demo

The demo is ready when these commands work:

```bash
conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy random \
  --num_episodes 3 \
  --save_dataset ./data/random_rollouts.h5 \
  --save_metrics ./logs/random_eval.json \
  --save_gif ./out/gifs/random_demo.gif
```

```bash
conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy heuristic \
  --num_episodes 3 \
  --save_dataset ./data/heuristic_rollouts.h5 \
  --save_metrics ./logs/heuristic_eval.json \
  --save_gif ./out/gifs/heuristic_demo.gif
```

```bash
conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy replay \
  --replay_dataset ./data/heuristic_rollouts.h5 \
  --num_episodes 3 \
  --save_dataset ./data/replay_from_heuristic_rollouts.h5 \
  --save_metrics ./logs/replay_from_heuristic_eval.json \
  --save_gif ./out/gifs/replay_from_heuristic_demo.gif
```

And these files exist:

```text
out/gifs/random_demo.gif
out/gifs/heuristic_demo.gif
out/gifs/replay_from_heuristic_demo.gif

logs/random_eval.json
logs/heuristic_eval.json
logs/replay_from_heuristic_eval.json

data/random_rollouts.h5
data/heuristic_rollouts.h5
data/replay_from_heuristic_rollouts.h5
```

Optional full test command for the demo slice:

```bash
conda run -n isaac_arm python -m pytest \
  tests/test_project_scaffold.py \
  tests/test_task_contract.py \
  tests/test_observation_wrapper.py \
  tests/test_demo_policies.py \
  tests/test_demo_dataset.py \
  tests/test_eval_metrics.py \
  tests/test_visual_outputs.py \
  tests/test_demo_data_loop.py \
  -v
```

---

## 15. Interview Walkthrough

Suggested explanation order:

1. Problem framing  
   Before training a large policy, I built the data and evaluation loop first.

2. Interface  
   The robot interface is 7D continuous control: 6D end-effector delta plus 1D gripper.

3. Observation  
   Every method consumes the same image-proprio observation dictionary.

4. Rollout dataset  
   Rollouts are stored by episode, so future action chunks never cross episode boundaries.

5. Metrics  
   I compute return, success, episode length, and action jerk.

6. GIF  
   I record visual outputs because robotics failures are often easiest to debug visually.

7. Next step  
   SAC can plug in as the expert oracle, then Diffusion Policy BC and DAgger can use the same dataset and evaluation interfaces.

One-sentence summary:

> This demo shows a robotics data/evaluation loop first; trained SAC and Diffusion policies are drop-in replacements for the same policy interface.
