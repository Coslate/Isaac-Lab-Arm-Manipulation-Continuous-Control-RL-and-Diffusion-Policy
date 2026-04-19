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
| PR 2 | Formal Isaac Lab observation wrapper | Done | Adds a tested formal Isaac Lab adapter contract for `Isaac-Lift-Cube-Franka-IK-Rel-v0`; local unit tests use an injected Gymnasium test double. The **2026-04-17 no-camera env.reset()/env.step() confirmation was a stock Isaac runtime smoke**, not a proof that the wrapper has live image-proprio observations yet. |
| Runtime env | WSL2 Xvfb fix | Done | `_app.update()` deadlock resolved. No-camera smoke test passes (`status: ok`, reward on `cuda:0`). See CLAUDE.md WSL2 section. |
| Runtime env | Camera renderer boot | Done | Camera mode boots in the current `isaac_arm` Isaac Sim / Isaac Lab runtime when run with GPU access. `gym.make()`, `env.reset()`, `env.step()`, and `env.close()` succeed with `--enable-cameras`. |
| Runtime env | Stock task camera observation | Blocked | Confirmed 2026-04-19: stock `Isaac-Lift-Cube-Franka-IK-Rel-v0` still exposes only `policy` obs with shape `(num_envs, 35)` in camera mode. `--enable-cameras` enables rendering but does not add RGB observation terms. |
| PR 2.5 | Camera-enabled Franka lift cfg | Pending | Keep the same IK-relative lift task, but customize `env_cfg` before `gym.make()` to add `wrist_cam`, optional debug camera, and named 40D proprio terms. |
| PR 8-lite | Rollout dataset | Pending | Store rollouts by episode so future action chunks never cross episode boundaries. |
| PR 11-lite | Evaluation metrics | Pending | Compute return, success, episode length, and action jerk. |
| PR 12-lite | GIF output | Pending | Save visual rollout GIFs from a fixed debug camera, while policy/dataset images come from wrist camera. |
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
- PR 2.5 Camera-enabled Franka lift cfg
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

Important scope correction from live testing:
- Do not switch to a different task just because another Isaac Lab task already has camera observations.
- Keep `Isaac-Lift-Cube-Franka-IK-Rel-v0` and the 7D IK-relative action contract.
- Add camera and named observation terms by customizing the Isaac Lab env config.
- Treat stock 35D `policy` observations as a diagnostic only, not as the formal project contract.
- Use wrist RGB as the learning image. Use a fixed table/front camera only for GIF/debug output.

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

Create a stable Python wrapper contract around the intended Isaac Lab task:

```text
Isaac-Lift-Cube-Franka-IK-Rel-v0
```

PR2 is the local/unit-tested adapter layer. It defines how project code calls `reset()`, `step()`, and receives observations. It does **not** create Isaac camera sensors, mutate Isaac Lab `env_cfg`, or prove live camera observations. Those live environment changes belong to PR2.5.

**Observation Contract**

```python
obs = {
    "image": np.ndarray,    # policy image, shape: (num_envs, 3, 84, 84), dtype uint8
    "proprio": np.ndarray,  # shape: (num_envs, 40), dtype float32
}
```

PR2's unit tests may inject this observation with a Gymnasium-compatible test double. The test double is not a project backend and must not appear in user-facing demo commands.

Formal 40D proprio feature order:

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

Wrapper-owned derived terms:

```text
ee_to_cube = cube_pos_base - ee_pos_base
cube_to_target = target_pos_base - cube_pos_base
```

Expected named low-dimensional inputs:

```text
arm_joint_pos_rel
arm_joint_vel_rel
gripper_finger_pos
gripper_finger_vel
ee_pos_base
cube_pos_base or object_position
target_pos_base or target_object_position already reduced to 3D
previous_action or actions
```

PR2 should reject a flat stock 35D observation by shape. Converting stock 35D into the final live `image + 40D` contract is not PR2's job; PR2.5 must make the Isaac cfg expose the needed image and named terms.

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
- Batched reset/step preserves the configured `num_envs`.
- `render()` and `close()` forward to the underlying environment when available.

**Boundary With PR2.5**

PR2.5 owns the live Isaac details that PR2 intentionally does not solve:

```text
camera sensors
policy wrist camera selection
debug camera separation
7D target_object_position -> 3D target_pos_base
custom Isaac Lab env_cfg
dedicated live camera observation smoke
```

**Suggested Commit**

```bash
git commit -m "feat(env): add Isaac Lab image-proprio wrapper"
```

---

## 7.5 PR 2.5 - Camera-Enabled Franka Lift Config

**Status:** Pending

**Goal / Why**

Make the live Isaac task satisfy the tested image-proprio wrapper contract. Keep the same task and action interface:

```text
env_id = Isaac-Lift-Cube-Franka-IK-Rel-v0
action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

but customize the Isaac Lab `env_cfg` before `gym.make()` so the scene includes a policy wrist camera, optional debug camera, and named observation terms for the 40D proprio vector.

**Camera Design**

Policy camera:

```text
name: wrist_cam
mount: near Robot/panda_hand / end-effector
purpose: policy input, rollout dataset image, future Diffusion Policy image
wrapper field: obs["image"]
shape after wrapper: (num_envs, 3, 84, 84), uint8
```

Debug camera:

```text
name: table_cam or front_cam
mount: fixed in the workcell, not attached to robot
purpose: human inspection, GIF recording, failure diagnosis
wrapper access: get_debug_frame("table_cam") or render_debug()
not used by: policy.act(), SAC replay buffer, Diffusion Policy dataset image
```

The fixed debug camera is allowed because it is an experiment recorder, not the robot policy's eye. The robot-learning observation remains eye-in-hand wrist RGB plus proprio/task-state features.

**Scope Clarification**

PR2.5 includes all work needed to move from the current unit-tested wrapper contract to a live camera-capable Isaac observation contract. In other words, the next seven steps belong to PR2.5:

1. Fix `target_object_position` handling so a 7D stock target pose becomes a 3D `target_pos_base`.
2. Make policy image extraction explicitly use the wrist policy camera, not any generic/debug camera key.
3. Add the camera-enabled Franka lift cfg helper.
4. Wire `IsaacArmEnv` to use that helper when camera observations are enabled.
5. Add PR2.5 cfg/wrapper unit tests.
6. Add a dedicated camera observation smoke script.
7. Run the live camera observation smoke and require `image + 40D proprio`, not stock `policy: (1, 35)`.

This means PR2.5 is not just "add a camera object." It is the acceptance gate for the real live observation contract:

```text
obs["image"]   = wrist camera RGB, shape (num_envs, 3, 84, 84), uint8
obs["proprio"] = named 40D contract, shape (num_envs, 40), float32
```

**Implementation**

- Fix the PR2 wrapper edge case before or inside this PR:
  - stock `target_object_position` is 7D: 3D target position plus 4D quaternion;
  - formal `target_pos_base` must use only `target_object_position[:, :3]`;
  - add a regression test where the fake Isaac observation uses a 7D target pose and the wrapper still returns `(num_envs, 40)` proprio.
- Make policy image extraction strict:
  - keep sensor names and observation keys distinct:
    - `wrist_cam` is the Isaac scene camera sensor;
    - `wrist_rgb` is the observation term / key mapped to wrapper `obs["image"]`;
    - `table_cam` is the fixed debug camera sensor;
    - `table_rgb` is the optional debug image term / key;
  - `obs["image"]` must come from `policy_image_obs_key`, defaulting to `wrist_rgb`;
  - do not accept `front_rgb`, `table_rgb`, or generic `camera` as a fallback for policy input;
  - expose debug camera frames only through `get_debug_frame(...)` or `render_debug(...)`.
- Add a helper such as `make_camera_enabled_franka_lift_cfg(...)` in `env/franka_lift_camera_cfg.py`.
- Internally call `parse_env_cfg(ISAAC_FRANKA_IK_REL_ENV_ID, device=device, num_envs=num_envs)`.
- Add `wrist_cam` using Isaac Lab `CameraCfg` or `TiledCameraCfg` with `data_types=["rgb"]` or `["rgb", "distance_to_image_plane"]`.
- Add optional fixed `table_cam` / `front_cam` for debug frames.
- Replace or extend the observation config so policy observations are non-concatenated named terms.
- Add `wrist_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False})`.
- Add named low-dimensional terms listed in PR2 instead of relying on stock flat 35D.
- Keep `policy_camera_name="wrist_cam"`, `policy_image_obs_key="wrist_rgb"`, `debug_camera_name="table_cam"`, and `debug_image_obs_key="table_rgb"` configurable.
- Keep debug camera out of the formal `obs["image"]` contract.
- Save sample wrist/debug frames during live smoke for manual inspection.

Keep `env/isaac_env.py` as the thin integration layer:

```python
if enable_cameras:
    env_cfg = make_camera_enabled_franka_lift_cfg(...)
else:
    env_cfg = parse_env_cfg(...)

env = gym.make(ISAAC_FRANKA_IK_REL_ENV_ID, cfg=env_cfg)
```

Do not place all camera sensor creation, observation-term replacement, and frame validation directly inside `IsaacArmEnv.__init__`.

**What To Test**

- Unit test 7D `target_object_position` is sliced to 3D `target_pos_base`.
- Unit test that when `policy_camera_name` / `policy_image_obs_key` are configured, generic or debug image keys are not silently used as policy `obs["image"]`.
- Unit test config helper sets the policy and debug camera names.
- Unit test wrapper maps `wrist_rgb` to `obs["image"]`.
- Unit test debug frames are accessible separately and are not returned as policy `obs["image"]`.
- Unit test 40D proprio assembly from named terms.
- Live test camera observation:

```bash
timeout 240s conda run -n isaac_arm python -m scripts.isaac_runtime_smoke \
  --device cuda:0 \
  --headless \
  --enable-cameras \
  --steps 1
```

plus a dedicated wrapper smoke proving:

```text
obs["image"].shape == (1, 3, 84, 84)
obs["image"].dtype == uint8
obs["proprio"].shape == (1, 40)
wrist image pixel variance > 0
sample wrist frame saved
sample debug frame saved, if debug camera is enabled
```

The old stock runtime smoke is still useful but is not a PR2.5 pass condition:

```text
stock result: observation.policy.shape == (1, 35)
```

PR2.5 passes only when the customized cfg plus wrapper returns:

```text
obs["image"].shape == (1, 3, 84, 84)
obs["proprio"].shape == (1, 40)
```

**Suggested Files**

```text
env/franka_lift_camera_cfg.py
env/isaac_env.py
tests/test_camera_enabled_env_cfg.py
scripts/isaac_camera_observation_smoke.py
```

**Suggested Commit**

```bash
git commit -m "feat(env): add camera-enabled Franka lift cfg"
```

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
  images        (T, 3, 84, 84) uint8    # source = wrist_cam (policy_camera_name); training image stream
  proprios      (T, proprio_dim) float32
  actions       (T, 7) float32
  rewards       (T,) float32
  dones         (T,) bool
  truncateds    (T,) bool
  debug_images  (optional) (T, H, W, 3) uint8  # source = table_cam (debug_camera_name); human/GIF only
  metadata/
    policy_name
    env_backend
    policy_camera_name
    policy_image_obs_key
    debug_camera_name
    debug_image_obs_key
    action_dim
    proprio_dim
    seed
```

Camera-source tagging is a hard rule:

- `images` **must** come from the wrist policy camera sensor (`policy_camera_name`, default `wrist_cam`) through the policy image observation key (`policy_image_obs_key`, default `wrist_rgb`). This is the stream a training loader reads.
- `debug_images` **must** come from the fixed debug camera (`debug_camera_name`, default `table_cam`). These are optional and never touch the training loader.
- Do not write fixed-camera frames under the `images` key "because the wrist frame is hard to see." Debugging ease is what `debug_images` exists for.
- The `policy_camera_name` / `policy_image_obs_key` and `debug_camera_name` / `debug_image_obs_key` metadata entries let a reader verify which camera produced each stream without re-running the collector.

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
- Metadata includes policy name, backend, seed, action dim, proprio dim, and policy camera name.
- If debug images are stored, they are under `debug_images` and not used by the training loader by default.
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
rollout debug frames -> save GIF -> optionally overlay policy/return/success/jerk
```

GIFs should use the fixed debug camera by default. The policy wrist camera is useful for learning and diagnostics, but it is narrow, moving, and may be hard for humans to interpret in a demo. The debug camera is the human-facing recorder.

Recommended wrapper usage:

```python
obs = env.reset()
frames = []
for _ in range(max_steps):
    action = policy.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.get_debug_frame("table_cam"))
save_gif(frames, out_path)
```

The debug camera must not be passed to `policy.act()` and must not replace `obs["image"]`.

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
- GIF recorder can consume frames from `env.get_debug_frame(...)`.
- Policy observations remain wrist-camera images while GIF frames come from the debug camera.

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
3. Camera-enabled Franka lift cfg with wrist policy camera and debug camera.
4. Random and heuristic policies.
5. Metrics JSON.
6. GIF output from debug camera.
7. Episode dataset with wrist policy images.
8. Replay policy.
9. Tests.

If time is tight:

- GIF and metrics are mandatory for the interview.
- Dataset can be simplified, but keep episode-safe design.
- Replay policy is nice to have.
- The final interview-facing demo runs against live Isaac. Unit-level PRs (PR0/PR1/PR2) intentionally use a Gymnasium test double injected through `gym_make` — that is constructor injection, not `unittest.mock`, and it is only for wrapper/contract tests. "No-mock" here means the demo rollout itself does not substitute a fake env for the live Isaac task; it does not mean every test in the slice avoids test doubles.
- The demo is not ready if live Isaac only returns stock `policy: (num_envs, 35)` observations.
- The demo is ready only when the customized cfg (PR2.5) produces wrist RGB plus 40D proprio.

---

## 14. Acceptance Criteria For Interview Demo

**Dependency note:** every acceptance criterion in this section assumes PR2.5 (Camera-Enabled Franka Lift Config) is merged. PR2.5 is currently Pending per Section 2; until it lands, the "runtime observation acceptance" block below cannot be satisfied because live Isaac only returns `policy: (num_envs, 35)` and no camera image term. Do not mark the demo accepted on the strength of PR0/PR1/PR2 passing alone.

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

Runtime observation acceptance:

```text
policy observation image = wrist_cam RGB, stored as dataset images
policy observation proprio = 40D named feature contract
GIF frames = fixed debug camera frames
debug camera = not passed to policy.act()
```

Optional full test command for the demo slice:

```bash
conda run -n isaac_arm python -m pytest \
  tests/test_project_scaffold.py \
  tests/test_task_contract.py \
  tests/test_observation_wrapper.py \
  tests/test_camera_enabled_env_cfg.py \
  tests/test_demo_policies.py \
  tests/test_demo_dataset.py \
  tests/test_eval_metrics.py \
  tests/test_visual_outputs.py \
  tests/test_demo_data_loop.py \
  -v
```

Of the demo-slice test files listed in the command above, only `test_project_scaffold.py`, `test_task_contract.py`, and `test_observation_wrapper.py` exist today. The repository also has separate Isaac installation/runtime smoke tests. The other six demo-slice files are created by their owning PRs (PR2.5 and the lite PRs above). The command is the full intended invocation after the demo slice is implemented; running it in the current tree will error on missing files.

---

## 15. Interview Walkthrough

Suggested explanation order:

1. Problem framing  
   Before training a large policy, I built the data and evaluation loop first.

2. Interface  
   The robot interface is 7D continuous control: 6D end-effector delta plus 1D gripper.

3. Observation  
   Every method consumes the same wrist-image plus 40D proprio observation dictionary. The stock 35D Isaac policy tensor is only a diagnostic baseline.

4. Rollout dataset  
   Rollouts are stored by episode, so future action chunks never cross episode boundaries.

5. Metrics  
   I compute return, success, episode length, and action jerk.

6. GIF  
   I record visual outputs from a fixed debug camera because robotics failures are often easiest to debug visually. The debug camera is not the policy input.

7. Next step  
   SAC can plug in as the expert oracle, then Diffusion Policy BC and DAgger can use the same dataset and evaluation interfaces.

One-sentence summary:

> This demo shows a robotics data/evaluation loop first; trained SAC and Diffusion policies are drop-in replacements for the same policy interface.
