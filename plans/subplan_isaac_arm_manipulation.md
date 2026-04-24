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

### Runtime env for live Isaac commands

Every live Isaac run (`scripts.collect_rollouts`, `scripts.benchmark_rollout_collection`, `scripts.isaac_camera_observation_smoke`, `scripts.isaac_runtime_smoke`) needs a working X11 display **plus** an X authority cookie. Isaac Sim's RTX / Vulkan WSI layer opens an X display surface to enumerate the GPU even with `--headless`. Missing cookie → cascade of `Authorization required`, `eglInitialize failed`, `xcb_connection_has_error()`.

**vast.ai bare-metal box (primary runtime):**

Recommend to put following two lines in ~/.bashrc

```bash
export DISPLAY=:0
export XAUTHORITY="/var/run/sddm/$(ls /var/run/sddm/ | head -1)"   # one-time per shell session
```

Put both lines in your shell rc or run them before the first Isaac command. `ls /var/run/sddm/` shows the curly-braced cookie file that SDDM creates at boot. `scripts.benchmark_rollout_collection` auto-discovers that cookie if `XAUTHORITY` is unset in the shell, so the benchmark works without the `export`; the other scripts inherit straight from the shell env and need the `export` explicitly. `--xauthority PATH` on the benchmark overrides everything.

**WSL2 + Docker (secondary):**

```bash
pkill Xvfb 2>/dev/null; Xvfb :1 -screen 0 1280x720x24 &
export DISPLAY=:1
```

Isaac scripts already set `DISPLAY=:1` automatically when it is not present, so the `export` is usually redundant here. `XAUTHORITY` is not needed on Xvfb.

### Status table

| Demo PR | Name | Status | Notes |
|---|---|---|---|
| PR 0 | Project scaffold | Done | Package layout, config, reproducibility utilities, output directory helpers, and scaffold tests are already implemented. |
| PR 1 | Task/action contract | Done | Defines the tested 7D Franka IK-relative action contract, clipping helper, action splitter, and gripper open/close rule. |
| PR 2 | Formal Isaac Lab observation wrapper | Done | Adds a tested formal Isaac Lab adapter contract for `Isaac-Lift-Cube-Franka-IK-Rel-v0`; local unit tests use an injected Gymnasium test double. The **2026-04-17 no-camera env.reset()/env.step() confirmation was a stock Isaac runtime smoke**, not a proof that the wrapper has live image-proprio observations yet. |
| Runtime env | WSL2 Xvfb fix | Done | `_app.update()` deadlock resolved. No-camera smoke test passes (`status: ok`, reward on `cuda:0`). See CLAUDE.md WSL2 section. |
| Runtime env | Camera renderer boot | Done | Camera mode boots in the current `isaac_arm` Isaac Sim / Isaac Lab runtime when run with GPU access. `gym.make()`, `env.reset()`, `env.step()`, and `env.close()` succeed with `--enable-cameras`. |
| Runtime env | Stock task camera observation | Blocked | Confirmed 2026-04-19: stock `Isaac-Lift-Cube-Franka-IK-Rel-v0` still exposes only `policy` obs with shape `(num_envs, 35)` in camera mode. `--enable-cameras` enables rendering but does not add RGB observation terms. |
| PR 2.5 | Camera-enabled Franka lift cfg | Done | Keeps the same IK-relative lift task, customizes `env_cfg` before `gym.make()` to add `wrist_cam`, optional `table_cam`, named 40D proprio terms, and has a live camera smoke result. |
| Image aug | `utils/image_aug.py` | Done | Three-tier contract: wrapper = deterministic resize; training = `PadAndRandomCrop` (primary) or `CenterBiasedResizedCrop` (alternative); eval/GIF/smoke = `IdentityAug`. Utility layer and tests are done; wiring the aug into future trainers/loaders is owned by later training PRs. |
| PR 8-pre | Demo policies | Done | Adds the lightweight demo policy interface plus RandomPolicy and HeuristicPolicy so rollout collection has policies to call. HDF5-backed ReplayPolicy is implemented in the Demo PR now that the PR8-lite episode dataset schema exists. |
| PR 8-lite | Rollout dataset | Done | Stores rollouts by episode in HDF5, supports parallel env rollout collection by splitting each env into its own episode group, keeps resized wrist policy images separate from optional native-resolution wrist images and optional debug images, provides action-window sampling that never crosses done/truncated boundaries, and includes dataset inspection plus rollout-throughput benchmark helpers. Refactor 2026-04-24: renamed `--num-envs` → `--num-parallel-envs` (alias kept), metadata now records `reset_round` / `reset_seed` / `terminated_by` / `settle_steps`, per-lane collection no longer force-truncates sibling lanes when one lane ends early, CLI collection shows a tqdm episode progress bar by default (`--no-progress` disables it), and `--settle-steps` can skip reset-settling physics before recording dataset transitions. |
| PR 11-lite | Evaluation metrics | Done | `eval/eval_loop.py` computes return, project-level success, episode length, action jerk, and target XYZ metadata from episode-safe rollout HDF5 files. Success uses stored `info["success"]` / `info["is_success"]` flags when present, otherwise falls back to the 40D proprio `cube_to_target` threshold because the stock Lift task does not expose an active success signal today. |
| PR 12-lite | Visual rollout output | Done | `eval/gif_recorder.py` saves fixed-debug-camera GIFs, optional MP4s, sampled debug PNGs, and optional text overlays. `record_debug_gif()` drives `env + policy` while keeping policy observations on wrist images and visual frames on the debug camera. |
| Demo PR | One-command script | Done | `scripts.demo_data_loop` creates dataset, metrics JSON, GIF, and sampled debug PNGs in one command. Supports random, heuristic, HDF5 replay policies, and optional `--settle-steps` zero-action reset warmup; live Isaac is the interview path and `--backend fake` is kept for fast unit tests/sample artifacts. |

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
18 passed
```

PR2.5 verification commands:

```bash
conda run -n isaac_arm python -m pytest tests/test_camera_enabled_env_cfg.py -v
timeout 360s conda run -n isaac_arm python -m scripts.isaac_camera_observation_smoke --steps 1 --output-dir out/camera_smoke
timeout 360s env OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y DISPLAY=:0 \
  /root/miniconda3/bin/conda run -n isaac_arm python -m scripts.isaac_camera_observation_smoke \
  --num-envs 2 \
  --steps 1 \
  --table-cleanup matte-overlay \
  --min-clean-env-spacing 5.0 \
  --output-dir out/camera_smoke_clean \
  --device cuda:0
```

Known PR2.5 result:

```text
tests/test_camera_enabled_env_cfg.py: 12 passed
live camera smoke: status ok, image (1, 3, 224, 224) uint8, proprio (1, 40) float32
clean live camera smoke: status ok, num_envs 2, --table-cleanup matte-overlay, --min-clean-env-spacing 5.0, saved out/camera_smoke_clean/debug_rgb.png without cloned-neighbor arms
clean rollout probe: --table-cleanup matte-overlay, max_steps 3, episode_005/009 step002 saved from data/random_rollouts_10eps_numenvs2_raw_debug_overlay_probe.h5 without the red table visual artifact
```

Current full local test result:

```text
conda run -n isaac_arm python -m pytest -q
144 passed, 1 skipped
skipped: tests/test_isaac_runtime_smoke.py requires RUN_ISAAC_RUNTIME_SMOKE=1 to launch Isaac Sim / Isaac Lab
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
- PR 8-pre Demo policies
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
  --settle-steps 20 \
  --target-overlay text-reticle \
  --save_dataset ./data/heuristic_rollouts.h5 \
  --save_metrics ./logs/heuristic_eval.json \
  --save_gif ./out/gifs/heuristic_demo.gif \
  --save_mp4 ./out/gifs/heuristic_demo.mp4
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
    "image": np.ndarray,    # policy image, shape: (num_envs, 3, 224, 224), dtype uint8
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
- Image shape is `(num_envs, 3, 224, 224)` and dtype is `uint8`.
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

**Status:** Done

Implemented 2026-04-19:

- Added `env/franka_lift_camera_cfg.py` to build the camera-enabled Franka lift cfg from the stock `Isaac-Lift-Cube-Franka-IK-Rel-v0` config.
- Added wrist policy camera `wrist_cam`, policy RGB term `wrist_rgb`, optional debug camera `table_cam`, and debug RGB term `table_rgb`.
- Replaced the stock concatenated 35D policy obs with non-concatenated named terms for the 40D project proprio contract.
- Added 7D `target_object_position` / command pose slicing so only the XYZ target position contributes to `target_pos_base`.
- Made `IsaacArmEnv` strict about policy image source: `obs["image"]` comes from `policy_image_obs_key` only, not from debug or generic camera keys.
- Added `get_debug_frame(...)` for human-facing debug images, separate from policy input.
- Added a torch action bridge for the live Isaac backend while keeping numpy actions for injected unit-test envs.
- Disabled debug-only visualizers before adding cameras: the stock Lift `object_pose` command starts with `debug_vis=True`, so the helper turns off command and first-level scene debug visualizers by default. This keeps helper markers out of `images`, `raw_policy_images`, and `debug_images`.
- Added opt-in table visual cleanup through `table_cleanup` / `--table-cleanup` with enum values `none`, `matte`, `overlay`, and `matte-overlay`. The default remains `none`, so camera streams preserve the stock Isaac scene render apart from disabled debug visualizers. `--clean-demo-scene` is kept as shorthand for the recommended clean mode (`matte-overlay` plus the default `--min-clean-env-spacing 5.0`). A live stage check confirmed the remaining red patch was not `/Visuals/Command/*`; it came from the SeattleLabTable visual surface. `matte` changes the table material, `overlay` adds a visual-only matte tabletop cover without changing physics, and `matte-overlay` applies both.
- Added `tests/test_camera_enabled_env_cfg.py`, PR2.5 wrapper regressions, and `scripts/isaac_camera_observation_smoke.py`.
- Live camera smoke passed in `isaac_arm` with GPU access:

```text
Observation Manager policy terms:
wrist_rgb, arm_joint_pos_rel, arm_joint_vel_rel, gripper_finger_pos,
gripper_finger_vel, ee_pos_base, cube_pos_base, target_pos_base, previous_action

obs["image"]:   (1, 3, 224, 224), uint8, nonzero variance
obs["proprio"]: (1, 40), float32
debug table_rgb: (720, 1280, 3), uint8
saved frames: out/camera_smoke/wrist_policy_rgb.png, out/camera_smoke/debug_rgb.png
```

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
shape after wrapper: (num_envs, 3, 224, 224), uint8
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

Demo/debug frames should keep debug-only markers out of camera streams by default: no command target frame markers and no scene sensor debug visualizers in `images`, `raw_policy_images`, or `debug_images`. The red table-surface visual mark from the stock SeattleLabTable asset and wider cloned-lane spacing are opt-in cleanup behavior, enabled with `table_cleanup != "none"` / `--table-cleanup`, so the default camera output remains the closest stock Isaac render. The recommended clean-demo mode is `--table-cleanup matte-overlay --min-clean-env-spacing 5.0`; `--clean-demo-scene` remains a shorthand for that default-clean path.

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
obs["image"]   = wrist camera RGB, shape (num_envs, 3, 224, 224), uint8
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
- Disable command and scene debug visualizers by default before camera observations are generated.
- Keep stock Isaac table visuals and cloned-lane spacing by default; expose `table_cleanup` / `--table-cleanup` and `min_clean_env_spacing` / `--min-clean-env-spacing` as the opt-in path for matte table material, tabletop overlay, and larger lane spacing.
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
obs["image"].shape == (1, 3, 224, 224)
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
obs["image"].shape == (1, 3, 224, 224)
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

## 7.6 Image Augmentation Utilities

**Status:** Done

Implemented in `utils/image_aug.py`. Augmentation is applied in the training pipeline only; the env wrapper stays deterministic.

Three-tier contract:

```text
Env wrapper     : native → deterministic resize → 224×224        (obs contract)
Training aug    : 224×224 → pad 8 px → random crop 224×224       (primary)
Eval/GIF/smoke  : no augmentation
```

**Primary: `PadAndRandomCrop`**

DrQ/RAD-style augmentation. Takes wrapper-output 224×224 images and applies a small random translation (pad 8 px → random crop back to 224×224). Safe for this task because the maximum pixel shift is ≤ 16 px, unlikely to remove gripper or cube.

**Alternative: `CenterBiasedResizedCrop`**

Takes native-resolution images (e.g. 400×400 from `get_policy_frame()`). Randomly samples a crop with scale ∈ [0.75, 1.0] biased toward image center, then resizes to 224×224. Requires the dataset to store native-resolution frames; if the dataset stores 224×224 (PR8-lite default), use `PadAndRandomCrop` instead.

**Eval: `IdentityAug`**

No-op. Used for eval, GIF recording, and smoke tests. Ensures the obs contract is deterministic and reproducible.

**Factory helpers:**

```python
from utils.image_aug import make_train_aug, make_eval_aug

train_aug = make_train_aug(mode="pad_crop", pad=8)        # primary
train_aug = make_train_aug(mode="resized_crop", min_scale=0.75)  # alternative
eval_aug  = make_eval_aug()
```

**Test command:**

```bash
pytest tests/test_image_aug.py -v
```

Known result: `24 passed`.

**Boundary:** this PR provides the reusable augmentation utilities and tests. It does not yet connect augmentation to an actual SAC/TD3/PPO/Diffusion Policy trainer or dataloader. That integration belongs to the future training/dataset PRs.

---

## 8. PR 8-pre - Demo Policies

**Status:** Done

Implemented in:

```text
policies/__init__.py
policies/base.py
policies/random_policy.py
policies/heuristic_policy.py
tests/test_demo_policies.py
```

Known result:

```text
conda run -n isaac_arm python -m pytest tests/test_demo_policies.py -v
10 passed
```

**Goal / Why**

Define the lightweight policy interface used by the demo collector and evaluator before implementing the episode dataset. This PR provides policies that can be called as:

```python
action = policy.act(obs)
```

The action contract remains the project-wide normalized 7D action:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

PR 8-pre owns the base policy interface, RandomPolicy, and HeuristicPolicy. HDF5-backed ReplayPolicy is implemented in the Demo PR because replay needs the finalized PR8-lite episode dataset schema.

### RandomPolicy

```python
action = np.random.uniform(-1.0, 1.0, size=(7,))
```

Purpose:

- Baseline.
- Verifies action handling.
- Verifies dataset/eval/GIF pipeline works.
- Serves as the first end-to-end policy once rollout collection exists.

### HeuristicPolicy

Simple rules:

```text
1. Move gripper toward cube.
2. When close, close gripper.
3. Lift upward.
```

Current implementation detail: the Isaac cube blocks the gripper fingers before the raw finger joints reach a fully closed empty-gripper value. The heuristic therefore treats the gripper as "closed around the cube" at about `0.025m` mean finger position, not `0.015m`; otherwise the policy can keep pushing/closing and never enter the lift branch during the 100-step demo.

The lift phase is a target-servo, not a blind upward push. After grasp:

- Far from target, command motion along `cube_to_target`.
- Within `target_slow_radius_m = 0.08`, scale the command down as the cube approaches the target to reduce overshoot.
- Within `target_hold_radius_m = 0.02`, hold translation at zero and keep the gripper closed. This matches the current success threshold.
- If the cube overshoots above or past the target, allow corrective motion back toward the target instead of forcing a minimum upward command.

Purpose:

- Sanity policy.
- Should be easier to interpret than random in the formal Isaac env.
- Produces a more meaningful GIF for interview discussion.

### ReplayPolicy

ReplayPolicy reads actions from a saved dataset and replays them. It is implemented in `policies/replay_policy.py` as part of the Demo PR.

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

ReplayPolicy file:

```text
policies/replay_policy.py
```

**Suggested Files**

```text
policies/__init__.py
policies/base.py
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

- Base policy interface documents `act(obs) -> action`.
- RandomPolicy returns shape `(7,)`.
- HeuristicPolicy returns shape `(7,)`.
- Actions are finite.
- Actions are clipped to `[-1, 1]`.
- RandomPolicy is deterministic when constructed with a fixed seed.
- Heuristic policy commands open when far from the cube.
- Heuristic policy eventually commands close when near the cube.
- Heuristic policy commands lift/target-servo motion after close/grasp conditions.
- Heuristic policy slows near the target, holds inside the 2cm success radius, and can correct downward after target overshoot.

**Suggested Commit**

```bash
git commit -m "feat(policy): add random and heuristic demo policies"
```

---

## 9. PR 8-lite - Rollout Dataset

**Status:** Done

Implemented in:

```text
dataset/episode_dataset.py
scripts/collect_rollouts.py
scripts/inspect_rollout_dataset.py
scripts/benchmark_rollout_collection.py
tests/test_demo_dataset.py
tests/test_rollout_benchmark.py
```

Known result:

```text
conda run -n isaac_arm python -m pytest tests/test_demo_dataset.py -v
18 passed
conda run -n isaac_arm python -m pytest tests/test_rollout_benchmark.py -v
7 passed
```
(13 original + 1 reset_round/reset_seed round-trip + 1 per-lane staggered termination; plus 4 benchmark command/CSV plumbing tests + 3 `_subprocess_env` XAUTHORITY auto-discovery / override / preservation tests added 2026-04-24)

**Depends On**

- PR 8-pre demo policy interface.
- The collector assumes policies expose `act(obs) -> action`.
- `tqdm>=4.66` is a direct project dependency for CLI progress bars. The collector has a no-op fallback if `tqdm` is unavailable, but the restored project env should include it.

**Goal / Why**

Save rollout data in an episode-safe format. This is the most DeepReach-relevant part of the demo because it shows data infrastructure, not only model code.

PR8-lite supports both the interview-demo default (`--num-parallel-envs 1`) and the full-plan direction (`--num-parallel-envs > 1`). The old `--num-envs` flag is still accepted as an alias, but the clearer CLI name is `--num-parallel-envs` because this value controls Isaac Lab vectorized-physics lanes, not the total number of trajectories to collect.

For vectorized Isaac runs, the collector does **not** store a single mixed `(T, N, ...)` episode. It splits each environment lane into separate HDF5 episode groups:

```text
parallel env batch
  env 0 -> episode_000
  env 1 -> episode_001
  env 2 -> episode_002
```

Each episode metadata includes `source_env_index`, `reset_round`, `reset_seed`, and `terminated_by` so later inspection can trace exactly which `(reset round, vectorized lane)` produced the episode and why it ended. The policy interface remains `policy.act(single_env_obs) -> (7,)`; the collector applies it per env and stacks actions back into `(num_parallel_envs, 7)` before one vectorized `env.step(...)`.

The CLI enables a `tqdm` episode progress bar by default. It advances whenever an episode group is appended, regardless of whether the episode ended through env `done`, env `truncated`, or the collector's `--max-steps` cap. Pass `--no-progress` for cleaner logs in scripts or CI. Direct Python calls to `collect_rollout_episodes(...)` default to `show_progress=False` so unit tests stay quiet.

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
  images        (T, 3, 224, 224) uint8    # source = wrist_cam (policy_camera_name); training image stream
  raw_policy_images (optional) (T, H, W, 3) uint8  # native wrist_cam frames, e.g. 400x400; debug/inspection only
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
    seed               # legacy field, mirrors reset_seed
    source_env_index   # vectorized-env lane (0..num_parallel_envs-1) that produced this episode
    reset_round        # which env.reset() round produced this episode (0-indexed)
    reset_seed         # seed actually passed to env.reset(seed=...) for this round
    terminated_by      # "done", "truncated", or collector-side "max_steps"
    settle_steps       # zero-action warmup steps run after explicit reset before recording
    clean_demo_scene   # whether opt-in visual table/spacing cleanup was enabled during collection
    table_cleanup      # "none", "matte", "overlay", or "matte-overlay"
    min_clean_env_spacing
```

Generate rollout:

```bash
timeout 360s env OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.collect_rollouts \
  --backend isaac \
  --policy random \
  --num-parallel-envs 2 \
  --num-episodes 10 \
  --max-steps 100 \
  --settle-steps 20 \
  --include-raw-policy-images \
  --include-debug-images \
  --save-dataset data/random_rollouts_10eps_numenvs2_raw_debug.h5 \
  --seed 0 \
  --device cuda:0 \
  --progress

timeout 360s env OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.collect_rollouts \
  --backend isaac \
  --policy random \
  --num-parallel-envs 2 \
  --num-episodes 10 \
  --max-steps 100 \
  --settle-steps 20 \
  --include-raw-policy-images \
  --include-debug-images \
  --save-dataset data/random_rollouts_10eps_numenvs2_tc_matte.h5 \
  --seed 0 \
  --device cuda:0 \
  --table-cleanup matte \
  --progress

timeout 360s env OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.collect_rollouts \
  --backend isaac \
  --policy random \
  --num-parallel-envs 2 \
  --num-episodes 10 \
  --max-steps 100 \
  --settle-steps 20 \
  --include-raw-policy-images \
  --include-debug-images \
  --save-dataset data/random_rollouts_10eps_numenvs2_tc_matte_overlay.h5 \
  --seed 0 \
  --device cuda:0 \
  --table-cleanup matte-overlay \
  --progress
```

Use `--no-progress` instead of `--progress` when writing logs to files or running non-interactive CI.

Use `--settle-steps 20` for live Isaac datasets intended for training or demo review. The collector runs those zero-action steps after each explicit `env.reset(seed=...)`, then starts writing `images`, `proprios`, `actions`, rewards, and optional raw/debug images. The warmup transitions are not stored, and each episode metadata records `settle_steps`. The default remains `0` when you intentionally want the raw reset behavior.

The default collector disables Isaac debug visualizers but otherwise preserves the stock rendered scene for all three image streams (`images`, `raw_policy_images`, and `debug_images`). To opt into cleaner demo visuals that hide the stock table red mark and increase cloned-lane spacing, use the recommended enum mode:

```bash
  --table-cleanup matte-overlay \
  --min-clean-env-spacing 5.0
```

For targeted probes, `--table-cleanup matte` applies only the table material change and `--table-cleanup overlay` applies only the visual-only tabletop cover. `--clean-demo-scene` remains supported as a shorthand for `matte-overlay` with the default `5.0` minimum spacing.

Camera-source tagging is a hard rule:

- `images` **must** come from the wrist policy camera sensor (`policy_camera_name`, default `wrist_cam`) through the policy image observation key (`policy_image_obs_key`, default `wrist_rgb`). This is the stream a training loader reads.
- `raw_policy_images` are optional native-resolution wrist camera frames captured from `env.get_policy_frame()`. They exist for camera/debug inspection and must not replace the training image stream.
- `debug_images` **must** come from the fixed debug camera (`debug_camera_name`, default `table_cam`). These are optional and never touch the training loader.
- Do not write fixed-camera frames under the `images` key "because the wrist frame is hard to see." Debugging ease is what `debug_images` exists for.
- The `policy_camera_name` / `policy_image_obs_key` and `debug_camera_name` / `debug_image_obs_key` metadata entries let a reader verify which camera produced each stream without re-running the collector.
- `source_env_index` records the vectorized Isaac env lane that produced the episode. This matters when `--num-parallel-envs > 1`.
- `reset_round` and `reset_seed` identify the `env.reset(seed=...)` call that started the episode.
- `terminated_by` records whether the lane ended through the env's `done`, the env's `truncated`, or the collector's `--max-steps` cap.
- `settle_steps` records how many zero-action warmup steps ran after the explicit reset before dataset recording started. Training loaders read only the stored post-settle transitions.
- The progress bar reports episode groups written, not environment steps. With `--num-parallel-envs 2`, a single vectorized reset round may advance the bar by two episodes when both lanes flush.
- `table_cleanup` is `none` by default. When it is `matte`, `overlay`, or `matte-overlay`, collection uses the opt-in visual cleanup path and applies `min_clean_env_spacing` (default `5.0`) unless that spacing is set to `none`.

Inspection helper:


inspecting for dataset summary：
```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.inspect_rollout_dataset \
  --dataset data/random_rollouts_10eps_numenvs2_raw_debug.h5
```

inspecting for 10th episode 3rd step
```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.inspect_rollout_dataset \
  --dataset data/random_rollouts_10eps_numenvs2_raw_debug.h5 \
  --episode episode_009 \
  --step 2 \
  --save-policy-frame out/debug_frames/random_10eps_numenvs2_ep009_step002_policy.png \
  --save-raw-policy-frame out/debug_frames/random_10eps_numenvs2_ep009_step002_raw_policy.png \
  --save-debug-frame out/debug_frames/random_10eps_numenvs2_ep009_step002_debug.png  

/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.inspect_rollout_dataset \
  --dataset data/random_rollouts_10eps_numenvs2_raw_debug.h5 \
  --episode episode_003 \
  --step 49 \
  --save-policy-frame out/debug_frames/random_10eps_numenvs2_ep003_step049_policy.png \
  --save-raw-policy-frame out/debug_frames/random_10eps_numenvs2_ep003_step049_raw_policy.png \
  --save-debug-frame out/debug_frames/random_10eps_numenvs2_ep003_step049_debug.png  

/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.inspect_rollout_dataset \
  --dataset data/random_rollouts_10eps_numenvs2_raw_debug.h5 \
  --episode episode_005 \
  --step 49 \
  --save-policy-frame out/debug_frames/random_10eps_numenvs2_ep005_step049_policy.png \
  --save-raw-policy-frame out/debug_frames/random_10eps_numenvs2_ep005_step049_raw_policy.png \
  --save-debug-frame out/debug_frames/random_10eps_numenvs2_ep005_step049_debug.png  

/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.inspect_rollout_dataset \
  --dataset data/random_rollouts_10eps_numenvs2_tc_matte.h5 \
  --episode episode_005 \
  --step 49 \
  --save-policy-frame out/debug_frames/random_10eps_numenvs2_ep005_step049_policy_tc_matte.png \
  --save-raw-policy-frame out/debug_frames/random_10eps_numenvs2_ep005_step049_raw_policy_tc_matte.png \
  --save-debug-frame out/debug_frames/random_10eps_numenvs2_ep005_step049_debug_tc_matte.png  

/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.inspect_rollout_dataset \
  --dataset data/random_rollouts_10eps_numenvs2_tc_matte_overlay.h5 \
  --episode episode_005 \
  --step 49 \
  --save-policy-frame out/debug_frames/random_10eps_numenvs2_ep005_step049_policy_tc_matte_overlay.png \
  --save-raw-policy-frame out/debug_frames/random_10eps_numenvs2_ep005_step049_raw_policy_tc_matte_overlay.png \
  --save-debug-frame out/debug_frames/random_10eps_numenvs2_ep005_step049_debug_tc_matte_overlay.png
```

This helper prints the HDF5 schema/metadata and can export single PNG frames from `images`, `raw_policy_images`, and `debug_images`. It is a dataset inspection tool, not the full GIF pipeline. PR12-lite owns GIF creation and broader visual rollout artifacts.

Rollout throughput benchmark helper (for X11 / DISPLAY setup see §2 "Runtime env for live Isaac commands"; `--display :1` below is the WSL2 form and should be dropped on vast.ai, which uses `DISPLAY=:0` + SDDM cookie auto-discovery):

```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.benchmark_rollout_collection \
  --parallel-envs 1,2 \
  --repeats 3 \
  --num-episodes 10 \
  --max-steps 100 \
  --policy random \
  --include-raw-policy-images \
  --include-debug-images \
  --output-csv logs/rollout_collection_benchmark.csv \
  --dataset-dir data/benchmark_rollouts \
  --device cuda:0 \
  --display :1
```

The benchmark script launches `scripts.collect_rollouts` in a fresh subprocess for each `(num_parallel_envs, repeat)` condition, so its `wall_time_s` is an end-to-end measurement that includes Isaac App startup, camera rendering, Python policy calls, and HDF5 writes. It writes one CSV row per run with `num_parallel_envs`, `wall_time_s`, `episodes_per_s`, `steps_per_s`, `actual_steps`, dataset size, status, and the exact command. Child collector progress is disabled by default for cleaner timing; pass `--collect-progress` only for interactive supervision.

Recommended benchmark protocol:

1. End-to-end demo benchmark with camera data enabled. This matches the interview data-loop workload because it includes wrist policy images, raw wrist frames, debug camera frames, and HDF5 writes.

```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.benchmark_rollout_collection \
  --parallel-envs 1,2,4 \
  --repeats 3 \
  --num-episodes 20 \
  --max-steps 100 \
  --policy random \
  --include-raw-policy-images \
  --include-debug-images \
  --output-csv logs/rollout_collection_benchmark_camera.csv \
  --dataset-dir data/benchmark_rollouts_camera \
  --device cuda:0 \
  --display :1
```

2. Physics/control-loop benchmark with large image storage disabled. This isolates vectorized rollout throughput more cleanly because it reduces raw/debug image I/O.

```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.benchmark_rollout_collection \
  --parallel-envs 1,2,4 \
  --repeats 3 \
  --num-episodes 20 \
  --max-steps 100 \
  --policy random \
  --no-include-raw-policy-images \
  --no-include-debug-images \
  --output-csv logs/rollout_collection_benchmark_no_aux_images.csv \
  --dataset-dir data/benchmark_rollouts_no_aux_images \
  --device cuda:0
```

Useful CSV fields:

```text
num_parallel_envs       vectorized Isaac env lane count for this run
repeat                  repeat index for averaging
wall_time_s             end-to-end subprocess wall time
episodes_written        number of HDF5 episode groups actually written
actual_steps            sum of per-episode action rows in the generated HDF5
episodes_per_s          episodes_written / wall_time_s
steps_per_s             actual_steps / wall_time_s
dataset_size_bytes      HDF5 output size; useful for spotting I/O-heavy runs
status / returncode     command success/failure
command                 exact child collect_rollouts command
```

To report the speed difference, compare the mean `episodes_per_s` or `steps_per_s` for `num_parallel_envs=2` against `num_parallel_envs=1` across repeats. The expected speedup is sublinear because the benchmark still includes fixed Isaac startup cost, camera rendering, HDF5 writes, and a Python loop that calls `policy.act(...)` once per lane.

Example interview wording:

> I benchmarked serial rollout collection against Isaac Lab vectorized collection by holding policy, episode count, max steps, camera streams, seed schedule, and device fixed while varying `num_parallel_envs`. The vectorized run improved rollout throughput, but the speedup was sublinear because camera rendering, HDF5 writes, and per-lane Python policy calls remain shared bottlenecks.

Known live raw/debug rollout check for the current collector contract:

```text
command: scripts.collect_rollouts with --num-parallel-envs 2, --include-raw-policy-images, and --include-debug-images
status: ok
clean_demo_scene: false
table_cleanup: none
min_clean_env_spacing: 5.0
dataset: data/random_rollouts_numenvs2_raw_debug.h5
num_episodes: 2
episode_000 source_env_index: 0
episode_001 source_env_index: 1
episode_000 reset_round/reset_seed: 0/0
episode_001 reset_round/reset_seed: 0/0
episode_000 terminated_by: max_steps
episode_001 terminated_by: max_steps
per-episode images: (2, 3, 224, 224) uint8
per-episode raw_policy_images: (2, 400, 400, 3) uint8
per-episode debug_images: (2, 720, 1280, 3) uint8
sample frames:
  out/debug_frames/random_numenvs2_ep000_step000_policy.png
  out/debug_frames/random_numenvs2_ep000_step000_raw_policy.png
  out/debug_frames/random_numenvs2_ep000_step000_debug.png
```

Note: historical HDF5 artifacts generated before the 2026-04-24 metadata refactor may omit `reset_round`, `reset_seed`, and `terminated_by`. Regenerate the dataset with the current collector before using those fields in an interview demo or analysis.

**Suggested Files**

```text
dataset/episode_dataset.py
scripts/collect_rollouts.py
scripts/inspect_rollout_dataset.py
scripts/benchmark_rollout_collection.py
tests/test_demo_dataset.py
tests/test_rollout_benchmark.py
```

**Test Command**

```bash
conda run -n isaac_arm python -m pytest tests/test_demo_dataset.py -v
```

**What To Test**

- Dataset file is created.
- Each episode has required keys.
- `actions` shape is `(T, 7)`.
- `images` shape is `(T, 3, 224, 224)`.
- If native wrist images are stored, they are under `raw_policy_images` and are omitted by the loader by default.
- `proprios` shape is `(T, proprio_dim)`.
- Metadata includes policy name, backend, seed/reset seed, reset round, termination reason, action dim, proprio dim, and policy camera name.
- Parallel env collection writes one episode group per env lane and records `source_env_index`.
- If debug images are stored, they are under `debug_images` and not used by the training loader by default.
- Action chunk sampling never crosses episode boundaries.
- Done/truncated flags terminate sampling windows correctly.
- Dataset inspection can summarize schema/metadata and export sample policy/raw/debug PNG frames.
- CLI collection shows progress by default and accepts `--no-progress`; direct collector calls keep progress disabled unless `show_progress=True`.
- Benchmark helper builds current collector commands, measures end-to-end wall time in subprocesses, summarizes generated HDF5 files, and writes CSV rows without launching Isaac in unit tests.

**Suggested Commit**

```bash
git commit -m "feat(data): add episode-safe rollout dataset"
```

---

## 10. PR 11-lite - Evaluation Metrics

**Status:** Done

**Goal / Why**

Quantify behavior even without a trained policy. The first comparison should be random vs heuristic.

**Metrics**

```text
mean_return
success_rate
mean_episode_length
mean_action_jerk
episode_successes
closest_target_approach_by_episode
```

Action jerk for the lightweight demo:

```python
jerk = mean(norm(action[t] - action[t - 1]))
```

This is a simple smoothness proxy. Later the full project can use higher-order jerk based on acceleration changes if needed.

Success for the lightweight demo has two sources, in priority order:

1. If the rollout collector receives `info["success"]` or `info["is_success"]`, it stores those per-step flags as an optional HDF5 `successes` dataset and PR11 uses them first.
2. If no explicit success flags exist, PR11 falls back to the camera-enabled wrapper's 40D proprio vector:

```python
cube_to_target = proprio[:, 30:33]
success = any(norm(cube_to_target[t]) <= 0.02 for t in episode)
```

This fallback distance is a 3D Euclidean point-to-point distance in the robot/base frame: `norm(proprio[:, 30:33])`, where the vector is `target_pos_base - cube_pos_base`.

The JSON records `success_source` as `"info_success"` when explicit flags are used, `"proprio_cube_to_target_norm"` when the fallback is used, or `"mixed_info_success_and_proprio_cube_to_target_norm"` for mixed old/new datasets. It also records `success_threshold_m = 0.02`, `consecutive_success_steps = 1`, `success_distance_metric = "euclidean_3d_cube_to_target_base_m"`, and `success_distance_source = "norm(proprio[:, 30:33]) where proprio[:, 30:33] = target_pos_base - cube_pos_base"`. A stricter future run can require multiple consecutive success hits without changing the dataset schema. The fallback remains important because `Isaac-Lift-Cube-Franka-IK-Rel-v0` has a helper termination for reaching the goal, but the stock task config does not wire it as an active `success` termination/info field today.

Per-episode diagnostics are now part of the metrics contract:

- `episode_successes`: map from `episode_000`, `episode_001`, ... to whether each episode succeeded under the active success source/threshold.
- `closest_target_approach_by_episode`: map from each episode key to the closest recorded approach to the target, including `closest_step`, `closest_distance_m`, `closest_cube_position_base_m`, `target_position_base_m_at_closest_step`, `cube_to_target_base_m_at_closest_step`, `success`, and `success_source`.

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
  "mean_action_jerk": 0.0,
  "success_threshold_m": 0.02,
  "consecutive_success_steps": 1,
  "success_source": "proprio_cube_to_target_norm",
  "success_distance_metric": "euclidean_3d_cube_to_target_base_m",
  "success_distance_source": "norm(proprio[:, 30:33]) where proprio[:, 30:33] = target_pos_base - cube_pos_base",
  "episode_successes": {
    "episode_000": false
  },
  "closest_target_approach_by_episode": {
    "episode_000": {
      "closest_step": 0,
      "closest_distance_m": 0.0,
      "closest_cube_position_base_m": [0.0, 0.0, 0.0],
      "target_position_base_m_at_closest_step": [0.0, 0.0, 0.0],
      "cube_to_target_base_m_at_closest_step": [0.0, 0.0, 0.0],
      "success": false,
      "success_source": "proprio_cube_to_target_norm"
    }
  }
}
```

Values above are placeholders. Tests should validate keys/types/ranges; live Isaac performance should be measured in an installed Isaac Lab runtime.

Current implementation files:

```text
eval/eval_loop.py
tests/test_eval_metrics.py
```

Verified:

```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m pytest tests/test_eval_metrics.py -q
/root/miniconda3/bin/conda run -n isaac_arm python -m pytest tests/test_demo_dataset.py tests/test_demo_policies.py -q
```

**Suggested Commit**

```bash
git commit -m "feat(eval): add rollout metrics for data-loop demo"
```

---

## 11. PR 12-lite - GIF Visual Output

**Status:** Done

**Goal / Why**

Create the most interview-friendly artifact. Robotics failures are often easier to understand visually than from scalar metrics.

**Pipeline**

```text
rollout debug frames -> save GIF + sampled debug PNGs -> optionally overlay policy/return/success/jerk/target
```

GIFs should use the fixed debug camera by default. The policy wrist camera is useful for learning and diagnostics, but it is narrow, moving, and may be hard for humans to interpret in a demo. The debug camera is the human-facing recorder.

In addition to GIFs, PR12-lite should save a small number of still debug images for quick inspection, for example:

```text
out/debug_frames/heuristic_ep000_step000_debug.png
out/debug_frames/heuristic_ep000_step025_debug.png
out/debug_frames/heuristic_ep000_step050_debug.png
```

These stills come from the same fixed debug camera stream as the GIF. They are separate from the wrist-camera `images` dataset used for policy learning.

Recommended wrapper usage:

```python
obs = env.reset()
frames = []
for _ in range(max_steps):
    action = policy.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.get_debug_frame("table_cam"))
save_gif(frames, gif_path)
save_mp4(frames, mp4_path)
```

The debug camera must not be passed to `policy.act()` and must not replace `obs["image"]`.

The env does provide the target position through the camera-enabled proprio contract: `target_pos_base` is stored in proprio slice `24:27`, and `cube_to_target` is stored in slice `30:33`. PR12-lite can mark the target in GIFs in two levels:

- Always available from saved datasets: overlay target XYZ, cube-to-target distance, and success state as text on debug frames.
- Exact 2D reticle on the debug-camera image: requires projecting the 3D target point with the debug camera intrinsics/extrinsics or recording that projection during live rollout. Do this as a post-processing overlay for GIF/debug output only; do not re-enable Isaac debug visualizers or inject target markers into policy images.

**Suggested Files**

```text
eval/gif_recorder.py
tests/test_visual_outputs.py
```

Current implementation:

- `save_gif(frames, out_path, fps=...)` writes RGB frames to a GIF and creates missing parent directories.
- `save_mp4(frames, out_path, fps=...)` writes RGB frames to an MP4 through `imageio` / `imageio-ffmpeg`, avoiding GIF's 256-color palette limit.
- `save_sampled_debug_frames(...)` writes evenly sampled debug PNGs such as `heuristic_ep000_step002_debug.png`.
- `record_debug_gif(env, policy, ...)` runs one visual rollout, calls `policy.act(obs)` with the normal wrist-camera observation, and records frames only from `env.get_debug_frame(debug_camera_name)`. When `mp4_output_path` is provided, the same frames/overlay are also saved as MP4.
- Optional text overlays are post-processing only; they do not re-enable Isaac debug visualizers and do not touch `obs["image"]`.

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
- Sampled debug PNG frames are saved from the same fixed debug camera stream.
- Policy observations remain wrist-camera images while GIF frames come from the debug camera.

Verified:

```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m pytest tests/test_visual_outputs.py -q
/root/miniconda3/bin/conda run -n isaac_arm python -m pytest tests/test_visual_outputs.py tests/test_observation_wrapper.py tests/test_demo_dataset.py tests/test_eval_metrics.py -q
```

**Suggested Commit**

```bash
git commit -m "feat(viz): add rollout GIF recording"
```

---

## 12. Demo PR - One-Command Data Loop

**Status:** Done

**Goal / Why**

One command should generate dataset, metrics, and GIF. This is the command to run before the interview and potentially screen-share.

Implemented files:

```text
scripts/demo_data_loop.py
policies/replay_policy.py
tests/test_demo_data_loop.py
```

The one-command loop:

- creates a rollout HDF5 dataset through the same collector used by PR8-lite;
- evaluates it with PR11-lite metrics and writes metrics JSON;
- records a fixed-debug-camera GIF through PR12-lite with 2x scaled text overlay for readability;
- saves sampled debug PNG stills next to the GIF when `--save-debug-frames-dir` is provided;
- supports `--policy random`, `--policy heuristic`, and `--policy replay --replay_dataset ...`;
- supports `--backend isaac` for the real demo and `--backend fake` only for fast tests/sample artifacts.
- supports `--settle-steps N`, which runs zero-action physics warmup after explicit `env.reset(...)` calls and before any dataset/GIF frame is recorded. `scripts.collect_rollouts` owns this for HDF5 dataset collection, and `scripts.demo_data_loop` applies the same behavior to GIF recording. The default is `0` to preserve raw behavior; the recommended live demo/training-data value is `20` so the cube settles on the table before the visible rollout starts.

Current GIF text overlay:

```text
step    current GIF frame index in the visual rollout
policy  policy used for the demo run: random, heuristic, or replay
return  mean_return from the dataset/eval rollout, i.e. mean episode reward sum
success success_rate from the dataset/eval rollout
jerk    mean_action_jerk from the dataset/eval rollout
```

Only `step` changes per GIF frame today. The other text fields summarize the metrics JSON generated before the GIF is recorded.

Current target metadata in the demo metrics JSON:

- `target_position_base_m`: convenience summary target XYZ in the robot/base frame, currently `episode_000` step 0 from `proprio[:, 24:27]`.
- `target_position_base_m_source`: exact provenance for the summary target, for example `episode_000_step_000_proprio_24_27`.
- `target_positions_base_m_by_episode`: first target XYZ per saved HDF5 episode.
- `target_position_constant_by_episode`: true when every episode's target stays constant over its recorded steps.
- `target_debug_pixel`: projected first target pixel in the fixed debug-camera frame when the live wrapper can read camera intrinsics/extrinsics.
- `target_debug_pixel_by_episode`: projected target pixel per episode when available.
- `target_debug_pixel_visible`: whether the first projected target lands inside the debug image.
- `target_debug_pixel_source`: `debug_camera_projection` when projection succeeded, or a `not_available...` / `projection_failed...` reason when it did not.

Current target reticle overlay:

Goal:

```text
3D target position in robot/base frame
-> transform to debug camera frame
-> project using camera intrinsics
-> draw reticle and pixel-coordinate text on the debug image pixel
```

Implementation plan:

1. Keep target information sourced from the formal proprio contract: `target_pos_base = obs["proprio"][0, 24:27]`. Do not add Isaac debug visualizers and do not draw target markers into `images` or `raw_policy_images`.
2. Add/use the live Isaac wrapper method `project_base_point_to_debug_pixel(point_base, camera_name="table_cam")`.
3. Inside that method, get the robot/base frame pose in world coordinates and the debug camera pose/intrinsics from the Isaac camera sensor.
4. Convert the target point from base frame to world frame: `target_w = T_world_base @ target_base`.
5. Convert world frame to debug-camera frame: `target_cam = T_camera_world @ target_w`.
6. If the target is behind the camera or outside the image plane, return `None` and skip the reticle for that frame.
7. Project with camera intrinsics: `u = fx * x / z + cx`, `v = fy * y / z + cy`.
8. `demo_data_loop.py` exposes opt-in `--target-overlay {none,text,reticle,text-reticle}`. The default is `none`; final interview GIFs should use `--target-overlay text-reticle`.
9. Reticle/text drawing is post-processing on fixed debug-camera GIF/debug PNG frames only. The policy observation remains wrist RGB plus proprio, and HDF5 dataset image streams are not modified.
10. Fake projection tests verify that a deterministic target point draws a reticle plus coordinate label such as `(100, 32)`, and that out-of-view targets do not draw anything.

**Command**

```bash
conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy heuristic \
  --num_episodes 3 \
  --settle-steps 20 \
  --target-overlay text-reticle \
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
  --settle-steps 20 \
  --target-overlay text-reticle \
  --save_dataset ./data/replay_from_heuristic_rollouts.h5 \
  --save_metrics ./logs/replay_from_heuristic_eval.json \
  --save_gif ./out/gifs/replay_from_heuristic_demo.gif \
  --save_mp4 ./out/gifs/replay_from_heuristic_demo.mp4
```

**Suggested File**

```text
scripts/demo_data_loop.py
policies/replay_policy.py
tests/test_demo_data_loop.py
```

**Test Command**

```bash
conda run -n isaac_arm python -m pytest tests/test_demo_data_loop.py -v
```

**What To Test**

- CLI accepts `--backend`, `--policy`, `--num_episodes`, `--settle-steps`, `--save_dataset`, `--save_metrics`, `--save_gif`, and optional `--save_mp4`.
- CLI accepts `--target-overlay {none,text,reticle,text-reticle}` and uses it only for GIF/debug PNG post-processing.
- CLI accepts `--replay_dataset` when `--policy replay`.
- Running with `--policy random` creates all requested outputs.
- Running with `--policy heuristic` creates all requested outputs.
- Running with `--policy replay --replay_dataset ...` creates metrics/GIF outputs and optionally a replay rollout dataset.
- Running with `--policy replay` and no `--replay_dataset` fails with a readable error.
- Metrics JSON has the expected keys.
- Dataset has at least one episode.
- GIF exists and has multiple frames.
- Output directories are created if missing.
- `--settle-steps` advances physics after reset without recording those warmup transitions in the dataset or GIF.

Verified in `isaac_arm`:

```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m pytest tests/test_demo_data_loop.py -q
/root/miniconda3/bin/conda run -n isaac_arm python -m pytest tests/test_demo_data_loop.py tests/test_demo_dataset.py tests/test_demo_policies.py tests/test_eval_metrics.py tests/test_visual_outputs.py -q
/root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q
```

Known result on 2026-04-24:

```text
tests/test_demo_data_loop.py: 8 passed
demo/eval/GIF related set: 60 passed
full pytest: 144 passed, 1 skipped
```

Generated sample artifacts for inspection:

```text
data/demo_pr_sample_isaac.h5
logs/demo_pr_sample_isaac_metrics.json
out/gifs/demo_pr_sample_isaac.gif
out/debug_frames/demo_pr_sample_isaac/heuristic_ep000_step000_debug.png
out/debug_frames/demo_pr_sample_isaac/heuristic_ep000_step002_debug.png
out/debug_frames/demo_pr_sample_isaac/heuristic_ep000_step005_debug.png
data/demo_pr_sample_isaac_settled.h5
logs/demo_pr_sample_isaac_settled_metrics.json
out/gifs/demo_pr_sample_isaac_settled.gif
```

To generate the settled live Isaac sample used to inspect the final demo visuals:

```bash
export DISPLAY=:0
export XAUTHORITY="/var/run/sddm/$(ls /var/run/sddm/ | head -1)"
timeout 360s env OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
  /root/miniconda3/bin/conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy heuristic \
  --num_episodes 1 \
  --max-steps 20 \
  --gif-max-steps 20 \
  --settle-steps 20 \
  --target-overlay text-reticle \
  --table-cleanup matte-overlay \
  --min-clean-env-spacing 5.0 \
  --save_dataset data/demo_pr_sample_isaac_settled.h5 \
  --save_metrics logs/demo_pr_sample_isaac_settled_metrics.json \
  --save_gif out/gifs/demo_pr_sample_isaac_settled.gif \
  --save_mp4 out/gifs/demo_pr_sample_isaac_settled.mp4 \
  --save-debug-frames-dir out/debug_frames/demo_pr_sample_isaac_settled \
  --seed 0 \
  --device cuda:0 \
  --no-progress
```

Expected inspection outputs:

```text
data/demo_pr_sample_isaac_settled.h5
logs/demo_pr_sample_isaac_settled_metrics.json
out/gifs/demo_pr_sample_isaac_settled.gif
out/gifs/demo_pr_sample_isaac_settled.mp4
out/debug_frames/demo_pr_sample_isaac_settled/
```

Known settled sample check on 2026-04-24:

```text
status: ok
gif_num_frames: 20
mean_episode_length: 20
cube z during recorded dataset: min 0.0209998768, max 0.0210000407
```

**Suggested Commit**

```bash
git commit -m "feat(demo): add one-command robotics data loop"
```

---

## 13. Recommended Implementation Order

The demo slice was implemented in this order:

1. PR 11-lite: Evaluation metrics JSON.
2. PR 12-lite: GIF output from the fixed debug camera.
3. ReplayPolicy over saved rollout datasets.
4. Demo PR: One-command data loop.

If time is tight during a live interview run:

- GIF and metrics are mandatory for the interview.
- Dataset can be simplified, but keep episode-safe design.
- Replay policy is nice to have.
- The final interview-facing demo runs against live Isaac. Unit-level PRs (PR0/PR1/PR2) intentionally use a Gymnasium test double injected through `gym_make` — that is constructor injection, not `unittest.mock`, and it is only for wrapper/contract tests. "No-mock" here means the demo rollout itself does not substitute a fake env for the live Isaac task; it does not mean every test in the slice avoids test doubles.
- The demo is not ready if live Isaac only returns stock `policy: (num_envs, 35)` observations.
- The demo is ready only when the customized cfg (PR2.5) produces wrist RGB plus 40D proprio.

---

## 14. Final Runbook

Use this section as the operational checklist for final artifacts. For live Isaac commands on the vast.ai bare-metal runtime, start each shell with:

```bash
export DISPLAY=:0
export XAUTHORITY="/var/run/sddm/$(ls /var/run/sddm/ | head -1)"
```

### Final Dataset Collection

Collect a settled, camera-rich dataset for later training or inspection:

```bash
timeout 360s env OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
  /root/miniconda3/bin/conda run -n isaac_arm python -m scripts.collect_rollouts \
  --backend isaac \
  --policy random \
  --num-parallel-envs 2 \
  --num-episodes 10 \
  --max-steps 100 \
  --settle-steps 20 \
  --include-raw-policy-images \
  --include-debug-images \
  --table-cleanup matte-overlay \
  --min-clean-env-spacing 5.0 \
  --save-dataset data/final_random_rollouts_10eps_numenvs2_settled.h5 \
  --seed 0 \
  --device cuda:0 \
  --progress
```

Change `--num-episodes` and `--max-steps` upward for a larger training corpus. Keep `--settle-steps 20` for training-intended Isaac datasets so reset-settling motion is not stored. Keep `--include-raw-policy-images` and `--include-debug-images` for inspection/debug datasets; omit them for leaner training-only HDF5 files if storage becomes the bottleneck.

Inspect the dataset summary and metadata:

```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.inspect_rollout_dataset \
  --dataset data/final_random_rollouts_10eps_numenvs2_settled.h5
```

Inspect and export one representative frame triplet:

```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.inspect_rollout_dataset \
  --dataset data/final_random_rollouts_10eps_numenvs2_settled.h5 \
  --episode episode_000 \
  --step 0 \
  --save-policy-frame out/debug_frames/final_dataset_ep000_step000_policy.png \
  --save-raw-policy-frame out/debug_frames/final_dataset_ep000_step000_raw_policy.png \
  --save-debug-frame out/debug_frames/final_dataset_ep000_step000_debug.png
```

Inspect a later episode/step if the dataset has enough steps:

```bash
/root/miniconda3/bin/conda run -n isaac_arm python -m scripts.inspect_rollout_dataset \
  --dataset data/final_random_rollouts_10eps_numenvs2_settled.h5 \
  --episode episode_009 \
  --step 49 \
  --save-policy-frame out/debug_frames/final_dataset_ep009_step049_policy.png \
  --save-raw-policy-frame out/debug_frames/final_dataset_ep009_step049_raw_policy.png \
  --save-debug-frame out/debug_frames/final_dataset_ep009_step049_debug.png
```

Dataset inspection output should be read as:

- `num_episodes`: number of HDF5 episode groups, not vectorized env count.
- `source_env_index`: vectorized Isaac lane that produced that episode.
- `reset_round` / `reset_seed`: explicit `env.reset(seed=...)` round that started the episode.
- `terminated_by`: whether the episode group ended via env `done`, env `truncated`, or collector `max_steps`.
- `settle_steps`: zero-action reset warmup count skipped before recording. For final training/demo datasets this should be `20`.
- `images`: wrist policy camera stream, resized to `(T, 3, 224, 224)`, and this is the future training image input.
- `raw_policy_images`: optional native-resolution wrist camera frames for inspection only.
- `debug_images`: optional fixed table-camera frames for human inspection only.

Frame files from inspection should be read as:

- `*_policy.png`: exact policy-image stream after resize; this is what a training loader sees.
- `*_raw_policy.png`: native wrist camera view; useful to debug camera placement and crop/resize.
- `*_debug.png`: fixed debug camera view; useful for human understanding and screenshots, not policy input.

### Final Demo Run

Run the main final demo artifact:

```bash
timeout 360s env OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
  /root/miniconda3/bin/conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy heuristic \
  --num_episodes 3 \
  --max-steps 100 \
  --gif-max-steps 100 \
  --settle-steps 20 \
  --target-overlay text-reticle \
  --table-cleanup matte-overlay \
  --min-clean-env-spacing 5.0 \
  --save_dataset data/final_heuristic_demo_rollouts.h5 \
  --save_metrics logs/final_heuristic_demo_metrics.json \
  --save_gif out/gifs/final_heuristic_demo.gif \
  --save_mp4 out/gifs/final_heuristic_demo.mp4 \
  --save-debug-frames-dir out/debug_frames/final_heuristic_demo \
  --seed 0 \
  --device cuda:0 \
  --progress
```

Optional comparison demo commands:

```bash
timeout 360s env OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
  /root/miniconda3/bin/conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy random \
  --num_episodes 3 \
  --max-steps 100 \
  --gif-max-steps 100 \
  --settle-steps 20 \
  --target-overlay text-reticle \
  --table-cleanup matte-overlay \
  --min-clean-env-spacing 5.0 \
  --save_dataset data/final_random_demo_rollouts.h5 \
  --save_metrics logs/final_random_demo_metrics.json \
  --save_gif out/gifs/final_random_demo.gif \
  --save_mp4 out/gifs/final_random_demo.mp4 \
  --save-debug-frames-dir out/debug_frames/final_random_demo \
  --seed 0 \
  --device cuda:0 \
  --progress
```

```bash
timeout 360s env OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
  /root/miniconda3/bin/conda run -n isaac_arm python -m scripts.demo_data_loop \
  --backend isaac \
  --policy replay \
  --replay_dataset data/final_heuristic_demo_rollouts.h5 \
  --num_episodes 3 \
  --max-steps 100 \
  --gif-max-steps 100 \
  --settle-steps 20 \
  --target-overlay text-reticle \
  --table-cleanup matte-overlay \
  --min-clean-env-spacing 5.0 \
  --save_dataset data/final_replay_from_heuristic_demo_rollouts.h5 \
  --save_metrics logs/final_replay_from_heuristic_demo_metrics.json \
  --save_gif out/gifs/final_replay_from_heuristic_demo.gif \
  --save_mp4 out/gifs/final_replay_from_heuristic_demo.mp4 \
  --save-debug-frames-dir out/debug_frames/final_replay_from_heuristic_demo \
  --seed 0 \
  --device cuda:0 \
  --progress
```

### Final Demo Outputs

The main heuristic demo run should create:

```text
data/final_heuristic_demo_rollouts.h5
logs/final_heuristic_demo_metrics.json
out/gifs/final_heuristic_demo.gif
out/gifs/final_heuristic_demo.mp4
out/debug_frames/final_heuristic_demo/
```

Interpret the outputs as:

- `data/final_heuristic_demo_rollouts.h5`: episode-safe rollout dataset. This proves the policy/env loop can produce training-ready HDF5. `images` are wrist policy images; `debug_images` are human-facing and optional.
- `logs/final_heuristic_demo_metrics.json`: scalar eval summary plus target-position metadata. Read `mean_return`, `success_rate`, `mean_episode_length`, and `mean_action_jerk` together; also check `success_source` to know whether success came from explicit `info` flags or the proprio fallback.
- `out/gifs/final_heuristic_demo.gif`: interview-facing visual artifact from the fixed debug camera. With `--target-overlay text-reticle`, it draws a reticle on the projected target and writes the pixel coordinate beside it, for example `(100, 32)`. It is for explanation/debugging and is not the policy observation stream.
- `out/gifs/final_heuristic_demo.mp4`: color-faithful visual artifact from the same debug-camera frames/overlays. Prefer this for visual inspection when GIF palette quantization makes colors look wrong.
- `out/debug_frames/final_heuristic_demo/`: sampled fixed-camera PNGs from the GIF rollout. They receive the same target reticle/text post-processing as the GIF when `--target-overlay` is enabled. Use them for quick image inspection without opening the full GIF.

Metrics JSON fields:

- `policy_name`: policy evaluated, for example `heuristic`, `random`, or `replay`.
- `env_backend`: expected `isaac` for final demo artifacts.
- `num_episodes`: number of rollout episodes evaluated.
- `mean_return`: mean undiscounted reward sum per episode.
- `success_rate`: fraction of episodes judged successful.
- `success_source`: `info_success` if Isaac/env emitted success flags, `proprio_cube_to_target_norm` if using the fallback distance threshold, or mixed for mixed datasets.
- `success_threshold_m`: cube-to-target distance threshold used by the fallback, currently `0.02`.
- `consecutive_success_steps`: number of consecutive successful steps required, currently `1`.
- `success_distance_metric`: current fallback distance definition, `euclidean_3d_cube_to_target_base_m`.
- `success_distance_source`: exact proprio source for the fallback, `norm(proprio[:, 30:33])` where `proprio[:, 30:33] = target_pos_base - cube_pos_base`.
- `episode_successes`: per-episode boolean success map, for example `{"episode_000": true, "episode_001": false}`.
- `closest_target_approach_by_episode`: per-episode closest approach diagnostics. Each entry records `closest_step`, `closest_distance_m`, `closest_cube_position_base_m`, `target_position_base_m_at_closest_step`, `cube_to_target_base_m_at_closest_step`, `success`, and `success_source`.
- `mean_episode_length`: average recorded transition count per episode, after settle warmup.
- `mean_action_jerk`: mean norm of first action differences; lower means smoother commands.
- `target_position_base_m`: convenience summary target XYZ in robot/base frame, currently `episode_000` step 0 from proprio slice `24:27`.
- `target_position_base_m_source`: exact source of the summary value, for example `episode_000_step_000_proprio_24_27`.
- `target_positions_base_m_by_episode`: per-episode first target XYZ in robot/base frame.
- `target_position_constant_by_episode`: whether every episode's target remained constant across all recorded steps.
- `target_debug_camera_name`: fixed debug camera used for target projection, usually `table_cam`.
- `target_debug_pixel`: projected first target pixel `[u, v]` in the debug image, or `null` if projection was unavailable.
- `target_debug_pixel_by_episode`: per-episode projected target pixel map when available.
- `target_debug_pixel_visible`: whether `target_debug_pixel` is inside the debug camera frame.
- `target_debug_pixel_source`: projection provenance or failure reason, for example `debug_camera_projection`.

The expected story for the interview is: the HDF5 proves the data interface, the metrics JSON proves the evaluation loop, the GIF proves human-readable visual rollout inspection, and the debug PNGs make quick camera/source checks easy.

---

## 15. Acceptance Criteria For Interview Demo

This section is intentionally short. The exact commands live in §14 Final Runbook so there is one source of truth for final artifact generation.

The demo is accepted when:

- the §14 final dataset collection command completes and the dataset can be inspected with `scripts.inspect_rollout_dataset`;
- the §14 main final heuristic demo command completes and writes `data/final_heuristic_demo_rollouts.h5`, `logs/final_heuristic_demo_metrics.json`, `out/gifs/final_heuristic_demo.gif`, `out/gifs/final_heuristic_demo.mp4`, and sampled debug PNGs;
- optional §14 random and replay comparison commands complete when comparison artifacts are needed;
- all final live Isaac commands use `--settle-steps 20`, `--table-cleanup matte-overlay`, and `--min-clean-env-spacing 5.0` unless intentionally probing raw Isaac visuals.

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

All demo-slice test files listed in the command above exist in the current tree. The full repository test run on 2026-04-24 passed with `144 passed, 1 skipped`; the skip is the opt-in Isaac runtime smoke test, which still requires `RUN_ISAAC_RUNTIME_SMOKE=1`.

---

## 16. Interview Walkthrough

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
