# Plan: Isaac Lab Arm Manipulation — Continuous-Control RL and Diffusion Policy

## 1. Introduction

This project builds a complete robot-learning benchmark in **Isaac Lab / Isaac Sim** for a Franka Panda arm lifting a cube from a table. The goal is to compare standard continuous-control reinforcement learning (RL) algorithms and then distill the strongest RL behavior into a smoother **Diffusion Policy** imitation-learning (IL) controller.

Beginner explanation:
- The robot receives an image from a wrist camera and proprioception from its joints.
- The policy outputs continuous robot commands, not discrete buttons.
- RL methods learn by trial and error in simulation.
- Diffusion Policy learns from expert demonstrations and predicts short action sequences that tend to be smoother.

The project produces:
- reproducible Isaac Lab environment wrapper,
- PPO / pure GRPO / SAC / TD3 continuous-control baselines,
- SAC expert demonstration dataset,
- Diffusion Policy BC and DAgger training,
- evaluation metrics and result tables,
- rollout GIFs for visual comparison.

---

## 2. Motivation

### 2.1 Why move from ViZDoom to robot manipulation?

The previous ViZDoom project used a discrete-action game environment. That made several algorithms non-standard:
- SAC and TD3 had to become discrete-action variants.
- Diffusion Policy had to predict discrete logits instead of continuous actions.
- The final system was useful for representation-learning experiments, but less canonical for robotics.

Robot arm manipulation is a better fit because:
- PPO, GRPO, SAC, and TD3 all support continuous actions naturally.
- Diffusion Policy was designed for robot manipulation and action chunks.
- Isaac Lab provides GPU-parallel physics simulation for large-scale training.
- FrankaCubeLift is simple enough to implement, but realistic enough to show grasping, lifting, and recovery behavior.

### 2.2 Project research questions

1. How do PPO, pure GRPO, SAC, and TD3 compare in sample efficiency on FrankaCubeLift?
2. Does pure GRPO work without a critic/value network in continuous robot control?
3. Does SAC produce a strong enough oracle for demonstration collection?
4. Can Diffusion Policy behavior cloning match or approach SAC performance using offline demonstrations?
5. Does DAgger reduce covariate shift by relabeling states visited by the student policy?
6. Do Diffusion Policy action chunks reduce action jerk compared with step-by-step RL policies?

---

### 2.3 Current Status

Updated 2026-04-29. The repository has completed the interview/demo vertical slice from
`plans/subplan_isaac_arm_manipulation.md`:

```text
policy rollout -> episode-safe dataset -> metrics -> GIF/MP4/debug PNGs
```

The current code base proves the robot-learning data and evaluation loop against the live
Isaac backend, and now has SAC/TD3 train, logger, checkpoint-eval, live-monitor scaffolding,
running normalization, reward diagnostics, periodic/best checkpoints, reward curriculum, and
bucket-rarity prioritized replay, and eval-subskill dual-gated curriculum advancement. The
latest SAC diagnostic runs still have `success_rate=0`: PR6.8 made reach/grip/lift/goal
events observable and replayable, PR6.9 added dense lift progress diagnostics, and PR6.10
replaced bucket-ratio-only curriculum advancement with current-policy eval + minimum-exposure
dual gates. The next action is a serious SAC v6 run using `--curriculum-gating eval_dual_gate`.
The remaining research-training stack is PPO, pure
GRPO, SAC expert demonstrations, Diffusion Policy BC, and DAgger.

Runtime requirement for live Isaac commands:
- On the vast.ai bare-metal runtime, use `DISPLAY=:0` and set `XAUTHORITY` to the active SDDM cookie under `/var/run/sddm/`.
- On WSL2 + Docker, run Xvfb on `DISPLAY=:1`.
- Missing X authority can cause `Authorization required`, `eglInitialize failed`, or `xcb_connection_has_error()` even when Isaac is launched headless.

This is the single source of truth for implementation status. Do not maintain a second PR roadmap table elsewhere in this plan.

| PR / Area | Status | What it provides | Test / evidence | Next / notes |
|---|---|---|---|---|
| Runtime env | Ready with display setup | Live Isaac command path for collection, smoke, benchmark, and demo scripts | `scripts.collect_rollouts`, `scripts.demo_data_loop`, `scripts.isaac_camera_observation_smoke`, `scripts.benchmark_rollout_collection` | Live Isaac still needs working X11 display/auth; benchmark helper can auto-discover SDDM `XAUTHORITY` when unset. |
| PR 0 - Project scaffold | Done | Package layout, project config, output dirs, seed utilities | `tests/test_project_scaffold.py` -> `6 passed` | Foundation for all later packages. |
| PR 1 - Task/action contract | Done | `Isaac-Lift-Cube-Franka-IK-Rel-v0`, normalized 7D action order, clipping, gripper convention | `tests/test_task_contract.py` -> `9 passed` | Public action contract stays `dx, dy, dz, droll, dpitch, dyaw, gripper`. |
| PR 2 - Formal observation wrapper | Done | `IsaacArmEnv`, wrist image + 40D proprio contract, strict stock-35D rejection | `tests/test_observation_wrapper.py` -> `18 passed` | Formal policy obs is `obs["image"]` `(N,3,224,224)` and `obs["proprio"]` `(N,40)`. |
| PR 2.5 - Camera-enabled Franka cfg | Done | Customized Isaac cfg with `wrist_cam`, optional `table_cam`, named proprio terms, debug/policy camera separation | `tests/test_camera_enabled_env_cfg.py` -> `13 passed`; live camera smoke image `(1,3,224,224)`, proprio `(1,40)` | Stock `--enable_cameras` alone is not enough; use customized cfg. |
| Image augmentation utilities | Done | `PadAndRandomCrop`, `CenterBiasedResizedCrop`, `IdentityAug` | `tests/test_image_aug.py` -> `24 passed` | Training-only augmentation; env wrapper remains deterministic. |
| PR 8-pre - Demo policies | Done | `BasePolicy`, random, heuristic, HDF5 replay policy interface | `tests/test_demo_policies.py` -> `10 passed` | All policies expose `act(obs) -> 7D action`. |
| PR 8-lite - Rollout dataset | Done | Episode-safe HDF5 schema, collector, inspector, benchmark helper | `tests/test_demo_dataset.py`, `tests/test_rollout_benchmark.py` | Stores wrist `images`, `proprios`, actions, rewards, done/truncated, optional raw/debug images, lane/reset metadata. |
| PR 11-lite - Dataset metrics | Done | Dataset-level return, success, length, jerk, target diagnostics | `tests/test_eval_metrics.py` | Success falls back to `norm(proprio[:, 30:33]) <= 0.02` when explicit success flags are absent. |
| PR 12-lite - Visual output | Done | Fixed-debug-camera GIF/MP4, sampled PNGs, overlay support | `tests/test_visual_outputs.py` | Human-facing frames come from debug camera, not policy wrist image. |
| Demo PR - One-command data loop | Done | Dataset + metrics + GIF/MP4 + debug PNG artifacts in one command | `tests/test_demo_data_loop.py` -> `13 passed`; full pytest at demo completion -> `149 passed, 1 skipped` | Used for interview/demo vertical slice. |
| Final live demo artifacts | Generated | `final_heuristic_demo` HDF5/JSON/GIF/MP4/debug PNGs | `logs/final_heuristic_demo_metrics.json`: 3 eps, mean return `27.3684`, success `2/3`, jerk `0.2926` | Random and replay comparison artifacts also exist. |
| PR 3 - Shared backbone | Done | `ImageProprioBackbone` shared image-proprio encoder | `tests/test_nn_backbone.py` -> `8 passed`; full pytest after PR3 -> `157 passed, 1 skipped` | First research-model component is complete. |
| PR 3.5 - Agent primitives | Done | Shared distributions, actor/critic heads, replay/rollout batches, checkpoint helpers, fake-checkpoint factory, `CheckpointPolicy` adapter | `tests/test_agent_primitives.py` -> `21 passed` | Lays the off-policy + checkpoint infrastructure for PR6/PR7/PR11a/PR12a. |
| PR 6 - SAC train | Done | `SACAgent`, online replay training, deterministic oracle mode (`tanh_mu`), `scripts.train_sac_continuous`, fake-env smoke + reward-probe | `tests/test_sac_continuous.py` -> `18 passed` | Logger-less base training loop; long-run diagnostics now live in PR6.5/PR6.7. |
| PR 7 - TD3 train | Done | `TD3Agent`, deterministic actor, target smoothing, delayed actor updates, `scripts.train_td3_continuous`, shares replay format with SAC | `tests/test_td3_continuous.py` -> `17 passed` | Logger-less; long-run TB/wandb logs land in PR6.5. |
| PR 6.5 - Training logger + LR scheduler + live monitors | Done | TensorBoard/wandb/JSONL logger contract from §8.3, console/file progress for SAC/TD3, scheduler hooks, fake-env separate periodic `eval/*`, training rollout metrics (`train_rollout/*`), same-Isaac-env deterministic eval rollout lanes (`eval_rollout/*`) with delayed clean-episode start, initial and per-lane settle steps, checkpointed `scheduler_state` | `tests/test_training_logger_and_scheduler.py` -> `28 passed`; full pytest at PR6.5 commit -> `244 passed, 1 skipped` (superseded by PR6.7 row below for current full-suite count) | For live Isaac, use `--eval-every-env-steps 0`, `--same-env-eval-lanes N`, and `--same-env-eval-start-env-steps K` for train-time monitoring; PR11a remains the final checkpoint eval path. |
| PR 6.6 - Running obs/action normalization | Done | `agents.normalization` running per-dimension proprio mean/std, optional channel-wise image running mean/std (`--image-normalization none|per_channel_running_mean_std`, default off), explicit `bidirectional_env_learner_affine` action normalizer, checkpointed `normalizer_state`, SAC/TD3 train/eval/checkpoint policy consistency, optional angle sin/cos primitive only for true wrap-around angle features | `tests/test_normalization.py` + SAC/TD3 checks -> `47 passed`; full pytest -> `259 passed, 1 skipped` | Serious SAC/TD3 training can now use normalized proprio inputs, optional train-lane-only image channel stats, and stable learner-action/env-action conversion; replay still stores raw env observations/actions. |
| PR 6.7 - Training diagnostics + checkpoint controls | Done | Per-step visual reward trace in `record_gif_continuous` metrics, SAC/TD3 reward component logging under `reward/train/*` and `reward/eval_rollout/*`, periodic/best checkpoint manager, `--disable-reward-curriculum`, SAC `--alpha-min` floor | Targeted PR6.7 slice -> `70 passed`; full pytest -> `283 passed, 1 skipped` | Use this before the next serious SAC/TD3 run so failed runs leave reward traces, stock reward breakdown, intermediate checkpoints, and best-by-eval checkpoints for debugging. |
| PR 6.8 - Curriculum reward + bucket-rarity replay | Done | Opt-in `reach_grip_lift_goal` training reward curriculum, grip proxy bridge reward, task-progress bucket labels, frequency-based bucket rarity, mixed uniform/priority replay sampling, protected rare-transition retention, W&B/JSONL/progress diagnostics, TD-error priority feedback | PR6.8 targeted slice -> `97 passed`; full pytest in `isaac_arm` -> `294 passed, 1 skipped` | This is not vanilla SAC. It is a task-aware RL improvement that uses no demos, no BC, and no expert actions; final PR11a/PR12a eval remains stock-env evaluation. |
| PR 6.9 - Progress-gated lift curriculum + lift-aware diagnostics | Done | Progress-gated curriculum advancement, dense `lift_progress_proxy`, grip-attempt/effect diagnostics, lift-aware eval metrics, composite best-checkpoint selection, action gripper diagnostics, lower protected-replay defaults for the next run | PR6.9 targeted slice -> `79 passed`; full pytest in `isaac_arm` -> `310 passed, 1 skipped` | Use this before the next SAC run; it is designed to answer whether the policy is failing to reach, failing to close near the cube, failing to convert grip attempts into lift, or merely being hidden by the wrong best metric. |
| PR 6.10 - Eval-subskill dual-gated curriculum | Done | `eval_dual_gate` curriculum mode for SAC/TD3, current-policy deterministic eval episode subskill tracker, stage-local train exposure counters, strict stage advancement only when eval + exposure + min-stage-step gates all pass, W&B/JSONL/progress logs for why stages hold/advance | Targeted PR6.10 slice (`tests/test_reward_curriculum.py`, `tests/test_training_logger_and_scheduler.py`) -> `56 passed`; full pytest in `isaac_arm` -> `318 passed, 1 skipped` | Use this for the next serious SAC run so stage transitions mean "the current policy can do the subskill", not merely "the replay buffer has seen a few matching transitions." |
| PR 11a - SAC/TD3 eval | Done | `scripts.eval_checkpoint_continuous --agent-type/--agent_type sac|td3`, metrics JSON, optional eval HDF5 | `tests/test_eval_sac_td3_checkpoints.py` -> `13 passed` | First trained-checkpoint eval path. |
| PR 12a - SAC/TD3 visuals | Done | `scripts.record_gif_continuous --agent-type/--agent_type sac|td3`, GIF/MP4/debug PNGs, same-rollout metrics JSON, optional PR11a metrics overlay validation, shared target-reticle/settle helpers | `tests/test_visual_sac_td3_checkpoints.py` -> `11 passed`; visual/demo/eval regression slice -> `66 passed` | SAC/TD3 train -> eval -> GIF path is now wired. Live Isaac still needs an actual trained SAC/TD3 checkpoint plus display/camera runtime. |
| PR 8-full - SAC demonstrations | Pending | SAC expert rollout collection into existing HDF5 schema | Planned test: `tests/test_sac_demo_collection.py` | Depends on SAC checkpoint plus PR11a/PR12a sanity checks. |
| PR 8.5 - Diffusion sequence dataset | Pending | `(B,T_obs,3,224,224)`, `(B,T_obs,40)`, `(B,H,7)` sequence dataloader | Planned test: `tests/test_diffusion_sequence_dataset.py` | Bridge from HDF5 episodes to Diffusion training batches. |
| PR 9a - Diffusion core | Pending | Noise schedule, timestep embeddings, conditioned temporal denoiser | Planned test: `tests/test_diffusion_core.py` | Model only; no BC train script yet. |
| PR 9b - Diffusion BC training | Pending | BC train script, denoising loss, checkpoints, resume, synthetic overfit mode | Planned test: `tests/test_diffusion_bc_training.py` | Trains from SAC demos via PR8.5 dataloader. |
| PR 9c - Diffusion deployment | Pending | DDIM inference, action queue, `DiffusionPolicy.act(obs)` | Planned test: `tests/test_diffusion_policy_deployment.py` | Enables eval/visual rollout for BC policy. |
| PR 10 - DAgger | Pending | Student rollout, SAC oracle relabeling, aggregation, fine-tuning loop | Planned test: `tests/test_dagger_diffusion.py` | Uses SAC oracle and deployed Diffusion student. |
| PR 4 - PPO | Deferred | On-policy PPO baseline | Planned test: `tests/test_ppo_continuous.py` | Not blocking SAC/TD3-first path. |
| PR 5 - Pure GRPO | Deferred | No-critic group-relative on-policy baseline | Planned test: `tests/test_grpo_continuous.py` | Not blocking SAC/TD3-first path. |
| PR 11-full - General checkpoint eval | Pending | Extends PR11a to PPO/GRPO/Diffusion/DAgger | Planned test: `tests/test_eval_agent.py` | After more trained agents exist. |
| PR 12-full - Full visual comparison | Pending | Side-by-side GIF grids and plots across all methods | Planned test: `tests/test_visual_comparison_outputs.py` | After PR11-full and multiple trained checkpoints. |

Current measured demo-slice results from the final live artifacts:

| Policy/artifact | Episodes | Mean return | Success rate | Mean action jerk | Notes |
|---|---:|---:|---:|---:|---|
| `logs/final_random_demo_metrics.json` | 3 | `-0.0029` | `0.0000` | `2.0789` | Random sanity baseline; all episodes failed under the current 2 cm proprio fallback. |
| `logs/final_heuristic_demo_metrics.json` | 3 | `27.3684` | `0.6667` | `0.2926` | Heuristic policy succeeded on `episode_001` and `episode_002`; target debug projection succeeded for all three episodes. |
| `logs/final_replay_from_heuristic_demo_metrics.json` | 3 | `9.4850` | `0.0000` | `0.0636` | HDF5 replay path generated metrics and visual artifacts; replay smoothness is lower jerk, but this run did not hit the current success threshold. |

Important status interpretation:
- The stock Isaac task with `--enable_cameras` still exposes only the stock flat `policy` tensor unless the project applies its customized env cfg.
- The current accepted live observation contract is the customized cfg plus wrapper: wrist RGB policy image and named 40D proprio.
- The debug camera is only for GIFs, MP4s, sampled PNGs, and human inspection; it is not passed into `policy.act()`.

---

## 3. Environment And Task Contract

### 3.1 Isaac Lab task

Use the IK-relative Franka lift task:

```python
env_id = "Isaac-Lift-Cube-Franka-IK-Rel-v0"
```

This environment is chosen because the policy should command **relative end-effector pose deltas**, not raw joint targets. Plain joint-position lift variants are not the default choice for this project because they are not the cleanest match for 6D end-effector delta control.

Camera-based runs must launch Isaac Sim with camera rendering enabled:

```bash
--enable_cameras
```

If the camera is requested without `--enable_cameras`, the wrapper must fail with a readable error explaining how to launch the environment correctly.

Important runtime finding, confirmed 2026-04-19:
- The stock `Isaac-Lift-Cube-Franka-IK-Rel-v0` task can boot with `--enable_cameras`, `gym.make()`, `env.reset()`, `env.step()`, and `env.close()` in camera mode.
- `--enable_cameras` only enables Isaac Sim rendering and camera sensors. It does **not** automatically add RGB image terms to a task's observations.
- The stock task's active observation group is still only `policy` with shape `(num_envs, 35)`.
- The stock task has no RGB/image/camera observation term in its scene or policy observation contract.
- Therefore the project must not treat the stock 35D `policy` tensor as the final image-proprio observation contract.
- The correct project direction is to keep `Isaac-Lift-Cube-Franka-IK-Rel-v0` and customize its Isaac Lab `env_cfg` before `gym.make()` so the task exposes a policy wrist camera and named low-dimensional observation terms.

**WSL2 runtime note (confirmed 2026-04-17):** On WSL2 + Docker, `SimulationApp._app.update()` deadlocks at C++ GPU Foundation level without a virtual display. Always run Xvfb before Isaac Sim:

```bash
pkill Xvfb 2>/dev/null; Xvfb :1 -screen 0 1280x720x24 &
export DISPLAY=:1
```

`scripts/isaac_runtime_smoke.py` sets `DISPLAY=:1` automatically. No-camera (`--no-enable-cameras`) mode is confirmed working. Camera mode requires `nvcr.io/nvidia/isaac-sim:5.1.0` (official container with correct Vulkan passthrough).

### 3.2 Observation space

The policy receives a dictionary:

```python
obs_dict = {
    "image":   uint8 array,   # shape: (num_envs, 3, 224, 224), wrist RGB image
    "proprio": float32 array, # shape: (num_envs, proprio_dim)
}
```

`proprio_dim` is config-driven and must not be hardcoded into every module. The PR0 scaffold can start with the minimal joint-only setting:

```python
proprio_dim = 14
```

PR2 should update `proprio_dim` to match the exact enabled feature list and test that the configured dimension equals the sum of feature dimensions.

The stock Isaac Lab policy observation is useful as a diagnostic baseline, but it is **not** the formal project observation:

```text
stock policy observation = [
    joint_pos,                 # 9 dims, mdp.joint_pos_rel for all Franka joints
    joint_vel,                 # 9 dims, mdp.joint_vel_rel for all Franka joints
    object_position,           # 3 dims, cube/object position in robot root frame
    target_object_position,    # 7 dims, generated object_pose command: 3D position + 4D quaternion
    actions,                   # 7 dims, previous action
]
```

The stock 35D vector is not missing "just 5 padded values." It contains four target-orientation quaternion dimensions that the 40D project contract does not use, and it omits the end-effector position term needed by the project:

```text
35 stock dims - 4 target quaternion dims + 3 ee_pos_base dims + 3 ee_to_cube dims + 3 cube_to_target dims = 40 dims
```

This arithmetic only tracks the dimension count. It **does not** mean the 40D vector can be produced by slicing the 35D tensor. The gripper bookkeeping in particular breaks the slicing assumption:

- Stock `joint_pos`/`joint_vel` are `joint_pos_rel`/`joint_vel_rel` across all 9 Franka joints (arm 7 + gripper 2), i.e. relative-to-default.
- The formal contract wants `gripper_finger_pos`/`gripper_finger_vel` as **raw** (absolute) joint coordinates, not relative-to-default values.
- Therefore the wrapper must pull gripper terms from separate raw `mdp.joint_pos` / `mdp.joint_vel` queries over `panda_finger.*`, not by slicing the last 2 entries of the stock 9D relative joint vectors.

Do **not** pad the stock vector with zeros. Do **not** slice the 35D tensor to fabricate a 40D one. Do **not** infer the end-effector position from the flat vector. The formal wrapper must build the 40D vector from named observation terms and scene state.

`proprio` means the low-dimensional numeric state used by the policy. Strictly speaking,
robot proprioception is the robot's internal body state, such as joint positions, joint
velocities, and gripper finger positions. In this project, the `proprio` vector may also
include low-dimensional task state from Isaac Lab, such as cube and target positions.

Important distinction:
- `joint_pos`, `joint_vel`, and gripper finger joints are robot proprioception.
- `cube_pos_base` and `target_pos_base` are task state, not robot proprioception.
- In a real robot system, cube state would normally come from perception rather than privileged simulator state.
- The wrapper should keep the feature names explicit so this distinction is easy to explain.
- "Position" in `arm_joint_pos_rel` and `gripper_finger_pos` means joint coordinate, not world-space XYZ position.

Arm and gripper position convention:
- Use `arm_joint_pos_rel` for the Franka arm: `current_arm_joint_pos - default_arm_joint_pos`.
- Use `arm_joint_vel_rel` for the Franka arm: `current_arm_joint_vel - default_arm_joint_vel`; in practice this is usually the raw joint velocity because the default velocity is zero.
- Use `gripper_finger_pos` for the gripper: actual/raw left and right finger joint positions.
- Use `gripper_finger_vel` for the gripper: actual/raw left and right finger joint velocities.
- Do not encode gripper finger positions only as relative-to-default values, because the raw finger coordinate has direct physical meaning.
- The stock 35D `joint_pos` and `joint_vel` terms are `joint_pos_rel` and `joint_vel_rel` for all 9 Franka joints. The first 7 dimensions can supply arm-relative state, but the last 2 gripper position dimensions are relative-to-default and should not be used as the formal raw gripper opening state.
- For the formal contract, get gripper finger state from raw joint terms over `panda_finger.*`, not from the flat stock 35D vector.

Franka gripper intuition:

```text
closed: left_finger ~= 0.00, right_finger ~= 0.00
open:   left_finger ~= 0.04, right_finger ~= 0.04
width:  left_finger + right_finger
```

Why arm uses relative joint position but gripper uses actual finger position:
- Arm joint angles are easier for a policy to learn when centered around the default Franka pose.
- Gripper finger positions are already interpretable as opening distance.
- Actual finger position can reveal contact or blockage; previous gripper action cannot.

Recommended PR2 feature contract:

```text
proprio = [
    arm_joint_pos_rel,    # 7 dims: current arm q - default arm q, radians
    arm_joint_vel_rel,    # 7 dims: current arm qdot - default arm qdot, usually qdot
    gripper_finger_pos,   # 2 dims: actual/raw left/right finger joint positions, meters
    gripper_finger_vel,   # 2 dims: actual/raw left/right finger joint velocities
    ee_pos_base,          # 3 dims: end-effector position in robot base/root frame
    cube_pos_base,        # 3 dims: cube position in robot base/root frame
    target_pos_base,      # 3 dims: target position in robot base/root frame
    ee_to_cube,           # 3 dims: cube_pos_base - ee_pos_base
    cube_to_target,       # 3 dims: target_pos_base - cube_pos_base
    previous_action,      # 7 dims
]
```

This recommended contract has `proprio_dim = 40`. If the wrapper also exposes
`gripper_width = left_finger_pos + right_finger_pos` as a separate scalar, then
`proprio_dim = 41`.

Formal source of each 40D term:

| Feature | Dims | Source |
|---|---:|---|
| `arm_joint_pos_rel` | 7 | `mdp.joint_pos_rel` with `SceneEntityCfg("robot", joint_names=["panda_joint.*"])` |
| `arm_joint_vel_rel` | 7 | `mdp.joint_vel_rel` with `SceneEntityCfg("robot", joint_names=["panda_joint.*"])` |
| `gripper_finger_pos` | 2 | raw `mdp.joint_pos` with `SceneEntityCfg("robot", joint_names=["panda_finger.*"])` |
| `gripper_finger_vel` | 2 | raw `mdp.joint_vel` with `SceneEntityCfg("robot", joint_names=["panda_finger.*"])` |
| `ee_pos_base` | 3 | `scene["ee_frame"]` end-effector target position transformed from world frame into robot root/base frame |
| `cube_pos_base` | 3 | `mdp.object_position_in_robot_root_frame`, equivalent to stock `object_position` |
| `target_pos_base` | 3 | `mdp.generated_commands("object_pose")[:, :3]`, the position part of stock `target_object_position` |
| `ee_to_cube` | 3 | `cube_pos_base - ee_pos_base` computed by the wrapper |
| `cube_to_target` | 3 | `target_pos_base - cube_pos_base` computed by the wrapper |
| `previous_action` | 7 | `mdp.last_action`, equivalent to stock `actions` |

`ee_pos_base` details:
- The stock Franka lift config already defines `scene.ee_frame` using a `FrameTransformerCfg`.
- Its first and only target frame is named `end_effector`, attached to `Robot/panda_hand` with an offset of roughly 10.34 cm.
- The end-effector world position is:

```python
ee_frame = env.scene["ee_frame"]
ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
```

- The formal base-frame position is obtained with Isaac Lab's frame transform helper:

```python
robot = env.scene["robot"]
ee_pos_base, _ = subtract_frame_transforms(
    robot.data.root_pos_w,
    robot.data.root_quat_w,
    ee_pos_w,
)
```

- The index `0` is used because the current Franka lift `ee_frame.target_frames` list contains exactly one target frame: `end_effector`. If future configs add finger frames, the implementation must assert the target frame identity instead of silently assuming index 0.

Why include derived relative features:
- `cube_pos_base` and `target_pos_base` are useful, but the policy still has to subtract positions internally.
- `ee_to_cube` tells the policy how to move the gripper toward the cube.
- `cube_to_target` tells the policy where the lifted cube should move.
- These features make heuristic policies, behavior cloning, and Diffusion Policy easier to debug and usually easier to learn.

Why include actual gripper state:
- `previous_action[-1]` is only the commanded gripper action.
- It is not the same as the physical gripper state.
- Example: the previous action may command close, but the fingers may still be open, still moving, or blocked by the cube.
- Therefore PR2 should expose actual finger joint positions or `gripper_width`, not only the previous gripper command.

The wrapper is responsible for converting Isaac Lab native observations into this stable project contract.

### 3.2.1 Camera Observation Contract

The project uses an eye-in-hand policy camera by default:

```text
policy camera: wrist_cam
debug camera: table_cam or front_cam, optional, human/GIF only
```

The policy image must come from a camera attached to the robot wrist/end-effector area, not from a fixed external workcell camera. This is the more realistic setup for a single-arm manipulation policy and better matches the project's visuomotor / Diffusion Policy narrative.

Implementation requirements:
- Add `wrist_cam` to the customized Isaac Lab `env_cfg`, attached under the Franka hand/end-effector prim, for example under `Robot/panda_hand`.
- Add a policy RGB observation term using `mdp.image` / `SceneEntityCfg("wrist_cam")`.
- Keep the wrapper's public policy image as `obs["image"]` with shape `(num_envs, 3, 224, 224)` and dtype `uint8`.
- Keep `--enable_cameras` as a hard runtime requirement for camera observations.
- Do not assume the stock env's camera mode adds this term; the customized config must add the camera sensor and the observation term explicitly.

The wrist camera is meaningful only after its pose is validated. A valid wrist camera frame should show useful manipulation context such as the gripper fingers, table surface, and cube during the approach/grasp portion of a rollout. It may not see the full workspace at reset, so low-dimensional proprio/task state remains part of the observation contract.

### 3.2.2 Debug Camera And GIF Recording

A fixed external camera may be added for debugging and visualization, but it must not be used as the policy input by default.

```text
debug camera purpose = human inspection, GIFs, failure diagnosis
policy camera purpose = learning observation
```

Recommended debug design:
- Add an optional fixed `table_cam` or `front_cam` to the customized Isaac cfg.
- Position it outside the robot, fixed in the workcell, looking at the table, cube, gripper, and lift region.
- Use it only through a wrapper method such as `get_debug_frame(camera_name="table_cam")` or `render_debug()`.
- Return debug frames as `(H, W, 3)` `uint8` arrays for GIF recording.
- Do not pass debug frames into `policy.act()`, SAC replay buffers, Diffusion Policy datasets, or the formal `obs["image"]` field.
- If debug frames are stored in an HDF5 rollout file, store them under a clearly separate key such as `debug_images`, never under the training `images` key.

This separation keeps the project realistic: the robot policy uses the wrist camera, while the fixed camera acts like a lab recording camera for humans.

### 3.2.3 Image Augmentation Contract

Augmentation is applied in the **training pipeline**, not in the env wrapper. The wrapper always produces a deterministic 224×224 image.

Three-tier contract:

```text
Env wrapper     : native (e.g. 400×400) → deterministic resize → 224×224  (obs contract, no randomness)
Training aug    : 224×224 → pad 8 px → random crop 224×224                 (primary, DrQ/RAD-style)
Eval/GIF/smoke  : no augmentation
```

Rationale for keeping augmentation out of the wrapper:
- Random crop inside `env.step()` breaks reproducibility: same simulator state → different `obs["image"]`.
- Direct 400×400 → random 224×224 crop retains only 56 % of image area, risking removal of gripper or cube from frame.
- Pad + random crop with `pad ≤ 16 px` provides small translation variance without discarding task-relevant content.

Implementation: `utils/image_aug.py` provides:

| Class | Input | Use |
|---|---|---|
| `PadAndRandomCrop(pad=8)` | `(B, 3, 224, 224)` wrapper output | Primary training aug |
| `CenterBiasedResizedCrop(min_scale=0.75)` | `(B, 3, H, W)` native resolution | Alternative; requires dataset to store native-res images |
| `IdentityAug()` | any | Eval, GIF, smoke |

Factory helpers: `make_train_aug(mode="pad_crop"|"resized_crop")` and `make_eval_aug()`.

When `CenterBiasedResizedCrop` is used as the training aug, eval should fix `scale=1.0` (center resize) to minimise the train/eval distribution gap.

Test command:

```bash
pytest tests/test_image_aug.py -v
```

Known result: `24 passed`.

### 3.3 Action space: 6D arm + 1D gripper

All policies output a **7D continuous action**:

```text
a = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

Action shape:

```python
action_dim = 7
action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=float32)
```

Action semantics:
- `dx, dy, dz`: relative end-effector translation command.
- `droll, dpitch, dyaw`: relative end-effector rotation command.
- `gripper`: continuous gripper command.

Scaling:
- translation commands are scaled to a small per-step delta, for example +/-2 cm,
- rotation commands are scaled to a small per-step delta, for example +/-5 degrees,
- gripper command is mapped to open/close or target aperture depending on what the Isaac Lab task config supports.

Implementation rule:
- The learning code always sees 7D continuous actions.
- If Isaac Lab internally expects binary gripper commands, the environment wrapper follows Isaac Lab's binary gripper convention: `gripper >= 0` means open and `gripper < 0` means close.
- The conversion must be documented and unit-tested.

### 3.4 Reward and success

Primary task: lift the cube to a target height.

The project evaluates with Isaac Lab's task reward and success signals when available. If the wrapper exposes custom metrics, use:

```text
success = cube_z >= target_z
```

Metrics:
- episode return,
- success rate,
- episode length,
- action jerk,
- steps-to-threshold.

Before PR6/PR7 training, run a reward sanity probe:
- confirm `Isaac-Lift-Cube-Franka-IK-Rel-v0` produces a dense, non-constant reward during random and heuristic rollouts;
- log reward min/mean/max and at least a few per-step reward traces;
- compare random and heuristic reward distributions to verify the reward signal reflects meaningful progress;
- record whether explicit success flags are present or whether evaluation must use the 40D proprio fallback.

If the reward probe shows sparse or nearly constant rewards, do not start long SAC/TD3 runs. First add reward diagnostics, curriculum, or documented reward shaping as a separate PR.

### 3.5 Off-Policy Training Runtime Contracts

SAC and TD3 are image-based off-policy methods, so PR3.5/PR6/PR7 must lock down a few runtime contracts before long training.

**Replay storage and memory budget**

The current training GPU is an NVIDIA RTX 5000 Ada with 32 GB VRAM. That is not enough for a naive GPU replay buffer:

```text
200k * 3 * 224 * 224 uint8 ~= 30.1 GB decimal ~= 28.0 GiB
```

That estimate is for one image stream only. Storing both `obs_image` and `next_obs_image` doubles the image storage before proprio, actions, rewards, done flags, Python overhead, sampled GPU batches, model activations, or optimizer state.

Required replay-buffer contract:
- Replay storage lives in CPU RAM, memory-mapped files, or a disk-backed array store; never store the full replay buffer on CUDA.
- Store policy images as `uint8`; convert to float and move only sampled minibatches to GPU.
- Store `proprios`, `actions`, rewards, and done/truncated flags as compact numeric arrays.
- Prefer storing each transition once and reconstructing `next_obs` by index when possible. If `next_obs` must be stored explicitly because vectorized lanes reset independently, document the memory cost and keep it on CPU/disk.
- PR3.5 must expose a replay-buffer capacity/memory estimator so PR6/PR7 can print the expected storage footprint before training starts.

**Vectorized env terminal-transition rule**

Isaac Lab vectorized envs may auto-reset a finished lane inside `env.step()`. Therefore `next_obs` returned for a terminal lane may already be the reset observation for the next episode. Off-policy training must not treat that reset observation as a same-episode continuation.

Required transition rule:
- Store `terminated` and `truncated` separately when available.
- Store a scalar training mask such as `bootstrap_mask = 0.0` for true terminal transitions and for truncations when the trainer chooses not to bootstrap through time limits.
- Critic targets must use:

```text
target = reward + gamma * bootstrap_mask * target_q(next_obs, next_action)
```

- If a lane auto-resets, the replay buffer may store the post-reset `next_obs`, but `bootstrap_mask` must still prevent bootstrapping across the boundary.
- Tests must include a fake vectorized env where one lane terminates early while sibling lanes continue, matching the PR8-lite per-lane collection bug class.

**Deterministic action convention**

Use one convention everywhere:
- SAC deterministic eval/oracle action = `tanh(mu)`, where `mu` is the Gaussian actor mean before squashing.
- SAC training action = reparameterized squashed Gaussian sample.
- TD3 deterministic eval action = actor output without exploration noise.
- TD3 training action = actor output plus clipped exploration noise before env clipping.

PR11a, PR12a, PR8-full, and PR10 must use the same deterministic convention when loading checkpoints.

**Backbone sharing convention for off-policy agents**

For the first SAC/TD3 implementation, keep actor and critic encoders separate:
- actor has its own `ImageProprioBackbone`;
- critic/twin-Q module has its own `ImageProprioBackbone`;
- target networks own their own target encoders;
- no hidden actor/critic encoder parameter sharing in PR6/PR7.

This is more memory-heavy than shared encoders, but it avoids ambiguous gradient coupling and makes checkpoint loading simpler. Shared encoders or actor stop-gradient variants can be explored later as an explicit optimization PR.

**Settle-steps comparison profile**

Use one settle-step value for a given comparison table. The demo has used `--settle-steps 600` for final visual artifacts; quick training-data probes may use `20`.

Contract:
- PR11a eval and PR12a visual commands must record `settle_steps` in metrics/artifact metadata.
- Final SAC-vs-TD3 metric and GIF comparisons must use the same `settle_steps` value, recommended `600` for final demo-quality artifacts.
- Smoke tests may use smaller values, but their outputs should be labeled as smoke/debug, not final comparison results.

---

## 4. Baselines

| Method | Type | Purpose |
|---|---|---|
| Random Policy | Non-learning sanity baseline | Confirms env runs and task is non-trivial. |
| Heuristic Gripper Sanity Controller | Simple scripted baseline | Confirms action scaling and gripper mapping. |
| PPO | On-policy RL | Stable actor-critic baseline with GAE. |
| Pure GRPO | On-policy RL | Group-relative policy optimization without critic/value network. |
| SAC | Off-policy RL | Main sample-efficient expert/oracle candidate. |
| TD3 | Off-policy RL | Deterministic off-policy baseline with target smoothing. |
| BC-Diffusion | Offline IL | Trains Diffusion Policy from SAC demonstrations. |
| DAgger-Diffusion | Interactive IL | Reduces distribution shift by relabeling student-visited states. |
| SAC to DAgger | RL-to-IL pipeline | Uses SAC as oracle, then distills into a smoother diffusion controller. |

---

## 5. Proposed Method

### 5.1 High-level pipeline

```text
Isaac Lab FrankaCubeLift IK-Rel env
        |
        v
Train PPO / pure GRPO / SAC / TD3
        |
        v
Select SAC checkpoint as expert oracle
        |
        v
Collect episode-safe demonstrations
        |
        v
Train Diffusion Policy BC
        |
        v
Run DAgger: student rollout -> SAC relabel -> dataset aggregation -> fine-tune
        |
        v
Evaluate metrics + record GIFs + plot comparison
```

### 5.2 Why SAC is the oracle

SAC is expected to be the strongest oracle because:
- it is off-policy and sample-efficient,
- it uses twin critics to reduce overestimation,
- entropy maximization improves exploration,
- the deterministic mean action can be used for clean demonstration labels.

The oracle action for data collection:

```python
oracle_action = sac_agent.act(obs_dict, deterministic=True)
```

---

## 6. Model Architecture

### 6.1 Shared image-proprio fusion backbone

```text
ImageEncoder:
  image (B, 3, 224, 224)
  -> CNN
  -> image_feat (B, 256)

ProprioMLP:
  proprio (B, proprio_dim)
  -> MLP + LayerNorm
  -> proprio_feat (B, 64)

Fusion:
  concat(image_feat, proprio_feat)
  -> MLP
  -> obs_feat (B, 256)
```

This backbone is shared conceptually across RL and Diffusion Policy so every method uses the same observation information.

### 6.2 RL policy heads

All RL methods output 7D actions.

PPO:
```text
obs_feat -> Gaussian actor mean/log_std -> tanh-squashed 7D action
obs_feat -> value critic V(s)
```

Pure GRPO:
```text
obs_feat -> Gaussian actor mean/log_std -> tanh-squashed 7D action
```

Pure GRPO has **no critic**, no value head, no GAE, and no value loss.

SAC:
```text
actor: obs_feat -> squashed Gaussian 7D action
critic: concat(obs_feat, action_7d) -> Q1(s,a), Q2(s,a)
```

TD3:
```text
actor: obs_feat -> deterministic tanh 7D action
critic: concat(obs_feat, action_7d) -> Q1(s,a), Q2(s,a)
```

### 6.3 Diffusion Policy

Diffusion Policy predicts a short future action sequence:

```python
action_horizon = 8
exec_horizon = 4
action_dim = 7
```

Training target:

```text
actions: (B, H, 7)
images:  (B, T_obs, 3, 224, 224)
proprio: (B, T_obs, proprio_dim)
```

Architecture:
- image-proprio backbone encodes the last `T_obs` observations,
- 1D temporal U-Net denoises the action chunk,
- FiLM conditioning injects observation and timestep embeddings into each residual block,
- DDIM inference produces an action chunk,
- the controller executes only the first `exec_horizon` actions, then replans.

---

## 7. Algorithm Details And Corrections

### 7.1 PPO

PPO uses:
- Gaussian tanh-squashed 7D policy,
- learned or state-dependent log standard deviation,
- value critic,
- GAE,
- clipped policy objective,
- entropy bonus.

Log probability must include tanh correction:

```text
u = mu + sigma * epsilon
a = tanh(u)
log_pi(a|s) = log N(u; mu, sigma^2) - sum(log(1 - tanh(u)^2 + eps))
```

### 7.2 Pure GRPO

GRPO uses group-relative trajectory advantages and **does not use a critic**.

Training steps:
1. Collect trajectories with the current policy.
2. Compute one scalar return per trajectory.
3. Split trajectories into groups of size `G`.
4. Normalize each trajectory return relative to its group.
5. Assign each timestep in the trajectory the same group-relative advantage.
6. Update policy with PPO-style clipped objective.

No value network:
- no `V(s)`,
- no GAE,
- no return target,
- no value loss,
- no critic checkpoint.

This keeps GRPO conceptually clean: the baseline is the group, not a learned value function.

### 7.3 SAC

SAC uses:
- squashed Gaussian actor,
- twin Q critics,
- replay buffer,
- target critics,
- automatic entropy temperature tuning.

Correct actor update:

```python
u_new, a_new = policy_net(obs)
log_pi_new = policy_net.log_prob(u_new, a_new)
q1_new, q2_new = critic(obs, a_new)
q_min_new = torch.min(q1_new, q2_new)
actor_loss = (alpha * log_pi_new - q_min_new).mean()
```

Important correction:

```python
# Wrong for actor learning:
q_min_new = torch.min(q1_new, q2_new).detach()
```

Do not detach `q_min_new` from the actor path. The actor needs gradients through `Q(s, a_new)` with respect to `a_new`. Critic parameters may be excluded from the actor optimizer, but the computation graph from Q output to actor action must remain intact.

### 7.4 TD3

TD3 uses:
- deterministic actor,
- twin critics,
- replay buffer,
- target policy smoothing,
- delayed actor update.

Target smoothing:

```text
a_target = clip(mu_target(s_next) + clipped_noise, -1, 1)
```

All actor and critic actions are 7D.

### 7.5 Diffusion Policy BC and DAgger

BC trains on expert state-action chunks.

DAgger loop:
1. Roll out current diffusion student.
2. Save states visited by the student.
3. Query SAC oracle for labels on those states.
4. Append new labeled samples to the dataset.
5. Fine-tune the diffusion policy.

The dataset must be episode-safe. Sampling windows must never cross `done` or `truncated` boundaries.

---

## 8. Experiment Design

### 8.1 Training conditions

| Method | Interaction | Data source | Seeds |
|---|---|---|---|
| Random | Online eval only | None | 3 |
| Heuristic | Online eval only | Scripted controller | 3 |
| PPO | Online RL | Isaac Lab env | 3 |
| Pure GRPO | Online RL | Isaac Lab env | 3 |
| SAC | Online RL | Isaac Lab env | 3 |
| TD3 | Online RL | Isaac Lab env | 3 |
| BC-Diffusion | Offline IL | SAC demonstrations | 3 |
| DAgger-Diffusion | Interactive IL | Student states + SAC labels | 3 |

### 8.2 Default hyperparameters

| Parameter | PPO | Pure GRPO | SAC | TD3 | BC | DAgger |
|---|---:|---:|---:|---:|---:|---:|
| action_dim | 7 | 7 | 7 | 7 | 7 | 7 |
| num_envs | 64 | 64 | 64 | 64 | 1-64 | 1-16 |
| batch_size | 256 | 256 | 256 | 256 | 128 | 128 |
| learning_rate | 3e-4 | 3e-4 | 3e-4 | 3e-4 | 1e-4 | 1e-5 |
| gamma | 0.99 | 0.99 | 0.99 | 0.99 | N/A | N/A |
| gae_lambda | 0.95 | N/A | N/A | N/A | N/A | N/A |
| group_size | N/A | 8 | N/A | N/A | N/A | N/A |
| replay_size | N/A | N/A | 200k | 200k | N/A | N/A |
| replay_storage | N/A | N/A | CPU/disk uint8 images | CPU/disk uint8 images | N/A | N/A |
| proprio_normalization | N/A | N/A | running per-dim mean/std | running per-dim mean/std | dataset/global per-dim mean/std | running/dataset per-dim mean/std |
| image_normalization | N/A | N/A | `none` default; optional `per_channel_running_mean_std` | `none` default; optional `per_channel_running_mean_std` | dataset/global policy-image stats if enabled | dataset/global policy-image stats if enabled |
| action_normalization | normalized env action `[-1,1]` | normalized env action `[-1,1]` | learner action `[-1,1]` -> env action `[-1,1]` | learner action `[-1,1]` -> env action `[-1,1]` | dataset action normalized with same mapper | dataset/oracle action normalized with same mapper |
| warmup_steps | N/A | N/A | 5k transitions | 5k transitions | N/A | N/A |
| eval_every_env_steps | N/A | N/A | 10k | 10k | N/A | N/A |
| same_env_eval_lanes | N/A | N/A | 0 default; set `1+` for live monitor | 0 default; set `1+` for live monitor | N/A | N/A |
| same_env_eval_start_env_steps | N/A | N/A | 0 default; use warmup/50k+ for real runs | 0 default; use warmup/50k+ for real runs | N/A | N/A |
| rollout_metrics_window | N/A | N/A | 20 completed episodes | 20 completed episodes | N/A | N/A |
| settle_steps | N/A | N/A | 0 smoke; 20/600 if needed | 0 smoke; 20/600 if needed | N/A | N/A |
| per_lane_settle_steps | N/A | N/A | 0 default; 20 if reset settling is needed | 0 default; 20 if reset settling is needed | N/A | N/A |
| utd_ratio | N/A | N/A | 1 | 1 | N/A | N/A |
| polyak_tau | N/A | N/A | 0.005 | 0.005 | N/A | N/A |
| sac_target_entropy | N/A | N/A | `-action_dim` | N/A | N/A | N/A |
| sac_initial_alpha | N/A | N/A | 0.2 | N/A | N/A | N/A |
| policy_delay | N/A | N/A | 1 | 2 | N/A | N/A |
| td3_exploration_noise_sigma | N/A | N/A | N/A | 0.1 | N/A | N/A |
| td3_target_noise_sigma | N/A | N/A | N/A | 0.2 | N/A | N/A |
| td3_target_noise_clip | N/A | N/A | N/A | 0.5 | N/A | N/A |
| deterministic_eval_action | deferred to PR4 | deferred to PR5 | `tanh(mu)` | actor output, no noise | DDIM deterministic | DDIM deterministic |
| diffusion_T | N/A | N/A | N/A | N/A | 100 | 100 |
| ddim_steps | N/A | N/A | N/A | N/A | 10 | 10 |
| action_horizon | N/A | N/A | N/A | N/A | 8 | 8 |
| exec_horizon | N/A | N/A | N/A | N/A | 4 | 4 |

For SAC/TD3:
- `warmup_steps` is measured in individual replay transitions, not vectorized `env.step()` calls; one 64-env Isaac step can add up to 64 transitions.
- `proprio_normalization` must use running Welford-style per-dimension mean/std from train-lane observations entering replay, not mini-batch statistics. Eval lanes and checkpoint eval must freeze the normalizer.
- The current 40D proprio contains limited robot joint positions, Cartesian positions/deltas, gripper values, velocities, and previous normalized action. Add sin/cos features only for true periodic/wrap-around angle features; do not blindly transform all bounded joint positions or Cartesian values.
- Image observations are stored as CPU `uint8` replay tensors. SAC/TD3 update paths apply DrQ-style `PadAndRandomCropTorch(pad=8)` to sampled image batches, then convert to float `[0,1]` before CNN forward. `--image-normalization none` keeps this baseline. `--image-normalization per_channel_running_mean_std` additionally applies running RGB channel mean/std computed only from replay-writing train lanes; eval lanes, settle steps, and checkpoint eval do not update image stats.
- `action_normalization` is an explicit mapper between learner action space and the public env action contract. The code schema type is `bidirectional_env_learner_affine`: `env_to_learner(a_env) = clip((a_env - mid) / half, -1, 1)` and `learner_to_env(a_learner) = clip(mid + a_learner * half, env_low, env_high)`, where `mid=(env_high+env_low)/2` and `half=(env_high-env_low)/2`. The public Isaac wrapper already expects normalized 7D actions in `[-1,1]`; therefore PR6.6 denormalizes only back to this env-normalized contract before `env.step`, not directly to physical meters/radians.
- The current default action mapper uses seven per-dimension bounds matching `dx, dy, dz, droll, dpitch, dyaw, gripper`: `env_low=[-1,-1,-1,-1,-1,-1,-1]` and `env_high=[1,1,1,1,1,1,1]`. This is numerically identity today, but it keeps learner action and env action conventions explicit for BC/TD3+BC and future datasets.
- `eval_every_env_steps=10000` means run deterministic eval rollouts every 10k individual env transitions and log the §8.3 eval keys. This separate-env path is for fake-env smoke tests and future out-of-process eval; live Isaac training should keep it at `0` until a safe external evaluator owns that path.
- `same_env_eval_lanes=N` reserves the last `N` vectorized Isaac lanes for deterministic current-policy monitoring inside the already-running training env. Those lanes do not enter replay, do not drive warmup/update counts, and log under `eval_rollout/*`.
- `same_env_eval_start_env_steps=K` delays same-env eval metrics until `K` training transitions have elapsed, then waits for each eval lane's next done/reset boundary before recording a clean episode.
- `rollout_metrics_window=20` is the rolling completed-episode window for `train_rollout/*` and `eval_rollout/*` episode metrics.
- `settle_steps=S` runs `S` zero-action physics steps immediately after the explicit training env reset. These settle transitions are not logged, not stored in replay, and do not increment `env_steps`.
- `per_lane_settle_steps=S` runs `S` zero-action cooldown steps after an individual vectorized lane reports done/truncated and Isaac auto-resets that lane. Cooldown transitions are not stored in replay, do not increment `env_steps`, and are masked out of rollout/eval episode metrics while other lanes continue normally.
- `utd_ratio=1` means one gradient update per env step after warmup unless explicitly changed.
- `polyak_tau=0.005` means target parameters update as `target = (1 - tau) * target + tau * online`.
- TD3 exploration noise is added to actor actions during data collection; target smoothing noise is added only in target-Q computation.
- All replay storage is CPU/disk-backed; only sampled minibatches are moved to GPU.

### 8.3 Checkpoint And Logging Contract

Every trainable checkpoint from PR4-7 and PR9-10 must include metadata so PR11a/PR12a can evaluate without guessing:

```json
{
  "agent_type": "sac",
  "env_id": "Isaac-Lift-Cube-Franka-IK-Rel-v0",
  "action_dim": 7,
  "proprio_dim": 40,
  "image_shape": [3, 224, 224],
  "num_env_steps": 500000,
  "global_update_step": 12345,
  "seed": 0,
  "deterministic_action_mode": "tanh_mu",
  "backbone_config": {},
  "normalizer_config": {
    "version": 1,
    "proprio": {
      "type": "running_mean_std",
      "enabled": true,
      "dim": 40,
      "eps": 1e-6,
      "clip": 10.0
    },
    "image": {
      "type": "none",
      "enabled": false,
      "image_shape": [3, 224, 224],
      "channel_order": "rgb",
      "stats_space": "float_0_1",
      "eps": 1e-6,
      "clip": 10.0
    },
    "action": {
      "type": "bidirectional_env_learner_affine",
      "action_dim": 7,
      "env_low": [-1, -1, -1, -1, -1, -1, -1],
      "env_high": [1, 1, 1, 1, 1, 1, 1],
      "clip": true
    },
    "feature_transform": {
      "type": "angle_sin_cos",
      "input_dim": 40,
      "output_dim": 40,
      "angle_indices": []
    }
  },
  "algorithm_hparams": {},
  "replay_storage": "cpu_uint8_images"
}
```

Required checkpoint state:
- model weights;
- target model weights when the algorithm uses target networks;
- optimizer states;
- entropy temperature state for SAC;
- observation/action normalizer state when enabled (`proprio` running mean, Welford `m2`, `count`, `eps`, `clip`; optional `image` RGB channel running mean/M2/count over float `[0,1]` pixels; `bidirectional_env_learner_affine` action mapper state; and feature-transform metadata);
- global env-step and update counters;
- RNG seed/config needed for deterministic eval reproducibility.

PR11a may write `num_env_steps: null` only for legacy checkpoints that genuinely lack this metadata, and must include a warning field. New PR6/PR7 checkpoints must always write `num_env_steps`.

Logging key contract:
- `train/critic_loss`
- `train/actor_loss`
- `train/alpha_loss` for SAC
- `train/alpha` for SAC
- `train/q_mean`
- `train/replay_size`
- `train/num_env_steps`
- `normalizer/proprio_count`
- `normalizer/proprio_mean_abs_max`
- `normalizer/proprio_std_min`
- `normalizer/image_count` when `per_channel_running_mean_std` is enabled
- `normalizer/image_mean_min` when `per_channel_running_mean_std` is enabled
- `normalizer/image_mean_max` when `per_channel_running_mean_std` is enabled
- `normalizer/image_std_min` when `per_channel_running_mean_std` is enabled
- `train_rollout/mean_return`
- `train_rollout/success_rate`
- `train_rollout/mean_episode_length`
- `train_rollout/episode_count`
- `eval_rollout/mean_return`
- `eval_rollout/success_rate`
- `eval_rollout/mean_episode_length`
- `eval_rollout/episode_count`
- `reward/train/native_total`
- `reward/train/<stock_reward_term>` such as `reward/train/reaching_object`, `reward/train/lifting_object`, `reward/train/object_goal_tracking`, `reward/train/object_goal_tracking_fine_grained`, `reward/train/action_rate`, `reward/train/joint_vel`
- `reward/eval_rollout/native_total`
- `reward/eval_rollout/<stock_reward_term>` with the same term names for same-env deterministic eval lanes
- `reward/eval_rollout/eval_shaped` as a PR6.8 diagnostic only; it does not feed `eval_rollout/mean_return`
- `reward/eval_rollout/grip_proxy` as a PR6.8 deterministic eval diagnostic
- `reward/eval_rollout/lift_progress_proxy` as a PR6.9 deterministic eval diagnostic
- `curriculum/stage_index` and `curriculum/stage/<stage_name>` when PR6.8 reward curriculum is enabled
- `curriculum/stage_progress` when PR6.8 reward curriculum is enabled
- `curriculum/gate/reach_rate`, `curriculum/gate/grip_rate`, and `curriculum/gate/lift_rate` when PR6.9 progress-gated curriculum is enabled
- `curriculum/gate/held_stage` when PR6.9 progress-gated curriculum keeps the current stage because the next gate is not yet met
- `reward/train_shaped` when PR6.8 reward curriculum changes the reward stored in replay
- `reward/train/grip_proxy` when PR6.8 grip proxy is enabled
- `reward/train/lift_progress_proxy` when PR6.9 dense lift progress reward is enabled
- `action/train/gripper_mean` and `action/train/gripper_close_rate` for active train lanes
- `action/eval_rollout/gripper_mean`, `action/eval_rollout/gripper_close_rate`, and `action/eval_rollout/gripper_close_near_cube_rate` for same-env deterministic eval lanes
- `eval_rollout/max_cube_lift_m`, `eval_rollout/min_ee_to_cube_m`, `eval_rollout/min_cube_to_target_m`, and `eval_rollout/gripper_close_near_cube_rate` for PR6.9 lift-aware debugging and best-checkpoint selection
- `train/td_error_mean` when PR6.8 TD-error priority feedback is enabled through SAC/TD3 updates
- `priority_replay/batch_uniform`, `priority_replay/batch_priority`, `priority_replay/mean_priority_score`, and `priority_replay/protected_count` when PR6.8 prioritized replay is enabled
- `priority_replay/bucket_count/<bucket>` and `priority_replay/bucket_rarity/<bucket>` for `normal`, `reach`, `grip`, `lift`, and `goal` when PR6.8 bucket-rarity replay is enabled
- `priority_replay/bucket_count/grip_attempt` and `priority_replay/bucket_count/grip_effect` as PR6.9 diagnostic counts; these do not add a manual bucket importance order
- `eval/mean_return`
- `eval/success_rate`
- `eval/mean_episode_length`
- `eval/mean_action_jerk`

Use these exact keys in TensorBoard/CSV/JSON logs where applicable so comparison scripts do not need method-specific adapters.

After PR6.5, live Isaac SAC/TD3 training should set `--eval-every-env-steps 0` and use `--same-env-eval-lanes N` for train-time deterministic monitoring in the already-running Isaac env. Final comparison logs still come from PR11a checkpoint evaluation with deterministic actions and `eval_settle_steps=600`. Smoke/debug fake-env runs may use separate periodic `eval/*` with fewer settle steps, but must not overwrite final-comparison metrics.

---

## 9. Evaluation Metrics

| Metric | Meaning | Why it matters |
|---|---|---|
| Mean return | Average episode reward | Overall task performance. |
| Success rate | Fraction of episodes lifting cube to target | Most interpretable robotics metric. |
| Episode length | Number of steps before success/timeout | Fast success is better. |
| Steps-to-threshold | Env steps to reach target return/success | Measures sample efficiency. |
| Mean action jerk | Mean `||a_t - a_{t-1}||_2` | Lower means smoother control. |
| Demonstration count | Number of expert episodes used | Measures data efficiency. |
| Oracle query count | SAC labels requested during DAgger | Measures DAgger supervision cost. |

Evaluation command template:

```bash
python -m scripts.eval_checkpoint_continuous \
  --checkpoint ./checkpoints/sac_franka_final.pt \
  --agent_type sac \
  --num_episodes 20 \
  --out_path ./logs/eval_sac.npz
```

---

## 10. Evaluation Result

This section is a template. Values below are expected targets, not measured results. Fill measured values after running experiments.

### 10.1 RL results

| Method | Expected return | Measured return | Expected success | Measured success | Steps-to-80% | Notes |
|---|---:|---:|---:|---:|---:|---|
| Random | near 0.0 | TBD | near 0% | TBD | N/A | Sanity baseline. |
| Heuristic | low/medium | TBD | task-dependent | TBD | N/A | Verifies action mapping. |
| PPO | 0.65-0.80 | TBD | 60-80% | TBD | TBD | Stable but less sample-efficient. |
| Pure GRPO | 0.55-0.80 | TBD | 50-80% | TBD | TBD | Tests no-critic group advantage. |
| SAC | 0.80-0.90 | TBD | 80-90% | TBD | TBD | Main oracle candidate. |
| TD3 | 0.75-0.88 | TBD | 75-88% | TBD | TBD | Strong deterministic baseline. |

### 10.2 IL results

| Method | Expected success | Measured success | Demonstrations | Oracle queries | Expected jerk | Measured jerk |
|---|---:|---:|---:|---:|---:|---:|
| BC-Diffusion | 65-85% | TBD | 500 episodes | 0 after data collection | lower than RL | TBD |
| DAgger-Diffusion | 75-90% | TBD | BC + DAgger | TBD | lower than RL | TBD |
| SAC to DAgger | 80-90% | TBD | SAC demos + student states | TBD | lowest/smoothest | TBD |

### 10.3 Reporting rule

Do not put expected numbers into the resume or final report as measured results. Only report measured values after:

```bash
pytest tests/ -v
python -m eval.plot_comparison_continuous ...
python -m scripts.record_gif_continuous ...
```

---

## 11. Visual Output Comparison (GIF)

The project must produce GIFs for side-by-side visual inspection:

| Output | Agent | Purpose |
|---|---|---|
| `out/gifs/random_seed0.gif` | Random | Shows task is non-trivial. |
| `out/gifs/heuristic_seed0.gif` | Heuristic | Verifies gripper/action mapping. |
| `out/gifs/ppo_seed0.gif` | PPO | On-policy visual baseline. |
| `out/gifs/grpo_seed0.gif` | Pure GRPO | No-critic visual baseline. |
| `out/gifs/sac_seed0.gif` | SAC | Expert/oracle behavior. |
| `out/gifs/td3_seed0.gif` | TD3 | Deterministic baseline behavior. |
| `out/gifs/bc_diffusion_seed0.gif` | BC-Diffusion | Offline IL behavior. |
| `out/gifs/dagger_diffusion_seed0.gif` | DAgger-Diffusion | Final distilled behavior. |
| `out/gifs/comparison_grid_seed0.gif` | All methods | Same seed, same camera, side-by-side comparison. |

GIFs should be recorded from the debug camera by default, not the wrist policy camera. The wrist camera is the robot's observation; the debug camera is the human-readable rollout view. A GIF recorder may optionally save wrist-camera thumbnails for diagnosis, but comparison GIFs should use the fixed debug view so failures are understandable.

Each GIF should overlay:
- method name,
- episode return,
- success/failure,
- action jerk,
- seed.

Record command template:

```bash
python -m scripts.record_gif_continuous \
  --checkpoint ./checkpoints/dagger_diffusion_franka_final.pt \
  --agent_type diffusion \
  --seed 0 \
  --gif ./out/gifs/dagger_diffusion_seed0.gif
```

---

## 12. Work Breakdown — Pull Requests

Each PR should do one thing. Each PR must include tests and a copy-paste pytest command.

Planning rules for the remaining roadmap:
- Treat the completed interview/demo slice as a finished infrastructure layer, not as a substitute for the full research-training PRs.
- Every future PR must declare the input contract it consumes from previous PRs and the output contract it provides to later PRs.
- A PR is too large if it introduces a new model family, a new storage format, and a new runtime script at the same time.
- Tests should include interface/shape checks, mathematical unit checks where relevant, save/load round trips for trainable components, and at least one small integration or overfit/smoke test when a training loop is introduced.

Roadmap layers:

| Layer | Status | Purpose |
|---|---|---|
| Foundation env layer | Done | PR0, PR1, PR2, and PR2.5 define the Isaac task, 7D action contract, wrist-image + 40D proprio observation contract, and live camera-enabled cfg. |
| Demo data-loop layer | Done | PR8-pre, PR8-lite, PR11-lite, PR12-lite, and Demo PR prove rollout collection, HDF5 episodes, metrics, GIF/MP4/debug PNGs, and one-command artifacts without trained agents. |
| Research model layer | In progress | PR3 shared backbone, PR3.5 primitives, PR6/PR7 SAC/TD3, PR6.5-PR6.10 training instrumentation/normalization/diagnostics/curriculum/replay, PR11a eval, and PR12a visuals are done. The next work item is the SAC v6 dual-gated diagnostic run. |

Current priority path:

```text
PR3 done
  -> PR3.5 Agent Primitives
  -> PR6 SAC train
  -> PR6.5/PR6.6/PR6.7 Training instrumentation, normalization, diagnostics
  -> PR6.8 Curriculum reward + bucket-rarity replay
  -> PR6.9 Progress-gated lift curriculum + lift-aware diagnostics
  -> PR6.10 Eval-subskill dual-gated curriculum
  -> SAC v6 dual-gated diagnostic run
  -> PR7 TD3 train
  -> PR11a SAC/TD3 checkpoint eval
  -> PR12a SAC/TD3 checkpoint visual rollout
  -> PR8-full SAC expert demos
```

PPO and pure GRPO remain in the plan, but they are no longer blockers for the first complete train/eval/visualize loop.

Future PRs should use this handoff pattern:

```text
Inputs from previous PR -> implementation in this PR -> outputs guaranteed for later PRs -> tests that protect the handoff
```

### PR 0 — Project Scaffold

**Goal / Why**

Create the minimum repository structure so future PRs have a stable place for packages, configs, tests, logs, checkpoints, data, and outputs. This avoids mixing project setup with algorithm implementation.

**Implementation**
- Add package directories for `env`, `agents`, `train`, `dataset`, `configs`, `eval`, and `scripts`.
- Add `pyproject.toml` or equivalent dependency notes.
- Define standard output directories: `logs/`, `checkpoints/`, `data/`, `out/gifs/`, `plots/`.
- Add seed utility and config defaults.
- Do not implement Isaac Lab environment logic or any RL algorithm in this PR.

**How To Test**
- Import package modules.
- Verify config defaults are valid.
- Verify seed setting is deterministic for NumPy and PyTorch.
- Verify output directory creation is idempotent.

**Pytest**

```bash
pytest tests/test_project_scaffold.py -v
```

**Suggested Commit**

```bash
git commit -m "chore: scaffold Isaac Lab manipulation project"
```

---

### PR 1 — Isaac Task Config

**Goal / Why**

Define the exact Isaac Lab task and action contract. This PR makes the project use `Isaac-Lift-Cube-Franka-IK-Rel-v0` and standardizes all learning code around a 7D action: 6D relative end-effector control plus 1D gripper.

**Implementation**
- Add `IsaacArmEnvConfig` with `env_id="Isaac-Lift-Cube-Franka-IK-Rel-v0"`.
- Define `action_dim=7`.
- Define action names: `dx`, `dy`, `dz`, `droll`, `dpitch`, `dyaw`, `gripper`.
- Clip incoming actions to `[-1, 1]`.
- Convert gripper action to the task's expected gripper command.
- Require `--enable_cameras` when camera observations are enabled.
- Do not implement neural networks or training loops.

**How To Test**
- Verify env id is the IK-relative task.
- Verify action space has shape `(7,)`.
- Verify action clipping.
- Verify gripper conversion behavior.
- Verify readable error if camera mode is requested without camera support.

**Pytest**

```bash
conda run -n isaac_arm python -m pytest tests/test_task_contract.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(env): add Franka IK-relative lift task config"
```

---

### PR 2 — Observation Wrapper

**Goal / Why**

Define the stable project observation API:

```python
obs = {
    "image": ...,    # (num_envs, 3, 224, 224), uint8
    "proprio": ...,  # (num_envs, 40), float32
}
```

PR2 is the local/unit-tested wrapper contract. It lets project code call `reset()`, `step()`, `close()`, and consume a consistent observation dictionary. It does not create live Isaac camera sensors or mutate the Isaac Lab task config; that belongs to PR2.5.

**Implementation**
- Implement `IsaacArmEnv` wrapper around the configured Isaac Lab environment.
- Return batched observations with image shape `(num_envs, 3, 224, 224)`.
- Return `proprio` as float32 with config-driven `proprio_dim`.
- Define a stable 40D proprio feature order and expose it in config:

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

- Compute wrapper-owned derived features:
  - `ee_to_cube = cube_pos_base - ee_pos_base`,
  - `cube_to_target = target_pos_base - cube_pos_base`.
- Accept named low-dimensional terms from the configured environment:
  - `arm_joint_pos_rel`,
  - `arm_joint_vel_rel`,
  - `gripper_finger_pos`,
  - `gripper_finger_vel`,
  - `ee_pos_base`,
  - `cube_pos_base` or `object_position`,
  - `target_pos_base` or a 3D target-position term,
  - `previous_action` or `actions`.
- These named terms only exist in live Isaac after PR2.5 customizes the cfg. Until PR2.5 lands, PR2's unit tests source them from the `FakeIsaacGymEnv` test double via `gym_make` constructor injection. That is intentional; PR2 is the unit-tested wrapper contract, not the live-env proof.
- Reject flat stock 35D observations by shape. Do not pad 35D to 40D.
- Support `reset()`, `step(actions)`, `close()`, and `max_episode_steps`.
- Handle `done` and `truncated` separately.
- Do not implement reward shaping beyond exposing environment reward and success info.

**How To Test**
- Verify reset and step shapes.
- Verify image dtype is uint8 and proprio dtype is float32.
- Verify `proprio_dim` equals the sum of configured feature dimensions and that feature ordering is stable.
- Verify `ee_to_cube` is computed as `cube_pos_base - ee_pos_base`.
- Verify `cube_to_target` is computed as `target_pos_base - cube_pos_base`.
- Verify a flat stock 35D policy tensor is not accepted as formal 40D proprio.
- Verify batch size is preserved for `num_envs=1` and `num_envs>1`.
- Verify done/truncated arrays have shape `(num_envs,)`.
- Verify camera errors are readable.

PR2's original 13 passing tests covered shape/dtype/derived-feature assembly using an injected observation and a `wrist_rgb` image key. PR2.5 has now added the missing wrapper regressions:

- a 7D stock `target_object_position` / command pose is sliced to its `[:, :3]` position component;
- a flat `(num_envs, 35)` stock tensor is rejected instead of padded or sliced into 40D;
- debug/generic camera keys are refused as policy image input when `policy_image_obs_key` is configured;
- live Isaac actions are converted to torch tensors for the real backend.

Current wrapper regression result: `conda run -n isaac_arm python -m pytest tests/test_observation_wrapper.py -q` -> `18 passed`.

**Boundary With PR2.5**

PR2.5 owns the live Isaac work:

```text
camera sensors
policy wrist camera selection
debug camera separation
7D target_object_position -> 3D target_pos_base
custom Isaac Lab env_cfg
dedicated live camera observation smoke
```

**Pytest**

```bash
conda run -n isaac_arm python -m pytest tests/test_observation_wrapper.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(env): add image-proprio observation wrapper"
```

---

### PR 2.5 — Camera-Enabled Franka Lift Config

**Goal / Why**

The stock `Isaac-Lift-Cube-Franka-IK-Rel-v0` task does not expose RGB camera observations even when Isaac Sim is launched with `--enable_cameras`. This PR keeps the same task and 7D IK-relative action contract, but customizes the Isaac Lab `env_cfg` before `gym.make()` so the environment has a real policy wrist camera and named low-dimensional observation terms.

**Implementation**
- Add a small config helper rather than burying all cfg mutation inside `IsaacArmEnv.__init__`.
- Put that helper in a dedicated file such as `env/franka_lift_camera_cfg.py`.
- Start from `parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Rel-v0", device=device, num_envs=num_envs)`.
- Mutate or subclass the parsed cfg to add:
  - `wrist_cam`: an RGB camera attached under the Franka hand/end-effector prim, used as the policy camera.
  - optional `table_cam` or `front_cam`: fixed workcell camera used only for GIF/debug recording.
  - non-concatenated named low-dimensional policy terms for the 40D proprio contract.
- Use Isaac Lab `CameraCfg` or `TiledCameraCfg` according to the local Isaac Lab 2.3.2 camera API and batching requirements.
- Add a wrist RGB observation term with `mdp.image`, `SceneEntityCfg("wrist_cam")`, `data_type="rgb"`, and `normalize=False`.
- Keep sensor names and observation keys distinct:
  - `wrist_cam` is the Isaac scene camera sensor.
  - `wrist_rgb` is the observation term / key mapped to wrapper `obs["image"]`.
  - `table_cam` is the fixed debug camera sensor.
  - `table_rgb` is the optional debug image term / key.
- Keep `obs["image"]` mapped to the wrist RGB observation only.
- Provide a debug-frame accessor for the fixed camera, such as `get_debug_frame("table_cam")`, without adding it to the policy observation contract.
- Make camera names configurable, with defaults:

```python
policy_camera_name = "wrist_cam"
policy_image_obs_key = "wrist_rgb"
debug_camera_name = "table_cam"
debug_image_obs_key = "table_rgb"
```

- Fail readably if `enable_cameras=True` but the customized cfg does not produce the requested policy camera term.
- Do not change `env_id`, `action_dim`, or the 7D action order.
- Do not switch the project to another task just because another task already has camera examples. Existing Isaac Lab visuomotor examples should be used as implementation references only.

Keep `env/isaac_env.py` thin. It should select the cfg and then call `gym.make(...)`, not own all camera and observation mutation logic:

```python
if enable_cameras:
    env_cfg = make_camera_enabled_franka_lift_cfg(
        env_id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
        device=device,
        num_envs=num_envs,
        policy_camera_name="wrist_cam",
        policy_image_obs_key="wrist_rgb",
        debug_camera_name="table_cam",
        debug_image_obs_key="table_rgb",
    )
else:
    env_cfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Rel-v0",
        device=device,
        num_envs=num_envs,
    )

env = gym.make("Isaac-Lift-Cube-Franka-IK-Rel-v0", cfg=env_cfg)
```

**Policy Camera Semantics**
- `wrist_cam` is the robot's learning camera. It is the only camera that goes into `obs["image"]`, replay buffers, and Diffusion Policy datasets by default.
- It should be mounted close to `panda_hand` / the gripper frame so it moves with the end effector.
- It should be validated during approach/grasp, not only at reset, because a wrist camera may not see the cube before the arm moves toward the workspace.

**Debug Camera Semantics**
- `table_cam` / `front_cam` is a human inspection camera.
- It is fixed in the workcell and may see the full robot, table, cube, and lift region.
- It is used for GIFs, saved sample frames, and failure diagnosis.
- It must not be passed to `policy.act()` or used as the default training image.
- If stored in datasets, use a separate key such as `debug_images`.

**How To Test**
- Unit-test cfg creation without launching Isaac Sim by checking requested camera names, observation names, and feature dimensions where possible.
- Unit-test wrapper behavior with injected named observations:
  - `wrist_rgb` becomes `obs["image"]`.
  - debug camera frames do not appear in `obs["image"]`.
  - 40D proprio is assembled from named terms.
- Add wrapper-level regression tests (deferred from PR2, but strictly PR2.5's acceptance set):
  - 7D `target_object_position` (3D pos + 4D quat) is sliced to `[:, :3]` so proprio still shapes to `(num_envs, 40)`.
  - A flat stock `(num_envs, 35)` observation is rejected with a readable shape error; the wrapper must not zero-pad or slice it into 40D.
  - When `policy_camera_name` / `policy_image_obs_key` are set (for example `"wrist_cam"` / `"wrist_rgb"`), a debug-camera or generic `camera` key must not be silently used as `obs["image"]`; the wrapper raises rather than falling back to the first-match tuple.
  - `get_debug_frame(camera_name)` / `render_debug()` exists and returns a `(H, W, 3)` `uint8` array that is distinct from `obs["image"]` when both cameras are configured.
- Live-smoke camera mode in `isaac_arm`:

```bash
timeout 240s conda run -n isaac_arm python -m scripts.isaac_runtime_smoke \
  --device cuda:0 \
  --headless \
  --enable-cameras \
  --steps 1
```

- Add a dedicated live camera observation smoke that verifies:
  - `wrist_cam` RGB term exists.
  - wrapper returns `image.shape == (1, 3, 224, 224)`.
  - wrapper returns `proprio.shape == (1, 40)`.
  - image dtype is `uint8`.
  - pixel variance is nonzero.
  - a sample wrist frame is saved for manual inspection.
  - a sample debug frame is saved if `debug_camera_name` is configured.

**Suggested Commit**

```bash
git commit -m "feat(env): add camera-enabled Franka lift cfg"
```

---

### Completed Demo-Slice PRs — Already Landed

These PRs were implemented from `plans/subplan_isaac_arm_manipulation.md` to make an interview-ready robotics data loop before the full RL/IL training stack. They remain part of the project foundation because future SAC, PPO, TD3, Diffusion Policy, and DAgger agents should plug into the same interfaces.

| Demo PR | Status | Inputs | Outputs guaranteed for later work | Test coverage |
|---|---|---|---|---|
| PR 8-pre - Demo policies | Done | PR1 action contract and PR2/2.5 observation contract | `BasePolicy.act(obs) -> (7,)`, `RandomPolicy`, `HeuristicPolicy`, and HDF5 `ReplayPolicy` | `tests/test_demo_policies.py`; action shape/range, seeding, heuristic phase behavior, replay policy basics. |
| PR 8-lite - Episode-safe rollout dataset | Done | Policy interface, `IsaacArmEnv`, wrist policy image, 40D proprio, optional debug camera | HDF5 episode groups with `images`, `proprios`, `actions`, rewards, done/truncated flags, optional `successes`, optional native wrist/debug images, and metadata (`source_env_index`, `reset_round`, `reset_seed`, `terminated_by`, `settle_steps`) | `tests/test_demo_dataset.py`, `tests/test_rollout_benchmark.py`; schema, metadata, vectorized lane splitting, action windows, CLI/benchmark plumbing. |
| PR 11-lite - Dataset evaluation metrics | Done | Episode-safe HDF5 rollout files | JSON metrics for return, success, episode length, action jerk, per-episode success, closest target approach, target metadata | `tests/test_eval_metrics.py`; metric keys/ranges, jerk behavior, success-source fallback, JSON save. |
| PR 12-lite - Debug-camera visual outputs | Done | Env/policy loop plus fixed debug camera accessor | GIF, MP4, sampled debug PNGs, optional text/reticle overlays; policy still receives wrist RGB only | `tests/test_visual_outputs.py`; GIF/MP4/frame creation, debug-camera source separation, output-dir handling. |
| Demo PR - One-command data loop | Done | PR8-lite collector, PR11-lite metrics, PR12-lite recorder, demo policies | `scripts.demo_data_loop` creates dataset + metrics + GIF/MP4 + sampled debug PNGs for random, heuristic, or replay policies | `tests/test_demo_data_loop.py`; CLI modes, replay requirements, existing-dataset replay, target overlay, output creation. |

Current full local verification:

```text
conda run -n isaac_arm python -m pytest -q
149 passed, 1 skipped
```

The skipped test is the opt-in Isaac runtime smoke that requires `RUN_ISAAC_RUNTIME_SMOKE=1`.

---

### PR 3 — Shared Backbone

**Status:** Done

**Goal / Why**

Implement the common image-proprio encoder used by RL agents and Diffusion Policy. This isolates perception from algorithm-specific code.

**Inputs**
- PR2.5 observation contract:
  - `image`: `(B, 3, 224, 224)` `uint8` or normalized float.
  - `proprio`: `(B, 40)` `float32`.
- `utils/image_aug.py` augmentation utilities, used by trainers/dataloaders, not by the env wrapper.

**Outputs**
- A reusable encoder module with a stable call signature:

```python
obs_feat = backbone(images, proprios)  # (B, feat_dim)
```

- Configurable `proprio_dim` and `feat_dim`.
- No actor, critic, loss function, replay buffer, rollout buffer, or training script.
- Implemented in `agents/backbone.py` as `ImageProprioBackbone`.

**Implementation**
- Add image CNN encoder.
- Add proprio MLP encoder.
- Add fusion MLP returning `feat_dim=256`.
- Accept configurable `proprio_dim`.
- Normalize images from uint8 `[0, 255]` to float `[0, 1]`.
- Do not add actor, critic, PPO, SAC, TD3, or diffusion logic in this PR.

**How To Test**
- Verify output shape `(B, feat_dim)`.
- Verify uint8 and float image inputs are handled correctly.
- Verify gradients flow through image and proprio branches.
- Verify configurable `proprio_dim`.
- Verify batch sizes `B=1` and `B>1`.
- Verify bad image/proprio shapes raise readable errors.
- Verify train/eval mode does not change output shape.
- Verify the backbone can be serialized and loaded with identical output for deterministic weights/input.

**Pytest**

```bash
pytest tests/test_nn_backbone.py -v
```

Known result in `isaac_arm`:

```text
8 passed
```

**Suggested Commit**

```bash
git commit -m "feat(model): add image-proprio fusion backbone"
```

---

### PR 3.5 — Agent Primitives And Training Interfaces

**Goal / Why**

Add reusable RL building blocks before implementing PPO, GRPO, SAC, and TD3. This keeps the algorithm PRs similar in size and prevents four separate implementations of squashed Gaussian log-probs, actor/checkpoint conventions, and batch containers.

**Inputs**
- PR1 action contract: normalized 7D continuous actions in `[-1, 1]`.
- PR3 backbone output: `obs_feat` with fixed `feat_dim`.
- PR8-lite dataset/replay concepts for off-policy storage, but no SAC expert data yet.

**Outputs**
- Shared squashed Gaussian distribution with correct tanh log-prob correction.
- Deterministic actor head helper for TD3-style policies.
- Value/Q network head helpers that consume `obs_feat` and optional 7D action.
- Rollout batch dataclasses for on-policy methods.
- Replay batch/replay buffer dataclasses for off-policy methods.
- Checkpoint save/load helpers that preserve model config, normalizer/augmentation config if present, optimizer state, and global step.
- Replay-buffer storage tier config:
  - CPU RAM or disk-backed storage;
  - policy images stored as `uint8`;
  - sampled batches moved to GPU only after indexing;
  - memory estimator for image/proprio/action/reward/done storage.
- Explicit terminal-transition fields:
  - `terminated`;
  - `truncated`;
  - `bootstrap_mask`;
  - optional `reset_after_step` diagnostic for auto-reset lanes.
- Checkpoint metadata dataclass matching §8.3.
- Fake checkpoint factory for PR11a/PR12a tests, supporting both SAC-like and TD3-like deterministic policies before real training converges.
- A common evaluation/deployment policy adapter:

```python
action = policy.act(obs, deterministic=True)  # shape (B, 7) or (7,)
```

**Implementation**
- Add `agents/` or `models/` modules for distribution utilities, actor/critic heads, batch containers, replay buffer, and checkpoint helpers.
- Keep algorithm-specific losses out of this PR.
- Keep env rollout collection out of this PR except for tiny fake-env interface tests.
- Define which tensors are stored as `uint8` (`images`) versus `float32` (`proprios`, `actions`, rewards).
- Implement deterministic action modes as named metadata values:
  - SAC: `"tanh_mu"`;
  - TD3: `"actor_no_noise"`.
- Implement a replay memory estimator and print/warn when a requested capacity would exceed a configurable CPU RAM budget.
- Implement fake checkpoint writers/readers used only by tests for eval/GIF plumbing.

**How To Test**
- Verify squashed Gaussian sampled actions are in `[-1, 1]`.
- Verify tanh log-prob correction against a small hand-computed or finite-difference sanity case.
- Verify deterministic actor mode is repeatable and stochastic mode can vary.
- Verify replay buffer preserves image dtype as `uint8` and action/proprio dtype as `float32`.
- Verify rollout/replay batch shape validation rejects malformed tensors.
- Verify checkpoint save/load round trip reproduces action output for deterministic mode.
- Verify replay memory estimator reports about 28 GiB for `200k` single-image observations at `(3, 224, 224)` `uint8`.
- Verify terminal transitions set `bootstrap_mask=0` and nonterminal transitions set `bootstrap_mask=1`.
- Verify fake SAC/TD3 checkpoint factory writes metadata fields needed by PR11a/PR12a.
- Verify checkpoint loading rejects mismatched `action_dim`, `proprio_dim`, or `env_id` with readable errors.

**Pytest**

```bash
pytest tests/test_agent_primitives.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(agents): add continuous-control agent primitives"
```

---

### PR 4 — PPO Baseline

**Goal / Why**

Implement PPO as the stable on-policy actor-critic baseline.

**Inputs**
- PR3 backbone.
- PR3.5 squashed Gaussian actor, value head, rollout batch, checkpoint helpers.
- PR2.5 env wrapper for live rollout smoke and fake vectorized env for fast tests.

**Outputs**
- `PPOAgent` with `act(obs, deterministic=...)`, update, save/load, and train script.
- PPO checkpoints that PR11-full and PR12-full can evaluate/visualize.
- Training logs with policy loss, value loss, entropy, approximate KL, clip fraction, and explained variance.

**Implementation**
- Add 7D Gaussian tanh-squashed actor.
- Add value critic.
- Add rollout buffer with image, proprio, action, reward, done, value, and log probability.
- Implement GAE.
- Implement PPO clipped objective, value loss, and entropy bonus.
- Add train script and checkpoint save/load.
- Do not include GRPO, SAC, TD3, or diffusion code.

**How To Test**
- Verify action shape `(B, 7)`.
- Verify actions are in `[-1, 1]`.
- Verify tanh log-prob correction.
- Verify GAE against a hand-computed example.
- Verify update returns policy/value/entropy loss keys.
- Verify save/load round trip.
- Verify one PPO update changes actor parameters and value parameters on synthetic data.
- Verify a tiny deterministic fake env can run collect -> update -> eval without Isaac.
- Verify checkpoint resume preserves global step and optimizer state.

**Pytest**

```bash
pytest tests/test_ppo_continuous.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(rl): add continuous PPO baseline"
```

---

### PR 5 — Pure GRPO Baseline

**Goal / Why**

Implement pure GRPO as a no-critic on-policy baseline. This tests whether group-relative trajectory comparison can replace a learned value baseline.

**Inputs**
- PR3 backbone.
- PR3.5 squashed Gaussian actor and rollout batch helpers.
- PR4 rollout collection conventions, but without value/GAE fields.

**Outputs**
- `GRPOAgent` with no critic/value module and the same deployment `act(obs, deterministic=...)` interface as PPO.
- Group-return normalization utilities that can be unit-tested independently.
- GRPO checkpoints and logs that PR11-full and PR12-full can compare with PPO/SAC/TD3.

**Implementation**
- Add 7D Gaussian tanh-squashed actor.
- Do not create a critic or value head.
- Collect complete or segmented trajectories.
- Compute trajectory returns.
- Normalize returns within groups.
- Apply PPO-style clipped policy objective using group-relative advantages.
- Do not compute GAE, value targets, or value loss.

**How To Test**
- Verify model has no critic/value parameters.
- Verify group advantages have approximately zero mean and unit variance per valid group.
- Verify changing a fake value function has no effect because no value function exists.
- Verify update works with only policy loss and entropy terms.
- Verify checkpoint files contain no critic/value state.
- Verify a tiny fake env rollout can form groups, update, and evaluate without Isaac.

**Pytest**

```bash
pytest tests/test_grpo_continuous.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(rl): add pure GRPO baseline"
```

---

### PR 6 — SAC Baseline

**Goal / Why**

Implement SAC as the main off-policy sample-efficient baseline and expert oracle for demonstrations.

**Inputs**
- PR3 backbone.
- PR3.5 squashed Gaussian actor, twin-Q head helpers, replay buffer, checkpoint helpers.
- PR8-lite HDF5 collector only for optional evaluation/demo artifacts, not for SAC's online replay buffer.

**Outputs**
- `SACAgent` with stochastic training actions and deterministic oracle mode.
- SAC checkpoint format consumed by PR8-full demo collection and PR10 DAgger oracle labeling.
- Online replay buffer and train script with critic loss, actor loss, alpha loss, entropy, and Q-value logs.
- `scripts.train_sac_continuous` that writes checkpoints under `checkpoints/` and logs under `logs/`.
- Reward sanity probe command or train-script preflight output confirming non-constant dense rewards before long runs.

**Implementation**
- Add 7D squashed Gaussian actor.
- Add twin Q critics.
- Add target critics with soft update.
- Add replay buffer with uint8 images and float32 proprio/actions.
- Add automatic entropy temperature tuning.
- Use `polyak_tau=0.005`, `utd_ratio=1`, initial `alpha=0.2`, and target entropy `-action_dim` by default.
- Apply `PadAndRandomCrop(pad=8)` to sampled replay image batches before critic and actor forward passes, DrQ-style; keep eval/checkpoint action paths deterministic and unaugmented.
- Use separate actor and critic image-proprio backbones for the first implementation; no actor/critic encoder sharing in PR6.
- Ensure actor loss does not detach `Q(s, a_new)` from actor gradients.
- Define deterministic action mode for eval/oracle data collection as `tanh(mu)`.
- Store terminal transitions with `bootstrap_mask=0` so auto-reset `next_obs` cannot bootstrap across episode boundaries.
- Save checkpoint metadata from §8.3, including `num_env_steps`, `global_update_step`, `deterministic_action_mode="tanh_mu"`, replay storage config, and algorithm hyperparameters.
- Emit log keys from §8.3.
- Do not add TD3 or diffusion logic.

**How To Test**
- Verify action shape `(B, 7)` and range `[-1, 1]`.
- Verify actor receives nonzero gradient from Q term.
- Verify critic update reduces loss on synthetic data.
- Verify alpha changes in the correct direction.
- Verify target networks lag online networks.
- Verify replay buffer stores image as uint8 and action as float32.
- Verify sampled replay image batches receive `PadAndRandomCrop(pad=8)` during training updates and eval batches do not.
- Verify deterministic oracle action is repeatable after save/load.
- Verify a tiny fake continuous-control env can run warmup -> update -> eval without Isaac.
- Verify actor and critic optimizer states resume correctly from checkpoint.
- Verify terminal fake-env lanes set `bootstrap_mask=0` and Q targets do not bootstrap through terminal transitions.
- Verify checkpoint metadata includes `num_env_steps`, deterministic action mode, `polyak_tau`, `utd_ratio`, `target_entropy`, and replay storage config.
- Verify actor and critic do not share encoder parameters in the default config.
- Verify reward sanity probe detects constant rewards and fails before long training.

**Pytest**

```bash
pytest tests/test_sac_continuous.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(rl): add continuous SAC baseline"
```

---

### PR 7 — TD3 Baseline

**Goal / Why**

Implement TD3 as the deterministic off-policy baseline.

**Inputs**
- PR3 backbone.
- PR3.5 deterministic actor head, twin-Q head helpers, replay buffer, checkpoint helpers.
- PR6 replay-buffer conventions where possible.

**Outputs**
- `TD3Agent` with deterministic eval action and exploration-noise training action.
- TD3 checkpoints and logs comparable with SAC in PR11-full/PR12-full.
- Shared replay-buffer-compatible training script.
- `scripts.train_td3_continuous` that writes checkpoints under `checkpoints/` and logs under `logs/`.
- Reward sanity probe command or train-script preflight output shared with SAC.

**Implementation**
- Add deterministic 7D actor.
- Add twin Q critics.
- Add target actor and target critics.
- Add target policy smoothing.
- Add delayed actor updates.
- Use `polyak_tau=0.005`, `utd_ratio=1`, `policy_delay=2`, exploration noise sigma `0.1`, target smoothing noise sigma `0.2`, and target noise clip `0.5` by default.
- Apply `PadAndRandomCrop(pad=8)` to sampled replay image batches before critic and actor forward passes, DrQ-style; keep eval/checkpoint action paths deterministic and unaugmented.
- Use separate actor and critic image-proprio backbones for the first implementation; no actor/critic encoder sharing in PR7.
- Add checkpoint save/load.
- Reuse replay buffer format from SAC if already implemented.
- Define deterministic eval action as actor output with no exploration noise.
- Store terminal transitions with `bootstrap_mask=0` so auto-reset `next_obs` cannot bootstrap across episode boundaries.
- Save checkpoint metadata from §8.3, including `num_env_steps`, `global_update_step`, `deterministic_action_mode="actor_no_noise"`, noise hyperparameters, and replay storage config.
- Emit log keys from §8.3.
- Do not add SAC temperature logic or diffusion code.

**How To Test**
- Verify deterministic eval action is repeatable.
- Verify training action with noise varies.
- Verify policy delay skips actor update on the correct steps.
- Verify target smoothing noise is clipped.
- Verify sampled replay image batches receive `PadAndRandomCrop(pad=8)` during training updates and eval batches do not.
- Verify save/load round trip.
- Verify a tiny fake env can run warmup -> update -> eval without Isaac.
- Verify target actor/critic checkpoint state resumes exactly.
- Verify TD3 and SAC can share replay batch shapes without conversion glue.
- Verify terminal fake-env lanes set `bootstrap_mask=0` and Q targets do not bootstrap through terminal transitions.
- Verify target smoothing noise and exploration noise use distinct sigma/clip parameters.
- Verify checkpoint metadata includes `num_env_steps`, deterministic action mode, `polyak_tau`, `utd_ratio`, `policy_delay`, and TD3 noise hyperparameters.
- Verify actor and critic do not share encoder parameters in the default config.
- Verify reward sanity probe detects constant rewards and fails before long training.

**Pytest**

```bash
pytest tests/test_td3_continuous.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(rl): add continuous TD3 baseline"
```

---

### PR 6.5 — Training Logger, LR Scheduler, And Live Monitors

**Goal / Why**

PR6 and PR7 currently log only the final-step scalars to a JSON file. Long Isaac SAC/TD3 runs need:

- live training-loss curves (critic, actor, alpha, q_mean, entropy, replay_size, learning rate),
- terminal-side training progress that shows env-step progress plus the latest train losses, training rollout metrics, and eval metrics during long runs,
- training rollout episode metrics that summarize the exploratory lanes actually entering replay,
- same-Isaac-env deterministic eval lanes for live train-time monitoring without constructing a second Isaac env,
- delayed same-env eval metric start so untrained/warmup policy episodes do not pollute the live eval curve,
- training reset settle steps so replay collection starts from a post-reset stabilized scene when requested,
- per-lane post-auto-reset settle steps so vectorized lanes can cool down independently after done/truncated without stopping sibling lanes,
- periodic fake/separate-env evaluation rollouts that emit `eval/mean_return`, `eval/success_rate`, `eval/mean_episode_length`, `eval/mean_action_jerk`,
- the §8.3 log-key contract actually wired to TensorBoard and/or wandb,
- a learning-rate scheduler so the same code path can run a fixed-LR baseline and the recommended warmup + cosine annealing schedule.

This PR closes that gap before §10.1 measured results are populated.

**Inputs**
- PR6 `SACAgent` and PR7 `TD3Agent`, including their `update()` log dicts.
- PR3.5 `CheckpointPolicy` and PR11a `evaluate_episodes` for in-loop deterministic eval rollouts.
- PR2.5 camera-enabled Isaac env wrapper.
- §8.3 logging key contract.

**Outputs**
- New module `train/loggers.py` with:
  - abstract `TrainLogger` interface (`log_scalars(step, metrics)`, `log_hparams(hparams)`, `close()`),
  - `TensorBoardLogger` (PyTorch SummaryWriter under `--tb-log-dir`),
  - `WandbLogger` (`--wandb-project`, `--wandb-run-name`, `--wandb-mode online|offline|disabled`),
  - `JSONLinesLogger` always-on fallback that writes one JSON object per step to `logs/<run_name>_train.jsonl`,
  - `CompositeLogger` that fan-outs to any subset of the above without method-specific glue.
- New module `train/progress.py` with `TrainProgressReporter`, a tqdm-backed console progress helper with plain-stderr fallback and optional `--progress-log` text-file mirroring. It advances the bar on every vectorized env step, keeps moving during replay warmup, and prints/writes separate one-line `train` / `eval` metric records instead of hiding losses in the tqdm postfix. Loss printing can be throttled by optimizer-update count, so long runs can show losses every N SAC/TD3 train updates even when env-step reporting is sparse.
- New module `train/rollout_metrics.py` with per-lane episode trackers, success inference from `info["success"]` / `info["is_success"]`, and the same `norm(proprio[:, 30:33]) <= 0.02` fallback used by PR11-lite dataset metrics.
- New module `train/lr_scheduler.py` with:
  - `make_scheduler(scheduler_type, optimizer, *, warmup_steps, total_update_steps, ...)`,
  - `scheduler_type="constant"` (default for backwards compat),
  - `scheduler_type="step"` — `torch.optim.lr_scheduler.StepLR` exposed via `--lr-step-size`, `--lr-gamma`,
  - `scheduler_type="warmup_cosine"` — linear warmup over `--lr-warmup-updates` followed by cosine annealing to `--lr-min-lr` over the remaining update budget.
- Integration into `run_sac_train_loop` and `run_td3_train_loop`:
  - per-update logger calls,
  - scheduler step hooks for each optimizer (actor, critic, alpha for SAC; actor, critic for TD3),
  - training rollout episode summaries from replay-writing lanes under `train_rollout/mean_return`, `train_rollout/success_rate`, `train_rollout/mean_episode_length`, `train_rollout/episode_count`,
  - optional same-env deterministic eval rollout summaries under `eval_rollout/mean_return`, `eval_rollout/success_rate`, `eval_rollout/mean_episode_length`, `eval_rollout/episode_count`, with an optional delayed clean-episode start,
  - periodic `eval/*` rollouts every `eval_every_env_steps` individual train transitions using a deterministic `CheckpointPolicy`-equivalent path on a separate eval env when that path is enabled,
  - `train/learning_rate_actor`, `train/learning_rate_critic`, `train/learning_rate_alpha` (when applicable) emitted every step.
- CLI extensions in `scripts.train_sac_continuous` and `scripts.train_td3_continuous`:
  - `--lr-scheduler {constant,step,warmup_cosine}` (default `constant`),
  - `--lr-warmup-updates`, `--lr-step-size`, `--lr-gamma`, `--lr-min-lr`,
  - `--total-update-steps` optional override for scheduler horizon,
  - `--tb-log-dir`, `--wandb-project`, `--wandb-run-name`, `--wandb-mode`, `--jsonl-log` (default `logs/<run_name>_train.jsonl`),
  - `--progress-log PATH` to mirror the human-readable progress lines to a plain text file for post-run terminal-log review,
  - `--progress/--no-progress` (default auto-enabled only for interactive stderr), `--log-every-env-steps` (default `1000`), and `--log-every-train-steps` / `--log-every-updates` (default `100`) for terminal progress refresh cadence,
  - `--eval-every-env-steps` (default `10000` per §8.2), `--eval-num-episodes` (default `5`), `--eval-settle-steps` (default `600`), `--eval-seed` (default `args.seed + 1000`),
  - `--eval-backend {same-as-train,fake,isaac}` and eval env args mirroring the training env args when a separate eval env is needed,
  - `--same-env-eval-lanes N` (default `0`) to reserve the last `N` vectorized lanes for deterministic current-policy monitoring inside the same training env,
  - `--same-env-eval-start-env-steps K` (default `0`) to delay same-env eval metrics until training has reached `K` train transitions and the eval lane has crossed a clean reset boundary,
  - `--rollout-metrics-window N` (default `20`) for rolling completed-episode summaries,
  - `--settle-steps S` (default `0`) to run zero-action physics settle steps after the explicit training env reset and before replay collection,
  - `--per-lane-settle-steps S` (default `0`) to run per-lane zero-action cooldown after Isaac auto-resets a done/truncated lane.

**Implementation**
- Keep all loggers optional: missing wandb / TensorBoard installs degrade to JSONL only, with a single warning at startup. TensorBoard event-file assertions in tests should skip when `torch.utils.tensorboard` is unavailable; JSONL logging remains mandatory.
- Run `logger.log_scalars(env_steps, metrics)` after every `agent.update(...)` so `train/*` curves include warmup updates as soon as updates begin.
- Run `progress.update(env_steps, ...)` after every vectorized env step, after train update logs, after rollout episode summaries, and after periodic eval logs. Include `train/update_step` in train logs so `--log-every-train-steps N` means every N optimizer update calls, not every N env transitions. The progress reporter prints metric lines such as `sac train | train | env_step=... | update=... critic=...`, `sac train | train rollout | env_step=... | train_rollout_return=...`, `sac train | eval rollout | env_step=... | eval_rollout_return=...`, and `sac train | eval | env_step=... | eval_return=...`; `--progress-log` may mirror those exact human-readable lines to disk, while TensorBoard/wandb/JSONL scalar records remain separate machine-readable logs.
- `--log-every-env-steps` and `--log-every-train-steps` are independent cadences. Printing a train-loss line must not reset the env/replay progress cadence; otherwise frequent optimizer logs can starve `sac train | env | ...` lines.
- Eval cadence is measured in individual train transitions (matches §8.2 `eval_every_env_steps=10000`), not in update steps and not in raw vectorized `env.step()` calls. With `num_envs=64` and no same-env eval lanes, one vectorized step advances the cadence counter by 64; with reserved eval lanes, it advances by `num_envs - same_env_eval_lanes`.
- Separate-env periodic eval rollouts must use deterministic actions, the §3.5 `eval_settle_steps=600` default, a separate seed, and a separate eval env so they do not reset or advance the training env.
- Live Isaac separate-env train-time periodic eval is **not** currently supported in-process: constructing a second `IsaacArmEnv` while the training env is alive can cause Isaac Sim to close the app with exit code 0 before training resumes. For live Isaac SAC/TD3 runs, set `--eval-every-env-steps 0` and use `--same-env-eval-lanes N` for live monitoring; run PR11a `scripts.eval_checkpoint_continuous` after saving a checkpoint for final comparable eval. `--eval-backend fake` is acceptable only for logger/progress smoke tests, not for real task metrics.
- The eval rollouts call `evaluate_episodes` from PR11a; results write to `eval/mean_return`, `eval/success_rate`, `eval/mean_episode_length`, `eval/mean_action_jerk`, `eval/episode_successes_count`.
- Same-env eval lanes use deterministic current-policy actions every step inside the same vectorized Isaac env. They share the training env's reset/randomization stream and are meant for live trend monitoring, not final benchmark reporting. Their transitions are excluded from replay, warmup counts, scheduler horizon, and `total_env_steps`.
- When `same_env_eval_lanes > 0`, the last `N` lanes are eval lanes and the first `num_envs - N` lanes are train lanes. `env_steps` counts only train-lane transitions; for example `num_envs=64` and `--same-env-eval-lanes 4` advances the training counter by `60` per vectorized Isaac step.
- `same_env_eval_start_env_steps` must not start a metric from a partial episode. When the threshold is reached, keep stepping the eval lane but ignore metrics until that lane next reports done/truncated and auto-resets; the following episode is the first episode counted in `eval_rollout/*`.
- `settle_steps` mirrors the demo-data-loop reset warmup for the explicit training reset only: call `env.reset(seed)`, run zero actions for `S` steps, then start replay collection from the resulting observation. Do not increment `env_steps`, do not push settle transitions to replay, and do not step optimizers during initial settle.
- `per_lane_settle_steps` is a per-lane cooldown state machine after Isaac auto-reset. If a lane is cooling down, force its action to zero, skip replay insertion, skip rollout/eval metric accumulation, and do not count that lane toward `env_steps`. Sibling lanes whose cooldown is zero continue collecting replay and training normally. If a lane terminates again during cooldown, restart that lane's cooldown from `S`.
- Same-env eval lanes also respect per-lane cooldown: `eval_rollout/*` resumes only after the eval lane's cooldown reaches zero, so live eval metrics start from a post-settle observation.
- Off-policy update semantics must be explicit: the existing PR6/PR7 loops perform `utd_ratio` gradient updates per vectorized environment step after warmup, while `env_steps` counts individual transitions. Derive the default scheduler horizon from the expected number of actual update calls:

```text
num_train_lanes = num_envs - same_env_eval_lanes
num_vector_steps_after_warmup = ceil(max(total_env_steps - warmup_steps, 0) / num_train_lanes)
total_update_steps = num_vector_steps_after_warmup * utd_ratio
```

  If a later PR changes UTD to mean updates per individual transition, update this formula and tests in the same PR.
- Schedulers step when their paired optimizer actually steps. SAC actor, critic, and alpha schedulers step every SAC update. TD3 critic scheduler steps every critic update, but TD3 actor scheduler steps only on delayed actor updates; this prevents the actor LR from decaying on critic-only updates.
- When `--lr-scheduler warmup_cosine` is selected without an explicit `--total-update-steps`, use the formula above and warn if the result is non-positive.
- Persist scheduler state in checkpoints via a trainer-level checkpoint extension, e.g. `agent.save(..., extras_update={"scheduler_state": ...})`, or a dedicated trainer checkpoint writer that merges agent extras with scheduler extras. Resume must restore both optimizer and scheduler state before the next update.
- Keep `extras["scheduler_state"]` optional for old PR6/PR7 checkpoints, but newly saved PR6.5 checkpoints should include it when a non-constant scheduler is configured.
- Do not change SAC/TD3 loss formulas, replay format, or checkpoint metadata schema (other than the new optional `extras["scheduler_state"]` slot).

**How To Test**
- `tests/test_training_logger_and_scheduler.py` covers, on CPU + fake env:
  - `TensorBoardLogger` writes events under `tb-log-dir` and contains `train/critic_loss` after one update,
  - `JSONLinesLogger` writes one parseable JSON object per logged step,
  - `WandbLogger` is exercised with `mode="disabled"` so no network call happens; verify the log calls were forwarded to its proxy run object,
  - `CompositeLogger` fan-out to two backends does not duplicate events for a single backend,
  - `TrainProgressReporter` prints latest train losses every configured train-update interval and eval metrics immediately in its fallback mode, while loop-level progress hooks advance during replay warmup before losses exist,
  - `TrainProgressReporter` keeps env-step and train-update print cadences independent, so `--log-every-env-steps N` still emits env/replay lines even when `--log-every-train-steps` is much smaller,
  - `TrainProgressReporter` prints `train rollout` and `eval rollout` metric lines as standalone records, not as tqdm postfix text, and can mirror those lines to `--progress-log`,
  - `make_scheduler("step", ...)` reduces the actor LR by `gamma` after `step_size` updates,
  - `make_scheduler("warmup_cosine", ...)` LR rises linearly during warmup, then monotonically decays to `min_lr` near the end of the schedule (check three sample points: warmup tail, mid-schedule, last update),
  - `make_scheduler("constant", ...)` keeps LR equal to the optimizer's initial LR,
  - SAC and TD3 train loops emit `train/learning_rate_actor` matching the scheduler value at each logged step,
  - SAC and TD3 train loops reserve `same_env_eval_lanes`, exclude those lanes from replay, and emit both `train_rollout/*` and `eval_rollout/*` metrics to progress plus logger backends,
  - same-env eval metrics respect `same_env_eval_start_env_steps` and wait for the next clean eval-lane episode before logging,
  - training `settle_steps` uses zero actions before replay collection and does not increase replay size or `env_steps`,
  - `per_lane_settle_steps` forces zero actions after lane done/truncated, masks those cooldown transitions out of replay and metrics, and lets sibling lanes keep training,
  - vectorized fake env with `num_envs>1` triggers eval by individual transition count, not by raw `env.step()` count,
  - TD3 actor scheduler advances only on delayed actor optimizer steps while the critic scheduler advances every critic update,
  - periodic `eval_every_env_steps` actually runs at the configured cadence, uses a separate fake eval env instance, and writes `eval/mean_return` / `eval/success_rate` / `eval/mean_action_jerk`,
  - missing wandb install (simulated by patching the import) downgrades cleanly to JSONL-only without raising,
  - missing TensorBoard install downgrades cleanly to JSONL-only, while TensorBoard event assertions run only when the dependency is present,
  - scheduler state round-trips via checkpoint save/load and resume continues from the correct LR for both SAC and TD3.

**Acceptance Criteria**

PR6.5 is complete when:

```bash
pytest tests/test_training_logger_and_scheduler.py -v
```

passes on CPU + fake env, and a smoke command of the form

```bash
python -m scripts.train_sac_continuous \
  --backend fake --total-env-steps 64 --warmup-steps 8 --batch-size 4 \
  --replay-capacity 64 --device cpu --ram-budget-gib 4 \
  --reward-probe-steps 16 \
  --lr-scheduler warmup_cosine --lr-warmup-updates 4 --lr-min-lr 1e-5 \
  --tb-log-dir /tmp/sac_tb --jsonl-log /tmp/sac.jsonl --progress-log /tmp/sac_progress.log \
  --wandb-mode disabled \
  --progress --log-every-env-steps 16 --log-every-train-steps 4 \
  --same-env-eval-lanes 1 --same-env-eval-start-env-steps 32 \
  --rollout-metrics-window 10 --settle-steps 2 --per-lane-settle-steps 2 \
  --eval-every-env-steps 16 --eval-num-episodes 2 --eval-settle-steps 0
```

emits a JSONL log with `train/*`, `train_rollout/*`, `eval_rollout/*`, and at least one `eval/*` line, emits matching human-readable lines to `--progress-log`, emits non-empty TensorBoard event files when TensorBoard is installed, and writes a checkpoint whose `extras["scheduler_state"]` round-trips.

**Pytest**

```bash
pytest tests/test_training_logger_and_scheduler.py -v
```

Latest result: `19 passed`; full suite: `244 passed, 1 skipped`.

**Suggested Commit**

```bash
git commit -m "feat(train): add per-lane settle cooldown"
```

---

### PR 6.6 — Running Observation/Action Normalization

**Goal / Why**

Add the normalization contract needed before serious SAC/TD3 training and before any TD3+BC-style or BC/DAgger loss consumes dataset actions. The important rule is consistency: actor, critic, replay updates, checkpoint eval, visual rollout, and future BC losses must all see the same normalized inputs and action convention.

**Decision On The Proposed Rules**
- Use running per-dimension state normalization, not mini-batch statistics: yes. Online SAC/TD3 should update stats from train-lane observations that enter replay using Welford-style running mean/variance, then apply those frozen-at-use stats during gradient updates and eval.
- Use sin/cos for angles: conditionally yes. Only true periodic/wrap-around angle features should be expanded to sin/cos. The current 40D proprio mostly uses limited Franka joint positions, Cartesian positions/deltas, velocities, gripper values, and previous normalized action, so PR6.6 should keep angle expansion optional/configured instead of blindly changing all joint-position dimensions.
- Normalize state/action per dimension: yes. Proprio stats are per feature. Action normalization is also per action dimension via `bidirectional_env_learner_affine`; the default mapper is numerically identity because the public env action contract is already `[-1,1]` on all seven dimensions.
- Denormalize before `env.step`: yes, with one correction. Denormalize from the learner action space back to the env-normalized 7D action expected by `IsaacArmEnv`, then clip to `[-1,1]`. Do **not** denormalize directly to physical meters/radians unless a future env wrapper explicitly changes its backend action contract.
- Critic input is normalized state + learner-normalized action: yes.
- Actor input is normalized state: yes.
- Actor output is normalized action in `[-1,1]`: yes. The phrase "normalized state in `[-1,1]`" is treated as a typo; actor output is action, not state.
- BC / TD3+BC loss compares actor output with normalized dataset action: yes. When offline dataset actions are already env-normalized `[-1,1]`, the default action mapper keeps them unchanged; if a future dataset uses physical actions or nonstandard scaling, the mapper owns that conversion.

**Inputs**
- PR2.5 40D proprio contract and PR1 normalized 7D action contract.
- PR6 `SACAgent`, PR7 `TD3Agent`, replay buffer, and checkpoint helpers.
- PR6.5 logger/progress hooks and checkpoint `extras` path.
- PR11a/PR12a checkpoint eval/visual paths, which must load and apply the same normalizer state.

**Outputs**
- New module `agents/normalization.py` or `train/normalization.py` with:
  - `RunningMeanStd` / `RunningProprioNormalizer` using numerically stable running count/mean/M2 or equivalent variance accumulation,
  - optional `RunningImageChannelNormalizer` with `--image-normalization none|per_channel_running_mean_std`, default `none`; when enabled, stats are per RGB channel over float `[0,1]` policy images,
  - `normalize(x)`, `update(x)`, `state_dict()`, `load_state_dict()`, `freeze()` semantics,
  - epsilon/clamp config such as `eps=1e-6`, `clip=10.0`,
  - optional configured `AngleFeatureTransform` for explicit angle indices only.
- New `ActionNormalizer` abstraction with:
  - per-dimension env low/high from `TaskConfig`,
  - `env_to_learner(action_env)` and `learner_to_env(action_norm)`,
  - config/state type `bidirectional_env_learner_affine`,
  - default identity behavior for the current `[-1,1]` env contract with seven explicit bounds for `dx, dy, dz, droll, dpitch, dyaw, gripper`,
  - clipping after `learner_to_env` before calling `env.step`.
- SAC/TD3 integration:
  - training loop updates proprio stats and optional image channel stats only from train lanes whose transitions enter replay; same-env eval lanes, settle transitions, and PR11a eval episodes do not update stats,
  - replay stores raw env observations/actions; sampled batches are normalized immediately before actor/critic forward passes,
  - image replay batches are augmented first, then converted to float `[0,1]`, then optionally channel-normalized before actor/critic forward,
  - `agent.act(...)`, `AgentEvalPolicy`, and `CheckpointPolicy` apply the loaded proprio normalizer and optional image normalizer before actor forward,
  - actor outputs learner-normalized actions in `[-1,1]`; training loops convert to env-normalized actions before `env.step`,
  - critic receives normalized proprio/features and learner-normalized actions,
  - SAC/TD3 checkpoint save/load stores `extras["normalizer_state"]` and metadata describing the feature transform and action mapper.
- Logger keys:
  - `normalizer/proprio_count`
  - `normalizer/proprio_mean_abs_max`
  - `normalizer/proprio_std_min`
  - optional `normalizer/image_count`, `normalizer/image_mean_min`, `normalizer/image_mean_max`, `normalizer/image_std_min` when channel-wise image normalization is enabled,
  - optional `normalizer/action_mode` in hparams/config rather than a scalar metric.

**Implementation Notes**
- Do not use per-mini-batch mean/std inside `agent.update`; that would make target-Q computation depend on sampled batch composition and would make checkpoint eval irreproducible.
- Normalizer stats should be updated before gradient updates consume the latest replay transition, then treated as fixed tensors for that update.
- Evaluation must run with normalizer updates disabled. Loading an old checkpoint without `normalizer_state` uses explicit identity proprio/action behavior and baseline uint8-to-`[0,1]` image conversion, and this missing-state fallback is covered by policy tests.
- If angle features are enabled later, checkpoint metadata must include original proprio dim, transformed proprio dim, and angle indices so PR11a/PR12a can rebuild the exact actor input.
- For TD3+BC or future BC losses, dataset actions must pass through `env_to_learner(action_env)` before computing `MSE(actor_action_norm, dataset_action_norm)`.
- The public env wrapper remains stable: `IsaacArmEnv.step()` still receives normalized 7D actions in `[-1,1]`.

**How To Test**
- `tests/test_normalization.py`:
  - running mean/std matches known per-dimension statistics and does not depend on mini-batch order,
  - optional image normalizer computes per-channel RGB stats in float `[0,1]`,
  - `freeze()` prevents updates during eval,
  - zero-variance dimensions are protected by epsilon and finite outputs,
  - state dict round-trips exactly,
  - optional angle transform only changes configured angle indices and records transformed dimensions.
- SAC/TD3 loop tests:
  - proprio and optional image normalizer updates only for replay-writing train lanes, not same-env eval lanes or settle cooldown transitions,
  - critic update receives normalized proprio and learner-normalized action,
  - actor action is converted back to env-normalized action before `env.step`,
  - checkpoint save/load preserves normalizer state and PR11a eval uses it.
- Policy/eval tests:
  - `AgentEvalPolicy` and `CheckpointPolicy` produce identical actions before/after checkpoint round-trip when the same normalizer state is loaded,
  - missing normalizer state behavior is explicit and tested.

**Acceptance Criteria**

PR6.6 status: implemented. The code now keeps replay storage raw, updates proprio stats and optional image channel stats only from replay-writing train lanes, feeds SAC/TD3 actor/critic with normalized proprio, float `[0,1]` images plus optional channel normalization, and learner-normalized actions, converts learner actions back to env-normalized actions before `env.step`, and saves/loads `extras["normalizer_state"]` for checkpoint eval/visualization. Action normalizer metadata uses `type: "bidirectional_env_learner_affine"`.

PR6.6 is complete when:

```bash
pytest tests/test_normalization.py tests/test_sac_continuous.py tests/test_td3_continuous.py tests/test_eval_sac_td3_checkpoints.py -q
```

passes, and a fake SAC/TD3 smoke run shows nonzero `normalizer/proprio_count`; with `--image-normalization per_channel_running_mean_std`, it also shows nonzero `normalizer/image_count` in JSONL/W&B while checkpoint eval loads the same normalizer state.

Current verification:

```bash
timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm pytest -q tests/test_normalization.py tests/test_sac_continuous.py tests/test_td3_continuous.py
# 47 passed

timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm pytest -q
# 259 passed, 1 skipped
```

Tiny fake-backend SAC/TD3 smoke runs also produced nonzero `normalizer/proprio_count=16.0` in JSONL/final logs, and PR11a fake checkpoint eval loaded both checkpoints successfully.

**Suggested Commit**

```bash
git commit -m "feat(train): add running observation and action normalization"
```

---

### PR 6.7 — Training Diagnostics And Checkpoint Controls

**Goal / Why**

The first 500k SAC run reached `success_rate=0` and the final video showed the policy barely interacting with the cube. Before changing reward design or algorithm logic, add diagnostics that make the next run explainable: reward traces in visual rollouts, stock reward-term breakdown during training, intermediate checkpoints, an option to disable the stock reward-penalty curriculum, and a SAC entropy-temperature floor.

**Inputs**
- PR6/PR7 SAC and TD3 train loops.
- PR6.5 logger/progress/wandb/JSONL path.
- PR6.6 checkpoint extras and normalizer state.
- PR12a `scripts.record_gif_continuous` same-rollout metrics path.
- Isaac Lab `RewardManager._step_reward` / `active_terms` for manager-based reward terms.

**Outputs**
- `record_gif_continuous` metrics JSON includes:
  - `visual_rollout_reward_trace`: per-step reward list for the exact recorded episode,
  - `visual_rollout_reward_num_steps`, `visual_rollout_reward_sum`, `visual_rollout_reward_mean`, `visual_rollout_reward_min`, `visual_rollout_reward_max`, `visual_rollout_reward_first`, `visual_rollout_reward_last`.
- SAC/TD3 training logs include stock reward component means:
  - `reward/train/native_total` for active replay-writing train lanes,
  - `reward/train/<term>` for stock terms such as `reaching_object`, `lifting_object`, `object_goal_tracking`, `object_goal_tracking_fine_grained`, `action_rate`, `joint_vel`,
  - `reward/eval_rollout/native_total` and `reward/eval_rollout/<term>` for same-env deterministic eval lanes.
- New checkpoint controls for both SAC and TD3:
  - `--checkpoint-every-env-steps N`: save periodic checkpoints at the first loop step that reaches each N-env-step boundary,
  - `--keep-last-checkpoints K`: keep only the last K periodic checkpoints; final and best checkpoints are not pruned,
  - `--save-best-by KEY`: overwrite `{checkpoint_name}_best.pt` whenever metric `KEY` improves, typically `eval_rollout/mean_return`.
- New Isaac cfg switch:
  - `--disable-reward-curriculum`: removes the stock `action_rate` and `joint_vel` reward-weight curriculum terms while keeping the base reward terms unchanged.
- New SAC option:
  - `--alpha-min 0.05`: clamps learned SAC entropy temperature `alpha` to a lower bound so exploration pressure cannot collapse to near-zero during long runs.

**Implementation Notes**
- Reward component logs should prefer Isaac Lab `RewardManager._step_reward` when available and multiply by `step_dt` so component values are in the same scale as the env reward. Fake tests may provide `info["reward_components"]`.
- Reward components are diagnostics only; they must not change the reward stored in replay or the reward used by SAC/TD3 updates.
- Same-env eval lane component logs are live-trend monitors, not final benchmark metrics. Final comparable evaluation still comes from PR11a checkpoint eval.
- `--disable-reward-curriculum` is a training-control switch, not reward shaping. It keeps the stock reward terms but prevents the later curriculum jump that makes `action_rate` and `joint_vel` penalties much stronger.
- `--alpha-min` affects SAC only. TD3 has no entropy temperature.

**How To Use**

Recommended next SAC diagnostic run:

```bash
python -m scripts.train_sac_continuous \
  --backend isaac \
  --env-id Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num-envs 32 \
  --total-env-steps 500000 \
  --warmup-steps 5000 \
  --batch-size 256 \
  --replay-capacity 200000 \
  --ram-budget-gib 80 \
  --device cuda:0 \
  --learning-rate 3e-4 \
  --polyak-tau 0.005 \
  --utd-ratio 1 \
  --initial-alpha 0.2 \
  --alpha-min 0.05 \
  --target-entropy auto \
  --image-normalization none \
  --lr-scheduler warmup_cosine \
  --lr-warmup-updates 3000 \
  --lr-min-lr 1e-5 \
  --settle-steps 550 \
  --per-lane-settle-steps 20 \
  --same-env-eval-lanes 4 \
  --same-env-eval-start-env-steps 10000 \
  --rollout-metrics-window 20 \
  --eval-every-env-steps 0 \
  --disable-reward-curriculum \
  --checkpoint-every-env-steps 50000 \
  --keep-last-checkpoints 5 \
  --save-best-by eval_rollout/mean_return \
  --reward-probe-steps 200 \
  --progress \
  --log-every-train-steps 100 \
  --log-every-env-steps 1000 \
  --checkpoint-dir ./checkpoints \
  --checkpoint-name sac_franka_seed0_diag \
  --logs-dir ./logs \
  --jsonl-log ./logs/sac_franka_seed0_diag_train.jsonl \
  --progress-log ./logs/sac_franka_seed0_diag_progress.log \
  --tb-log-dir ./logs/tb/sac_franka_seed0_diag \
  --wandb-project isaac-arm \
  --wandb-run-name sac_franka_seed0_diag \
  --wandb-mode online
```

Record the final or best checkpoint with a reward trace:

```bash
python -m scripts.record_gif_continuous \
  --backend isaac \
  --agent-type sac \
  --checkpoint ./checkpoints/sac_franka_seed0_diag_best.pt \
  --save-gif ./logs/sac_franka_seed0_diag_best.gif \
  --save-mp4 ./logs/sac_franka_seed0_diag_best.mp4 \
  --save-metrics ./logs/sac_franka_seed0_diag_best_visual_metrics.json \
  --save-debug-frames-dir ./logs/sac_franka_seed0_diag_best_debug_frames \
  --num-envs 1 \
  --seed 0 \
  --device cuda:0 \
  --settle-steps 550 \
  --gif-max-steps 230 \
  --target-overlay text-reticle \
  --headless
```

**How To Test**
- `tests/test_visual_sac_td3_checkpoints.py` verifies `record_gif_continuous` writes the per-step reward trace and keeps it even when an external PR11a metrics payload is used for overlay.
- `tests/test_training_logger_and_scheduler.py` verifies SAC and TD3 log `reward/train/*`, `reward/eval_rollout/*`, parser support for checkpoint/curriculum switches, and periodic/best checkpoint save-prune behavior.
- `tests/test_camera_enabled_env_cfg.py` verifies `--disable-reward-curriculum` removes only the stock `action_rate` and `joint_vel` curriculum terms.
- `tests/test_sac_continuous.py` verifies `alpha_min` clamps SAC entropy temperature and is checkpointed in algorithm hparams.

Current targeted verification:

```bash
timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q \
  tests/test_sac_continuous.py \
  tests/test_training_logger_and_scheduler.py \
  tests/test_visual_sac_td3_checkpoints.py \
  tests/test_camera_enabled_env_cfg.py
# 70 passed
```

Full verification:

```bash
timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q
# 283 passed, 1 skipped
```

**Suggested Commit**

```bash
git commit -m "feat(train): add reward diagnostics and checkpoint controls"
```

---

### PR 6.8 — Curriculum Reward And Bucket-Rarity Replay

**Goal / Why**

Two 500k SAC diagnostic runs reached `success_rate=0`. PR6.7 reward breakdowns showed
that deterministic eval rollouts rarely get meaningful `reaching_object` reward and never
reliably enter the `lifting_object` or `object_goal_tracking` regions. The next opt-in
SAC/TD3 training improvement should make sparse manipulation progress easier to learn
without demonstrations, behavior cloning, expert actions, or heuristic labels.

PR6.8 adds:
- staged training reward curriculum for reach -> grip -> lift -> goal,
- a small grip-proxy bridge reward because the stock reward has reach and lift terms but
  no immediate "close the gripper near the cube" bridge,
- bucket-rarity prioritized replay so rare task-progress transitions found by the agent
  are sampled more often and retained longer.

This PR is **not vanilla SAC/TD3**. It should be reported as:

```text
SAC/TD3 with task-aware reward curriculum and bucket-rarity prioritized replay.
```

Final comparison still uses PR11a/PR12a stock-env evaluation with deterministic actions.

**Inputs**
- PR6/PR7 SAC and TD3 train loops.
- PR6.5 logger/progress/W&B/JSONL path.
- PR6.7 stock reward component extraction and checkpoint manager.
- Replay buffer transition storage in `agents.replay_buffer`.
- Formal 40D proprio contract:
  - `proprio[:, 14:16]`: gripper finger positions,
  - `proprio[:, 21:24]`: cube position in robot base frame,
  - `proprio[:, 27:30]`: `ee_to_cube`,
  - `proprio[:, 30:33]`: `cube_to_target`,
  - `action[:, 6]`: gripper command, where negative closes the gripper.

**Outputs**
- New reward curriculum module, recommended file: `train/reward_curriculum.py`.
- New prioritized replay metadata/sampling path, either in `agents.replay_buffer` or a
  helper such as `agents.prioritized_replay.py`.
- SAC/TD3 train CLI flags:
  - `--reward-curriculum none|reach_grip_lift_goal`, default `none`.
  - `--curriculum-stage-fracs 0.2,0.5,0.8`, interpreted as fractions of
    `total_env_steps`, not absolute env-step numbers.
  - `--grip-proxy-scale FLOAT`, default `1.0`.
  - `--grip-proxy-sigma-m FLOAT`, default `0.05`.
  - `--prioritize-replay`, default off.
  - `--priority-replay-ratio FLOAT`, default `0.5` when enabled.
  - `--priority-score-weights rarity,reward,return,td_error`, default
    `0.40,0.25,0.20,0.15`.
  - `--priority-rarity-power FLOAT`, default `0.5`, implementing
    `rarity = 1 / (count + eps) ** power`.
  - `--priority-rarity-eps FLOAT`, default `1.0`.
  - `--protect-rare-transitions`, default off.
  - `--protected-replay-fraction FLOAT`, default `0.2`.
  - `--protected-score-weights rarity,reward,return`, default
    `0.60,0.25,0.15`.
- New logs:
  - `curriculum/stage_index`, `curriculum/stage_progress`, and
    `curriculum/stage/<stage_name>` numeric mirrors for logger backends.
  - `reward/train_shaped`, the exact reward stored in replay when curriculum is enabled.
  - `reward/train/grip_proxy`.
  - `reward/eval_rollout/eval_shaped` and `reward/eval_rollout/grip_proxy` as same-env deterministic eval diagnostics; `eval_rollout/mean_return` remains stock reward.
  - `train/td_error_mean` from SAC/TD3 critic updates.
  - `priority_replay/batch_uniform`, `priority_replay/batch_priority`,
    `priority_replay/mean_priority_score`, `priority_replay/protected_count`.
  - `priority_replay/bucket_count/<bucket>` and `priority_replay/bucket_rarity/<bucket>`
    for `normal`, `reach`, `grip`, `lift`, and `goal`.

**Reward Curriculum Design**

Do not fork Isaac Lab's task cfg for this PR. Keep the env reward untouched. In the train
loop, convert the stock env reward/components into an opt-in training reward before
pushing the transition into replay:

```text
env.step(action)
  -> stock_reward, stock reward components, obs/proprio/action
  -> shaped_train_reward = curriculum_shaper(...)
  -> replay.push(reward=shaped_train_reward)
```

When `--reward-curriculum none`, `shaped_train_reward == stock_reward`.

`--disable-reward-curriculum` and `--reward-curriculum` are separate switches:
`--disable-reward-curriculum` disables Isaac Lab's stock penalty-weight curriculum for
`action_rate` and `joint_vel`, while `--reward-curriculum` enables this project's
training-reward shaping before replay insertion.

When `--reward-curriculum reach_grip_lift_goal`, compute:

```text
train_reward =
  w_reach(stage)  * reaching_object
+ w_grip(stage)   * grip_proxy
+ w_lift(stage)   * lifting_object
+ w_goal(stage)   * object_goal_tracking
+ w_fine(stage)   * object_goal_tracking_fine_grained
+ w_action(stage) * action_rate
+ w_joint(stage)  * joint_vel
```

`action_rate` and `joint_vel` are already negative stock penalties, so multiplying them
by values below `1.0` weakens the penalty during early exploration; it does not turn them
into rewards.

Stage boundaries are fractions of `total_env_steps`:

| Stage | Fraction range | Intent |
|---|---:|---|
| 1 | `0.00 <= progress < 0.20` | Make reaching the cube common. |
| 2 | `0.20 <= progress < 0.50` | Bridge reach into near-cube gripper closing. |
| 3 | `0.50 <= progress < 0.80` | Emphasize lift and start goal tracking. |
| 4 | `0.80 <= progress <= 1.00` | Return toward stock-like task reward. |

Default multipliers:

| Stage | `w_reach` | `w_grip` | `w_lift` | `w_goal` | `w_fine` | `w_action` | `w_joint` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 reach | `3.0` | `0.5` | `0.25` | `0.0` | `0.0` | `0.25` | `0.25` |
| 2 grip/pre-lift | `1.5` | `2.0` | `1.0` | `0.25` | `0.0` | `0.5` | `0.5` |
| 3 lift | `0.75` | `1.0` | `2.0` | `1.0` | `0.5` | `0.75` | `0.75` |
| 4 stock-like | `1.0` | `0.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |

Grip proxy:

```text
ee_to_cube = proprio[:, 27:30]
near_cube = exp(-norm(ee_to_cube) / grip_proxy_sigma_m)
close_cmd = clip(-action[:, 6], 0, 1)
grip_proxy = grip_proxy_scale * near_cube * close_cmd
```

This is intentionally gated by `near_cube`: closing the gripper far from the cube should
not receive meaningful reward. The proxy is a bridge between reach and lift, not a final
success metric, and it fades out in the stock-like final stage by default.

**Bucket-Rarity Replay Design**

Every transition can receive zero or more progress labels. There is no manual bucket
importance ordering. The sampler only uses bucket frequency.

| Label | Definition |
|---|---|
| `reach` | `norm(ee_to_cube) <= reach_threshold_m`; do not use `reaching_object > 0` because Isaac's dense reaching reward is positive for almost all finite distances and would make nearly every transition a reach sample. |
| `grip` | `norm(ee_to_cube) <= grip_threshold_m` and either `action[:, 6] < close_command_threshold` or the gripper finger gap is below `closed_finger_gap_threshold_m`. |
| `lift` | stock `lifting_object > 0` or cube height is above the lane's reset-time cube height by at least `lift_delta_m`. |
| `goal` | stock `object_goal_tracking > 0`, stock `object_goal_tracking_fine_grained > 0`, or `norm(cube_to_target) <= goal_threshold_m`. |
| `normal` | No progress labels apply. |

Recommended defaults:

```text
reach_threshold_m = 0.08
grip_threshold_m = 0.05
close_command_threshold = -0.25
closed_finger_gap_threshold_m = 0.035
lift_delta_m = 0.04
goal_threshold_m = 0.08
```

Because transitions are multi-label, a transition may be both `reach` and `grip`, or
`reach`, `grip`, and `lift`. Compute each bucket count from labels:

```text
bucket_count[label] = number of valid replay transitions currently carrying label
bucket_rarity[label] = 1 / (bucket_count[label] + eps) ** priority_rarity_power
```

For a multi-label transition, use:

```text
transition_rarity = max(bucket_rarity[label] for label in transition_labels)
```

The `max` rule is not a task-progress ranking. It only means that if a transition contains
one very rare label, that rarity should not be diluted by also carrying a common label. If
a transition has no progress labels, it carries only `normal`.

Priority score:

```text
priority_score =
  0.40 * transition_rarity_percentile
+ 0.25 * step_reward_percentile
+ 0.20 * episode_return_percentile
+ 0.15 * td_error_percentile
```

Notes:
- `step_reward_percentile` uses the reward actually stored in replay: stock reward when
  curriculum is off, shaped training reward when curriculum is on.
- `episode_return_percentile` is filled in when a vectorized lane episode ends; until then
  the transition can use the current partial return or a neutral default.
- `td_error_percentile` is updated after SAC/TD3 critic updates. New transitions should
  start with a neutral or optimistic default so they are not starved before their first
  TD-error measurement.

Mixed sampling:

```text
priority_count = round(batch_size * priority_replay_ratio)
uniform_count = batch_size - priority_count
```

With `batch_size=256` and `--priority-replay-ratio 0.5`, each update samples:

```text
128 uniform transitions from all valid replay entries
128 priority transitions with probability proportional to priority_score
```

Then concatenate and shuffle those 256 transitions before passing them to the SAC/TD3
update. If priority replay is disabled or has no valid scores yet, the entire batch falls
back to uniform sampling.

**Rare Transition Protection**

Protection prevents the circular replay cursor from immediately overwriting the rarest
high-value transitions. It should be off by default and capped.

Define:

```text
protected_score =
  0.60 * transition_rarity_percentile
+ 0.25 * step_reward_percentile
+ 0.15 * episode_return_percentile
```

The three protected-score weights are configurable through
`--protected-score-weights rarity,reward,return`. They are normalized internally,
so `0.60,0.25,0.15` and `60,25,15` are equivalent. Keep TD-error out of
`protected_score`: TD-error is useful for sampling priority, but it changes every
critic update and would make the protected set churn too aggressively.

A transition is eligible for protection when its `protected_score` is in the current top
decile among valid replay transitions. Protected entries must still obey:

```text
protected_count <= protected_replay_fraction * replay_capacity
```

When the protected pool is full, demote the protected transition with the lowest
`protected_score` before protecting a newer higher-score transition. Never protect settle
transitions, same-env eval-lane transitions, or invalid transitions that do not enter
replay.

This defines "rare" from observed replay frequency and rewards, not from a hard-coded
statement that one progress bucket is more important than another.

**How To Use**

Recommended SAC v3 run after implementing PR6.8:

```bash
python -m scripts.train_sac_continuous \
  --backend isaac \
  --env-id Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num-envs 32 \
  --seed 0 \
  --total-env-steps 500000 \
  --warmup-steps 5000 \
  --batch-size 256 \
  --replay-capacity 200000 \
  --ram-budget-gib 80 \
  --device cuda:0 \
  --learning-rate 3e-4 \
  --polyak-tau 0.005 \
  --utd-ratio 1 \
  --initial-alpha 0.2 \
  --alpha-min 0.05 \
  --target-entropy auto \
  --image-normalization none \
  --lr-scheduler warmup_cosine \
  --lr-warmup-updates 3000 \
  --lr-min-lr 5e-5 \
  --settle-steps 550 \
  --per-lane-settle-steps 20 \
  --same-env-eval-lanes 4 \
  --same-env-eval-start-env-steps 50000 \
  --rollout-metrics-window 20 \
  --eval-every-env-steps 0 \
  --reward-probe-steps 200 \
  --disable-reward-curriculum \
  --reward-curriculum reach_grip_lift_goal \
  --curriculum-stage-fracs 0.2,0.5,0.8 \
  --grip-proxy-scale 1.0 \
  --grip-proxy-sigma-m 0.05 \
  --prioritize-replay \
  --priority-replay-ratio 0.5 \
  --priority-score-weights 0.40,0.25,0.20,0.15 \
  --priority-rarity-power 0.5 \
  --priority-rarity-eps 1.0 \
  --protect-rare-transitions \
  --protected-replay-fraction 0.2 \
  --protected-score-weights 0.60,0.25,0.15 \
  --checkpoint-every-env-steps 50000 \
  --keep-last-checkpoints 5 \
  --save-best-by eval_rollout/mean_return \
  --progress \
  --log-every-train-steps 100 \
  --log-every-env-steps 1000 \
  --checkpoint-dir ./checkpoints \
  --checkpoint-name sac_franka_500k_seed0_v3_curriculum_priority \
  --logs-dir ./logs \
  --jsonl-log ./logs/sac_franka_500k_seed0_v3_curriculum_priority_train.jsonl \
  --progress-log ./logs/sac_franka_500k_seed0_v3_curriculum_priority_progress.log \
  --tb-log-dir ./logs/tb/sac_franka_500k_seed0_v3_curriculum_priority \
  --wandb-project isaac-arm \
  --wandb-run-name sac_franka_500k_seed0_v3_curriculum_priority \
  --wandb-mode online
```

Recommended ablations:

```text
A. SAC v2 diagnostic baseline:
   no PR6.8 flags

B. Curriculum only:
   --reward-curriculum reach_grip_lift_goal
   --curriculum-stage-fracs 0.2,0.5,0.8

C. Prioritized replay only:
   --prioritize-replay
   --priority-replay-ratio 0.5
   --protect-rare-transitions

D. Curriculum + prioritized replay:
   all PR6.8 flags above
```

Primary signals to watch:
- `reward/train_shaped` versus `reward/train/native_total` so shaping cannot hide stock
  reward regressions.
- `curriculum/stage_index` and `curriculum/stage/<stage_name>` to confirm stage transitions occur at the intended fractions.
- `priority_replay/bucket_count/lift` and `priority_replay/bucket_count/goal`; if these
  remain zero, prioritized replay cannot invent progress that exploration never finds.
- `eval_rollout/success_rate`, `eval_rollout/mean_return`, and final PR11a metrics.
- `reward/eval_rollout/eval_shaped` and `reward/eval_rollout/grip_proxy` to debug whether the deterministic eval policy is improving under the same curriculum objective, without replacing stock eval return.

**How To Test**

Add focused tests before live Isaac runs:

- `tests/test_reward_curriculum.py`
  - parses stage fractions as proportions of `total_env_steps`,
  - returns stock reward unchanged when curriculum is `none`,
  - applies the default stage multiplier table exactly,
  - computes `grip_proxy = scale * exp(-||ee_to_cube|| / sigma) * clip(-gripper_action, 0, 1)`,
  - gives near-zero grip proxy when the end effector is far from the cube,
  - assigns multi-label progress buckets without bucket importance ordering,
  - assigns `normal` only when no progress label applies,
  - logs `curriculum/*`, `reward/train_shaped`, and `reward/train/grip_proxy`.
- `tests/test_prioritized_replay.py`
  - stores multi-label bucket metadata and bucket counts,
  - computes bucket rarity from observed counts with `1 / (count + eps) ** power`,
  - uses `max` rarity for multi-label transitions,
  - samples the requested uniform/priority split for a deterministic seed,
  - tracks replay sample indices and updates TD-error priority scores after critic updates,
  - enforces `protected_replay_fraction`,
  - validates and applies configurable `protected_score_weights`,
  - keeps protected rare transitions from being overwritten while unprotected slots exist.
- SAC/TD3 loop/logger tests, likely in `tests/test_training_logger_and_scheduler.py`
  - verify same-env eval lanes and settle cooldown transitions do not enter replay,
  - verify curriculum reward, not stock reward, is stored when curriculum is enabled,
  - verify stock reward and stock reward components are still logged,
  - verify same-env eval logs include `reward/eval_rollout/native_total`, `reward/eval_rollout/eval_shaped`, `reward/eval_rollout/grip_proxy`, and the stock component terms,
  - verify JSONL/progress/W&B logger shims emit curriculum and priority-replay metrics.

Targeted verification command:

```bash
timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q \
  tests/test_reward_curriculum.py \
  tests/test_prioritized_replay.py \
  tests/test_sac_continuous.py \
  tests/test_td3_continuous.py \
  tests/test_training_logger_and_scheduler.py
```

Full verification command:

```bash
timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q
```

**Acceptance Criteria**

PR6.8 is complete when:
- all new flags default to off and reproduce PR6.7 behavior,
- `--reward-curriculum reach_grip_lift_goal` changes only the training reward stored in
  replay and never changes PR11a/PR12a stock-env eval,
- bucket-rarity replay uses bucket frequencies, not a hand-coded bucket importance order,
- mixed sampling produces the configured uniform/priority batch split,
- rare-transition protection obeys `protected_replay_fraction`,
- rare-transition protection uses configurable normalized `protected_score_weights`,
- logs make shaped reward, stock reward, bucket counts, rarity, and protected counts
  visible in console progress, JSONL, and W&B,
- targeted tests and full pytest pass in the `isaac_arm` environment.

**Suggested Commit**

```bash
git commit -m "feat(train): add curriculum reward and bucket-rarity replay"
```

---

### PR 6.9 — Progress-Gated Lift Curriculum And Lift-Aware Diagnostics

**Goal / Why**

PR6.8 made curriculum reward and bucket-rarity replay available, but the
`sac_franka_500k_seed0_v4_reachbucketfix_curr_prio` run still ended with
`success_rate=0`. The run did produce a small number of `reach`, `grip`, `lift`, and
`goal` labels, but the policy did not reliably convert near-cube closing into lift. The
best checkpoint was also selected by `eval_rollout/mean_return`, which can prefer a
reach-only policy because dense stock reaching reward can rise while lift/success remain
zero.

PR6.9 keeps PR6.8's no-demo, no-BC, no-expert-action constraint and tightens the SAC/TD3
training objective/debug loop around the missing behavior:

```text
reach cube -> close gripper near cube -> make cube height increase -> move lifted cube to target
```

**Inputs**
- PR6.8 reward curriculum, grip proxy, bucket labels, bucket-rarity replay, and protected replay.
- PR6.7 stock reward component extraction and checkpoint manager.
- PR6.5 same-Isaac-env deterministic eval lanes.
- 40D proprio contract:
  - `proprio[:, 21:24]`: cube position in robot base frame,
  - `proprio[:, 27:30]`: `ee_to_cube`,
  - `proprio[:, 30:33]`: `cube_to_target`,
  - `action[:, 6]`: gripper command; negative closes.

**Coverage Of The Nine Review Items**

| Item | PR6.9 decision |
|---|---|
| 1. Progress-gated curriculum | Add opt-in bucket-rate gates. The stage no longer advances only because env steps crossed a fraction. |
| 2. Dense lift proxy | Add `lift_progress_proxy = clip((next_cube_z - cube_reset_z - 0.002) / 0.04, 0, 1)`. Do not multiply it by gripper action. If the cube really moves upward, that is useful progress by itself. |
| 3. Proposed hyperparameter changes | The important PR code change is gating + lift proxy + better best metric. `--curriculum-stage-fracs 0.45,0.75,0.95` is not required once gates are active. Keep `--grip-proxy-scale`/`--grip-proxy-sigma-m` configurable but do not change defaults just to compensate for missing lift. Lower protected replay is recommended for the next run. `--alpha-min 0.10` is a run-level exploration knob, not a required PR6.9 code path. |
| 4. Lift-aware metrics | Add `eval_rollout/max_cube_lift_m`, `eval_rollout/min_ee_to_cube_m`, `eval_rollout/min_cube_to_target_m`, and `eval_rollout/gripper_close_near_cube_rate`. |
| 5. Grip bucket weakness | Keep the existing `grip` bucket for replay, but add diagnostic counts for `grip_attempt` and `grip_effect` so W&B shows whether the policy closes near the cube but fails to move it. |
| 6. Lift-aware best selection | Add composite best selection: success first, max lift second, mean return third. Do not use `max_lifting_object`; `max_cube_lift_m` is clearer and less redundant. |
| 7. Progress-gated curriculum details | Stage transitions are gated by recent observed bucket rates: reach gate, then grip gate, then lift gate. |
| 8. Lower protected replay | Keep the existing configurable protected score path and recommend `--protected-replay-fraction 0.02` plus `--protected-score-weights 0.80,0.10,0.10` for the next run. |
| 9. Action diagnostics | Add gripper action logs for active train lanes and deterministic eval lanes so W&B shows whether the policy is actually sending close commands. |

**Outputs / CLI Contract**

New or extended SAC/TD3 train CLI flags:

```text
--curriculum-gating none|bucket_rates
--curriculum-gate-window-transitions INT
--curriculum-gate-thresholds reach,grip,lift
--lift-progress-deadband-m FLOAT
--lift-progress-height-m FLOAT
--save-best-by composite:success_lift_return
```

Recommended defaults:

```text
--curriculum-gating none
--curriculum-gate-window-transitions 20000
--curriculum-gate-thresholds 0.002,0.0005,0.0001
--lift-progress-deadband-m 0.002
--lift-progress-height-m 0.04
```

Notes:
- Defaults must preserve PR6.8 behavior when the new flags are not used.
- `--curriculum-stage-fracs` remains the fixed-fraction PR6.8 behavior when
  `--curriculum-gating none`.
- When `--curriculum-gating bucket_rates`, stage advancement is controlled by the recent
  bucket-rate gates, not by fixed env-step fractions. Keep logging `curriculum/stage_progress`
  as context, but do not auto-advance only because `env_steps / total_env_steps` crossed a
  boundary.
- Composite best selection is a deterministic lexicographic comparison:

```text
(eval_rollout/success_rate, eval_rollout/max_cube_lift_m, eval_rollout/mean_return)
```

This means a checkpoint with any higher success rate beats lower-success checkpoints; when
success is tied, the checkpoint that lifts the cube higher wins; when both are tied, mean
return breaks the tie.

**Reward Design**

Keep PR6.8's grip proxy. Do not add a second finger-closed reward to the main training
objective in PR6.9: finger gap can shrink during an empty close, so it is not reliable
evidence that the cube is grasped.

Add a dense lift progress proxy:

```text
lift_delta = next_cube_z - cube_reset_z
lift_progress_proxy = clip((lift_delta - lift_progress_deadband_m) / lift_progress_height_m, 0, 1)
```

Default:

```text
lift_progress_proxy = clip((next_cube_z - cube_reset_z - 0.002) / 0.04, 0, 1)
```

Interpretation:
- Ignore the first `2mm` to avoid rewarding camera/physics jitter.
- Give smooth credit from `2mm` through roughly `4.2cm` above reset height.
- Do not multiply by gripper action. If the cube height truly increases, the transition is
  useful regardless of which action component caused it.

When `--reward-curriculum reach_grip_lift_goal` is enabled in PR6.9, compute:

```text
train_reward =
  w_reach(stage)         * reaching_object
+ w_grip(stage)          * grip_proxy
+ w_lift_progress(stage) * lift_progress_proxy
+ w_lift_stock(stage)    * lifting_object
+ w_goal(stage)          * object_goal_tracking
+ w_fine(stage)          * object_goal_tracking_fine_grained
+ w_action(stage)        * action_rate
+ w_joint(stage)         * joint_vel
```

Recommended PR6.9 stage weights:

| Stage | Intent | `w_reach` | `w_grip` | `w_lift_progress` | `w_lift_stock` | `w_goal` | `w_fine` | `w_action` | `w_joint` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 reach | Make near-cube states common. | `3.0` | `0.5` | `0.0` | `0.0` | `0.0` | `0.0` | `0.25` | `0.25` |
| 2 grip/pre-lift | Encourage closing at the cube and tiny lift progress. | `1.5` | `2.0` | `0.5` | `0.25` | `0.0` | `0.0` | `0.5` | `0.5` |
| 3 lift | Make cube-height increase the main training signal. | `0.75` | `1.0` | `2.0` | `1.0` | `0.5` | `0.25` | `0.75` | `0.75` |
| 4 stock-like | Return to stock task reward. | `1.0` | `0.0` | `0.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |

`reward/train_shaped` is still the reward stored in replay. `reward/train/native_total`
and stock component logs must remain visible so shaping cannot hide task regressions.

**Progress-Gated Curriculum**

Maintain a rolling window over active train-lane transitions that enter replay. Do not use
settle transitions and do not use same-env eval lanes.

For each inserted transition, PR6.8 already computes multi-label progress buckets. PR6.9
uses the recent window to compute:

```text
reach_rate = count(reach labels in window) / window_size
grip_rate  = count(grip labels in window) / window_size
lift_rate  = count(lift labels in window) / window_size
```

Stage advancement rule:

```text
stage 1 reach -> stage 2 grip/pre-lift when reach_rate >= gate_reach
stage 2 grip  -> stage 3 lift          when grip_rate  >= gate_grip
stage 3 lift  -> stage 4 stock-like    when lift_rate  >= gate_lift
```

If the next gate is not met, keep the current stage and log:

```text
curriculum/gate/held_stage = 1
```

If a transition advances the stage, log `held_stage = 0` for that update. The log is a
numeric backend-friendly value; progress messages can print the stage name.

Required W&B/JSONL/progress logs:

```text
curriculum/gate/reach_rate
curriculum/gate/grip_rate
curriculum/gate/lift_rate
curriculum/gate/held_stage
curriculum/stage_index
curriculum/stage/<stage_name>
reward/train/lift_progress_proxy
reward/eval_rollout/lift_progress_proxy
```

These logs should make it obvious whether a run is held in the reach, grip, or lift stage
because actual behavior has not appeared yet.

**Grip Attempt / Grip Effect Diagnostics**

Keep PR6.8's existing `grip` bucket for prioritized replay. Add two diagnostic counts:

```text
grip_attempt =
  norm(ee_to_cube) <= grip_threshold_m
  and action[:, 6] < close_command_threshold

grip_effect =
  grip_attempt
  and (
    next_cube_z - cube_reset_z > lift_progress_deadband_m
    or norm(next_cube_pos - cube_pos) > cube_motion_effect_threshold_m
  )
```

Recommended:

```text
cube_motion_effect_threshold_m = 0.005
```

Log:

```text
priority_replay/bucket_count/grip_attempt
priority_replay/bucket_count/grip_effect
```

These are diagnostic counts, not a manual bucket importance order. The immediate question is:

```text
grip_attempt high, grip_effect low -> policy closes near the cube but does not affect it
grip_attempt low                  -> policy is not even trying to close near the cube
grip_effect rising                -> policy starts causing useful cube motion
```

**Lift-Aware Eval Metrics**

Same-env deterministic eval lanes and `record_gif_continuous`/PR11a-compatible eval paths
should compute the following from the rollout:

```text
eval_rollout/max_cube_lift_m =
  max(cube_z - cube_reset_z)

eval_rollout/min_ee_to_cube_m =
  min(norm(ee_to_cube))

eval_rollout/min_cube_to_target_m =
  min(norm(cube_to_target))

eval_rollout/gripper_close_near_cube_rate =
  mean(norm(ee_to_cube) <= grip_threshold_m and action[:, 6] < close_command_threshold)
```

Do not add `eval_rollout/max_lifting_object` in PR6.9. It is redundant with
`max_cube_lift_m` and less readable. If a future report needs stock reward-term maxima,
add them as a separate diagnostics PR.

**Action Diagnostics**

Log gripper action behavior separately for active training lanes and deterministic eval
lanes:

```text
action/train/gripper_mean
action/train/gripper_close_rate
action/eval_rollout/gripper_mean
action/eval_rollout/gripper_close_rate
action/eval_rollout/gripper_close_near_cube_rate
```

Definitions:
- `gripper_mean`: mean of `action[:, 6]`; more negative means the policy is sending more
  close-side commands.
- `gripper_close_rate`: fraction of actions with `action[:, 6] < close_command_threshold`.
- `gripper_close_near_cube_rate`: fraction of steps where the policy is both near the cube
  and sending a close-side command.

This answers whether the failure is "does not close", "closes but not near the cube", or
"closes near the cube but still does not move the cube".

**Protected Replay Defaults For Next Run**

PR6.9 should not remove PR6.8 protected replay. It should document and support a more
conservative next-run setting:

```text
--protected-replay-fraction 0.02
--protected-score-weights 0.80,0.10,0.10
```

Meaning:

```text
protected_score =
  0.80 * transition_rarity_percentile
+ 0.10 * step_reward_percentile
+ 0.10 * episode_return_percentile
```

This keeps the protected pool small and rarity-focused. It reduces the risk that a large
protected set becomes mostly old normal/reach transitions.

**How To Use**

Recommended SAC v5 run after PR6.9:

```bash
python -m scripts.train_sac_continuous \
  --backend isaac \
  --env-id Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num-envs 32 \
  --seed 0 \
  --total-env-steps 500000 \
  --warmup-steps 5000 \
  --batch-size 256 \
  --replay-capacity 200000 \
  --ram-budget-gib 80 \
  --device cuda:0 \
  --learning-rate 3e-4 \
  --polyak-tau 0.005 \
  --utd-ratio 1 \
  --initial-alpha 0.2 \
  --alpha-min 0.10 \
  --target-entropy auto \
  --image-normalization none \
  --lr-scheduler warmup_cosine \
  --lr-warmup-updates 3000 \
  --lr-min-lr 5e-5 \
  --settle-steps 550 \
  --per-lane-settle-steps 20 \
  --same-env-eval-lanes 4 \
  --same-env-eval-start-env-steps 50000 \
  --rollout-metrics-window 20 \
  --eval-every-env-steps 0 \
  --reward-probe-steps 200 \
  --disable-reward-curriculum \
  --reward-curriculum reach_grip_lift_goal \
  --curriculum-gating bucket_rates \
  --curriculum-gate-window-transitions 20000 \
  --curriculum-gate-thresholds 0.002,0.0005,0.0001 \
  --lift-progress-deadband-m 0.002 \
  --lift-progress-height-m 0.04 \
  --grip-proxy-scale 1.0 \
  --grip-proxy-sigma-m 0.05 \
  --prioritize-replay \
  --priority-replay-ratio 0.5 \
  --priority-score-weights 0.40,0.25,0.20,0.15 \
  --priority-rarity-power 0.5 \
  --priority-rarity-eps 1.0 \
  --protect-rare-transitions \
  --protected-score-weights 0.80,0.10,0.10 \
  --protected-replay-fraction 0.02 \
  --checkpoint-every-env-steps 50000 \
  --keep-last-checkpoints 5 \
  --save-best-by composite:success_lift_return \
  --progress \
  --log-every-train-steps 100 \
  --log-every-env-steps 1000 \
  --checkpoint-dir ./checkpoints \
  --checkpoint-name sac_franka_500k_seed0_v5_gated_liftprogress \
  --logs-dir ./logs \
  --jsonl-log ./logs/sac_franka_500k_seed0_v5_gated_liftprogress_train.jsonl \
  --progress-log ./logs/sac_franka_500k_seed0_v5_gated_liftprogress_progress.log \
  --tb-log-dir ./logs/tb/sac_franka_500k_seed0_v5_gated_liftprogress \
  --wandb-project isaac-arm \
  --wandb-run-name sac_franka_500k_seed0_v5_gated_liftprogress \
  --wandb-mode online
```

What to inspect in W&B during the run:

```text
curriculum/gate/reach_rate
curriculum/gate/grip_rate
curriculum/gate/lift_rate
curriculum/gate/held_stage

reward/train/grip_proxy
reward/train/lift_progress_proxy
reward/train/native_total
reward/train_shaped

priority_replay/bucket_count/reach
priority_replay/bucket_count/grip
priority_replay/bucket_count/lift
priority_replay/bucket_count/goal
priority_replay/bucket_count/grip_attempt
priority_replay/bucket_count/grip_effect
priority_replay/protected_count

action/train/gripper_mean
action/train/gripper_close_rate
action/eval_rollout/gripper_mean
action/eval_rollout/gripper_close_rate
action/eval_rollout/gripper_close_near_cube_rate

eval_rollout/max_cube_lift_m
eval_rollout/min_ee_to_cube_m
eval_rollout/min_cube_to_target_m
eval_rollout/mean_return
eval_rollout/success_rate
```

Debug interpretation:

| Observation | Likely issue |
|---|---|
| `reach_rate` low | Still not reaching cube; keep reach stage and inspect exploration/action scale. |
| `reach_rate` ok, `grip_rate` low | Policy reaches but does not close near cube; inspect gripper action logs. |
| `grip_attempt` high, `grip_effect` low | Policy closes near cube but does not affect cube; tune grip/lift behavior or gripper command handling. |
| `lift_progress_proxy` rises but `success_rate=0` | Lift is starting; keep goal stage locked until lift becomes common enough. |
| `max_cube_lift_m` improves but `mean_return` does not | Best checkpoint should use composite/lift-aware selection, not mean return alone. |

**How To Test**

Add or extend focused tests:

- `tests/test_reward_curriculum.py`
  - computes `lift_progress_proxy = clip((next_cube_z - cube_reset_z - deadband) / height, 0, 1)`,
  - returns zero lift progress for jitter below the deadband,
  - applies the PR6.9 stage table with `w_lift_progress` and `w_lift_stock`,
  - keeps `grip_proxy` unchanged from PR6.8,
  - advances a gated curriculum only when the relevant recent bucket rate meets the threshold,
  - holds a stage and logs `curriculum/gate/held_stage` when the threshold is not met,
  - logs `curriculum/gate/reach_rate`, `curriculum/gate/grip_rate`, and `curriculum/gate/lift_rate`.
- `tests/test_prioritized_replay.py`
  - keeps PR6.8 bucket-rarity sampling unchanged,
  - logs diagnostic `grip_attempt` and `grip_effect` counts without imposing a bucket importance order,
  - verifies lower `protected_replay_fraction` caps protected entries,
  - verifies `protected_score_weights=0.80,0.10,0.10` is normalized and applied.
- SAC/TD3 loop/logger tests
  - verify active train-lane action diagnostics emit `action/train/gripper_mean` and `action/train/gripper_close_rate`,
  - verify same-env eval emits lift-aware metrics and eval gripper diagnostics,
  - verify `reward/eval_rollout/lift_progress_proxy` is logged when curriculum diagnostics are enabled,
  - verify settle transitions and eval lanes still do not enter replay or curriculum gate windows.
- Checkpoint manager tests
  - verify `--save-best-by composite:success_lift_return` compares success first, max lift second, mean return third,
  - verify missing composite inputs fail readably instead of silently selecting a bad checkpoint.

Targeted verification command:

```bash
timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q \
  tests/test_reward_curriculum.py \
  tests/test_prioritized_replay.py \
  tests/test_sac_continuous.py \
  tests/test_td3_continuous.py \
  tests/test_training_logger_and_scheduler.py
```

Full verification command:

```bash
timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q
```

**Acceptance Criteria**

PR6.9 is complete when:
- all new flags default to PR6.8-compatible behavior,
- dense lift progress is available in shaped training reward and W&B diagnostics,
- progress-gated curriculum can hold stages until observed behavior appears,
- grip attempt/effect counts make near-cube closing failures visible,
- lift-aware eval metrics are logged for deterministic same-env eval lanes,
- action diagnostics show whether the policy sends gripper close commands,
- composite best-checkpoint selection is implemented and tested,
- protected replay can be run with a smaller rarity-focused protected pool,
- targeted tests and full pytest pass in the `isaac_arm` environment.

Implementation status on 2026-04-29:
- Completed in code for SAC and TD3.
- Targeted PR6.9 verification: `79 passed`.
- Full `isaac_arm` pytest: `310 passed, 1 skipped`.

**Suggested Commit**

```bash
git commit -m "feat(train): add progress-gated lift curriculum diagnostics"
```

---

### PR 6.10 — Eval-Subskill Dual-Gated Curriculum

**Status**

Done. Implemented for SAC and TD3 train loops, CLI parsers, progress/W&B/JSONL logging,
and checkpoint-compatible loop config serialization.

Verification:

```bash
timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q \
  tests/test_reward_curriculum.py \
  tests/test_training_logger_and_scheduler.py
# 56 passed

timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q
# 318 passed, 1 skipped
```

**Goal / Why**

PR6.9's `bucket_rates` curriculum gate is useful but still indirect. It answers:

```text
Did recent replay data contain enough reach/grip/lift-looking transitions?
```

That is not the same as:

```text
Can the current policy reliably do the subskill now?
```

PR6.10 upgrades curriculum advancement to a dual gate:

```text
A. current-policy deterministic eval gate
B. minimum training-exposure gate
```

The stage advances only when both are true. This keeps the no-demo, no-BC, no-expert-action
constraint, but makes curriculum stages mean actual policy competence instead of a few lucky
events in replay.

**Design Decision**

Keep PR6.9 `bucket_rates` as a fallback mode for experiments, but do not use it as the main
recommendation for serious SAC runs. The preferred mode after PR6.10 is:

```text
--curriculum-gating eval_dual_gate
```

This mode requires same-env deterministic eval lanes:

```text
--same-env-eval-lanes N, where N > 0
```

If `--curriculum-gating eval_dual_gate` is requested with `same_env_eval_lanes=0`, training
must fail readably before the run starts.

**Inputs**

- PR6.9 same-env eval lift-aware metrics:
  - `eval_rollout/max_cube_lift_m`
  - `eval_rollout/min_ee_to_cube_m`
  - `eval_rollout/min_cube_to_target_m`
  - `eval_rollout/gripper_close_near_cube_rate`
- PR6.9 action diagnostics:
  - `action/eval_rollout/gripper_close_rate`
  - `action/eval_rollout/gripper_close_near_cube_rate`
- PR6.8/PR6.9 replay labels and diagnostics:
  - progress buckets: `reach`, `grip`, `lift`, `goal`
  - diagnostic buckets: `grip_attempt`, `grip_effect`
- 40D proprio contract:
  - `proprio[:, 21:24]`: cube position in robot base frame
  - `proprio[:, 27:30]`: `ee_to_cube`
  - `proprio[:, 30:33]`: `cube_to_target`
  - `action[:, 6]`: gripper command; negative closes

**CLI Contract**

Extend SAC and TD3 train scripts:

```text
--curriculum-gating none|bucket_rates|eval_dual_gate
--curriculum-gate-eval-window-episodes INT
--curriculum-gate-min-eval-episodes INT
--curriculum-gate-eval-thresholds reach,grip_attempt,grip_effect,lift_2cm
--curriculum-gate-min-train-exposures reach,grip_attempt,grip_effect,lift_progress
--curriculum-gate-lift-success-height-m FLOAT
--curriculum-gate-min-stage-env-steps INT
```

Recommended defaults:

```text
--curriculum-gate-eval-window-episodes 20
--curriculum-gate-min-eval-episodes 20
--curriculum-gate-eval-thresholds 0.40,0.30,0.05,0.10
--curriculum-gate-min-train-exposures 400,100,20,20
--curriculum-gate-lift-success-height-m 0.02
--curriculum-gate-min-stage-env-steps 10000
```

Interpretation:

```text
reach threshold       = 40% of recent eval episodes reach cube
grip_attempt threshold= 30% of recent eval episodes close near cube
grip_effect threshold = 5% of recent eval episodes close near cube and move/lift cube
lift_2cm threshold    = 10% of recent eval episodes lift cube at least 2cm
```

`--curriculum-gate-min-stage-env-steps` prevents a stage from advancing immediately after
one lucky clean eval episode. It counts active train transitions collected since the current
stage began, not settle steps and not same-env eval lanes.

**Current-Policy Eval Gate**

Add a completed-episode tracker for deterministic same-env eval lanes. It should track
per-episode subskill booleans, not only scalar means.

For each completed eval episode:

```text
reach_episode =
  min(norm(ee_to_cube)) <= reach_threshold_m

grip_attempt_episode =
  any(norm(ee_to_cube) <= grip_threshold_m
      and action[:, 6] < close_command_threshold)

grip_effect_episode =
  any(grip_attempt at step t
      and (
        cube_z[t+1] - cube_reset_z > lift_progress_deadband_m
        or norm(cube_pos[t+1] - cube_pos[t]) > cube_motion_effect_threshold_m
      ))

lift_2cm_episode =
  max(cube_z - cube_reset_z) >= curriculum_gate_lift_success_height_m
```

Default thresholds:

```text
reach_threshold_m = 0.08
grip_threshold_m = 0.05
close_command_threshold = -0.25
lift_progress_deadband_m = 0.002
cube_motion_effect_threshold_m = 0.005
curriculum_gate_lift_success_height_m = 0.02
```

Maintain a rolling window over completed deterministic eval episodes:

```text
eval_reach_episode_rate =
  mean(reach_episode over window)

eval_grip_attempt_episode_rate =
  mean(grip_attempt_episode over window)

eval_grip_effect_episode_rate =
  mean(grip_effect_episode over window)

eval_lift_2cm_episode_rate =
  mean(lift_2cm_episode over window)
```

Do not advance stages until the eval window has at least
`curriculum_gate_min_eval_episodes` completed episodes.

**Minimum Training-Exposure Gate**

The eval gate answers whether the current deterministic policy can perform the subskill.
The exposure gate ensures the replay buffer has enough recent/current-stage data to train
on that behavior.

Track stage-local counts for active train-lane transitions only:

```text
stage_exposure/reach_count
stage_exposure/grip_attempt_count
stage_exposure/grip_effect_count
stage_exposure/lift_progress_count
stage_exposure/active_train_transition_count
```

Reset these counts each time the curriculum advances to the next stage.

Definitions:

```text
reach_count:
  number of active train transitions with reach=True

grip_attempt_count:
  number of active train transitions with grip_attempt=True

grip_effect_count:
  number of active train transitions with grip_effect=True

lift_progress_count:
  number of active train transitions where
  next_cube_z - cube_reset_z >= curriculum_gate_lift_success_height_m
```

The exposure gate uses raw counts rather than percentages. This makes it easy to reason
about minimum sample availability.

Recommended default:

```text
--curriculum-gate-min-train-exposures 400,100,20,20
```

Meaning:

```text
reach exposure       >= 400 train transitions
grip_attempt exposure>= 100 train transitions
grip_effect exposure >= 20 train transitions
lift_progress exposure >= 20 train transitions
```

**Stage Advancement Rule**

Use four curriculum stages from PR6.9:

```text
0 reach
1 grip_pre_lift
2 lift
3 stock_like
```

Stage 0 -> 1:

```text
eval_reach_episode_rate >= 0.40
and stage_exposure/reach_count >= 400
and stage_env_steps >= curriculum_gate_min_stage_env_steps
```

Stage 1 -> 2:

```text
eval_grip_attempt_episode_rate >= 0.30
and eval_grip_effect_episode_rate >= 0.05
and stage_exposure/grip_attempt_count >= 100
and stage_exposure/grip_effect_count >= 20
and stage_env_steps >= curriculum_gate_min_stage_env_steps
```

Stage 2 -> 3:

```text
eval_lift_2cm_episode_rate >= 0.10
and stage_exposure/lift_progress_count >= 20
and stage_env_steps >= curriculum_gate_min_stage_env_steps
```

Stage 3:

```text
hold stock_like; no further advancement
```

Important behavior:
- If eval gate passes but exposure gate fails, hold the stage.
- If exposure gate passes but eval gate fails, hold the stage.
- If eval window is not yet full enough, hold the stage.
- If there are no same-env eval lanes, fail before training starts.

**Logging Contract**

Log scalar metrics to W&B/TensorBoard/JSONL/progress:

```text
curriculum/gate/mode_eval_dual_gate
curriculum/gate/eval_window_size
curriculum/gate/eval_reach_episode_rate
curriculum/gate/eval_grip_attempt_episode_rate
curriculum/gate/eval_grip_effect_episode_rate
curriculum/gate/eval_lift_2cm_episode_rate

curriculum/gate/exposure_reach_count
curriculum/gate/exposure_grip_attempt_count
curriculum/gate/exposure_grip_effect_count
curriculum/gate/exposure_lift_progress_count
curriculum/gate/stage_env_steps

curriculum/gate/eval_gate_passed
curriculum/gate/exposure_gate_passed
curriculum/gate/min_stage_steps_passed
curriculum/gate/held_stage
curriculum/gate/advanced_stage
```

Use numeric flags (`0.0` or `1.0`) for logger backends.

For human-readable progress/file logs, include a note when a stage advances:

```text
sac train | curriculum_advance | env_step=... | from=reach to=grip_pre_lift reason=eval_dual_gate
```

For JSONL, optionally include a non-scalar diagnostic event:

```json
{
  "type": "curriculum_gate",
  "event": "held",
  "stage": "grip_pre_lift",
  "blocked_by": ["eval_grip_effect_episode_rate", "stage_exposure/grip_effect_count"]
}
```

If the logger stack only supports scalars, keep the structured event in `progress_log` and
the final train JSON summary.

**Best-Checkpoint Selection**

Keep PR6.9:

```text
--save-best-by composite:success_lift_return
```

Do not use `eval_rollout/mean_return` alone while curriculum reward is shaped. The best
checkpoint should prefer actual success, then cube lift, then return.

**Recommended SAC Run After PR6.10**

```bash
python -m scripts.train_sac_continuous \
  --backend isaac \
  --env-id Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num-envs 32 \
  --seed 0 \
  --total-env-steps 500000 \
  --warmup-steps 5000 \
  --batch-size 256 \
  --replay-capacity 200000 \
  --ram-budget-gib 80 \
  --device cuda:0 \
  --learning-rate 3e-4 \
  --polyak-tau 0.005 \
  --utd-ratio 1 \
  --initial-alpha 0.2 \
  --alpha-min 0.10 \
  --target-entropy auto \
  --image-normalization none \
  --lr-scheduler warmup_cosine \
  --lr-warmup-updates 3000 \
  --lr-min-lr 5e-5 \
  --settle-steps 550 \
  --per-lane-settle-steps 20 \
  --same-env-eval-lanes 4 \
  --same-env-eval-start-env-steps 50000 \
  --rollout-metrics-window 20 \
  --eval-every-env-steps 0 \
  --reward-probe-steps 200 \
  --disable-reward-curriculum \
  --reward-curriculum reach_grip_lift_goal \
  --curriculum-gating eval_dual_gate \
  --curriculum-gate-eval-window-episodes 20 \
  --curriculum-gate-min-eval-episodes 20 \
  --curriculum-gate-eval-thresholds 0.40,0.30,0.05,0.10 \
  --curriculum-gate-min-train-exposures 400,100,20,20 \
  --curriculum-gate-lift-success-height-m 0.02 \
  --curriculum-gate-min-stage-env-steps 10000 \
  --lift-progress-deadband-m 0.002 \
  --lift-progress-height-m 0.04 \
  --grip-proxy-scale 1.0 \
  --grip-proxy-sigma-m 0.05 \
  --prioritize-replay \
  --priority-replay-ratio 0.5 \
  --priority-score-weights 0.40,0.25,0.20,0.15 \
  --priority-rarity-power 0.5 \
  --priority-rarity-eps 1.0 \
  --protect-rare-transitions \
  --protected-score-weights 0.80,0.10,0.10 \
  --protected-replay-fraction 0.02 \
  --checkpoint-every-env-steps 50000 \
  --keep-last-checkpoints 5 \
  --save-best-by composite:success_lift_return \
  --progress \
  --log-every-train-steps 100 \
  --log-every-env-steps 1000 \
  --checkpoint-dir ./checkpoints \
  --checkpoint-name sac_franka_500k_seed0_v6_eval_dual_gate \
  --logs-dir ./logs \
  --jsonl-log ./logs/sac_franka_500k_seed0_v6_eval_dual_gate_train.jsonl \
  --progress-log ./logs/sac_franka_500k_seed0_v6_eval_dual_gate_progress.log \
  --tb-log-dir ./logs/tb/sac_franka_500k_seed0_v6_eval_dual_gate \
  --wandb-project isaac-arm \
  --wandb-run-name sac_franka_500k_seed0_v6_eval_dual_gate \
  --wandb-mode online
```

**What To Watch In W&B**

The key question is whether the stage is held for a good reason:

```text
curriculum/gate/eval_reach_episode_rate
curriculum/gate/eval_grip_attempt_episode_rate
curriculum/gate/eval_grip_effect_episode_rate
curriculum/gate/eval_lift_2cm_episode_rate

curriculum/gate/exposure_reach_count
curriculum/gate/exposure_grip_attempt_count
curriculum/gate/exposure_grip_effect_count
curriculum/gate/exposure_lift_progress_count

curriculum/gate/eval_gate_passed
curriculum/gate/exposure_gate_passed
curriculum/gate/held_stage
curriculum/stage_index
```

Debug interpretation:

| Observation | Meaning |
|---|---|
| eval gate fails, exposure gate passes | Replay has examples, but current policy cannot reproduce the subskill. Keep stage. |
| eval gate passes, exposure gate fails | Policy got lucky or has too little replay support. Keep stage until enough samples exist. |
| reach eval rate high, grip attempt low | Policy reaches but does not send close commands near cube. |
| grip attempt high, grip effect low | Policy closes near cube but does not move/lift cube. |
| lift 2cm rate rises, success still zero | Lift behavior exists; goal stage can start once exposure also passes. |

**Tests**

Add focused coverage:

- `tests/test_reward_curriculum.py`
  - parse and validate `eval_dual_gate` config,
  - reject bad eval threshold counts and exposure counts,
  - verify stage 0 advances only when eval reach rate, reach exposure, and min stage steps all pass,
  - verify stage 1 requires both grip attempt and grip effect eval/exposure gates,
  - verify stage-local exposure counters reset after stage advancement.
- `tests/test_rollout_metrics.py` or `tests/test_training_logger_and_scheduler.py`
  - verify completed eval episodes compute `reach_episode`, `grip_attempt_episode`, `grip_effect_episode`, and `lift_2cm_episode`,
  - verify partial episodes do not count,
  - verify settle/cooldown steps do not count toward eval gate.
- SAC/TD3 loop tests:
  - `eval_dual_gate` fails readably when `same_env_eval_lanes=0`,
  - stage does not advance from replay exposure alone,
  - stage does not advance from eval success alone,
  - stage advances when both gates pass,
  - logs all `curriculum/gate/*` scalars to JSONL/W&B logger and progress.
- Parser tests:
  - SAC and TD3 accept all new CLI flags,
  - invalid threshold/exposure values raise `ValueError`.

Targeted verification command:

```bash
timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q \
  tests/test_reward_curriculum.py \
  tests/test_training_logger_and_scheduler.py \
  tests/test_sac_continuous.py \
  tests/test_td3_continuous.py
```

Full verification command:

```bash
timeout 360s env PYTHONPATH=. /root/miniconda3/bin/conda run -n isaac_arm python -m pytest -q
```

**Acceptance Criteria**

PR6.10 is complete when:
- `eval_dual_gate` is available for SAC and TD3 train scripts,
- `bucket_rates` remains available but is no longer the recommended main gate,
- stage advancement requires both current-policy eval performance and minimum train exposure,
- same-env eval lanes produce subskill episode rates,
- active train lanes produce stage-local exposure counts,
- stage-local exposure counters reset after every advancement,
- W&B/JSONL/progress clearly show why a stage is held or advanced,
- training fails early if `eval_dual_gate` is requested without same-env eval lanes,
- targeted tests and full pytest pass in the `isaac_arm` environment.

**Suggested Commit**

```bash
git commit -m "feat(train): add eval-subskill dual curriculum gates"
```

---

### PR 11a — SAC/TD3 Checkpoint Evaluation

**Goal / Why**

Add the first online checkpoint evaluation path for the two off-policy baselines. This is intentionally narrower than PR11-full: it only needs to load SAC and TD3 checkpoints, run Isaac/fake env episodes, and save metrics compatible with the already-completed PR11-lite dataset metrics.

**Inputs**
- PR6 SAC checkpoints from `scripts.train_sac_continuous`.
- PR7 TD3 checkpoints from `scripts.train_td3_continuous`.
- PR2.5 env wrapper and PR11-lite metric definitions.
- PR8-lite collector only if the eval run also saves rollout HDF5 for inspection.

**Outputs**
- `scripts.eval_checkpoint_continuous` with `--agent-type/--agent_type sac|td3`.
- Metrics JSON files such as `logs/eval_sac.json` and `logs/eval_td3.json`.
- Optional eval rollout HDF5 files using the PR8-lite schema.

**CLI Contract**

```bash
python -m scripts.eval_checkpoint_continuous \
  --agent_type sac \
  --checkpoint ./checkpoints/sac_franka_final.pt \
  --num_episodes 20 \
  --num-parallel-envs 1 \
  --settle-steps 600 \
  --seed 0 \
  --device cuda:0 \
  --save_metrics ./logs/eval_sac.json \
  --save_dataset ./data/eval_sac_rollouts.h5
```

Required args:
- `--agent_type {sac,td3}`
- `--checkpoint PATH`
- `--save_metrics PATH`

Optional but recommended args:
- `--save_dataset PATH` for post-hoc inspection with the existing HDF5 tools.
- `--include-debug-images` when eval rollouts should be visually inspectable.
- `--deterministic` defaulting to true for checkpoint comparison.
- `--settle-steps 600` for final SAC-vs-TD3 comparisons; smaller values such as `20` are allowed only for smoke/debug runs and must be recorded in metrics.

**Metrics JSON Contract**

Minimum required fields:

```json
{
  "agent_type": "sac",
  "checkpoint": "checkpoints/sac_franka_final.pt",
  "env_id": "Isaac-Lift-Cube-Franka-IK-Rel-v0",
  "num_eval_episodes": 20,
  "num_env_steps": 500000,
  "mean_return": 0.0,
  "success_rate": 0.0,
  "mean_episode_length": 100.0,
  "mean_action_jerk": 0.0,
  "success_threshold_m": 0.02,
  "success_source": "proprio_cube_to_target_norm",
  "episode_successes": {}
}
```

`num_env_steps` should be loaded from checkpoint metadata when available. If an older checkpoint does not contain it, write `null` and include a warning field rather than fabricating the value.

**Implementation**
- Load SAC or TD3 checkpoint and construct the matching deterministic eval policy.
- Run `num_episodes` with the camera-enabled Isaac env or a fake env in tests.
- Compute mean return, success rate, mean episode length, mean action jerk, and per-episode diagnostics using the same definitions as PR11-lite.
- Compute action jerk inline during rollout even when `--save_dataset` is not provided. If `--save_dataset` is provided, optionally verify the offline HDF5 jerk matches the inline value on deterministic fake rollouts.
- Support `--settle-steps`, `--num-parallel-envs`, `--seed`, `--device`, and camera/debug-camera names.
- Fail readably for missing checkpoints, unknown agent types, or checkpoint/action-dim mismatches.
- Read `num_env_steps`, deterministic action mode, env/action/proprio/image contract, and algorithm hyperparameters from checkpoint metadata.
- Do not add PPO/GRPO/Diffusion checkpoint loading in this PR.

**How To Test**
- Verify fake SAC and fake TD3 checkpoints route to the correct policy loader.
- Verify returned metrics match PR11-lite metrics when saving and re-reading a deterministic fake rollout.
- Verify `num_envs=1` and `num_envs>1` evaluation preserves episode counts.
- Verify deterministic eval is repeatable with fixed seed.
- Verify missing checkpoint, unknown agent type, and wrong action dimension fail readably.
- Verify metrics JSON contains the minimum required fields and does not invent `num_env_steps`.
- Verify optional `--save_dataset` writes PR8-lite-compatible HDF5 episodes.
- Verify action jerk is computed when `--save_dataset` is omitted.
- Verify `settle_steps` is recorded in metrics and eval HDF5 metadata when present.
- Verify PR6/PR7 checkpoints without required new metadata fail, while explicitly marked legacy checkpoints may write `num_env_steps: null` with a warning.

**Acceptance Criteria**

PR11a is complete when both commands work against fake checkpoints in tests:

```bash
python -m scripts.eval_checkpoint_continuous --agent_type sac ...
python -m scripts.eval_checkpoint_continuous --agent_type td3 ...
```

and live Isaac evaluation can be run after real PR6/PR7 checkpoints exist without changing the metrics schema.

**Pytest**

```bash
pytest tests/test_eval_sac_td3_checkpoints.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(eval): add SAC and TD3 checkpoint evaluation"
```

---

### PR 12a — SAC/TD3 Checkpoint Visual Rollouts

**Goal / Why**

Add the first trained-checkpoint visualization path, with outputs matching the style of `scripts.demo_data_loop`: fixed-debug-camera GIF/MP4 plus sampled debug PNGs and optional overlay text. This proves SAC and TD3 can be inspected visually before the full multi-method comparison grid exists.

**Inputs**
- PR6 SAC checkpoint and PR7 TD3 checkpoint.
- PR11a checkpoint loading/eval policy adapters.
- PR12-lite GIF/MP4/debug-frame recorder.
- PR2.5 debug-camera accessor and target-overlay projection support.

**Outputs**
- `scripts.record_gif_continuous` with `--agent-type/--agent_type sac|td3`.
- Shared visual wrappers in `eval.visual_helpers`:
  - `SettledResetEnv` for zero-action reset warmup,
  - `TargetOverlayEnv` for debug-camera target text/reticle,
  - `target_projection_payload(...)` for metrics JSON target-pixel diagnostics.
- Visual artifacts:

```text
out/gifs/sac_seed0.gif
out/gifs/sac_seed0.mp4
out/gifs/td3_seed0.gif
out/gifs/td3_seed0.mp4
out/debug_frames/sac_seed0/
out/debug_frames/td3_seed0/
```

**CLI Contract**

```bash
python -m scripts.record_gif_continuous \
  --agent_type sac \
  --checkpoint ./checkpoints/sac_franka_final.pt \
  --seed 0 \
  --device cuda:0 \
  --settle-steps 600 \
  --gif-max-steps 100 \
  --target-overlay text-reticle \
  --save_gif ./out/gifs/sac_seed0.gif \
  --save_mp4 ./out/gifs/sac_seed0.mp4 \
  --save-debug-frames-dir ./out/debug_frames/sac_seed0 \
  --save_metrics ./logs/visual_sac_seed0_metrics.json
```

Required args:
- `--agent_type {sac,td3}`
- `--checkpoint PATH`
- `--save_gif PATH`

Optional outputs:
- `--save_mp4 PATH`
- `--save-debug-frames-dir PATH`
- `--save_metrics PATH`, written after the visual rollout so overlay metrics and saved JSON agree.
- `--metrics-payload PATH` / `--pr11a-metrics PATH`, optional PR11a eval metrics used for final-comparison overlay text. The script rejects mismatched checkpoint, seed, settle steps, agent type, or deterministic mode before recording.

For final comparisons, use the same `--settle-steps` value as PR11a evaluation. The recommended final comparison value is `600`; smaller values are acceptable only for smoke/debug artifacts and must be visible in metrics/metadata.

**Output Semantics**

- GIF/MP4/debug PNGs use the fixed debug camera.
- Policy actions are computed only from wrist RGB plus 40D proprio.
- Overlay text may show policy, return, success, jerk, step, seed, and target reticle/pixel when projection is available.
- The script should not write fixed-debug-camera frames into the policy `images` stream.
- By default, metrics are computed from the exact rollout being recorded, so `--save_metrics` and overlay text refer to the same episode. The saved JSON also includes `visual_rollout_reward_trace` and reward summary fields for per-step debugging. If an external PR11a metrics payload is supplied, the JSON keeps those final-comparison metrics at top level and stores the recorded single-episode metrics under `visual_rollout_metrics`.

**Implementation**
- Load SAC or TD3 checkpoint and roll out deterministic actions.
- Keep policy input as wrist RGB + 40D proprio.
- Record visual frames from the fixed debug camera only.
- Reuse PR12-lite recorder plus shared `eval.visual_helpers` for settle reset, target text/reticle, and target pixel projection diagnostics.
- Support `--settle-steps`, `--target-overlay`, `--gif-max-steps`, `--save-gif`, `--save-mp4`, and `--save-debug-frames-dir`.
- Compute return/success/jerk inline during the visual rollout by wrapping the env and buffering the selected lane's policy images, proprio, actions, per-step rewards, done/truncated, and optional success flags. External PR11a metrics payloads are accepted only after contract validation.
- Do not implement side-by-side grids or PPO/GRPO/Diffusion visualization in this PR.

**How To Test**
- `tests/test_visual_sac_td3_checkpoints.py` verifies fake SAC/TD3 checkpoints create GIF, MP4, sampled PNGs, and metrics JSON.
- It checks debug frames come from the fixed debug camera rather than `obs["image"]`, output directories are created automatically, target reticle overlays draw when projection is available, and projection absence falls back cleanly.
- It checks the metrics JSON contains the per-step reward trace and keeps it available even when top-level overlay metrics come from an external PR11a payload.
- It checks missing checkpoint and unknown agent types fail readably.
- It checks external PR11a metrics payloads can drive overlay/top-level saved metrics and rejects mismatched seed/checkpoint/settle/agent contracts before recording.

**Acceptance Criteria**

PR12a is complete when fake checkpoint tests produce non-empty GIF/MP4/PNG files for both SAC and TD3, same-rollout metrics JSON is written, and live Isaac can run:

```bash
python -m scripts.record_gif_continuous --agent_type sac ...
python -m scripts.record_gif_continuous --agent_type td3 ...
```

after PR6/PR7 checkpoints exist.

**Pytest**

```bash
pytest tests/test_visual_sac_td3_checkpoints.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(viz): add SAC and TD3 checkpoint GIFs"
```

---

### PR 8-full — SAC Expert Demonstration Collection

**Goal / Why**

Collect SAC expert demonstrations in the episode-safe HDF5 format that PR8-lite already established. This PR turns SAC from a trained RL baseline into the oracle/data source for imitation learning.

**Inputs**
- PR6 `SACAgent` checkpoint with deterministic oracle mode.
- PR11a/PR12a metrics and visual checks confirming the SAC checkpoint is worth using as an oracle.
- PR8-lite rollout dataset schema and collector conventions.
- PR2.5 camera-enabled Isaac env and 7D action contract.

**Outputs**
- SAC demonstration files such as `data/franka_sac_demos.h5`.
- Dataset metadata recording SAC checkpoint path/hash, env id, action dim, proprio dim, policy/debug camera names, seed schedule, and collection command.
- Optional PR11-lite metrics and PR12-lite visual artifacts for the SAC oracle dataset.

**Implementation**
- Add SAC demo collection script.
- Reuse the HDF5 episode schema from PR8-lite instead of inventing a second format.
- Save images, proprios, actions, rewards, done, truncated, optional success flags, episode id, and metadata.
- Support `--num-parallel-envs`, `--settle-steps`, optional raw wrist images, and optional debug images consistently with `scripts.collect_rollouts`.
- Evaluate the collected SAC demos with existing dataset metrics so bad oracle checkpoints are caught before IL training.
- Do not train Diffusion Policy in this PR.

**How To Test**
- Verify saved file contains required keys.
- Verify metadata includes action dim, proprio dim, env id, seed/reset seed, SAC checkpoint identity, camera names, and collection command.
- Verify a fake SAC oracle with deterministic actions writes those exact labels to HDF5.
- Verify vectorized collection writes separate episode groups per env lane.
- Verify collected actions are clipped 7D float32 arrays and images remain uint8 wrist-camera frames.
- Verify collection fails readably if the SAC checkpoint/env action dimension does not match the project contract.

**Pytest**

```bash
pytest tests/test_sac_demo_collection.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(data): collect SAC expert demonstrations"
```

---

### PR 8.5 — Diffusion Sequence Dataset

**Goal / Why**

Make the exact supervised-learning input/output contract for Diffusion Policy explicit before implementing the model. This is the bridge from episode HDF5 files to batches of observation histories and future action chunks.

**Inputs**
- PR8-lite or PR8-full episode-safe HDF5 files.
- PR3 backbone image/proprio expectations.
- Image augmentation utilities from `utils/image_aug.py`.

**Outputs**
- `DiffusionSequenceDataset` or equivalent dataloader that yields:

```text
images:   (B, T_obs, 3, 224, 224) uint8 or normalized float after transform
proprios: (B, T_obs, 40) float32
actions:  (B, action_horizon, 7) float32
mask:     optional, marks padded timesteps when short episodes are allowed
```

- Clear rule for padding versus rejecting short episodes.
- Augmentation hook applied only in training loaders, not in the env wrapper or eval/GIF path.

**Implementation**
- Index valid `(episode_key, start, stop)` windows without crossing `done` or `truncated`.
- Support configurable `T_obs`, `action_horizon`, and `exec_horizon`.
- Preserve chronological order of image/proprio history and action targets.
- Provide a small collate function if padding/masking is needed.
- Keep model architecture and diffusion loss out of this PR.

**How To Test**
- Verify sequence batches have exact expected shapes and dtypes.
- Verify windows never cross episode boundaries or terminal/truncated steps.
- Verify short episodes are either rejected or padded according to the documented rule.
- Verify `PadAndRandomCrop` can be plugged into the training loader and `IdentityAug` leaves eval batches unchanged.
- Verify action chunks loaded from HDF5 match the underlying stored rows exactly.

**Pytest**

```bash
pytest tests/test_diffusion_sequence_dataset.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(data): add diffusion sequence dataset"
```

---

### PR 9a — Diffusion Core Model

**Goal / Why**

Implement the reusable diffusion-policy model components without a full training loop. This keeps the largest IL component small enough to review.

**Inputs**
- PR3 backbone output or frozen observation feature tensors.
- PR8.5 action chunk shape `(B, action_horizon, 7)`.

**Outputs**
- Noise schedule utilities.
- Sinusoidal timestep embeddings.
- Observation-conditioned 1D temporal U-Net or equivalent denoiser.
- Model forward API:

```python
pred_noise = model(noisy_actions, timesteps, obs_features)
```

**Implementation**
- Add 1D temporal U-Net noise predictor.
- Add sinusoidal diffusion timestep embedding.
- Add FiLM or equivalent conditioning from image-proprio observation features.
- Add DDPM forward-diffusion helpers.
- Do not add full BC train script, DDIM deployment, DAgger, or live Isaac rollout in this PR.

**How To Test**
- Verify noise schedule is monotonic and numerically stable.
- Verify forward diffusion variance matches expected behavior.
- Verify predicted noise shape is `(B, H, 7)`.
- Verify gradients flow from predicted noise to model parameters and observation conditioning.
- Verify deterministic forward pass with fixed weights/input.

**Pytest**

```bash
pytest tests/test_diffusion_core.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(il): add diffusion policy core model"
```

---

### PR 9b — Diffusion Policy Behavior Cloning Training

**Goal / Why**

Train a Diffusion Policy from SAC demonstrations. This tests whether an offline imitation learner can reproduce the expert's manipulation behavior.

**Inputs**
- PR8-full SAC demonstration HDF5 files.
- PR8.5 sequence dataloader batches.
- PR9a diffusion core model.

**Outputs**
- BC training script and checkpoint format.
- Training logs with denoising loss, learning rate, dataset size, seed, and validation loss if a validation split is configured.
- Checkpoints loadable by PR9c deployment.

**Implementation**
- Add DDPM noise prediction loss.
- Add train/validation split over episode-safe windows.
- Add checkpoint save/load and resume.
- Add a small synthetic overfit mode for tests.
- Do not add online DAgger or live Isaac deployment in this PR.

**How To Test**
- Verify training loss is positive and decreases on toy data.
- Verify checkpoint save/load reproduces model output for deterministic inputs.
- Verify resume preserves optimizer state, global step, and scheduler state if used.
- Verify train loader applies training aug and eval loader uses identity/no-random eval transform.
- Verify a tiny synthetic dataset can be overfit within a bounded number of steps.

**Pytest**

```bash
pytest tests/test_diffusion_bc_training.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(il): train diffusion policy behavior cloning"
```

---

### PR 9c — Diffusion Policy Deployment And DDIM Inference

**Goal / Why**

Turn a trained BC-Diffusion checkpoint into a rollout policy that can be evaluated by the existing metrics and visual-output infrastructure.

**Inputs**
- PR9b BC-Diffusion checkpoint.
- PR2.5 env observation contract.
- PR11-lite/PR12-lite evaluation and visualization paths.

**Outputs**
- `DiffusionPolicy.act(obs)` adapter with an internal action queue.
- DDIM inference producing action chunks.
- Deployment config for `action_horizon`, `exec_horizon`, diffusion steps, deterministic/stochastic sampling, and observation history length.

**Implementation**
- Add DDIM inference with action chunking.
- Maintain an observation-history buffer and an action queue.
- Output only the next 7D action to the env while replanning every `exec_horizon` steps.
- Keep DAgger out of this PR.

**How To Test**
- Verify DDIM deterministic mode returns identical actions for identical inputs.
- Verify action chunk output shape is `(E, 7)`.
- Verify the action queue consumes chunk actions before replanning.
- Verify `DiffusionPolicy.act(obs)` returns normalized 7D actions and preserves policy/debug camera separation.
- Verify a fake env rollout can call the deployed policy for multiple replanning cycles.

**Pytest**

```bash
pytest tests/test_diffusion_policy_deployment.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(il): deploy diffusion policy with DDIM"
```

---

### PR 10 — DAgger

**Goal / Why**

Add dataset aggregation so the diffusion student learns how to recover from states it visits itself, not only states from the expert's original demonstrations.

**Inputs**
- PR6 SAC checkpoint as deterministic oracle.
- PR9c deployed Diffusion Policy student.
- PR8.5 sequence dataset and PR8-full SAC demonstration dataset.
- PR8-lite HDF5 append/schema rules.

**Outputs**
- Aggregated DAgger dataset with expert labels on student-visited states.
- DAgger training/fine-tuning checkpoints.
- Logs for oracle query count, dataset size by iteration, success/return/jerk by iteration, and DAgger dataset fraction.

**Implementation**
- Load BC-Diffusion checkpoint.
- Roll out student policy in Isaac Lab.
- Query SAC oracle on student-visited states.
- Append relabeled samples to the demonstration dataset.
- Fine-tune diffusion policy after each DAgger iteration.
- Track oracle query count and DAgger dataset fraction.
- Do not modify RL training algorithms.

**How To Test**
- Verify dataset size increases after append.
- Verify oracle is called once per relabeled student state.
- Verify appended samples preserve episode boundaries.
- Verify fine-tuning reduces BC loss on a small synthetic dataset.
- Verify checkpoint save/load after DAgger iteration.
- Verify a fake student rollout -> fake oracle relabel -> aggregate -> dataloader path produces valid PR8.5 batches.
- Verify oracle query counts and iteration metadata are written deterministically.
- Verify DAgger does not overwrite the original SAC demo episodes unless explicitly requested.

**Pytest**

```bash
pytest tests/test_dagger_diffusion.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(il): add DAgger fine-tuning loop"
```

---

### PR 11-full — Online Checkpoint Evaluation

**Goal / Why**

Provide one full online evaluator that works for RL agents and Diffusion Policy agents, so all methods are compared with the same metrics. PR11-lite already evaluates rollout HDF5 files; PR11-full adds checkpoint/agent evaluation against envs.

**Inputs**
- PR11a SAC/TD3 checkpoint evaluation path.
- PR11-lite dataset metrics functions.
- Agent deployment interfaces from PR4-7 and PR9c/PR10.
- PR2.5 env wrapper and optional PR8-lite collector when saving eval rollouts.

**Outputs**
- `evaluate_agent()` for live/fake envs.
- Agent-level eval JSON/NPZ files with the same metric definitions as dataset-level metrics.
- Optional eval rollout HDF5 files for post-hoc inspection.

**Implementation**
- Generalize `evaluate_agent()` beyond the PR11a SAC/TD3-only scope.
- Track mean return, success rate, episode length, action jerk, and optional steps-to-threshold.
- Add adapter for diffusion action chunks using an action queue.
- Support deterministic evaluation.
- Save evaluation output as `.npz` or `.json`.
- Do not record GIFs or plot curves in this PR.

**How To Test**
- Verify returned metric keys.
- Verify metric ranges.
- Verify action jerk is zero for constant-action dummy policy.
- Verify diffusion action queue consumes chunk actions before replanning.
- Verify deterministic eval is repeatable with fixed seed.
- Verify fake vectorized env evaluation handles `num_envs=1` and `num_envs>1`.
- Verify checkpoint loading routes to the correct agent type.
- Verify online `evaluate_agent()` and offline `evaluate_rollout_dataset()` agree on a saved deterministic fake rollout.
- Verify missing/invalid checkpoint and unknown agent type fail readably.

**Pytest**

```bash
conda run -n isaac_arm python -m pytest tests/test_eval_agent.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(eval): add checkpoint evaluation loop"
```

---

### PR 12-full — Trained-Policy Visual Comparison

**Goal / Why**

Create visual artifacts that make the project easy to inspect. PR12-lite already records debug-camera GIF/MP4 outputs for demo policies; PR12-full adds checkpoint-based visual comparison, side-by-side grids, and summary plots for trained methods.

**Inputs**
- PR12a SAC/TD3 checkpoint visual rollout path.
- PR12-lite GIF/MP4/debug-frame recorder.
- PR11-full checkpoint evaluator and metric files.
- Agent checkpoints from PR4-7, PR9c, and PR10.

**Outputs**
- Per-method GIFs/MP4s from a consistent debug-camera view.
- Side-by-side comparison grids for matching seeds/episodes.
- Plots for return, success rate, steps-to-threshold, action jerk, and data/oracle-query efficiency.

**Implementation**
- Add rollout GIF recording script.
- Add side-by-side GIF/grid generation.
- Add plots for return, success rate, steps-to-threshold, and jerk.
- Use consistent seed, debug camera pose, and episode length across methods.
- Use the fixed debug camera for human-facing GIFs.
- Keep the wrist policy camera separate from the GIF/debug camera unless explicitly recording policy-view diagnostics.
- Overlay method name, return, success, jerk, and seed.
- Do not change agent training code.

**How To Test**
- Verify GIF file is created.
- Verify GIF is non-empty.
- Verify expected number of frames is recorded.
- Verify comparison plot file is created.
- Verify missing checkpoint gives a readable error.
- Verify debug-camera frames are used for human-facing outputs and wrist-camera frames are never substituted silently.
- Verify side-by-side comparison aligns frame counts/seeds or reports why alignment is impossible.
- Verify plotting code handles missing methods/partial runs without fabricating measured results.

**Pytest**

```bash
pytest tests/test_visual_comparison_outputs.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(viz): add trained-policy comparison outputs"
```

---

## 13. Dependency Graph

```text
PR 0 Project Scaffold
  |
  v
PR 1 Isaac Task Config
  |
  v
PR 2 Observation Wrapper
  |
  v
PR 2.5 Camera-Enabled Franka Lift Cfg
  |
  v
Completed demo/data-loop branch:
  |
  +--> PR 8-pre Demo policies
  |      |
  |      v
  |   PR 8-lite Episode-safe rollout dataset
  |      |
  |      +--> PR 11-lite Dataset metrics
  |      +--> PR 12-lite Debug-camera visual output
  |              |
  |              v
  |           Demo PR One-command data loop
  |
  v
Future research branch:
  |
  v
PR 3 Shared Backbone
  |
  v
PR 3.5 Agent Primitives
  |
  +--> SAC/TD3-first branch:
  |      |
  |      +--> PR 6 SAC train/checkpoints -------+
  |      +--> PR 7 TD3 train/checkpoints -------+
  |                                            |
  |                                            v
  |                                      PR 6.5 training logs/schedulers/eval
  |                                            |
  |                                            v
  |                                      PR 6.6 running obs/action normalization
  |                                            |
  |                                            v
  |                                      PR 11a SAC/TD3 eval
  |                                            |
  |                                            v
  |                                      PR 12a SAC/TD3 GIF/MP4
  |                                            |
  |                                            v
  |                                      PR 8-full SAC expert demos
  |                                            |
  |                                            v
  |                                      PR 8.5 Diffusion sequence data
  |                                            |
  |                                            v
  |                                      PR 9a Diffusion core
  |                                            |
  |                                            v
  |                                      PR 9b Diffusion BC training
  |                                            |
  |                                            v
  |                                      PR 9c Diffusion deployment
  |                                            |
  |                                            v
  |                                      PR 10 DAgger
  |
  +--> Later on-policy branch:
         |
         +--> PR 4 PPO
         +--> PR 5 Pure GRPO

PR11-full generalizes PR11a to PPO/GRPO/Diffusion/DAgger after those agents exist.
PR12-full generalizes PR12a to full side-by-side visual comparison and plots.

PR11-lite and PR12-lite remain reusable by full PRs for dataset metrics and debug-camera visual artifact generation.
```

---

## 14. Global Test Commands

Fast CPU/unit tests, no Isaac Sim required:

```bash
pytest tests/ -v \
  --ignore=tests/test_task_contract.py \
  --ignore=tests/test_observation_wrapper.py
```

Isaac Lab / GPU / camera tests:

```bash
conda run -n isaac_arm python -m pytest \
  tests/test_task_contract.py \
  tests/test_observation_wrapper.py \
  tests/test_camera_enabled_env_cfg.py \
  -v
```

`tests/test_camera_enabled_env_cfg.py` is now part of PR2.5 and passes in `isaac_arm`.

Dedicated live PR2.5 camera smoke:

```bash
timeout 360s conda run -n isaac_arm python -m scripts.isaac_camera_observation_smoke \
  --steps 1 \
  --output-dir out/camera_smoke
```

Known result:

```text
status: ok
obs["image"]:   (1, 3, 224, 224), uint8
obs["proprio"]: (1, 40), float32
debug table_rgb: (720, 1280, 3), uint8
```

Camera runtime verification must prove that the customized cfg produces an RGB observation term. A successful stock `--enable_cameras` smoke with only `policy: (num_envs, 35)` is not sufficient.

Full project test suite after Isaac Lab is installed:

```bash
pytest tests/ -v
```

Single PR examples:

```bash
pytest tests/test_nn_backbone.py -v
pytest tests/test_agent_primitives.py -v
pytest tests/test_sac_continuous.py -v
pytest tests/test_td3_continuous.py -v
pytest tests/test_training_logger_and_scheduler.py -v
pytest tests/test_eval_sac_td3_checkpoints.py -v
pytest tests/test_visual_sac_td3_checkpoints.py -v
pytest tests/test_sac_demo_collection.py -v
pytest tests/test_diffusion_sequence_dataset.py -v
pytest tests/test_diffusion_core.py -v
pytest tests/test_diffusion_bc_training.py -v
pytest tests/test_diffusion_policy_deployment.py -v
pytest tests/test_eval_agent.py -v
pytest tests/test_visual_comparison_outputs.py -v
```

---

## 15. Example Run Commands

### 15.1 SAC/TD3-First Train + Eval + Visualize

> **Live-Isaac configuration note (see §8.3 line 908).** For real Isaac runs, set
> `--eval-every-env-steps 0` so the loop does NOT spin up a second Isaac env, and use
> `--same-env-eval-lanes N --same-env-eval-start-env-steps K` to reserve N of the
> training env lanes as deterministic eval rollouts that begin reporting `eval_rollout/*`
> after K total env transitions. The fake-backend smoke command in PR6.5 is the only
> place separate-env periodic `eval/*` is appropriate.

Train SAC (live Isaac):

```bash
python -m scripts.train_sac_continuous \
  --backend isaac \
  --env_id Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 64 \
  --total_envsteps 500000 \
  --batch_size 256 \
  --replay_buffer_size 200000 \
  --warmup_steps 5000 \
  --replay_storage cpu \
  --utd-ratio 1 \
  --polyak-tau 0.005 \
  --initial-alpha 0.2 \
  --alpha-min 0.05 \
  --target-entropy auto \
  --lr-scheduler warmup_cosine \
  --lr-warmup-updates 1000 \
  --lr-min-lr 1e-5 \
  --eval-every-env-steps 0 \
  --same-env-eval-lanes 8 \
  --same-env-eval-start-env-steps 50000 \
  --eval-settle-steps 600 \
  --settle-steps 600 \
  --per-lane-settle-steps 20 \
  --disable-reward-curriculum \
  --checkpoint-every-env-steps 50000 \
  --keep-last-checkpoints 5 \
  --save-best-by eval_rollout/mean_return \
  --learning_rate 3e-4 \
  --tb-log-dir ./logs/sac_franka \
  --jsonl-log ./logs/sac_franka_train.jsonl \
  --progress-log ./logs/sac_franka_progress.txt \
  --wandb-project isaac-arm-rl \
  --wandb-run-name sac_franka \
  --wandb-mode online \
  --checkpoint_dir ./checkpoints \
  --checkpoint_name sac_franka
```

Train TD3 (live Isaac):

```bash
python -m scripts.train_td3_continuous \
  --backend isaac \
  --env_id Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 64 \
  --total_envsteps 500000 \
  --batch_size 256 \
  --replay_buffer_size 200000 \
  --warmup_steps 5000 \
  --replay_storage cpu \
  --utd-ratio 1 \
  --polyak-tau 0.005 \
  --policy-delay 2 \
  --exploration-noise-sigma 0.1 \
  --target-noise-sigma 0.2 \
  --target-noise-clip 0.5 \
  --lr-scheduler warmup_cosine \
  --lr-warmup-updates 1000 \
  --lr-min-lr 1e-5 \
  --eval-every-env-steps 0 \
  --same-env-eval-lanes 8 \
  --same-env-eval-start-env-steps 50000 \
  --eval-settle-steps 600 \
  --settle-steps 600 \
  --per-lane-settle-steps 20 \
  --disable-reward-curriculum \
  --checkpoint-every-env-steps 50000 \
  --keep-last-checkpoints 5 \
  --save-best-by eval_rollout/mean_return \
  --learning_rate 3e-4 \
  --tb-log-dir ./logs/td3_franka \
  --jsonl-log ./logs/td3_franka_train.jsonl \
  --progress-log ./logs/td3_franka_progress.txt \
  --wandb-project isaac-arm-rl \
  --wandb-run-name td3_franka \
  --wandb-mode online \
  --checkpoint_dir ./checkpoints \
  --checkpoint_name td3_franka
```

Evaluate SAC and TD3 checkpoints:

```bash
python -m scripts.eval_checkpoint_continuous \
  --agent_type sac \
  --checkpoint ./checkpoints/sac_franka_final.pt \
  --num_episodes 20 \
  --settle-steps 600 \
  --save_metrics ./logs/eval_sac.json

python -m scripts.eval_checkpoint_continuous \
  --agent_type td3 \
  --checkpoint ./checkpoints/td3_franka_final.pt \
  --num_episodes 20 \
  --settle-steps 600 \
  --save_metrics ./logs/eval_td3.json
```

Record SAC and TD3 debug-camera GIF/MP4 artifacts:

```bash
python -m scripts.record_gif_continuous \
  --agent_type sac \
  --checkpoint ./checkpoints/sac_franka_final.pt \
  --settle-steps 600 \
  --target-overlay text-reticle \
  --save_gif ./out/gifs/sac_seed0.gif \
  --save_mp4 ./out/gifs/sac_seed0.mp4 \
  --save-debug-frames-dir ./out/debug_frames/sac_seed0

python -m scripts.record_gif_continuous \
  --agent_type td3 \
  --checkpoint ./checkpoints/td3_franka_final.pt \
  --settle-steps 600 \
  --target-overlay text-reticle \
  --save_gif ./out/gifs/td3_seed0.gif \
  --save_mp4 ./out/gifs/td3_seed0.mp4 \
  --save-debug-frames-dir ./out/debug_frames/td3_seed0
```

### 15.2 Diffusion / DAgger Follow-Up

Collect demonstrations:

```bash
python -m scripts.collect_demos_continuous \
  --oracle_checkpoint ./checkpoints/sac_franka_final.pt \
  --num_episodes 500 \
  --out_path ./data/franka_sac_demos.h5
```

Train BC-Diffusion:

```bash
python -m scripts.train_bc_diffusion \
  --data_path ./data/franka_sac_demos.h5 \
  --epochs 200 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --tb_log_dir ./logs/bc_diffusion_franka \
  --checkpoint_dir ./checkpoints \
  --checkpoint_name bc_diffusion_franka
```

Train DAgger-Diffusion:

```bash
python -m scripts.train_dagger_diffusion \
  --base_checkpoint ./checkpoints/bc_diffusion_franka_final.pt \
  --oracle_checkpoint ./checkpoints/sac_franka_final.pt \
  --dagger_iterations 10 \
  --episodes_per_iter 50 \
  --fine_tune_epochs 50 \
  --fine_tune_lr 1e-5 \
  --tb_log_dir ./logs/dagger_diffusion_franka \
  --checkpoint_dir ./checkpoints \
  --checkpoint_name dagger_diffusion_franka
```

Plot comparison:

```bash
python -m eval.plot_comparison_continuous \
  --logdirs ./logs/ppo_franka ./logs/grpo_franka ./logs/sac_franka ./logs/td3_franka \
  --labels PPO Pure-GRPO SAC TD3 \
  --tag Eval_AverageReturn \
  --output ./plots/rl_comparison_franka.png
```
