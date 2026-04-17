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
    "image":   uint8 array,   # shape: (num_envs, 3, 84, 84), wrist RGB image
    "proprio": float32 array, # shape: (num_envs, proprio_dim)
}
```

`proprio_dim` is config-driven and must not be hardcoded into every module. The PR0 scaffold can start with the minimal joint-only setting:

```python
proprio_dim = 14
```

PR2 should update `proprio_dim` to match the exact enabled feature list and test that the configured dimension equals the sum of feature dimensions.

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
  image (B, 3, 84, 84)
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
images:  (B, T_obs, 3, 84, 84)
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
| warmup_steps | N/A | N/A | 5k | 5k | N/A | N/A |
| policy_delay | N/A | N/A | 1 | 2 | N/A | N/A |
| diffusion_T | N/A | N/A | N/A | N/A | 100 | 100 |
| ddim_steps | N/A | N/A | N/A | N/A | 10 | 10 |
| action_horizon | N/A | N/A | N/A | N/A | 8 | 8 |
| exec_horizon | N/A | N/A | N/A | N/A | 4 | 4 |

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
pytest tests/test_isaac_task_config.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(env): add Franka IK-relative lift task config"
```

---

### PR 2 — Observation Wrapper

**Goal / Why**

Convert Isaac Lab native observations into one stable project observation format: `{"image": ..., "proprio": ...}`. This lets all algorithms share the same input contract.

**Implementation**
- Implement `IsaacArmEnv` wrapper around the configured Isaac Lab environment.
- Return batched observations with image shape `(num_envs, 3, 84, 84)`.
- Return `proprio` as float32 with config-driven `proprio_dim`.
- Define a stable proprio feature order and expose it in config, for example:

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

- Compute relative task features in the wrapper:
  - `ee_to_cube = cube_pos_base - ee_pos_base`,
  - `cube_to_target = target_pos_base - cube_pos_base`.
- Compute arm joint features as relative-to-default:
  - `arm_joint_pos_rel = arm_joint_pos - default_arm_joint_pos`,
  - `arm_joint_vel_rel = arm_joint_vel - default_arm_joint_vel`.
- Expose actual gripper state using finger joint positions/velocities or `gripper_width`, not only the previous gripper command.
- Treat `gripper_finger_pos` as actual/raw finger joint coordinates, not relative-to-default arm-style features.
- Keep privileged simulator task state clearly named so future real-robot variants can replace cube state with perception output.
- Support `reset()`, `step(actions)`, `close()`, and `max_episode_steps`.
- Handle `done` and `truncated` separately.
- Auto-reset completed environments if the underlying Isaac Lab vector env supports it; otherwise document and implement explicit reset handling.
- Do not implement reward shaping beyond exposing environment reward and success info.

**How To Test**
- Verify reset and step shapes.
- Verify image dtype is uint8 and proprio dtype is float32.
- Verify `proprio_dim` equals the sum of configured feature dimensions and that feature ordering is stable.
- Verify `arm_joint_pos_rel` is computed as `arm_joint_pos - default_arm_joint_pos`.
- Verify `arm_joint_vel_rel` is computed as `arm_joint_vel - default_arm_joint_vel`.
- Verify `gripper_finger_pos` equals actual/raw finger joint positions, for example open fingers near `[0.04, 0.04]` and closed fingers near `[0.0, 0.0]` in mock data.
- Verify optional `gripper_width` is computed as `left_finger_pos + right_finger_pos`.
- Verify `ee_to_cube` is computed as `cube_pos_base - ee_pos_base`.
- Verify `cube_to_target` is computed as `target_pos_base - cube_pos_base`.
- Verify actual gripper state comes from finger joint positions or `gripper_width`, not from `previous_action[-1]`.
- Verify a test case where `previous_action[-1]` commands close while finger joints remain open; the wrapper must report the actual open finger state.
- Verify batch size is preserved for `num_envs=1` and `num_envs>1`.
- Verify done/truncated arrays have shape `(num_envs,)`.
- Verify camera errors are readable.

**Pytest**

```bash
pytest tests/test_isaac_observation_wrapper.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(env): add image-proprio observation wrapper"
```

---

### PR 3 — Shared Backbone

**Goal / Why**

Implement the common image-proprio encoder used by RL agents and Diffusion Policy. This isolates perception from algorithm-specific code.

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

**Pytest**

```bash
pytest tests/test_nn_backbone.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(model): add image-proprio fusion backbone"
```

---

### PR 4 — PPO Baseline

**Goal / Why**

Implement PPO as the stable on-policy actor-critic baseline.

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

**Implementation**
- Add 7D squashed Gaussian actor.
- Add twin Q critics.
- Add target critics with soft update.
- Add replay buffer with uint8 images and float32 proprio/actions.
- Add automatic entropy temperature tuning.
- Ensure actor loss does not detach `Q(s, a_new)` from actor gradients.
- Add deterministic action mode for oracle data collection.
- Do not add TD3 or diffusion logic.

**How To Test**
- Verify action shape `(B, 7)` and range `[-1, 1]`.
- Verify actor receives nonzero gradient from Q term.
- Verify critic update reduces loss on synthetic data.
- Verify alpha changes in the correct direction.
- Verify target networks lag online networks.
- Verify replay buffer stores image as uint8 and action as float32.

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

**Implementation**
- Add deterministic 7D actor.
- Add twin Q critics.
- Add target actor and target critics.
- Add target policy smoothing.
- Add delayed actor updates.
- Add checkpoint save/load.
- Reuse replay buffer format from SAC if already implemented.
- Do not add SAC temperature logic or diffusion code.

**How To Test**
- Verify deterministic eval action is repeatable.
- Verify training action with noise varies.
- Verify policy delay skips actor update on the correct steps.
- Verify target smoothing noise is clipped.
- Verify save/load round trip.

**Pytest**

```bash
pytest tests/test_td3_continuous.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(rl): add continuous TD3 baseline"
```

---

### PR 8 — Demo Dataset

**Goal / Why**

Collect SAC expert demonstrations in an episode-safe format for Diffusion Policy training. Diffusion Policy needs observation histories and future action chunks, so flat `(s, a)` rows are not enough.

**Implementation**
- Add SAC demo collection script.
- Store demonstrations in HDF5 or zarr-style episode groups.
- Save images, proprios, actions, rewards, done, truncated, episode_id, and metadata.
- Implement dataset indexing for `(T_obs, image/proprio)` and `(H, 7)` action chunks.
- Prevent windows from crossing episode boundaries.
- Do not train Diffusion Policy in this PR.

**How To Test**
- Verify saved file contains required keys.
- Verify metadata includes action_dim, action_horizon, obs_history, env_id, and seed.
- Verify sampled windows never cross `done` or `truncated`.
- Verify action chunks have shape `(H, 7)`.
- Verify observation histories have shape `(T_obs, 3, 84, 84)` and `(T_obs, proprio_dim)`.

**Pytest**

```bash
pytest tests/test_demo_dataset.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(data): add episode-safe demonstration dataset"
```

---

### PR 9 — Diffusion Policy Behavior Cloning

**Goal / Why**

Train a Diffusion Policy from SAC demonstrations. This tests whether an offline imitation learner can reproduce the expert's manipulation behavior.

**Implementation**
- Add 1D temporal U-Net noise predictor.
- Add sinusoidal diffusion timestep embedding.
- Add FiLM conditioning from image-proprio observation history.
- Add DDPM noise prediction loss.
- Add DDIM inference with action chunking.
- Output `(exec_horizon, 7)` actions during deployment.
- Do not implement DAgger in this PR.

**How To Test**
- Verify noise schedule is monotonic.
- Verify forward diffusion variance matches expected behavior.
- Verify predicted noise shape is `(B, H, 7)`.
- Verify training loss is positive and decreases on toy data.
- Verify DDIM deterministic mode returns identical actions for identical inputs.
- Verify action chunk output shape is `(E, 7)`.

**Pytest**

```bash
pytest tests/test_diffusion_policy.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(il): add diffusion policy behavior cloning"
```

---

### PR 10 — DAgger

**Goal / Why**

Add dataset aggregation so the diffusion student learns how to recover from states it visits itself, not only states from the expert's original demonstrations.

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

**Pytest**

```bash
pytest tests/test_dagger_diffusion.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(il): add DAgger fine-tuning loop"
```

---

### PR 11 — Evaluation Metrics

**Goal / Why**

Provide one evaluator that works for RL agents and Diffusion Policy agents, so all methods are compared with the same metrics.

**Implementation**
- Add `evaluate_agent()`.
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

**Pytest**

```bash
pytest tests/test_eval_continuous.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(eval): add continuous-control evaluation metrics"
```

---

### PR 12 — Visual Outputs

**Goal / Why**

Create visual artifacts that make the project easy to inspect. GIFs show whether policies really grasp and lift the cube, not just whether scalar metrics improved.

**Implementation**
- Add rollout GIF recording script.
- Add side-by-side GIF/grid generation.
- Add plots for return, success rate, steps-to-threshold, and jerk.
- Use consistent seed, camera pose, and episode length across methods.
- Overlay method name, return, success, jerk, and seed.
- Do not change agent training code.

**How To Test**
- Verify GIF file is created.
- Verify GIF is non-empty.
- Verify expected number of frames is recorded.
- Verify comparison plot file is created.
- Verify missing checkpoint gives a readable error.

**Pytest**

```bash
pytest tests/test_visual_outputs.py -v
```

**Suggested Commit**

```bash
git commit -m "feat(viz): add rollout GIFs and comparison plots"
```

---

## 13. PR Roadmap Summary

| PR | What it does | Test command | Suggested commit |
|---|---|---|---|
| PR 0 | Project scaffold | `pytest tests/test_project_scaffold.py -v` | `chore: scaffold Isaac Lab manipulation project` |
| PR 1 | IK-relative Franka task + 7D action | `pytest tests/test_isaac_task_config.py -v` | `feat(env): add Franka IK-relative lift task config` |
| PR 2 | Image-proprio observation wrapper | `pytest tests/test_isaac_observation_wrapper.py -v` | `feat(env): add image-proprio observation wrapper` |
| PR 3 | Shared backbone | `pytest tests/test_nn_backbone.py -v` | `feat(model): add image-proprio fusion backbone` |
| PR 4 | PPO baseline | `pytest tests/test_ppo_continuous.py -v` | `feat(rl): add continuous PPO baseline` |
| PR 5 | Pure GRPO baseline | `pytest tests/test_grpo_continuous.py -v` | `feat(rl): add pure GRPO baseline` |
| PR 6 | SAC baseline | `pytest tests/test_sac_continuous.py -v` | `feat(rl): add continuous SAC baseline` |
| PR 7 | TD3 baseline | `pytest tests/test_td3_continuous.py -v` | `feat(rl): add continuous TD3 baseline` |
| PR 8 | Demo dataset | `pytest tests/test_demo_dataset.py -v` | `feat(data): add episode-safe demonstration dataset` |
| PR 9 | Diffusion BC | `pytest tests/test_diffusion_policy.py -v` | `feat(il): add diffusion policy behavior cloning` |
| PR 10 | DAgger | `pytest tests/test_dagger_diffusion.py -v` | `feat(il): add DAgger fine-tuning loop` |
| PR 11 | Evaluation metrics | `pytest tests/test_eval_continuous.py -v` | `feat(eval): add continuous-control evaluation metrics` |
| PR 12 | GIFs and plots | `pytest tests/test_visual_outputs.py -v` | `feat(viz): add rollout GIFs and comparison plots` |

---

## 14. Dependency Graph

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
PR 3 Shared Backbone
  |
  +--> PR 4 PPO
  +--> PR 5 Pure GRPO
  +--> PR 6 SAC ----+
  +--> PR 7 TD3     |
                    v
                 PR 8 Demo Dataset
                    |
                    v
                 PR 9 Diffusion BC
                    |
                    v
                 PR 10 DAgger

PR 11 Evaluation depends on PR 2 and agent interfaces from PR 4-10.
PR 12 Visual Outputs depends on PR 11 and trained checkpoints.
```

---

## 15. Global Test Commands

Fast CPU/unit tests, no Isaac Sim required:

```bash
pytest tests/ -v \
  --ignore=tests/test_isaac_task_config.py \
  --ignore=tests/test_isaac_observation_wrapper.py
```

Isaac Lab / GPU / camera tests:

```bash
pytest tests/test_isaac_task_config.py tests/test_isaac_observation_wrapper.py -v
```

Full project test suite after Isaac Lab is installed:

```bash
pytest tests/ -v
```

Single PR examples:

```bash
pytest tests/test_sac_continuous.py -v
pytest tests/test_diffusion_policy.py -v
pytest tests/test_visual_outputs.py -v
```

---

## 16. Example Run Commands

Train SAC expert:

```bash
python -m scripts.train_sac_continuous \
  --env_id Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 64 \
  --total_envsteps 500000 \
  --batch_size 256 \
  --replay_buffer_size 200000 \
  --learning_rate 3e-4 \
  --tb_log_dir ./logs/sac_franka \
  --checkpoint_dir ./checkpoints \
  --checkpoint_name sac_franka
```

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
