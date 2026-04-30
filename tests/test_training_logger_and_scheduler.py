"""Tests for PR6.5 training logger, LR scheduler, and periodic eval hooks."""

from __future__ import annotations

import json
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.checkpointing import load_checkpoint
from agents.normalization import ActionNormalizer
from agents.sac import SACAgent, SACConfig
from agents.td3 import TD3Agent, TD3Config
from configs import ACTION_DIM
from scripts.train_sac_continuous import _FakeSACEnv, _build_fake_env, parse_args as parse_sac_args
from scripts.train_td3_continuous import parse_args as parse_td3_args
from train.checkpoint_manager import (
    COMPOSITE_SUCCESS_LIFT_RETURN,
    STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN,
    TrainingCheckpointManager,
)
from train.loggers import CompositeLogger, JSONLinesLogger, TensorBoardLogger, TrainLogger, WandbLogger
from train.lr_scheduler import (
    estimate_total_update_steps,
    load_scheduler_collection_state,
    make_scheduler,
)
from train.progress import TrainProgressReporter
from train.sac_loop import (
    SACTrainLoopConfig,
    _merge_same_step_eval_checkpoint_metrics,
    run_sac_train_loop,
)
from train.td3_loop import TD3TrainLoopConfig, run_td3_train_loop


def _tiny_sac_config(**overrides) -> SACConfig:
    defaults = dict(
        action_dim=ACTION_DIM,
        proprio_dim=40,
        feat_dim=64,
        hidden_dim=64,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        utd_ratio=1,
        apply_image_aug=False,
    )
    defaults.update(overrides)
    return SACConfig(**defaults)


def _tiny_td3_config(**overrides) -> TD3Config:
    defaults = dict(
        action_dim=ACTION_DIM,
        proprio_dim=40,
        feat_dim=64,
        hidden_dim=64,
        actor_lr=3e-4,
        critic_lr=3e-4,
        utd_ratio=1,
        policy_delay=2,
        apply_image_aug=False,
    )
    defaults.update(overrides)
    return TD3Config(**defaults)


class _CountingLogger(TrainLogger):
    def __init__(self) -> None:
        self.scalar_calls: list[tuple[int, dict]] = []
        self.hparams: list[dict] = []
        self.closed = False

    def log_scalars(self, step: int, metrics: dict) -> None:
        self.scalar_calls.append((step, dict(metrics)))

    def log_hparams(self, hparams: dict) -> None:
        self.hparams.append(dict(hparams))

    def close(self) -> None:
        self.closed = True


class _ProxyWandbRun:
    def __init__(self) -> None:
        self.logs: list[dict] = []
        self.config: dict = {}
        self.finished = False

    def log(self, data: dict, *, step: int | None = None) -> None:
        self.logs.append({"step": step, **data})

    def finish(self) -> None:
        self.finished = True


class _RecordingProgress:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict, bool]] = []
        self.notes: list[tuple[int, str, dict]] = []

    def update(self, step: int, metrics: dict | None = None, *, force: bool = False) -> None:
        self.calls.append((step, dict(metrics or {}), force))

    def note(self, step: int, kind: str, fields: dict | None = None) -> None:
        self.notes.append((step, kind, dict(fields or {})))


class _ActionRecordingFakeEnv(_FakeSACEnv):
    def __init__(self, *, num_envs: int = 1, seed: int = 0, terminal_step: int = 50) -> None:
        super().__init__(num_envs=num_envs, seed=seed, terminal_step=terminal_step)
        self.step_actions: list[torch.Tensor] = []

    def step(self, action):
        self.step_actions.append(torch.as_tensor(action).detach().clone())
        return super().step(action)


def test_jsonlines_logger_writes_parseable_objects(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    logger = JSONLinesLogger(path)
    logger.log_hparams({"lr": 3e-4})
    logger.log_scalars(12, {"train/critic_loss": 1.25, "train/replay_size": 4})
    logger.close()

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["type"] == "hparams"
    assert lines[0]["hparams"]["lr"] == pytest.approx(3e-4)
    assert lines[1]["step"] == 12
    assert lines[1]["train/critic_loss"] == pytest.approx(1.25)


def test_composite_logger_deduplicates_backend_instances():
    backend = _CountingLogger()
    logger = CompositeLogger([backend, backend])
    logger.log_scalars(3, {"train/q_mean": 0.5})
    logger.close()

    assert len(backend.scalar_calls) == 1
    assert backend.scalar_calls[0][0] == 3
    assert backend.closed


def test_wandb_disabled_proxy_receives_logs_without_network():
    proxy = _ProxyWandbRun()
    logger = WandbLogger(mode="disabled", run=proxy)
    logger.log_hparams({"seed": 7})
    logger.log_scalars(4, {"train/actor_loss": 0.25})
    logger.close()

    assert proxy.config["seed"] == 7
    assert proxy.logs == [{"step": 4, "train/actor_loss": 0.25}]
    assert proxy.finished


def test_tensorboard_logger_optional_dependency(tmp_path: Path):
    logger = TensorBoardLogger(tmp_path / "tb")
    logger.log_scalars(1, {"train/critic_loss": 1.0})
    logger.close()

    if logger.enabled:
        assert any((tmp_path / "tb").glob("events.out.tfevents.*"))


def test_train_progress_reporter_prints_losses_and_eval_metrics():
    stream = io.StringIO()
    progress = TrainProgressReporter(
        total_env_steps=10,
        log_every_env_steps=100,
        log_every_train_steps=2,
        enabled=True,
        description="sac train",
        use_tqdm=False,
        stream=stream,
    )

    progress.update(4, {"train/update_step": 1, "train/critic_loss": 1.25, "train/replay_size": 4})
    assert stream.getvalue() == ""
    progress.note(4, "eval_start", {"eval_count": 1, "episodes": 2, "backend": "fake"})
    progress.update(4, {"train/update_step": 2, "train/critic_loss": 1.0, "train/replay_size": 4})
    progress.update(10, {"eval/mean_return": 2.5, "eval/success_rate": 0.5}, force=True)
    progress.close()

    output = stream.getvalue()
    assert "sac train | eval_start | env_step=4/10" in output
    assert "episodes=2" in output
    assert "backend=fake" in output
    assert "sac train | train | env_step=4/10" in output
    assert "sac train | eval | env_step=10/10" in output
    assert "update=2" in output
    assert "critic=1.000" in output
    assert "replay=4" in output
    assert "eval_return=2.500" in output
    assert "eval_success=0.500" in output


def test_train_progress_reporter_keeps_env_and_train_cadence_independent():
    stream = io.StringIO()
    progress = TrainProgressReporter(
        total_env_steps=30,
        log_every_env_steps=10,
        log_every_train_steps=2,
        enabled=True,
        description="sac train",
        use_tqdm=False,
        stream=stream,
    )

    progress.update(10, {"train/replay_size": 10, "train/num_env_steps": 10})
    progress.update(12, {"train/update_step": 2, "train/critic_loss": 0.5, "train/replay_size": 12})
    progress.update(20, {"train/replay_size": 20, "train/num_env_steps": 20})
    progress.update(22, {"train/update_step": 4, "train/critic_loss": 0.25, "train/replay_size": 22})
    progress.close()

    output = stream.getvalue()
    assert "sac train | env | env_step=10/30" in output
    assert "sac train | env | env_step=20/30" in output
    assert "sac train | train | env_step=12/30" in output
    assert "sac train | train | env_step=22/30" in output


def test_train_progress_reporter_prints_rollout_and_same_env_eval_metrics():
    stream = io.StringIO()
    progress = TrainProgressReporter(
        total_env_steps=12,
        log_every_env_steps=100,
        log_every_train_steps=100,
        enabled=True,
        description="sac train",
        use_tqdm=False,
        stream=stream,
    )

    progress.update(
        6,
        {
            "train_rollout/mean_return": 1.25,
            "train_rollout/success_rate": 0.5,
            "train_rollout/mean_episode_length": 3.0,
            "train_rollout/episode_count": 2.0,
        },
        force=True,
    )
    progress.update(
        6,
        {
            "eval_rollout/mean_return": 2.0,
            "eval_rollout/success_rate": 1.0,
            "eval_rollout/mean_episode_length": 3.0,
            "eval_rollout/episode_count": 1.0,
        },
        force=True,
    )
    progress.close()

    output = stream.getvalue()
    assert "sac train | train rollout | env_step=6/12" in output
    assert "train_rollout_return=1.250" in output
    assert "train_rollout_success=0.500" in output
    assert "sac train | eval rollout | env_step=6/12" in output
    assert "eval_rollout_return=2.000" in output
    assert "eval_rollout_success=1.000" in output


def test_train_progress_reporter_can_write_messages_to_log_file(tmp_path: Path):
    stream = io.StringIO()
    progress_log = tmp_path / "progress.log"
    progress = TrainProgressReporter(
        total_env_steps=12,
        log_every_env_steps=100,
        log_every_train_steps=100,
        enabled=False,
        description="sac train",
        use_tqdm=False,
        stream=stream,
        log_path=progress_log,
    )

    progress.update(
        6,
        {"train_rollout/mean_return": 1.25, "train_rollout/episode_count": 2.0},
        force=True,
    )
    progress.close()

    assert stream.getvalue() == ""
    output = progress_log.read_text(encoding="utf-8")
    assert "sac train | train rollout | env_step=6/12" in output
    assert "train_rollout_return=1.250" in output


def test_constant_and_step_schedulers():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)

    constant = make_scheduler("constant", optimizer)
    constant.step()
    constant.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

    step = make_scheduler("step", optimizer, step_size=2, gamma=0.1)
    step.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    step.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_warmup_cosine_scheduler_and_state_roundtrip():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = make_scheduler("warmup_cosine", optimizer, warmup_steps=2, total_update_steps=6, min_lr=0.1)

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    scheduler.step()
    mid_lr = optimizer.param_groups[0]["lr"]
    assert 0.1 < mid_lr < 1.0
    scheduler.step()

    state = scheduler.state_dict()
    restored_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(2.0))], lr=1.0)
    restored = make_scheduler("warmup_cosine", restored_optimizer, warmup_steps=2, total_update_steps=6, min_lr=0.1)
    restored.load_state_dict(state)
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
    restored.step()
    restored.step()
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_estimate_total_update_steps_counts_vectorized_steps():
    assert estimate_total_update_steps(
        total_env_steps=24,
        warmup_steps=8,
        num_envs=4,
        utd_ratio=2,
    ) == 8


def test_sac_train_loop_logs_lr_periodic_eval_and_checkpoint_scheduler_state(tmp_path: Path):
    torch.manual_seed(0)
    env = _build_fake_env(num_envs=4, seed=0)
    eval_factory_calls = {"count": 0}

    def _eval_factory():
        eval_factory_calls["count"] += 1
        return _build_fake_env(num_envs=2, seed=100 + eval_factory_calls["count"])

    agent = SACAgent(_tiny_sac_config())
    total_updates = estimate_total_update_steps(
        total_env_steps=24,
        warmup_steps=8,
        num_envs=4,
        utd_ratio=agent.config.utd_ratio,
    )
    schedulers = {
        "actor": make_scheduler(
            "warmup_cosine",
            agent.actor_optimizer,
            warmup_steps=1,
            total_update_steps=total_updates,
            min_lr=1e-5,
        ),
        "critic": make_scheduler(
            "warmup_cosine",
            agent.critic_optimizer,
            warmup_steps=1,
            total_update_steps=total_updates,
            min_lr=1e-5,
        ),
        "alpha": make_scheduler(
            "warmup_cosine",
            agent.alpha_optimizer,
            warmup_steps=1,
            total_update_steps=total_updates,
            min_lr=1e-5,
        ),
    }
    jsonl = tmp_path / "sac.jsonl"
    logger = JSONLinesLogger(jsonl)
    progress = _RecordingProgress()
    cfg = SACTrainLoopConfig(
        replay_capacity=64,
        warmup_steps=8,
        batch_size=4,
        total_env_steps=24,
        seed=0,
        ram_budget_gib=4.0,
        eval_every_env_steps=8,
        eval_num_episodes=2,
        eval_max_steps=3,
        eval_settle_steps=0,
        eval_seed=500,
        eval_backend="fake",
    )
    report = run_sac_train_loop(
        env,
        agent,
        loop_config=cfg,
        logger=logger,
        schedulers=schedulers,
        eval_env_factory=_eval_factory,
        progress=progress,
    )
    logger.close()

    assert report.num_updates == total_updates
    assert eval_factory_calls["count"] == 3
    assert len(report.eval_history) == 3
    assert "train/learning_rate_actor" in report.final_logs
    assert report.final_logs["train/update_step"] == pytest.approx(total_updates)
    assert "eval/mean_return" in report.eval_history[-1]
    progress_steps = [step for step, _metrics, _force in progress.calls]
    assert 4 in progress_steps
    assert 8 in progress_steps
    assert any("train/update_step" in metrics for _step, metrics, _force in progress.calls)
    assert any(force and "eval/mean_return" in metrics for _step, metrics, force in progress.calls)
    assert any(kind == "eval_start" for _step, kind, _fields in progress.notes)
    payloads = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert any("train/critic_loss" in payload for payload in payloads)
    assert any("eval/success_rate" in payload for payload in payloads)

    checkpoint = tmp_path / "sac.pt"
    agent.save(checkpoint, num_env_steps=report.num_env_steps, extras_update={"scheduler_state": report.scheduler_state})
    payload = load_checkpoint(checkpoint, expected_agent_type="sac")
    assert set(payload.extras["scheduler_state"]) == {"actor", "critic", "alpha"}

    reloaded_schedulers = {
        "actor": make_scheduler("warmup_cosine", agent.actor_optimizer, warmup_steps=1, total_update_steps=total_updates),
        "critic": make_scheduler("warmup_cosine", agent.critic_optimizer, warmup_steps=1, total_update_steps=total_updates),
        "alpha": make_scheduler("warmup_cosine", agent.alpha_optimizer, warmup_steps=1, total_update_steps=total_updates),
    }
    load_scheduler_collection_state(reloaded_schedulers, payload.extras["scheduler_state"])
    assert reloaded_schedulers["actor"].step_count == schedulers["actor"].step_count


def test_training_checkpoint_manager_saves_periodic_prunes_and_best(tmp_path: Path):
    torch.manual_seed(0)
    agent = SACAgent(_tiny_sac_config())
    manager = TrainingCheckpointManager(
        checkpoint_dir=tmp_path,
        checkpoint_name="sac_ckpt",
        checkpoint_every_env_steps=4,
        keep_last_checkpoints=2,
        save_best_by="eval_rollout/mean_return",
        seed=7,
        env_id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    )

    manager(agent, 4, {"train/num_env_steps": 4.0}, {"actor": {"step_count": 1}})
    manager(agent, 8, {"eval_rollout/mean_return": 1.0}, {"actor": {"step_count": 2}})
    manager(agent, 12, {"eval_rollout/mean_return": 0.5}, {"actor": {"step_count": 3}})
    manager(agent, 14, {"eval_rollout/mean_return": 2.0}, {"actor": {"step_count": 4}})

    assert not (tmp_path / "sac_ckpt_step_000000004.pt").exists()
    assert (tmp_path / "sac_ckpt_step_000000008.pt").exists()
    assert (tmp_path / "sac_ckpt_step_000000012.pt").exists()
    best = tmp_path / "sac_ckpt_best.pt"
    assert best.exists()
    best_payload = load_checkpoint(best, expected_agent_type="sac")
    assert best_payload.metadata.num_env_steps == 14
    assert best_payload.extras["scheduler_state"]["actor"]["step_count"] == 4
    assert manager.best_metric_value == pytest.approx(2.0)
    assert any(event["kind"] == "pruned" for event in manager.history)


def test_training_checkpoint_manager_composite_best_prefers_success_then_lift_then_return(tmp_path: Path):
    torch.manual_seed(0)
    agent = SACAgent(_tiny_sac_config())
    manager = TrainingCheckpointManager(
        checkpoint_dir=tmp_path,
        checkpoint_name="sac_composite",
        save_best_by=COMPOSITE_SUCCESS_LIFT_RETURN,
        seed=7,
        env_id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    )

    manager(
        agent,
        10,
        {
            "eval_rollout/success_rate": 0.0,
            "eval_rollout/max_cube_lift_m": 0.01,
            "eval_rollout/mean_return": 10.0,
        },
        {"actor": {"step_count": 1}},
    )
    manager(
        agent,
        20,
        {
            "eval_rollout/success_rate": 0.0,
            "eval_rollout/max_cube_lift_m": 0.02,
            "eval_rollout/mean_return": 0.0,
        },
        {"actor": {"step_count": 2}},
    )
    manager(
        agent,
        30,
        {
            "eval_rollout/success_rate": 1.0,
            "eval_rollout/max_cube_lift_m": 0.0,
            "eval_rollout/mean_return": -10.0,
        },
        {"actor": {"step_count": 3}},
    )

    best = tmp_path / "sac_composite_best.pt"
    assert best.exists()
    best_payload = load_checkpoint(best, expected_agent_type="sac")
    assert best_payload.metadata.num_env_steps == 30
    assert manager.best_metric_value == pytest.approx((1.0, 0.0, -10.0))
    assert manager.history[-1]["metric_key"] == COMPOSITE_SUCCESS_LIFT_RETURN
    assert manager.history[-1]["metric_value"] == pytest.approx([1.0, 0.0, -10.0])


def test_training_checkpoint_manager_composite_best_requires_all_eval_rollout_metrics(tmp_path: Path):
    agent = SACAgent(_tiny_sac_config())
    manager = TrainingCheckpointManager(
        checkpoint_dir=tmp_path,
        checkpoint_name="sac_bad_composite",
        save_best_by=COMPOSITE_SUCCESS_LIFT_RETURN,
        env_id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    )

    with pytest.raises(ValueError, match="requires"):
        manager(agent, 10, {"eval_rollout/mean_return": 1.0}, None)


def test_training_checkpoint_manager_stage_aware_best_keeps_per_stage_records(tmp_path: Path):
    torch.manual_seed(0)
    agent = SACAgent(_tiny_sac_config())
    manager = TrainingCheckpointManager(
        checkpoint_dir=tmp_path,
        checkpoint_name="sac_stage",
        save_best_by=STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN,
        seed=7,
        env_id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    )

    manager(
        agent,
        10,
        {
            "curriculum/stage_index": 0.0,
            "curriculum/gate/eval_reach_episode_rate": 0.2,
            "eval_rollout/min_ee_to_cube_m": 0.20,
            "eval_rollout/mean_return": 0.0,
        },
        None,
    )
    manager(
        agent,
        20,
        {
            "curriculum/stage_index": 0.0,
            "curriculum/gate/eval_reach_episode_rate": 0.2,
            "eval_rollout/min_ee_to_cube_m": 0.10,
            "eval_rollout/mean_return": -1.0,
        },
        None,
    )
    manager(
        agent,
        30,
        {
            "curriculum/stage_index": 1.0,
            "curriculum/gate/eval_grip_effect_episode_rate": 0.1,
            "curriculum/gate/eval_grip_attempt_episode_rate": 0.3,
            "eval_rollout/min_ee_to_cube_m": 0.15,
        },
        None,
    )

    assert (tmp_path / "sac_stage_best_stage0.pt").exists()
    assert (tmp_path / "sac_stage_best_stage1.pt").exists()
    stage0_payload = load_checkpoint(tmp_path / "sac_stage_best_stage0.pt", expected_agent_type="sac")
    stage1_payload = load_checkpoint(tmp_path / "sac_stage_best_stage1.pt", expected_agent_type="sac")
    assert stage0_payload.metadata.num_env_steps == 20
    assert stage1_payload.metadata.num_env_steps == 30
    assert manager.stage_best_metric_values[0] == pytest.approx((0.2, -0.10, -1.0))
    assert manager.stage_best_metric_values[1] == pytest.approx((0.1, 0.3, -0.15))
    assert manager.history[-1]["stage_index"] == 1
    assert manager.history[-1]["metric_key"] == STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN


def test_training_checkpoint_manager_stage_aware_best_requires_stage_metrics(tmp_path: Path):
    agent = SACAgent(_tiny_sac_config())
    manager = TrainingCheckpointManager(
        checkpoint_dir=tmp_path,
        checkpoint_name="sac_stage_bad",
        save_best_by=STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN,
        env_id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    )

    with pytest.raises(ValueError, match="requires"):
        manager(
            agent,
            10,
            {
                "curriculum/stage_index": 0.0,
                "curriculum/gate/eval_reach_episode_rate": 0.2,
                "eval_rollout/min_ee_to_cube_m": 0.2,
            },
            None,
        )


def test_training_checkpoint_manager_stage_aware_ignores_train_and_gate_only_logs(tmp_path: Path):
    agent = SACAgent(_tiny_sac_config())
    manager = TrainingCheckpointManager(
        checkpoint_dir=tmp_path,
        checkpoint_name="sac_stage_train_only",
        save_best_by=STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN,
        env_id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    )

    manager(
        agent,
        10,
        {
            "curriculum/stage_index": 0.0,
            "curriculum/gate/eval_reach_episode_rate": 0.0,
            "reward/train/reach_progress": 0.01,
            "train/update_step": 1.0,
        },
        None,
    )

    assert not (tmp_path / "sac_stage_train_only_best_stage0.pt").exists()
    assert manager.history == []


def test_stage_aware_checkpoint_uses_merged_same_step_eval_and_gate_metrics(tmp_path: Path):
    agent = SACAgent(_tiny_sac_config())
    manager = TrainingCheckpointManager(
        checkpoint_dir=tmp_path,
        checkpoint_name="sac_stage_merged",
        save_best_by=STAGE_AWARE_REACH_LIFT_SUCCESS_RETURN,
        env_id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    )
    eval_rollout_metrics = {
        "eval_rollout/min_ee_to_cube_m": 0.03,
        "eval_rollout/mean_return": 0.18,
        "eval_rollout/window_size": 20.0,
    }
    gate_metrics = {
        "curriculum/stage_index": 0.0,
        "curriculum/gate/eval_reach_episode_rate": 0.30,
        "curriculum/gate/advanced_stage": 1.0,
        "train/replay_size": 200000.0,
    }

    manager(agent, 198800, eval_rollout_metrics, None)
    manager(agent, 198800, gate_metrics, None)

    assert not (tmp_path / "sac_stage_merged_best_stage0.pt").exists()

    merged_metrics = _merge_same_step_eval_checkpoint_metrics(gate_metrics, eval_rollout_metrics)
    manager(agent, 198800, merged_metrics, {"actor": {"step_count": 12}})

    stage0_best = tmp_path / "sac_stage_merged_best_stage0.pt"
    assert stage0_best.exists()
    assert (tmp_path / "sac_stage_merged_best.pt").exists()
    payload = load_checkpoint(stage0_best, expected_agent_type="sac")
    assert payload.metadata.num_env_steps == 198800
    assert payload.extras["scheduler_state"]["actor"]["step_count"] == 12
    assert manager.stage_best_metric_values[0] == pytest.approx((0.30, -0.03, 0.18))


def test_train_script_parsers_accept_checkpoint_curriculum_and_alpha_controls():
    sac_args = parse_sac_args(
        [
            "--backend",
            "fake",
            "--alpha-min",
            "0.05",
            "--checkpoint-every-env-steps",
            "50000",
            "--keep-last-checkpoints",
            "5",
            "--save-best-by",
            "eval_rollout/mean_return",
            "--disable-reward-curriculum",
            "--reward-curriculum",
            "reach_grip_lift_goal",
            "--curriculum-stage-fracs",
            "0.2,0.5,0.8",
            "--curriculum-gating",
            "bucket_rates",
            "--curriculum-gate-window-transitions",
            "123",
            "--curriculum-gate-thresholds",
            "0.1,0.2,0.3",
            "--curriculum-gate-eval-window-episodes",
            "12",
            "--curriculum-gate-min-eval-episodes",
            "8",
            "--curriculum-gate-eval-thresholds",
            "0.4,0.3,0.05,0.1",
            "--curriculum-gate-min-train-exposures",
            "400,100,20,20",
            "--curriculum-gate-lift-success-height-m",
            "0.02",
            "--curriculum-gate-min-stage-env-steps",
            "10000",
            "--grip-proxy-scale",
            "1.25",
            "--lift-progress-deadband-m",
            "0.003",
            "--lift-progress-height-m",
            "0.05",
            "--reach-progress-stage-scales",
            "0.5,0.1,0,0",
            "--reach-progress-clip-m",
            "0.01",
            "--vertical-alignment-penalty-scale",
            "0.1",
            "--vertical-alignment-penalty-stages",
            "reach",
            "--vertical-alignment-deadband-m",
            "0.04",
            "--rotation-action-penalty-scale",
            "0.005",
            "--rotation-action-penalty-stages",
            "reach",
            "--prioritize-replay",
            "--priority-replay-ratio",
            "0.5",
            "--priority-score-weights",
            "0.4,0.25,0.2,0.15",
            "--protect-rare-transitions",
            "--protected-replay-fraction",
            "0.2",
            "--protected-score-weights",
            "0.6,0.3,0.1",
            "--protected-max-age-env-steps",
            "100000",
            "--protected-refresh-every-env-steps",
            "10000",
            "--protected-min-score",
            "0.01",
            "--protected-stage-local",
            "--protected-stage-grace-env-steps",
            "50000",
            "--protected-old-stage-retain-fraction",
            "0.5",
        ]
    )
    td3_args = parse_td3_args(
        [
            "--backend",
            "fake",
            "--checkpoint-every-env-steps",
            "50000",
            "--keep-last-checkpoints",
            "5",
            "--save-best-by",
            "eval_rollout/mean_return",
            "--disable-reward-curriculum",
            "--reward-curriculum",
            "reach_grip_lift_goal",
            "--curriculum-gating",
            "bucket_rates",
            "--curriculum-gate-thresholds",
            "0.1,0.2,0.3",
            "--curriculum-gate-eval-thresholds",
            "0.4,0.3,0.05,0.1",
            "--curriculum-gate-min-train-exposures",
            "400,100,20,20",
            "--lift-progress-deadband-m",
            "0.003",
            "--lift-progress-height-m",
            "0.05",
            "--reach-progress-stage-scales",
            "0.5,0.1,0,0",
            "--reach-progress-clip-m",
            "0.01",
            "--vertical-alignment-penalty-scale",
            "0.1",
            "--vertical-alignment-penalty-stages",
            "reach",
            "--vertical-alignment-deadband-m",
            "0.04",
            "--rotation-action-penalty-scale",
            "0.005",
            "--rotation-action-penalty-stages",
            "reach",
            "--prioritize-replay",
            "--priority-replay-ratio",
            "0.5",
            "--protect-rare-transitions",
            "--protected-score-weights",
            "0.7,0.2,0.1",
            "--protected-max-age-env-steps",
            "100000",
            "--protected-refresh-every-env-steps",
            "10000",
            "--protected-min-score",
            "0.01",
            "--protected-stage-local",
            "--protected-stage-grace-env-steps",
            "50000",
            "--protected-old-stage-retain-fraction",
            "0.5",
        ]
    )

    assert sac_args.alpha_min == pytest.approx(0.05)
    assert sac_args.checkpoint_every_env_steps == 50000
    assert sac_args.keep_last_checkpoints == 5
    assert sac_args.save_best_by == "eval_rollout/mean_return"
    assert sac_args.disable_reward_curriculum is True
    assert sac_args.reward_curriculum == "reach_grip_lift_goal"
    assert sac_args.curriculum_stage_fracs == "0.2,0.5,0.8"
    assert sac_args.curriculum_gating == "bucket_rates"
    assert sac_args.curriculum_gate_window_transitions == 123
    assert sac_args.curriculum_gate_thresholds == "0.1,0.2,0.3"
    assert sac_args.curriculum_gate_eval_window_episodes == 12
    assert sac_args.curriculum_gate_min_eval_episodes == 8
    assert sac_args.curriculum_gate_eval_thresholds == "0.4,0.3,0.05,0.1"
    assert sac_args.curriculum_gate_min_train_exposures == "400,100,20,20"
    assert sac_args.curriculum_gate_lift_success_height_m == pytest.approx(0.02)
    assert sac_args.curriculum_gate_min_stage_env_steps == 10000
    assert sac_args.grip_proxy_scale == pytest.approx(1.25)
    assert sac_args.lift_progress_deadband_m == pytest.approx(0.003)
    assert sac_args.lift_progress_height_m == pytest.approx(0.05)
    assert sac_args.reach_progress_stage_scales == "0.5,0.1,0,0"
    assert sac_args.reach_progress_clip_m == pytest.approx(0.01)
    assert sac_args.vertical_alignment_penalty_scale == pytest.approx(0.1)
    assert sac_args.vertical_alignment_penalty_stages == "reach"
    assert sac_args.vertical_alignment_deadband_m == pytest.approx(0.04)
    assert sac_args.rotation_action_penalty_scale == pytest.approx(0.005)
    assert sac_args.rotation_action_penalty_stages == "reach"
    assert sac_args.prioritize_replay is True
    assert sac_args.priority_replay_ratio == pytest.approx(0.5)
    assert sac_args.priority_score_weights == "0.4,0.25,0.2,0.15"
    assert sac_args.protect_rare_transitions is True
    assert sac_args.protected_replay_fraction == pytest.approx(0.2)
    assert sac_args.protected_score_weights == "0.6,0.3,0.1"
    assert sac_args.protected_max_age_env_steps == 100000
    assert sac_args.protected_refresh_every_env_steps == 10000
    assert sac_args.protected_min_score == pytest.approx(0.01)
    assert sac_args.protected_stage_local is True
    assert sac_args.protected_stage_grace_env_steps == 50000
    assert sac_args.protected_old_stage_retain_fraction == pytest.approx(0.5)
    assert td3_args.checkpoint_every_env_steps == 50000
    assert td3_args.keep_last_checkpoints == 5
    assert td3_args.save_best_by == "eval_rollout/mean_return"
    assert td3_args.disable_reward_curriculum is True
    assert td3_args.reward_curriculum == "reach_grip_lift_goal"
    assert td3_args.curriculum_gating == "bucket_rates"
    assert td3_args.curriculum_gate_thresholds == "0.1,0.2,0.3"
    assert td3_args.curriculum_gate_eval_thresholds == "0.4,0.3,0.05,0.1"
    assert td3_args.curriculum_gate_min_train_exposures == "400,100,20,20"
    assert td3_args.lift_progress_deadband_m == pytest.approx(0.003)
    assert td3_args.lift_progress_height_m == pytest.approx(0.05)
    assert td3_args.reach_progress_stage_scales == "0.5,0.1,0,0"
    assert td3_args.reach_progress_clip_m == pytest.approx(0.01)
    assert td3_args.vertical_alignment_penalty_scale == pytest.approx(0.1)
    assert td3_args.vertical_alignment_penalty_stages == "reach"
    assert td3_args.vertical_alignment_deadband_m == pytest.approx(0.04)
    assert td3_args.rotation_action_penalty_scale == pytest.approx(0.005)
    assert td3_args.rotation_action_penalty_stages == "reach"
    assert td3_args.prioritize_replay is True
    assert td3_args.priority_replay_ratio == pytest.approx(0.5)
    assert td3_args.protect_rare_transitions is True
    assert td3_args.protected_score_weights == "0.7,0.2,0.1"
    assert td3_args.protected_max_age_env_steps == 100000
    assert td3_args.protected_refresh_every_env_steps == 10000
    assert td3_args.protected_min_score == pytest.approx(0.01)
    assert td3_args.protected_stage_local is True
    assert td3_args.protected_stage_grace_env_steps == 50000
    assert td3_args.protected_old_stage_retain_fraction == pytest.approx(0.5)


def test_sac_train_loop_logs_training_rollouts_and_same_env_eval_lanes():
    torch.manual_seed(0)
    env = _FakeSACEnv(num_envs=4, seed=0, terminal_step=3)
    agent = SACAgent(_tiny_sac_config())
    logger = _CountingLogger()
    progress = _RecordingProgress()
    cfg = SACTrainLoopConfig(
        replay_capacity=64,
        warmup_steps=6,
        batch_size=4,
        total_env_steps=18,
        seed=0,
        ram_budget_gib=4.0,
        same_env_eval_lanes=1,
        rollout_metrics_window=10,
    )

    report = run_sac_train_loop(env, agent, loop_config=cfg, logger=logger, progress=progress)

    assert report.num_env_steps == 18
    assert report.final_logs["train/replay_size"] == pytest.approx(18)
    assert report.final_logs["normalizer/proprio_count"] == pytest.approx(18)
    assert agent.normalizers.proprio.count == 18
    assert any("train_rollout/mean_return" in logs for logs in report.log_history)
    assert any("eval_rollout/mean_return" in logs for logs in report.log_history)
    assert any("train_rollout/success_rate" in metrics for _step, metrics in logger.scalar_calls)
    assert any("eval_rollout/success_rate" in metrics for _step, metrics in logger.scalar_calls)
    assert any("train_rollout/mean_return" in metrics for _step, metrics, _force in progress.calls)
    assert any("eval_rollout/mean_return" in metrics for _step, metrics, _force in progress.calls)


@pytest.mark.parametrize(
    ("agent", "config", "runner"),
    [
        (lambda: SACAgent(_tiny_sac_config()), SACTrainLoopConfig, run_sac_train_loop),
        (lambda: TD3Agent(_tiny_td3_config()), TD3TrainLoopConfig, run_td3_train_loop),
    ],
)
def test_train_loop_logs_stock_reward_component_breakdown(agent, config, runner):
    torch.manual_seed(0)
    env = _FakeSACEnv(num_envs=4, seed=0, terminal_step=3)
    logger = _CountingLogger()
    progress = _RecordingProgress()
    cfg = config(
        replay_capacity=64,
        warmup_steps=6,
        batch_size=4,
        total_env_steps=18,
        seed=0,
        ram_budget_gib=4.0,
        same_env_eval_lanes=1,
        rollout_metrics_window=10,
    )

    report = runner(env, agent(), loop_config=cfg, logger=logger, progress=progress)

    assert any("reward/train/native_total" in logs for logs in report.log_history)
    assert any("reward/train/reaching_object" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/reaching_object" in logs for logs in report.log_history)
    assert any("reward/train/action_rate" in metrics for _step, metrics in logger.scalar_calls)
    assert any("reward/eval_rollout/action_rate" in metrics for _step, metrics in logger.scalar_calls)
    assert any("reward/train/native_total" in metrics for _step, metrics, _force in progress.calls)


@pytest.mark.parametrize(
    ("agent", "config", "runner"),
    [
        (lambda: SACAgent(_tiny_sac_config()), SACTrainLoopConfig, run_sac_train_loop),
        (lambda: TD3Agent(_tiny_td3_config()), TD3TrainLoopConfig, run_td3_train_loop),
    ],
)
def test_train_loop_logs_reward_curriculum_and_prioritized_replay(agent, config, runner):
    torch.manual_seed(0)
    env = _FakeSACEnv(num_envs=4, seed=0, terminal_step=3)
    logger = _CountingLogger()
    progress = _RecordingProgress()
    cfg = config(
        replay_capacity=64,
        warmup_steps=4,
        batch_size=4,
        total_env_steps=16,
        seed=0,
        ram_budget_gib=4.0,
        reward_curriculum="reach_grip_lift_goal",
        prioritize_replay=True,
        priority_replay_ratio=0.5,
        protect_rare_transitions=True,
        protected_replay_fraction=0.25,
        protected_score_weights=(0.6, 0.3, 0.1),
    )

    report = runner(env, agent(), loop_config=cfg, logger=logger, progress=progress)

    assert report.num_updates > 0
    assert any("reward/train_shaped" in logs for logs in report.log_history)
    assert any("curriculum/stage_index" in logs for logs in report.log_history)
    assert any("priority_replay/bucket_count/reach" in logs for logs in report.log_history)
    assert any("train/td_error_mean" in logs for logs in report.log_history)
    assert any(
        metrics.get("priority_replay/batch_priority") == pytest.approx(2.0)
        for _step, metrics in logger.scalar_calls
    )
    assert any("reward/train_shaped" in metrics for _step, metrics, _force in progress.calls)


@pytest.mark.parametrize(
    ("agent", "config", "runner"),
    [
        (lambda: SACAgent(_tiny_sac_config()), SACTrainLoopConfig, run_sac_train_loop),
        (lambda: TD3Agent(_tiny_td3_config()), TD3TrainLoopConfig, run_td3_train_loop),
    ],
)
def test_train_loop_logs_eval_rollout_curriculum_diagnostics(agent, config, runner):
    torch.manual_seed(0)
    env = _FakeSACEnv(num_envs=4, seed=0, terminal_step=3)
    logger = _CountingLogger()
    progress = _RecordingProgress()
    cfg = config(
        replay_capacity=64,
        warmup_steps=4,
        batch_size=4,
        total_env_steps=18,
        seed=0,
        ram_budget_gib=4.0,
        same_env_eval_lanes=1,
        rollout_metrics_window=10,
        reward_curriculum="reach_grip_lift_goal",
    )

    report = runner(env, agent(), loop_config=cfg, logger=logger, progress=progress)

    assert any("reward/eval_rollout/native_total" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/eval_shaped" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/grip_proxy" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/reaching_object" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/lifting_object" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/object_goal_tracking" in logs for logs in report.log_history)
    assert any(
        "reward/eval_rollout/object_goal_tracking_fine_grained" in logs for logs in report.log_history
    )
    assert any("reward/eval_rollout/action_rate" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/joint_vel" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/eval_shaped" in metrics for _step, metrics in logger.scalar_calls)
    assert any("reward/eval_rollout/grip_proxy" in metrics for _step, metrics, _force in progress.calls)


@pytest.mark.parametrize(
    ("agent", "config", "runner"),
    [
        (lambda: SACAgent(_tiny_sac_config()), SACTrainLoopConfig, run_sac_train_loop),
        (lambda: TD3Agent(_tiny_td3_config()), TD3TrainLoopConfig, run_td3_train_loop),
    ],
)
def test_train_loop_logs_pr69_lift_gate_action_and_diagnostic_replay_metrics(agent, config, runner):
    torch.manual_seed(0)
    env = _FakeSACEnv(num_envs=4, seed=0, terminal_step=3)
    logger = _CountingLogger()
    progress = _RecordingProgress()
    cfg = config(
        replay_capacity=64,
        warmup_steps=4,
        batch_size=4,
        total_env_steps=18,
        seed=0,
        ram_budget_gib=4.0,
        same_env_eval_lanes=1,
        rollout_metrics_window=10,
        reward_curriculum="reach_grip_lift_goal",
        curriculum_gating="bucket_rates",
        curriculum_gate_window_transitions=8,
        curriculum_gate_thresholds=(0.0, 0.0, 0.0),
        prioritize_replay=True,
        priority_replay_ratio=0.5,
        protect_rare_transitions=True,
        protected_replay_fraction=0.25,
    )

    report = runner(env, agent(), loop_config=cfg, logger=logger, progress=progress)

    assert report.num_updates > 0
    assert any("reward/train/lift_progress_proxy" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/lift_progress_proxy" in logs for logs in report.log_history)
    assert any("curriculum/gate/reach_rate" in logs for logs in report.log_history)
    assert any("curriculum/gate/held_stage" in logs for logs in report.log_history)
    assert any("priority_replay/bucket_count/grip_attempt" in logs for logs in report.log_history)
    assert any("priority_replay/bucket_count/grip_effect" in logs for logs in report.log_history)
    assert any("action/train/gripper_mean" in logs for logs in report.log_history)
    assert any("action/train/gripper_close_rate" in logs for logs in report.log_history)
    assert any("action/train/translation_norm" in logs for logs in report.log_history)
    assert any("action/train/rotation_norm" in logs for logs in report.log_history)
    assert any("action/train/gripper_abs_mean" in logs for logs in report.log_history)
    assert any("action/eval_rollout/gripper_mean" in logs for logs in report.log_history)
    assert any("action/eval_rollout/gripper_close_rate" in logs for logs in report.log_history)
    assert any("action/eval_rollout/gripper_close_near_cube_rate" in logs for logs in report.log_history)
    assert any("action/eval_rollout/translation_norm" in logs for logs in report.log_history)
    assert any("action/eval_rollout/rotation_norm" in logs for logs in report.log_history)
    assert any("action/eval_rollout/gripper_abs_mean" in logs for logs in report.log_history)
    assert any("reward/train/reach_progress" in logs for logs in report.log_history)
    assert any("reward/train/vertical_alignment_penalty" in logs for logs in report.log_history)
    assert any("reward/train/rotation_action_penalty" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/reach_progress" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/vertical_alignment_penalty" in logs for logs in report.log_history)
    assert any("reward/eval_rollout/rotation_action_penalty" in logs for logs in report.log_history)
    assert any("eval_rollout/max_cube_lift_m" in logs for logs in report.log_history)
    assert any("eval_rollout/min_ee_to_cube_m" in logs for logs in report.log_history)
    assert any("eval_rollout/min_cube_to_target_m" in logs for logs in report.log_history)
    assert any("eval_rollout/gripper_close_near_cube_rate" in logs for logs in report.log_history)
    assert any("action/train/gripper_mean" in metrics for _step, metrics in logger.scalar_calls)
    assert any("reward/train/reach_progress" in metrics for _step, metrics in logger.scalar_calls)
    assert any("eval_rollout/max_cube_lift_m" in metrics for _step, metrics in logger.scalar_calls)
    assert any("action/eval_rollout/gripper_mean" in metrics for _step, metrics, _force in progress.calls)
    assert any("reward/eval_rollout/reach_progress" in metrics for _step, metrics, _force in progress.calls)


@pytest.mark.parametrize(
    ("agent", "config", "runner"),
    [
        (lambda: SACAgent(_tiny_sac_config()), SACTrainLoopConfig, run_sac_train_loop),
        (lambda: TD3Agent(_tiny_td3_config()), TD3TrainLoopConfig, run_td3_train_loop),
    ],
)
def test_train_loop_logs_pr610_eval_dual_gate_metrics(agent, config, runner):
    torch.manual_seed(0)
    env = _FakeSACEnv(num_envs=4, seed=0, terminal_step=3)
    logger = _CountingLogger()
    progress = _RecordingProgress()
    cfg = config(
        replay_capacity=64,
        warmup_steps=4,
        batch_size=4,
        total_env_steps=18,
        seed=0,
        ram_budget_gib=4.0,
        same_env_eval_lanes=1,
        rollout_metrics_window=10,
        reward_curriculum="reach_grip_lift_goal",
        curriculum_gating="eval_dual_gate",
        curriculum_gate_eval_window_episodes=4,
        curriculum_gate_min_eval_episodes=1,
        curriculum_gate_eval_thresholds=(0.0, 0.0, 0.0, 0.0),
        curriculum_gate_min_train_exposures=(0, 0, 0, 0),
        curriculum_gate_lift_success_height_m=0.02,
        curriculum_gate_min_stage_env_steps=0,
    )

    report = runner(env, agent(), loop_config=cfg, logger=logger, progress=progress)

    assert report.num_updates > 0
    assert any("curriculum/gate/mode_eval_dual_gate" in logs for logs in report.log_history)
    assert any("curriculum/gate/eval_reach_episode_rate" in logs for logs in report.log_history)
    assert any("curriculum/gate/eval_grip_attempt_episode_rate" in logs for logs in report.log_history)
    assert any("curriculum/gate/eval_grip_effect_episode_rate" in logs for logs in report.log_history)
    assert any("curriculum/gate/eval_lift_2cm_episode_rate" in logs for logs in report.log_history)
    assert any("curriculum/gate/exposure_reach_count" in logs for logs in report.log_history)
    assert any("curriculum/gate/exposure_grip_attempt_count" in logs for logs in report.log_history)
    assert any("curriculum/gate/exposure_grip_effect_count" in logs for logs in report.log_history)
    assert any("curriculum/gate/exposure_lift_progress_count" in logs for logs in report.log_history)
    assert any("curriculum/gate/eval_gate_passed" in metrics for _step, metrics in logger.scalar_calls)
    assert any("curriculum/gate/advanced_stage" in metrics for _step, metrics, _force in progress.calls)
    assert any(kind == "curriculum_advance" for _step, kind, _fields in progress.notes)


@pytest.mark.parametrize(
    ("agent", "config", "runner"),
    [
        (lambda: SACAgent(_tiny_sac_config()), SACTrainLoopConfig, run_sac_train_loop),
        (lambda: TD3Agent(_tiny_td3_config()), TD3TrainLoopConfig, run_td3_train_loop),
    ],
)
def test_eval_dual_gate_requires_same_env_eval_lanes(agent, config, runner):
    env = _FakeSACEnv(num_envs=2, seed=0, terminal_step=3)
    cfg = config(
        replay_capacity=64,
        warmup_steps=0,
        batch_size=4,
        total_env_steps=4,
        seed=0,
        ram_budget_gib=4.0,
        reward_curriculum="reach_grip_lift_goal",
        curriculum_gating="eval_dual_gate",
    )

    with pytest.raises(ValueError, match="same_env_eval_lanes"):
        runner(env, agent(), loop_config=cfg)


def test_sac_same_env_eval_waits_for_clean_episode_after_start_threshold():
    torch.manual_seed(0)
    env = _FakeSACEnv(num_envs=4, seed=0, terminal_step=3)
    agent = SACAgent(_tiny_sac_config())
    logger = _CountingLogger()
    cfg = SACTrainLoopConfig(
        replay_capacity=64,
        warmup_steps=6,
        batch_size=4,
        total_env_steps=18,
        seed=0,
        ram_budget_gib=4.0,
        same_env_eval_lanes=1,
        same_env_eval_start_env_steps=9,
        rollout_metrics_window=10,
    )

    run_sac_train_loop(env, agent, loop_config=cfg, logger=logger)

    eval_lane_calls = [
        (step, metrics)
        for step, metrics in logger.scalar_calls
        if "eval_rollout/mean_return" in metrics
    ]
    assert len(eval_lane_calls) == 1
    step, metrics = eval_lane_calls[0]
    assert step == 18
    assert metrics["eval_rollout/episode_count"] == pytest.approx(1.0)
    assert metrics["eval_rollout/mean_episode_length"] == pytest.approx(3.0)


def test_sac_train_loop_settles_with_zero_actions_before_replay_collection():
    torch.manual_seed(0)
    env = _ActionRecordingFakeEnv(num_envs=2, seed=0, terminal_step=100)
    agent = SACAgent(_tiny_sac_config())
    cfg = SACTrainLoopConfig(
        replay_capacity=64,
        warmup_steps=0,
        batch_size=4,
        total_env_steps=4,
        seed=0,
        ram_budget_gib=4.0,
        settle_steps=3,
    )

    report = run_sac_train_loop(env, agent, loop_config=cfg)

    assert report.num_env_steps == 4
    assert report.final_logs["train/replay_size"] == pytest.approx(4)
    assert len(env.step_actions) == 5
    for action in env.step_actions[:3]:
        assert torch.allclose(action, torch.zeros((2, ACTION_DIM)))
    assert not torch.allclose(env.step_actions[3], torch.zeros((2, ACTION_DIM)))


@pytest.mark.parametrize(
    ("agent", "config", "runner"),
    [
        (lambda: SACAgent(_tiny_sac_config()), SACTrainLoopConfig, run_sac_train_loop),
        (lambda: TD3Agent(_tiny_td3_config()), TD3TrainLoopConfig, run_td3_train_loop),
    ],
)
def test_train_loop_reports_settle_and_warmup_progress(agent, config, runner):
    torch.manual_seed(0)
    env = _ActionRecordingFakeEnv(num_envs=2, seed=0, terminal_step=3)
    progress = _RecordingProgress()
    cfg = config(
        replay_capacity=64,
        warmup_steps=2,
        batch_size=4,
        total_env_steps=8,
        seed=0,
        ram_budget_gib=4.0,
        settle_steps=3,
        per_lane_settle_steps=2,
    )

    runner(env, agent(), loop_config=cfg, progress=progress)

    note_kinds = [kind for _step, kind, _fields in progress.notes]
    assert "initial_settle_start" in note_kinds
    assert "initial_settle" in note_kinds
    assert "initial_settle_done" in note_kinds
    assert "warmup_start" in note_kinds
    assert "warmup" in note_kinds
    assert "warmup_done" in note_kinds
    assert "per_lane_settle" in note_kinds
    assert any(
        fields.get("current_steps") == 3 and fields.get("total_steps") == 3
        for _step, kind, fields in progress.notes
        if kind == "initial_settle"
    )
    assert any(
        fields.get("settling_train_lanes") == 2
        for _step, kind, fields in progress.notes
        if kind == "per_lane_settle"
    )
    assert any(
        fields.get("current_steps", 0) > 0 and fields.get("total_steps") == 2
        for _step, kind, fields in progress.notes
        if kind == "warmup"
    )
    assert any("train/warmup_remaining" in metrics for _step, metrics, _force in progress.calls)


def test_sac_per_lane_settle_masks_replay_and_uses_zero_actions_after_done():
    torch.manual_seed(0)
    env = _ActionRecordingFakeEnv(num_envs=2, seed=0, terminal_step=3)
    agent = SACAgent(_tiny_sac_config())
    cfg = SACTrainLoopConfig(
        replay_capacity=64,
        warmup_steps=0,
        batch_size=4,
        total_env_steps=8,
        seed=0,
        ram_budget_gib=4.0,
        per_lane_settle_steps=2,
    )

    report = run_sac_train_loop(env, agent, loop_config=cfg)

    assert report.num_env_steps == 8
    assert report.final_logs["train/replay_size"] == pytest.approx(8)
    assert report.final_logs["normalizer/proprio_count"] == pytest.approx(8)
    assert agent.normalizers.proprio.count == 8
    assert len(env.step_actions) == 6
    for action in env.step_actions[3:5]:
        assert torch.allclose(action, torch.zeros((2, ACTION_DIM)))
    assert any(logs.get("train_rollout/mean_episode_length") == pytest.approx(3.0) for logs in report.log_history)
    assert any(logs.get("train_rollout/mean_episode_length") == pytest.approx(2.0) for logs in report.log_history)


def test_sac_per_lane_settle_does_not_restart_when_cooldown_episode_times_out():
    torch.manual_seed(0)
    env = _ActionRecordingFakeEnv(num_envs=1, seed=0, terminal_step=2)
    agent = SACAgent(_tiny_sac_config())
    progress = _RecordingProgress()
    cfg = SACTrainLoopConfig(
        replay_capacity=64,
        warmup_steps=0,
        batch_size=16,
        total_env_steps=3,
        seed=0,
        ram_budget_gib=4.0,
        per_lane_settle_steps=5,
    )

    report = run_sac_train_loop(env, agent, loop_config=cfg, progress=progress)

    assert report.num_env_steps == 3
    settle_remaining = [
        fields["max_settle_remaining"]
        for _step, kind, fields in progress.notes
        if kind == "per_lane_settle"
    ]
    assert settle_remaining == sorted(settle_remaining, reverse=True)
    assert settle_remaining[:2] == [5, 4]


def test_sac_same_env_eval_start_waits_for_per_lane_settle_completion():
    torch.manual_seed(0)
    env = _FakeSACEnv(num_envs=4, seed=0, terminal_step=3)
    agent = SACAgent(_tiny_sac_config())
    logger = _CountingLogger()
    cfg = SACTrainLoopConfig(
        replay_capacity=64,
        warmup_steps=6,
        batch_size=4,
        total_env_steps=12,
        seed=0,
        ram_budget_gib=4.0,
        same_env_eval_lanes=1,
        same_env_eval_start_env_steps=9,
        rollout_metrics_window=10,
        per_lane_settle_steps=2,
    )

    run_sac_train_loop(env, agent, loop_config=cfg, logger=logger)

    eval_lane_calls = [
        (step, metrics)
        for step, metrics in logger.scalar_calls
        if "eval_rollout/mean_return" in metrics
    ]
    assert len(eval_lane_calls) == 1
    step, metrics = eval_lane_calls[0]
    assert step == 12
    assert metrics["eval_rollout/episode_count"] == pytest.approx(1.0)
    assert metrics["eval_rollout/mean_episode_length"] == pytest.approx(1.0)


def test_td3_train_loop_logs_training_rollouts_and_same_env_eval_lanes():
    torch.manual_seed(0)
    env = _FakeSACEnv(num_envs=4, seed=1, terminal_step=3)
    agent = TD3Agent(_tiny_td3_config())
    logger = _CountingLogger()
    cfg = TD3TrainLoopConfig(
        replay_capacity=64,
        warmup_steps=6,
        batch_size=4,
        total_env_steps=18,
        seed=1,
        ram_budget_gib=4.0,
        same_env_eval_lanes=1,
        rollout_metrics_window=10,
    )

    report = run_td3_train_loop(env, agent, loop_config=cfg, logger=logger)

    assert report.num_env_steps == 18
    assert report.final_logs["train/replay_size"] == pytest.approx(18)
    assert report.final_logs["normalizer/proprio_count"] == pytest.approx(18)
    assert agent.normalizers.proprio.count == 18
    assert any("train_rollout/mean_return" in logs for logs in report.log_history)
    assert any("eval_rollout/mean_return" in logs for logs in report.log_history)
    assert any("train_rollout/success_rate" in metrics for _step, metrics in logger.scalar_calls)
    assert any("eval_rollout/success_rate" in metrics for _step, metrics in logger.scalar_calls)


def test_td3_actor_scheduler_steps_only_on_delayed_actor_updates():
    torch.manual_seed(0)
    env = _build_fake_env(num_envs=4, seed=0)
    agent = TD3Agent(_tiny_td3_config(policy_delay=2))
    schedulers = {
        "actor": make_scheduler("step", agent.actor_optimizer, step_size=1, gamma=0.5),
        "critic": make_scheduler("step", agent.critic_optimizer, step_size=1, gamma=0.5),
    }
    cfg = TD3TrainLoopConfig(
        replay_capacity=64,
        warmup_steps=8,
        batch_size=4,
        total_env_steps=24,
        seed=0,
        ram_budget_gib=4.0,
    )

    report = run_td3_train_loop(env, agent, loop_config=cfg, schedulers=schedulers)

    assert report.num_updates == 4
    assert agent._actor_update_count == 2
    assert schedulers["critic"].step_count == 4
    assert schedulers["actor"].step_count == 2
    assert report.final_logs["train/learning_rate_critic"] == pytest.approx(3e-4 * 0.5**4)
    assert report.final_logs["train/learning_rate_actor"] == pytest.approx(3e-4 * 0.5**2)


def test_sac_train_loop_denormalizes_learner_actions_before_env_step():
    torch.manual_seed(0)
    env = _ActionRecordingFakeEnv(num_envs=1, seed=0, terminal_step=100)
    agent = SACAgent(_tiny_sac_config())
    agent.normalizers.action = ActionNormalizer(
        action_dim=ACTION_DIM,
        env_low=np.full((ACTION_DIM,), -0.25, dtype=np.float32),
        env_high=np.full((ACTION_DIM,), 0.25, dtype=np.float32),
    )
    cfg = SACTrainLoopConfig(
        replay_capacity=16,
        warmup_steps=10,
        batch_size=4,
        total_env_steps=4,
        seed=0,
        ram_budget_gib=4.0,
    )

    run_sac_train_loop(env, agent, loop_config=cfg)

    assert env.step_actions
    for action in env.step_actions:
        assert float(torch.max(torch.abs(action))) <= 0.25 + 1e-6


def test_td3_train_loop_denormalizes_learner_actions_before_env_step():
    torch.manual_seed(0)
    env = _ActionRecordingFakeEnv(num_envs=1, seed=0, terminal_step=100)
    agent = TD3Agent(_tiny_td3_config())
    agent.normalizers.action = ActionNormalizer(
        action_dim=ACTION_DIM,
        env_low=np.full((ACTION_DIM,), -0.5, dtype=np.float32),
        env_high=np.full((ACTION_DIM,), 0.5, dtype=np.float32),
    )
    cfg = TD3TrainLoopConfig(
        replay_capacity=16,
        warmup_steps=10,
        batch_size=4,
        total_env_steps=4,
        seed=0,
        ram_budget_gib=4.0,
    )

    run_td3_train_loop(env, agent, loop_config=cfg)

    assert env.step_actions
    for action in env.step_actions:
        assert float(torch.max(torch.abs(action))) <= 0.5 + 1e-6
