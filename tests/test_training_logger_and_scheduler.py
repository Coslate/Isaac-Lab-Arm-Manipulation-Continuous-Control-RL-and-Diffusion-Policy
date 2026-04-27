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
from scripts.train_sac_continuous import _FakeSACEnv, _build_fake_env
from train.loggers import CompositeLogger, JSONLinesLogger, TensorBoardLogger, TrainLogger, WandbLogger
from train.lr_scheduler import (
    estimate_total_update_steps,
    load_scheduler_collection_state,
    make_scheduler,
)
from train.progress import TrainProgressReporter
from train.sac_loop import SACTrainLoopConfig, run_sac_train_loop
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
