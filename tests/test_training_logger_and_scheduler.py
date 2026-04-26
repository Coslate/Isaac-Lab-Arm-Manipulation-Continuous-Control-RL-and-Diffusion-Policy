"""Tests for PR6.5 training logger, LR scheduler, and periodic eval hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from agents.checkpointing import load_checkpoint
from agents.sac import SACAgent, SACConfig
from agents.td3 import TD3Agent, TD3Config
from configs import ACTION_DIM
from scripts.train_sac_continuous import _build_fake_env
from train.loggers import CompositeLogger, JSONLinesLogger, TensorBoardLogger, TrainLogger, WandbLogger
from train.lr_scheduler import (
    estimate_total_update_steps,
    load_scheduler_collection_state,
    make_scheduler,
)
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
    )
    logger.close()

    assert report.num_updates == total_updates
    assert eval_factory_calls["count"] == 3
    assert len(report.eval_history) == 3
    assert "train/learning_rate_actor" in report.final_logs
    assert "eval/mean_return" in report.eval_history[-1]
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
