"""Checkpoint metadata and save/load helpers for continuous-control agents.

The metadata schema is the contract between PR6/PR7 (training) and
PR11a/PR12a (eval/visualization). See plans §8.3 for the JSON layout this
mirrors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from configs import ACTION_DIM, ISAAC_FRANKA_IK_REL_ENV_ID


SUPPORTED_AGENT_TYPES = ("sac", "td3")
DETERMINISTIC_MODE_SAC = "tanh_mu"
DETERMINISTIC_MODE_TD3 = "actor_no_noise"
DEFAULT_IMAGE_SHAPE = (3, 224, 224)
DEFAULT_PROPRIO_DIM = 40
REPLAY_STORAGE_CPU_UINT8 = "cpu_uint8_images"


@dataclass
class CheckpointMetadata:
    """Metadata embedded in every PR6+ checkpoint. Mirrors plans §8.3."""

    agent_type: str
    env_id: str = ISAAC_FRANKA_IK_REL_ENV_ID
    action_dim: int = ACTION_DIM
    proprio_dim: int = DEFAULT_PROPRIO_DIM
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE
    num_env_steps: int | None = 0
    global_update_step: int = 0
    seed: int = 0
    deterministic_action_mode: str = DETERMINISTIC_MODE_SAC
    backbone_config: dict[str, Any] = field(default_factory=dict)
    algorithm_hparams: dict[str, Any] = field(default_factory=dict)
    replay_storage: str = REPLAY_STORAGE_CPU_UINT8
    legacy_warning: str | None = None

    def validate(self) -> None:
        if self.agent_type not in SUPPORTED_AGENT_TYPES:
            raise ValueError(
                f"agent_type must be one of {SUPPORTED_AGENT_TYPES}; got {self.agent_type!r}"
            )
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.proprio_dim <= 0:
            raise ValueError("proprio_dim must be positive")
        if tuple(self.image_shape) != DEFAULT_IMAGE_SHAPE:
            raise ValueError(
                f"image_shape must be {DEFAULT_IMAGE_SHAPE}; got {tuple(self.image_shape)}"
            )
        if self.global_update_step < 0:
            raise ValueError("global_update_step must be non-negative")
        if self.num_env_steps is not None and self.num_env_steps < 0:
            raise ValueError("num_env_steps must be None or non-negative")
        expected_mode = (
            DETERMINISTIC_MODE_SAC if self.agent_type == "sac" else DETERMINISTIC_MODE_TD3
        )
        if self.deterministic_action_mode != expected_mode:
            raise ValueError(
                f"deterministic_action_mode for agent_type={self.agent_type!r} "
                f"must be {expected_mode!r}; got {self.deterministic_action_mode!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_shape"] = list(self.image_shape)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        normalized = dict(data)
        if "image_shape" in normalized:
            normalized["image_shape"] = tuple(normalized["image_shape"])
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in normalized.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class CheckpointPayload:
    """Container for everything written/read in a single .pt file."""

    metadata: CheckpointMetadata
    model_state: dict[str, dict[str, torch.Tensor]]
    target_state: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(path: str | Path, payload: CheckpointPayload) -> Path:
    """Write ``payload`` to ``path`` as a torch ``.pt`` file. Returns the path."""

    payload.metadata.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": payload.metadata.to_dict(),
            "model_state": payload.model_state,
            "target_state": payload.target_state,
            "optimizer_state": payload.optimizer_state,
            "extras": payload.extras,
        },
        output,
    )
    return output


def load_checkpoint(
    path: str | Path,
    *,
    expected_agent_type: str | None = None,
    expected_action_dim: int | None = None,
    expected_proprio_dim: int | None = None,
    expected_env_id: str | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointPayload:
    """Load a checkpoint and validate its metadata against caller expectations."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    raw = torch.load(source, map_location=map_location, weights_only=False)
    if "metadata" not in raw:
        raise ValueError(f"checkpoint at {source} is missing 'metadata' field")
    metadata = CheckpointMetadata.from_dict(raw["metadata"])
    metadata.validate()

    if expected_agent_type is not None and metadata.agent_type != expected_agent_type:
        raise ValueError(
            f"checkpoint agent_type {metadata.agent_type!r} != expected {expected_agent_type!r}"
        )
    if expected_action_dim is not None and metadata.action_dim != expected_action_dim:
        raise ValueError(
            f"checkpoint action_dim {metadata.action_dim} != expected {expected_action_dim}"
        )
    if expected_proprio_dim is not None and metadata.proprio_dim != expected_proprio_dim:
        raise ValueError(
            f"checkpoint proprio_dim {metadata.proprio_dim} != expected {expected_proprio_dim}"
        )
    if expected_env_id is not None and metadata.env_id != expected_env_id:
        raise ValueError(
            f"checkpoint env_id {metadata.env_id!r} != expected {expected_env_id!r}"
        )

    return CheckpointPayload(
        metadata=metadata,
        model_state=raw.get("model_state", {}),
        target_state=raw.get("target_state", {}),
        optimizer_state=raw.get("optimizer_state", {}),
        extras=raw.get("extras", {}),
    )


__all__ = [
    "CheckpointMetadata",
    "CheckpointPayload",
    "DEFAULT_IMAGE_SHAPE",
    "DEFAULT_PROPRIO_DIM",
    "DETERMINISTIC_MODE_SAC",
    "DETERMINISTIC_MODE_TD3",
    "REPLAY_STORAGE_CPU_UINT8",
    "SUPPORTED_AGENT_TYPES",
    "load_checkpoint",
    "save_checkpoint",
]
