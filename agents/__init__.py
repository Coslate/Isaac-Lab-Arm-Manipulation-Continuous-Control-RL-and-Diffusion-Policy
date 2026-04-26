"""Agent primitives shared across continuous-control RL methods."""

from .backbone import (
    DEFAULT_FUSED_FEATURE_DIM,
    DEFAULT_IMAGE_FEATURE_DIM,
    DEFAULT_PROPRIO_DIM,
    DEFAULT_PROPRIO_FEATURE_DIM,
    ImageProprioBackbone,
    ImageProprioBackboneConfig,
    POLICY_IMAGE_SHAPE,
)
from .checkpointing import (
    CheckpointMetadata,
    CheckpointPayload,
    DEFAULT_IMAGE_SHAPE,
    DETERMINISTIC_MODE_SAC,
    DETERMINISTIC_MODE_TD3,
    REPLAY_STORAGE_CPU_UINT8,
    SUPPORTED_AGENT_TYPES,
    load_checkpoint,
    save_checkpoint,
)
from .distributions import SquashedGaussian
from .fake_checkpoints import (
    FakeSACActor,
    FakeTD3Actor,
    build_fake_actor,
    make_fake_sac_checkpoint,
    make_fake_td3_checkpoint,
)
from .heads import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_OBS_FEAT_DIM,
    DeterministicActorHead,
    GaussianActorHead,
    HeadConfig,
    QHead,
)
from .replay_buffer import (
    DEFAULT_ACTION_DIM,
    DEFAULT_RAM_BUDGET_GIB,
    ReplayBatch,
    ReplayBuffer,
    ReplayMemoryEstimate,
    estimate_replay_memory,
    make_dummy_transition,
)
from .sac import SAC_AGENT_TYPE, SACAgent, SACConfig
from .torch_image_aug import PadAndRandomCropTorch

__all__ = [
    "CheckpointMetadata",
    "CheckpointPayload",
    "DEFAULT_ACTION_DIM",
    "DEFAULT_FUSED_FEATURE_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_IMAGE_FEATURE_DIM",
    "DEFAULT_IMAGE_SHAPE",
    "DEFAULT_OBS_FEAT_DIM",
    "DEFAULT_PROPRIO_DIM",
    "DEFAULT_PROPRIO_FEATURE_DIM",
    "DEFAULT_RAM_BUDGET_GIB",
    "DETERMINISTIC_MODE_SAC",
    "DETERMINISTIC_MODE_TD3",
    "DeterministicActorHead",
    "FakeSACActor",
    "FakeTD3Actor",
    "GaussianActorHead",
    "HeadConfig",
    "ImageProprioBackbone",
    "ImageProprioBackboneConfig",
    "POLICY_IMAGE_SHAPE",
    "PadAndRandomCropTorch",
    "QHead",
    "REPLAY_STORAGE_CPU_UINT8",
    "SACAgent",
    "SACConfig",
    "SAC_AGENT_TYPE",
    "ReplayBatch",
    "ReplayBuffer",
    "ReplayMemoryEstimate",
    "SUPPORTED_AGENT_TYPES",
    "SquashedGaussian",
    "build_fake_actor",
    "estimate_replay_memory",
    "load_checkpoint",
    "make_dummy_transition",
    "make_fake_sac_checkpoint",
    "make_fake_td3_checkpoint",
    "save_checkpoint",
]
