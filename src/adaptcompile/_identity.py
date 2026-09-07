"""Internal validation for public model and episode identity keys."""

from __future__ import annotations

from typing import Any, cast

from ._validation import nonempty_name
from .dataset import ModelIdentityKey
from .episode import EpisodeIdentityKey
from .errors import ValidationError


def _validate_model_key(value: Any) -> ModelIdentityKey:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValidationError("model_key must be a ModelContext identity tuple")
    model_id, revision, base_state_id = value
    nonempty_name(model_id, field="model_key model_id")
    for name, item in (("revision", revision), ("base_state_id", base_state_id)):
        if item is not None:
            nonempty_name(item, field=f"model_key {name}")
    return cast(ModelIdentityKey, value)


def _validate_episode_key(value: Any) -> EpisodeIdentityKey:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValidationError("episode_key must be a LearningEpisode identity tuple")
    kind, identity = value
    if kind not in {"episode_id", "configuration"}:
        raise ValidationError("episode_key has an unknown identity kind")
    nonempty_name(identity, field="episode_key identity")
    return cast(EpisodeIdentityKey, value)
