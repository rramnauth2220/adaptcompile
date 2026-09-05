"""Public API for adaptcompile."""

from importlib.metadata import version as _distribution_version

from .dataset import AdaptationDataset
from .episode import LearningEpisode
from .errors import AdaptCompileError, SerializationError, ValidationError
from .family import ProgramFamily
from .geometry import AdaptationGeometry
from .model import ModelContext
from .program import ProgramSpec
from .result import AdaptationResult
from .study import AdaptationStudy

__version__ = _distribution_version("adaptcompile")

__all__ = [
    "AdaptCompileError",
    "AdaptationDataset",
    "AdaptationGeometry",
    "AdaptationResult",
    "AdaptationStudy",
    "LearningEpisode",
    "ModelContext",
    "ProgramFamily",
    "ProgramSpec",
    "SerializationError",
    "ValidationError",
    "__version__",
]
