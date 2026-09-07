"""Compiler-ready supervised data assembly."""

from .data import CompilerDataset, CompilerRecord, build_compiler_dataset
from .prediction import GeometryPrediction, GeometryPredictor
from .selection import (
    CandidateScore,
    GeometryConstraint,
    ProgramSelection,
    ProgramSelector,
    SelectionObjective,
)

__all__ = [
    "CandidateScore",
    "CompilerDataset",
    "CompilerRecord",
    "GeometryConstraint",
    "GeometryPrediction",
    "GeometryPredictor",
    "ProgramSelection",
    "ProgramSelector",
    "SelectionObjective",
    "build_compiler_dataset",
]
