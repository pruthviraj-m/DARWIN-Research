"""DARWIN Handwriting Research & Analysis Software Suite.

Captures, validates, extracts, and analyzes online handwriting dynamics for cognitive research.
DISCLAIMER: Academic research prototype. NOT a certified medical diagnostic device.
"""

__version__ = "2.0.0"
__protocol_version__ = "DARWIN_Protocol_2.0"

from .feature_extractor import HandwritingFeatureExtractor, extract_features_from_csv
from .quality_validator import DataQualityValidator, QualityStatus
from .dataset_generator import ResearchDatasetGenerator
from .model_runner import DarwinModelRunner
from .report_generator import ResearchReportGenerator

__all__ = [
    "HandwritingFeatureExtractor",
    "extract_features_from_csv",
    "DataQualityValidator",
    "QualityStatus",
    "ResearchDatasetGenerator",
    "DarwinModelRunner",
    "ResearchReportGenerator",
]
