"""
inference package
=================
Modules for document burn detection, physical budget estimation, and multimodal restoration.
"""

from inference.burn_detector import BurnDetector, BurnDetectionResult, SeveredLineRegion
from inference.budget_estimator import BudgetEstimator, BudgetComplianceReport
from inference.restoration_engine import RestorationEngine

__all__ = [
    "BurnDetector",
    "BurnDetectionResult",
    "SeveredLineRegion",
    "BudgetEstimator",
    "BudgetComplianceReport",
    "RestorationEngine",
]

