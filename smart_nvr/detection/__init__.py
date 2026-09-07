"""Detection package for Smart NVR.

Provides two-phase hybrid motion and object detection:
- Phase 1: MOG2MotionDetector (downscaled background subtraction, shadow elimination)
- ROI Filtering: ROIFilter (polygon boundaries, intersection and point containment)
- Phase 2: Multi-tier AI detectors (ONNXRuntimeDetector, OpenCVDNNDetector, MockDetector)
- Pipeline: HybridDetectionPipeline orchestrating MOG2 and rate-limited AI inference
"""

from smart_nvr.detection.inference import (
    COCO_CLASSES,
    BaseDetector,
    DetectionBox,
    MockDetector,
    ONNXRuntimeDetector,
    OpenCVDNNDetector,
    create_detector,
)
from smart_nvr.detection.mog2 import MOG2MotionDetector
from smart_nvr.detection.pipeline import DetectionResult, HybridDetectionPipeline
from smart_nvr.detection.roi import ROIFilter

__all__ = [
    "BaseDetector",
    "COCO_CLASSES",
    "DetectionBox",
    "DetectionResult",
    "HybridDetectionPipeline",
    "MOG2MotionDetector",
    "MockDetector",
    "ONNXRuntimeDetector",
    "OpenCVDNNDetector",
    "ROIFilter",
    "create_detector",
]
