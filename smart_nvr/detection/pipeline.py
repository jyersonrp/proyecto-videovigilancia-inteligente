"""Hybrid Two-Phase Detection Pipeline.

Coordinates Phase 1 (MOG2 motion detection on downscaled frames) and Phase 2
(Lightweight deep learning object detection with rate limiting).
- In idle state (<10% CPU, <2% detection load): MOG2 rapidly returns without invoking AI.
- When motion is confirmed in the monitored ROI: triggers AI inference rate-limited to 4-6 FPS.
- Generates clean annotated frames with bounding boxes and confidence labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from smart_nvr.config import settings
from smart_nvr.detection.inference import BaseDetector, DetectionBox, create_detector
from smart_nvr.detection.mog2 import MOG2MotionDetector
from smart_nvr.detection.roi import ROIFilter

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Represents the output of the two-phase hybrid detection pipeline for a frame.

    Attributes:
        camera_id: Identifier of the camera stream.
        timestamp: Epoch timestamp when frame was processed.
        motion_detected: True if Phase 1 MOG2 detected motion within ROI.
        ai_triggered: True if Phase 2 DL inference ran on this frame (subject to rate-limit).
        confirmed_detections: List of confirmed person/vehicle DetectionBox objects.
        annotated_frame: Optional BGR image with drawn bounding boxes and labels.
    """

    camera_id: str
    timestamp: float
    motion_detected: bool
    ai_triggered: bool
    confirmed_detections: List[DetectionBox] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None

    @property
    def has_detections(self) -> bool:
        """Return True if any confirmed person or vehicle was detected."""
        return len(self.confirmed_detections) > 0

    @property
    def primary_class(self) -> Optional[str]:
        """Return class name of highest confidence detection, if any."""
        if not self.confirmed_detections:
            return None
        return max(self.confirmed_detections, key=lambda d: d.confidence).class_name

    @property
    def max_confidence(self) -> float:
        """Return highest detection confidence score, or 0.0."""
        if not self.confirmed_detections:
            return 0.0
        return max(d.confidence for d in self.confirmed_detections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to serializable dict."""
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "motion_detected": self.motion_detected,
            "ai_triggered": self.ai_triggered,
            "has_detections": self.has_detections,
            "primary_class": self.primary_class,
            "max_confidence": round(self.max_confidence, 4),
            "detections": [d.to_dict() for d in self.confirmed_detections],
        }


class HybridDetectionPipeline:
    """Two-phase detection pipeline orchestrator.

    Phase 1: MOG2 background subtractor downscales the frame to 320x180, eliminates shadows,
    and isolates valid motion contours inside the camera's ROI. If no motion is detected,
    execution returns immediately to preserve CPU.

    Phase 2: When motion occurs within ROI, Phase 2 AI classification is triggered and
    rate-limited to 4-6 FPS (default 5 FPS) to prevent compute saturation.
    """

    def __init__(
        self,
        camera_id: str,
        motion_detector: Optional[MOG2MotionDetector] = None,
        roi_filter: Optional[ROIFilter] = None,
        ai_detector: Optional[BaseDetector] = None,
        ai_fps: float = 5.0,
        annotate: bool = True,
        draw_roi: bool = False,
        draw_motion_boxes: bool = False,
    ) -> None:
        self.camera_id = str(camera_id)
        self.motion_detector = motion_detector or MOG2MotionDetector(
            history=settings.MOG2_HISTORY,
            var_threshold=settings.MOG2_VAR_THRESHOLD,
            detect_shadows=settings.MOG2_DETECT_SHADOWS,
            shadow_threshold=settings.MOG2_SHADOW_THRESHOLD,
            downscale_width=settings.MOG2_DOWNSCALE_WIDTH,
            downscale_height=settings.MOG2_DOWNSCALE_HEIGHT,
            min_contour_area=settings.MOG2_MIN_CONTOUR_AREA // 4 if settings.MOG2_MIN_CONTOUR_AREA > 200 else 100,
        )
        self.roi_filter = roi_filter or ROIFilter()
        self.ai_detector = ai_detector or create_detector(
            model_path=settings.YOLO_MODEL_PATH,
            preferred_tier=settings.AI_ENGINE_TIER,
            confidence_threshold=settings.AI_CONFIDENCE_THRESHOLD,
            target_classes=settings.AI_TARGET_CLASSES,
        )
        self.ai_fps = max(0.5, float(ai_fps or settings.AI_INFERENCE_FPS))
        self.annotate = bool(annotate)
        self.draw_roi = bool(draw_roi)
        self.draw_motion_boxes = bool(draw_motion_boxes)

        # Rate-limiting state
        self._last_ai_time: float = 0.0
        self._min_ai_interval: float = 1.0 / self.ai_fps
        self._last_detections: List[DetectionBox] = []
        self._last_motion_boxes: List[Tuple[int, int, int, int]] = []

        # Operational telemetry
        self._total_frames: int = 0
        self._motion_frames: int = 0
        self._ai_inferences: int = 0

    def update_roi(
        self,
        polygons: Optional[Union[Sequence[Sequence[Sequence[float]]], Sequence[Sequence[Tuple[float, float]]]]],
    ) -> None:
        """Dynamically update ROI polygons without restarting the pipeline."""
        self.roi_filter.set_polygons(polygons)

    def set_ai_fps(self, fps: float) -> None:
        """Update the rate-limiting FPS target for Phase 2 AI inference."""
        self.ai_fps = max(0.5, float(fps))
        self._min_ai_interval = 1.0 / self.ai_fps

    def reset(self) -> None:
        """Reset internal detector state and background models."""
        self.motion_detector.reset()
        self._last_ai_time = 0.0
        self._last_detections = []
        self._last_motion_boxes = []

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> DetectionResult:
        """Execute the two-phase detection pipeline on an incoming video frame.

        Args:
            frame: Native BGR frame array (H, W, 3).
            timestamp: Optional frame timestamp (defaults to current time.time()).

        Returns:
            DetectionResult containing motion flags, AI trigger status, detections,
            and optional annotated frame.
        """
        now = timestamp if timestamp is not None else time.time()
        self._total_frames += 1

        if frame is None or frame.size == 0:
            return DetectionResult(
                camera_id=self.camera_id,
                timestamp=now,
                motion_detected=False,
                ai_triggered=False,
                confirmed_detections=[],
                annotated_frame=None,
            )

        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # Phase 1: MOG2 Low-CPU Motion Detection
        # -------------------------------------------------------------
        # Get downscaled ROI mask for MOG2
        down_shape = (self.motion_detector.downscale_height, self.motion_detector.downscale_width)
        roi_mask_mog2 = self.roi_filter.get_mask(down_shape) if not self.roi_filter.is_empty else None

        motion_detected, motion_boxes = self.motion_detector.detect(frame, roi_mask=roi_mask_mog2)
        self._last_motion_boxes = motion_boxes

        # If motion is outside ROI or negligible, return immediately (<2% CPU)
        if not motion_detected or len(motion_boxes) == 0:
            self._last_detections = []
            annotated = frame.copy() if self.annotate else None
            return DetectionResult(
                camera_id=self.camera_id,
                timestamp=now,
                motion_detected=False,
                ai_triggered=False,
                confirmed_detections=[],
                annotated_frame=annotated,
            )

        # -------------------------------------------------------------
        # Phase 2: Lightweight DL Inference with Rate-Limiting (4-6 FPS)
        # -------------------------------------------------------------
        self._motion_frames += 1
        time_since_ai = now - self._last_ai_time
        can_run_ai = (time_since_ai >= self._min_ai_interval) or (self._last_ai_time == 0.0)

        if can_run_ai:
            self._last_ai_time = now
            self._ai_inferences += 1
            ai_triggered = True

            # Execute AI detector
            raw_detections = self.ai_detector.detect(frame)

            # Filter through camera ROI
            confirmed = self.roi_filter.filter_detections(raw_detections, (h, w))
            self._last_detections = confirmed
        else:
            # Rate-limited frame: do not burn CPU running inference,
            # persist previous detections to prevent UI and recorder flicker
            ai_triggered = False
            confirmed = self._last_detections

        # -------------------------------------------------------------
        # Snapshot Annotation
        # -------------------------------------------------------------
        annotated_frame = None
        if self.annotate:
            annotated_frame = self._annotate_frame(frame, confirmed, motion_boxes)

        return DetectionResult(
            camera_id=self.camera_id,
            timestamp=now,
            motion_detected=True,
            ai_triggered=ai_triggered,
            confirmed_detections=confirmed,
            annotated_frame=annotated_frame,
        )

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[DetectionBox],
        motion_boxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Render clean bounding boxes, labels, and optional ROI boundaries."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. Draw ROI polygon boundary if requested
        if self.draw_roi and not self.roi_filter.is_empty:
            for poly in self.roi_filter.polygons:
                pts = np.array(
                    [[int(p[0] * (w - 1)), int(p[1] * (h - 1))] for p in poly],
                    dtype=np.int32,
                )
                cv2.polylines(annotated, [pts], isClosed=True, color=(255, 180, 0), thickness=2)

        # 2. Draw raw motion boxes in light yellow if requested
        if self.draw_motion_boxes:
            for mx, my, mw, mh in motion_boxes:
                cv2.rectangle(annotated, (mx, my), (mx + mw, my + mh), (120, 240, 255), 1)

        # 3. Draw confirmed AI detection bounding boxes and styled tags
        for det in detections:
            bx, by, bw, bh = det.bbox
            cls_name = det.class_name.lower()

            # Color styling: Green for person, Orange/Blue for vehicle, Red for other
            if cls_name == "person":
                box_color = (0, 210, 30)  # Bright Green (BGR)
            elif cls_name in ("car", "motorcycle", "bus", "truck", "vehicle"):
                box_color = (255, 140, 0)  # Blue/Cyan (BGR)
            else:
                box_color = (0, 60, 240)  # Red (BGR)

            # Draw bounding box
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_color, 2)

            # Draw label banner with dark background
            label = f"{det.class_name.upper()} {det.confidence * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            tag_y1 = max(0, by - th - 8)
            tag_y2 = by
            tag_x1 = bx
            tag_x2 = min(w, bx + tw + 8)

            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (tag_x1 + 4, tag_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated

    @property
    def telemetry(self) -> Dict[str, Any]:
        """Return operational pipeline statistics."""
        return {
            "total_frames": self._total_frames,
            "motion_frames": self._motion_frames,
            "ai_inferences": self._ai_inferences,
            "motion_ratio": round(self._motion_frames / max(1, self._total_frames), 4),
            "ai_inference_ratio": round(self._ai_inferences / max(1, self._total_frames), 4),
        }
