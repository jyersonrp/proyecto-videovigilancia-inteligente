"""Phase 1: MOG2 Low-CPU Background Subtraction and Motion Detection.

Implements MOG2MotionDetector with downscaling to 320x180, Gaussian blur pre-filtering,
shadow elimination by thresholding (fg_mask > 200), morphological cleaning,
and contour extraction with area filtering and resolution re-scaling.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np

from smart_nvr.config import settings

logger = logging.getLogger(__name__)


class MOG2MotionDetector:
    """OpenCV MOG2 background subtractor optimized for minimal CPU usage.

    Features:
    - Downscales frames to 320x180 (or configurable size) for low CPU footprint (<10% idle).
    - Gaussian blur pre-filtering to eliminate sensor noise.
    - Shadow elimination by thresholding fg_mask > 200 (strips gray shadow pixels at 127).
    - Morphological opening and dilation to remove isolated noise spots.
    - Contour extraction filtering by contourArea >= min_contour_area.
    - Methods to inspect contours, bounding boxes, and scale coordinates back to original resolution.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        downscale_width: int = 320,
        downscale_height: int = 180,
        shadow_threshold: int = 200,
        min_contour_area: int = 100,
    ) -> None:
        self.history = int(history)
        self.var_threshold = float(var_threshold)
        self.detect_shadows = bool(detect_shadows)
        self.downscale_width = int(downscale_width)
        self.downscale_height = int(downscale_height)
        self.shadow_threshold = int(shadow_threshold)
        self.min_contour_area = int(min_contour_area)

        # Initialize OpenCV BackgroundSubtractorMOG2
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

        # Morphological kernels
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Cached state from last processed frame
        self._last_raw_fg_mask: Optional[np.ndarray] = None
        self._last_processed_mask: Optional[np.ndarray] = None
        self._last_downscaled_contours: List[np.ndarray] = []
        self._last_downscaled_bboxes: List[Tuple[int, int, int, int]] = []
        self._last_native_bboxes: List[Tuple[int, int, int, int]] = []
        self._last_orig_shape: Optional[Tuple[int, int]] = None
        self._motion_detected: bool = False

    def reset(self) -> None:
        """Reset background model state and cached detections."""
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )
        self._last_raw_fg_mask = None
        self._last_processed_mask = None
        self._last_downscaled_contours = []
        self._last_downscaled_bboxes = []
        self._last_native_bboxes = []
        self._last_orig_shape = None
        self._motion_detected = False

    def detect(
        self,
        frame: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
        learning_rate: float = -1.0,
    ) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        """Process an incoming video frame and detect foreground motion.

        Args:
            frame: Full-resolution input BGR image (H, W, 3).
            roi_mask: Optional binary mask (same size as frame or downscaled)
                where 255/1 indicates regions of interest.
            learning_rate: Learning rate for MOG2 (-1 for auto).

        Returns:
            Tuple of (motion_detected: bool, native_bboxes: List[(x, y, w, h)]).
        """
        if frame is None or frame.size == 0:
            self._motion_detected = False
            self._last_native_bboxes = []
            return False, []

        orig_h, orig_w = frame.shape[:2]
        self._last_orig_shape = (orig_h, orig_w)

        # 1. Downscale frame for ultra-low CPU processing
        if (orig_w, orig_h) != (self.downscale_width, self.downscale_height):
            small_frame = cv2.resize(
                frame,
                (self.downscale_width, self.downscale_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small_frame = frame

        # 2. Gaussian blur pre-filtering to eliminate sensor noise
        blurred = cv2.GaussianBlur(small_frame, (5, 5), 0)

        # 3. Apply MOG2 background subtractor
        raw_mask = self.subtractor.apply(blurred, learningRate=learning_rate)
        self._last_raw_fg_mask = raw_mask

        # 4. Shadow elimination: MOG2 marks shadows as 127 when detectShadows=True.
        # Thresholding at > shadow_threshold (default 200) retains only true foreground (255).
        _, thresh = cv2.threshold(
            raw_mask, self.shadow_threshold, 255, cv2.THRESH_BINARY
        )

        # 5. Morphological cleaning: opening removes isolated noise, dilation merges blobs
        opened = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, self._morph_kernel, iterations=1
        )
        dilated = cv2.dilate(opened, self._morph_kernel, iterations=2)

        # 6. Apply ROI mask if specified
        if roi_mask is not None and roi_mask.size > 0:
            if roi_mask.shape[:2] != (self.downscale_height, self.downscale_width):
                roi_mask_scaled = cv2.resize(
                    roi_mask,
                    (self.downscale_width, self.downscale_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                roi_mask_scaled = roi_mask
            dilated = cv2.bitwise_and(dilated, dilated, mask=roi_mask_scaled)

        self._last_processed_mask = dilated

        # 7. Contour extraction
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by minimum area
        valid_contours = [
            c for c in contours if cv2.contourArea(c) >= self.min_contour_area
        ]
        self._last_downscaled_contours = valid_contours

        # 8. Compute bounding boxes and scale back to native resolution
        downscaled_bboxes: List[Tuple[int, int, int, int]] = []
        native_bboxes: List[Tuple[int, int, int, int]] = []

        scale_x = orig_w / float(self.downscale_width)
        scale_y = orig_h / float(self.downscale_height)

        for c in valid_contours:
            x, y, w, h = cv2.boundingRect(c)
            downscaled_bboxes.append((x, y, w, h))

            # Scale to original resolution and clamp
            nx = max(0, min(orig_w - 1, int(x * scale_x)))
            ny = max(0, min(orig_h - 1, int(y * scale_y)))
            nw = max(1, min(orig_w - nx, int(w * scale_x)))
            nh = max(1, min(orig_h - ny, int(h * scale_y)))
            native_bboxes.append((nx, ny, nw, nh))

        self._last_downscaled_bboxes = downscaled_bboxes
        self._last_native_bboxes = native_bboxes
        self._motion_detected = len(valid_contours) > 0

        return self._motion_detected, native_bboxes

    def get_foreground_mask(self) -> Optional[np.ndarray]:
        """Return the latest processed binary foreground mask (320x180)."""
        return self._last_processed_mask

    def get_raw_fg_mask(self) -> Optional[np.ndarray]:
        """Return the latest raw MOG2 foreground mask before thresholding."""
        return self._last_raw_fg_mask

    def get_motion_contours(self) -> List[np.ndarray]:
        """Return valid contours in downscaled coordinates."""
        return self._last_downscaled_contours

    def get_scaled_contours(
        self, orig_shape: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """Return motion contours scaled to native frame resolution (H, W)."""
        shape = orig_shape or self._last_orig_shape
        if shape is None or not self._last_downscaled_contours:
            return []

        orig_h, orig_w = shape[:2]
        scale_x = orig_w / float(self.downscale_width)
        scale_y = orig_h / float(self.downscale_height)

        scaled_contours = []
        for c in self._last_downscaled_contours:
            scaled_c = c.astype(np.float32).copy()
            scaled_c[:, :, 0] *= scale_x
            scaled_c[:, :, 1] *= scale_y
            scaled_contours.append(scaled_c.astype(np.int32))

        return scaled_contours

    def get_downscaled_bboxes(self) -> List[Tuple[int, int, int, int]]:
        """Return bounding boxes in downscaled coordinates."""
        return list(self._last_downscaled_bboxes)

    def get_native_bboxes(self) -> List[Tuple[int, int, int, int]]:
        """Return bounding boxes in original frame coordinates."""
        return list(self._last_native_bboxes)

    @property
    def has_motion(self) -> bool:
        """Return whether motion was detected on the latest frame."""
        return self._motion_detected
