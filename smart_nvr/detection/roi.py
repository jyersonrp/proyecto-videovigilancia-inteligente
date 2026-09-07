"""Region of Interest (ROI) Filtering for Smart NVR.

Supports configurable polygon ROIs defined with normalized coordinates ([0.0, 1.0]).
Generates binary OpenCV masks at any native or scaled resolution, performs
point-in-polygon tests, and evaluates bounding box and contour intersections to reject
motion or detections outside monitored zones.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Point type: (x, y) normalized between 0.0 and 1.0
Point = Tuple[float, float]
Polygon = List[Point]


class ROIFilter:
    """Configurable Region of Interest filter supporting multiple polygon boundaries.

    Coordinates are normalized to [0.0, 1.0] relative to frame width and height.
    If no polygons are configured, the ROI filter allows all frames, points, and detections.
    """

    def __init__(
        self,
        polygons: Optional[Union[Sequence[Sequence[Sequence[float]]], Sequence[Sequence[Tuple[float, float]]]]] = None,
    ) -> None:
        self._polygons: List[Polygon] = []
        self._mask_cache: dict[Tuple[int, int], np.ndarray] = {}
        if polygons:
            self.set_polygons(polygons)

    @property
    def is_empty(self) -> bool:
        """Return True if no ROI polygons are configured (entire frame monitored)."""
        return len(self._polygons) == 0

    @property
    def polygons(self) -> List[Polygon]:
        """Return list of normalized polygon vertices."""
        return [list(p) for p in self._polygons]

    def set_polygons(
        self,
        polygons: Optional[Union[Sequence[Sequence[Sequence[float]]], Sequence[Sequence[Tuple[float, float]]]]],
    ) -> None:
        """Set or update configured ROI polygons and invalidate mask caches.

        Args:
            polygons: Nested lists/tuples of normalized points, e.g.
                [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]]
        """
        self._polygons.clear()
        self._mask_cache.clear()

        if not polygons:
            return

        for poly in polygons:
            parsed_poly: Polygon = []
            for pt in poly:
                if len(pt) >= 2:
                    px = max(0.0, min(1.0, float(pt[0])))
                    py = max(0.0, min(1.0, float(pt[1])))
                    parsed_poly.append((px, py))
            if len(parsed_poly) >= 3:
                self._polygons.append(parsed_poly)

    def get_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate or retrieve a binary uint8 mask for the specified resolution.

        Args:
            shape: (height, width) of the target mask.

        Returns:
            np.ndarray uint8 of shape (H, W) where 255 indicates inside ROI, 0 outside.
        """
        h, w = int(shape[0]), int(shape[1])
        cache_key = (h, w)

        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        if self.is_empty:
            mask = np.full((h, w), 255, dtype=np.uint8)
            self._mask_cache[cache_key] = mask
            return mask

        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in self._polygons:
            pts = np.array(
                [[int(round(px * (w - 1))), int(round(py * (h - 1)))] for px, py in poly],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 255)

        self._mask_cache[cache_key] = mask
        return mask

    def contains_point(
        self,
        point: Tuple[float, float],
        normalized: bool = True,
        shape: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Check if a point lies inside any configured ROI polygon.

        Args:
            point: (x, y) coordinates.
            normalized: If True, point is in [0, 1]. If False, point is in pixels.
            shape: Required if normalized=False, tuple of (height, width).

        Returns:
            True if point is inside any ROI polygon (or if ROI is empty).
        """
        if self.is_empty:
            return True

        if not normalized:
            if shape is None:
                raise ValueError("shape=(H, W) is required when point is not normalized")
            h, w = shape[:2]
            nx = point[0] / float(w)
            ny = point[1] / float(h)
        else:
            nx, ny = point[0], point[1]

        # Check each polygon
        ref_h, ref_w = 1000, 1000
        pt = (float(nx * ref_w), float(ny * ref_h))

        for poly in self._polygons:
            pts = np.array(
                [[p[0] * ref_w, p[1] * ref_h] for p in poly],
                dtype=np.int32,
            )
            if cv2.pointPolygonTest(pts, pt, False) >= 0:
                return True

        return False

    def intersects_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        shape: Tuple[int, int],
        min_overlap_ratio: float = 0.0,
    ) -> bool:
        """Check if a pixel bounding box intersects the ROI mask.

        Args:
            bbox: (x, y, w, h) in pixel coordinates.
            shape: (height, width) of the frame.
            min_overlap_ratio: Optional minimum fraction of bbox area inside ROI.

        Returns:
            True if bbox intersects ROI (or if ROI is empty).
        """
        if self.is_empty:
            return True

        h, w = shape[:2]
        x, y, bw, bh = bbox

        x1 = max(0, min(w, x))
        y1 = max(0, min(h, y))
        x2 = max(0, min(w, x + bw))
        y2 = max(0, min(h, y + bh))

        if x2 <= x1 or y2 <= y1:
            return False

        mask = self.get_mask((h, w))
        roi_crop = mask[y1:y2, x1:x2]

        if min_overlap_ratio > 0.0:
            total_pixels = (x2 - x1) * (y2 - y1)
            active_pixels = int(np.count_nonzero(roi_crop))
            return (active_pixels / float(total_pixels)) >= min_overlap_ratio

        return bool(np.any(roi_crop > 0))

    def intersects_contour(
        self,
        contour: np.ndarray,
        shape: Tuple[int, int],
    ) -> bool:
        """Check if an OpenCV contour intersects the ROI mask.

        Args:
            contour: Contour point array.
            shape: (height, width) of the frame the contour belongs to.

        Returns:
            True if contour intersects ROI (or if ROI is empty).
        """
        if self.is_empty or contour is None or len(contour) == 0:
            return True

        x, y, w, h = cv2.boundingRect(contour)
        return self.intersects_bbox((x, y, w, h), shape)

    def filter_detections(
        self,
        detections: List[Any],
        shape: Tuple[int, int],
        min_overlap_ratio: float = 0.0,
    ) -> List[Any]:
        """Filter a list of detection objects to keep only those intersecting ROI.

        Args:
            detections: List of DetectionBox (or objects with .bbox attribute).
            shape: (height, width) of the frame.
            min_overlap_ratio: Minimum overlap ratio required.

        Returns:
            Filtered list of detections inside or intersecting ROI.
        """
        if self.is_empty:
            return list(detections)

        filtered = []
        for det in detections:
            bbox = getattr(det, "bbox", None)
            if bbox is None and hasattr(det, "normalized_bbox"):
                nb = det.normalized_bbox
                h, w = shape[:2]
                bbox = (int(nb[0] * w), int(nb[1] * h), int(nb[2] * w), int(nb[3] * h))

            if bbox is not None:
                if self.intersects_bbox(bbox, shape, min_overlap_ratio=min_overlap_ratio):
                    filtered.append(det)
            else:
                filtered.append(det)

        return filtered
