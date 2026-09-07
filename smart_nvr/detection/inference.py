"""Phase 2: Multi-Tier AI Inference Engine.

Provides person and vehicle classification across three detector tiers:
- Tier 1: ONNXRuntimeDetector (ONNX Runtime CPU with AVX2 acceleration)
- Tier 2: OpenCVDNNDetector (cv2.dnn.readNetFromONNX fallback)
- Tier 3: MockDetector (deterministic feature/sprite detection for hermetic CI/CD testing)

Filters detections by target classes and confidence thresholds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import cv2
import numpy as np

from smart_nvr.config import settings

logger = logging.getLogger(__name__)

# Standard 80 COCO dataset classes used by YOLO models
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class DetectionBox:
    """Represents a single confirmed detection with pixel and normalized coordinates.

    Attributes:
        class_name: Target classification, e.g. 'person', 'car', 'truck'.
        confidence: Prediction confidence in [0.0, 1.0].
        bbox: (x, y, w, h) in pixel coordinates relative to the input frame.
        normalized_bbox: (x, y, w, h) in [0.0, 1.0] relative to frame dimensions.
    """

    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    normalized_bbox: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to JSON-serializable dictionary."""
        return {
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox),
            "normalized_bbox": [round(float(v), 4) for v in self.normalized_bbox],
        }


class BaseDetector(ABC):
    """Abstract base class for object detection engines."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        target_classes: Optional[Sequence[str]] = None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        classes = target_classes or ["person", "car", "motorcycle", "bus", "truck"]
        self.target_classes: Set[str] = {c.lower().strip() for c in classes}

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionBox]:
        """Run object detection on the given BGR image frame.

        Args:
            frame: BGR image array (H, W, 3).

        Returns:
            List of confirmed DetectionBox instances.
        """
        pass

    def filter_predictions(
        self,
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        class_ids: List[int],
        frame_shape: Tuple[int, int],
        nms_threshold: float = 0.45,
    ) -> List[DetectionBox]:
        """Apply Non-Maximum Suppression (NMS) and target class filtering."""
        if not boxes:
            return []

        h, w = frame_shape[:2]
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=self.confidence_threshold,
            nms_threshold=nms_threshold,
        )

        detections: List[DetectionBox] = []
        if len(indices) == 0:
            return detections

        # cv2.dnn.NMSBoxes may return flat array or list of lists depending on OpenCV version
        flat_indices = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]

        for idx in flat_indices:
            cls_id = class_ids[idx]
            cls_name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else f"class_{cls_id}"

            if cls_name.lower() in self.target_classes:
                bx, by, bw, bh = boxes[idx]
                conf = float(scores[idx])

                # Clamp pixel box
                x1 = max(0, min(w - 1, bx))
                y1 = max(0, min(h - 1, by))
                bw_clamped = max(1, min(w - x1, bw))
                bh_clamped = max(1, min(h - y1, bh))

                # Compute normalized coordinates
                nx = max(0.0, min(1.0, x1 / float(w)))
                ny = max(0.0, min(1.0, y1 / float(h)))
                nw = max(0.0, min(1.0 - nx, bw_clamped / float(w)))
                nh = max(0.0, min(1.0 - ny, bh_clamped / float(h)))

                detections.append(
                    DetectionBox(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, bw_clamped, bh_clamped),
                        normalized_bbox=(nx, ny, nw, nh),
                    )
                )

        return detections


class ONNXRuntimeDetector(BaseDetector):
    """Tier 1 Detector: ONNX Runtime on CPU with AVX2 instruction optimization."""

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = 0.5,
        target_classes: Optional[Sequence[str]] = None,
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        super().__init__(confidence_threshold, target_classes)
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        self.input_size = input_size  # (width, height)
        # Configure session options for efficient multi-threaded CPU execution
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Inspect model inputs and outputs
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(
            "Initialized ONNXRuntimeDetector with %s on CPU (input=%s)",
            self.model_path.name,
            self.input_name,
        )

    def detect(self, frame: np.ndarray) -> List[DetectionBox]:
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        in_w, in_h = self.input_size

        # 1. Preprocessing: resize to 640x640, BGR to RGB, scale to [0, 1]
        resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        # 2. Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        raw_output = outputs[0]  # Shape: (1, 84, 8400) or (1, 25200, 85)

        # 3. Postprocess YOLO outputs
        return self._postprocess_yolo(raw_output, (orig_h, orig_w), (in_h, in_w))

    def _postprocess_yolo(
        self,
        output: np.ndarray,
        orig_shape: Tuple[int, int],
        input_shape: Tuple[int, int],
    ) -> List[DetectionBox]:
        orig_h, orig_w = orig_shape
        in_h, in_w = input_shape
        x_scale = orig_w / float(in_w)
        y_scale = orig_h / float(in_h)

        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        # Check YOLOv8 format: (1, 84, 8400) where 84 = [cx, cy, w, h, 80 scores]
        if output.ndim == 3 and output.shape[1] < output.shape[2]:
            preds = output[0].T  # Shape: (8400, 84)
            for row in preds:
                class_scores = row[4:]
                cls_id = int(np.argmax(class_scores))
                score = float(class_scores[cls_id])

                if score >= self.confidence_threshold:
                    cx, cy, w, h = row[:4]
                    x1 = int((cx - w / 2.0) * x_scale)
                    y1 = int((cy - h / 2.0) * y_scale)
                    bw = int(w * x_scale)
                    bh = int(h * y_scale)
                    boxes.append((x1, y1, bw, bh))
                    scores.append(score)
                    class_ids.append(cls_id)

        # Handle YOLOv5 format: (1, 25200, 85) where 85 = [cx, cy, w, h, obj_conf, 80 scores]
        elif output.ndim == 3:
            preds = output[0]  # Shape: (N, 85)
            for row in preds:
                obj_conf = float(row[4])
                if obj_conf < self.confidence_threshold:
                    continue
                class_scores = row[5:] * obj_conf
                cls_id = int(np.argmax(class_scores))
                score = float(class_scores[cls_id])

                if score >= self.confidence_threshold:
                    cx, cy, w, h = row[:4]
                    x1 = int((cx - w / 2.0) * x_scale)
                    y1 = int((cy - h / 2.0) * y_scale)
                    bw = int(w * x_scale)
                    bh = int(h * y_scale)
                    boxes.append((x1, y1, bw, bh))
                    scores.append(score)
                    class_ids.append(cls_id)

        return self.filter_predictions(boxes, scores, class_ids, orig_shape)


class OpenCVDNNDetector(BaseDetector):
    """Tier 2 Detector: Built-in OpenCV DNN module (cv2.dnn.readNetFromONNX)."""

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = 0.5,
        target_classes: Optional[Sequence[str]] = None,
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        super().__init__(confidence_threshold, target_classes)
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.input_size = input_size
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("Initialized OpenCVDNNDetector with %s", self.model_path.name)

    def detect(self, frame: np.ndarray) -> List[DetectionBox]:
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        in_w, in_h = self.input_size

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(in_w, in_h),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Parse outputs similarly to ONNX Runtime
        x_scale = orig_w / float(in_w)
        y_scale = orig_h / float(in_h)
        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        if outputs.ndim == 3 and outputs.shape[1] < outputs.shape[2]:
            preds = outputs[0].T
            for row in preds:
                class_scores = row[4:]
                cls_id = int(np.argmax(class_scores))
                score = float(class_scores[cls_id])
                if score >= self.confidence_threshold:
                    cx, cy, w, h = row[:4]
                    boxes.append(
                        (
                            int((cx - w / 2.0) * x_scale),
                            int((cy - h / 2.0) * y_scale),
                            int(w * x_scale),
                            int(h * y_scale),
                        )
                    )
                    scores.append(score)
                    class_ids.append(cls_id)

        return self.filter_predictions(boxes, scores, class_ids, (orig_h, orig_w))


class MockDetector(BaseDetector):
    """Tier 3 Detector: Fast deterministic detector for hermetic testing.

    Detects person/vehicle entities using:
    1. Pre-programmed or injected detections (set_detections).
    2. Synthetic ground-truth metadata from SyntheticCameraStream frames.
    3. Direct pixel feature extraction for synthetic humanoid/vehicle sprites.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        target_classes: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(confidence_threshold, target_classes)
        self._programmed_detections: Optional[List[DetectionBox]] = None
        self._ground_truth_registry: Dict[Any, List[DetectionBox]] = {}

    def set_detections(self, detections: Optional[List[DetectionBox]]) -> None:
        """Inject detections to be returned on subsequent call(s)."""
        self._programmed_detections = list(detections) if detections is not None else None

    def register_ground_truth(self, key: Any, detections: List[DetectionBox]) -> None:
        """Register ground-truth detections for a given frame index or identifier."""
        self._ground_truth_registry[key] = detections

    def detect(self, frame: np.ndarray) -> List[DetectionBox]:
        """Detect persons or vehicles in the given frame."""
        # 1. Check for programmed detections
        if self._programmed_detections is not None:
            ret = [
                d
                for d in self._programmed_detections
                if d.class_name.lower() in self.target_classes
                and d.confidence >= self.confidence_threshold
            ]
            return ret

        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detections: List[DetectionBox] = []

        # 2. Check if frame object carries synthetic metadata directly
        metadata = getattr(frame, "metadata", None)
        if isinstance(metadata, dict) and "ground_truth" in metadata:
            for gt in metadata["ground_truth"]:
                cls = str(gt.get("class_name", "")).lower()
                conf = float(gt.get("confidence", 0.90))
                if cls in self.target_classes and conf >= self.confidence_threshold:
                    bx = tuple(int(v) for v in gt.get("bbox", (0, 0, 0, 0)))
                    nb = tuple(float(v) for v in gt.get("normalized_bbox", (0, 0, 0, 0)))
                    detections.append(
                        DetectionBox(
                            class_name=cls,
                            confidence=conf,
                            bbox=bx,  # type: ignore
                            normalized_bbox=nb,  # type: ignore
                        )
                    )
            if detections:
                return detections

        # 3. Sprite feature extraction: detect synthetic humanoid or car sprites
        # Synthetic person has jacket with BGR (140, 50, 40) and head (170, 195, 220)
        if "person" in self.target_classes:
            person_boxes = self._detect_synthetic_person(frame)
            for bx in person_boxes:
                x, y, bw, bh = bx
                nb = (x / float(w), y / float(h), bw / float(w), bh / float(h))
                detections.append(
                    DetectionBox(
                        class_name="person",
                        confidence=0.92,
                        bbox=bx,
                        normalized_bbox=nb,
                    )
                )

        # Synthetic car has chassis with BGR (160, 65, 35) and cabin (140, 55, 30)
        if ("car" in self.target_classes or "vehicle" in self.target_classes) and not detections:
            car_boxes = self._detect_synthetic_car(frame)
            for bx in car_boxes:
                x, y, bw, bh = bx
                nb = (x / float(w), y / float(h), bw / float(w), bh / float(h))
                detections.append(
                    DetectionBox(
                        class_name="car",
                        confidence=0.95,
                        bbox=bx,
                        normalized_bbox=nb,
                    )
                )

        return [d for d in detections if d.confidence >= self.confidence_threshold]

    def _detect_synthetic_person(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect synthetic humanoid figure by jacket torso color (BGR: 140, 50, 40)."""
        lower = np.array([115, 30, 20], dtype=np.uint8)
        upper = np.array([165, 70, 60], dtype=np.uint8)
        mask = cv2.inRange(frame, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = frame.shape[:2]

        for c in contours:
            area = cv2.contourArea(c)
            # Torso is roughly 24x45 = ~1000 pixels in 640x360
            if area > 100:
                tx, ty, tw, th = cv2.boundingRect(c)
                # Expand torso box to full humanoid bounding box (include head and legs)
                full_h = int(th * 2.3)
                full_y = max(0, ty - int(th * 0.45))
                full_w = max(int(tw * 1.5), 30)
                full_x = max(0, tx - (full_w - tw) // 2)

                x1 = min(w - 1, full_x)
                y1 = min(h - 1, full_y)
                bw = min(w - x1, full_w)
                bh = min(h - y1, full_h)
                boxes.append((x1, y1, bw, bh))

        return boxes

    def _detect_synthetic_car(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect synthetic vehicle by chassis color (BGR: 160, 65, 35)."""
        lower = np.array([135, 45, 20], dtype=np.uint8)
        upper = np.array([185, 85, 55], dtype=np.uint8)
        mask = cv2.inRange(frame, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = frame.shape[:2]

        for c in contours:
            area = cv2.contourArea(c)
            # Car chassis is roughly 130x25 = ~3250 pixels
            if area > 200:
                cx, cy, cw, ch = cv2.boundingRect(c)
                # Expand chassis box to include roof cabin and wheels
                full_h = int(ch * 2.1)
                full_y = max(0, cy - int(ch * 0.85))
                full_w = int(cw * 1.05)
                full_x = max(0, cx - int(cw * 0.025))

                x1 = min(w - 1, full_x)
                y1 = min(h - 1, full_y)
                bw = min(w - x1, full_w)
                bh = min(h - y1, full_h)
                boxes.append((x1, y1, bw, bh))

        return boxes


def create_detector(
    model_path: Optional[Union[str, Path]] = None,
    preferred_tier: str = "onnx",
    confidence_threshold: float = 0.5,
    target_classes: Optional[Sequence[str]] = None,
) -> BaseDetector:
    """Factory function to instantiate the best available detector tier.

    Tiers:
    - "onnx": Tries ONNXRuntimeDetector, falls back to OpenCVDNNDetector, then MockDetector.
    - "opencv_dnn": Tries OpenCVDNNDetector, falls back to MockDetector.
    - "mock": Deterministic MockDetector for zero-download tests.

    Args:
        model_path: Path to ONNX model weights (e.g. models/yolov8n.onnx).
        preferred_tier: "onnx", "opencv_dnn", or "mock".
        confidence_threshold: Minimum confidence score [0.0, 1.0].
        target_classes: Sequence of classes to detect.

    Returns:
        Configured BaseDetector instance.
    """
    classes = target_classes or ["person", "car", "motorcycle", "bus", "truck"]
    tier = (preferred_tier or settings.AI_ENGINE_TIER).lower().strip()
    resolved_path = Path(model_path) if model_path else Path(settings.YOLO_MODEL_PATH)

    if tier == "mock":
        logger.info("Instantiating Tier 3 MockDetector as requested")
        return MockDetector(
            confidence_threshold=confidence_threshold,
            target_classes=classes,
        )

    # Check if model file is accessible on disk
    if resolved_path.is_file():
        if tier in ("onnx", "tier1"):
            try:
                return ONNXRuntimeDetector(
                    model_path=resolved_path,
                    confidence_threshold=confidence_threshold,
                    target_classes=classes,
                )
            except Exception as err:
                logger.warning(
                    "Failed to initialize ONNXRuntimeDetector: %s. Attempting OpenCV DNN fallback.",
                    err,
                )

        if tier in ("onnx", "opencv_dnn", "tier2"):
            try:
                return OpenCVDNNDetector(
                    model_path=resolved_path,
                    confidence_threshold=confidence_threshold,
                    target_classes=classes,
                )
            except Exception as err:
                logger.warning(
                    "Failed to initialize OpenCVDNNDetector: %s. Falling back to MockDetector.",
                    err,
                )

    # Fallback to MockDetector
    logger.info(
        "Model file '%s' not present or loadable. Initializing Tier 3 MockDetector.",
        resolved_path,
    )
    return MockDetector(
        confidence_threshold=confidence_threshold,
        target_classes=classes,
    )
