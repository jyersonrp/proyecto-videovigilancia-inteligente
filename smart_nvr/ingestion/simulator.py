"""Synthetic Camera Simulator for procedural video stream generation.

Generates realistic procedural video feeds at target FPS without physical cameras,
supporting programmable scenarios:
1. Static background with real-time timestamp and subtle sensor noise.
2. Moving humanoid/person sprite with walking gait, soft shadow, and ground-truth bounding box.
3. Moving vehicle/car sprite with wheels, headlights, and ground-truth bounding box.
4. Out-of-ROI motion (e.g., foliage or sky movement) for testing ROI mask filtering.
5. Global illumination / lighting shifts to test MOG2 shadow & ambient light resilience.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from smart_nvr.config import settings
from smart_nvr.ingestion.broadcaster import DualQueue, FrameBroadcaster
from smart_nvr.ingestion.stream import BaseCameraStream, CameraFrame

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    """Supported simulation scenarios."""

    STATIC = "static"
    IDLE = "idle"
    MOVING_PERSON = "moving_person"
    PERSON = "person"
    MOVING_CAR = "moving_car"
    CAR = "car"
    VEHICLE = "vehicle"
    OUT_OF_ROI_MOTION = "out_of_roi_motion"
    LIGHTING_SHIFT = "lighting_shift"


class SyntheticCameraStream(BaseCameraStream):
    """Procedural synthetic camera stream generator.

    Produces BGR frames at target FPS in a decoupled background thread,
    complete with ground-truth bounding boxes in frame metadata.
    """

    def __init__(
        self,
        camera_id: str = "synthetic_cam",
        fps_target: int = 15,
        width: int = 640,
        height: int = 360,
        scenario: str = "static",
        jpeg_quality: Optional[int] = None,
        broadcaster: Optional[FrameBroadcaster] = None,
    ) -> None:
        self.camera_id = str(camera_id)
        self.fps_target = max(1, int(fps_target))
        self.width = int(width)
        self.height = int(height)
        self.jpeg_quality = jpeg_quality or settings.JPEG_QUALITY

        # Threading state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()

        # Atomic single-slot latest frame storage
        self._latest_frame: Optional[CameraFrame] = None
        self._frame_index: int = 0

        # Scenario management
        self._scenario: str = scenario.lower()
        self._scenario_lock = threading.Lock()

        # Motion simulation state variables
        self._person_x: float = -40.0
        self._person_y: float = float(self.height * 0.60)
        self._person_speed: float = 65.0  # pixels per second

        self._car_x: float = -130.0
        self._car_y: float = float(self.height * 0.72)
        self._car_speed: float = 110.0  # pixels per second

        self._out_of_roi_x: float = 20.0
        self._out_of_roi_y: float = 35.0
        self._out_of_roi_speed: float = 40.0

        self._lighting_offset: float = 0.0
        self._lighting_direction: float = 1.0

        # Pre-render immutable background canvas
        self._base_background = self._render_base_scene()

        # Pub/sub broadcaster for live streaming
        if broadcaster is not None:
            self.broadcaster = broadcaster
        else:
            self.broadcaster = FrameBroadcaster(self.camera_id, jpeg_quality=self.jpeg_quality)

    @property
    def is_running(self) -> bool:
        """Return True if synthetic capture thread is active."""
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def current_scenario(self) -> str:
        """Return active simulation scenario."""
        with self._scenario_lock:
            return self._scenario

    def set_scenario(self, scenario: str) -> None:
        """Dynamically switch scenario without restarting the stream."""
        with self._scenario_lock:
            self._scenario = scenario.lower()
            # Reset object positions on scenario change
            self._person_x = -40.0
            self._car_x = -130.0
            self._out_of_roi_x = 20.0
            logger.info("Camera %s switched to scenario: %s", self.camera_id, self._scenario)

    def start(self) -> None:
        """Start generating frames in background thread."""
        if self.is_running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._generator_worker,
            name=f"SyntheticStream-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started SyntheticCameraStream %s at %d FPS", self.camera_id, self.fps_target)

    def stop(self, timeout: float = 2.0) -> None:
        """Cleanly stop synthetic stream generator."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("Stopped SyntheticCameraStream %s", self.camera_id)

    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Atomically fetch the freshest synthesized frame."""
        with self._frame_lock:
            return self._latest_frame

    def subscribe(self, maxsize: int = 1) -> DualQueue:
        """Subscribe to live JPEG broadcast stream."""
        return self.broadcaster.subscribe(maxsize=maxsize)

    def generate_next_frame(self, dt: Optional[float] = None) -> CameraFrame:
        """Generate the next frame synchronously.

        Useful for unit tests requiring deterministic frame stepping.
        """
        if dt is None:
            dt = 1.0 / self.fps_target

        now = time.time()
        frame, ground_truth = self._synthesize_frame(dt, now)

        with self._frame_lock:
            self._frame_index += 1
            cam_frame = CameraFrame(
                camera_id=self.camera_id,
                timestamp=now,
                frame=frame,
                frame_index=self._frame_index,
                metadata={
                    "scenario": self.current_scenario,
                    "ground_truth": ground_truth,
                },
            )
            self._latest_frame = cam_frame

        # Distribute to web clients
        self.broadcaster.broadcast_frame(frame)
        return cam_frame

    def _render_base_scene(self) -> np.ndarray:
        """Pre-render a realistic surveillance camera background canvas."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. Sky / Horizon gradient (top 30%)
        horizon_y = int(self.height * 0.35)
        for y in range(horizon_y):
            factor = y / horizon_y
            # Gradient from slate blue-gray to lighter hazy gray
            b = int(170 + 20 * factor)
            g = int(150 + 25 * factor)
            r = int(140 + 25 * factor)
            canvas[y, :] = (b, g, r)

        # 2. Building facade (middle section, x: 0 to 65% width)
        building_w = int(self.width * 0.65)
        building_top = int(self.height * 0.20)
        building_bot = int(self.height * 0.65)
        # Brick/stone gray-tan color
        canvas[building_top:building_bot, 0:building_w] = (120, 115, 110)

        # Draw building roof trim
        cv2.line(canvas, (0, building_top), (building_w, building_top), (80, 75, 70), 3)

        # Door on building
        door_x1, door_x2 = int(self.width * 0.35), int(self.width * 0.47)
        door_y1, door_y2 = int(self.height * 0.40), building_bot
        cv2.rectangle(canvas, (door_x1, door_y1), (door_x2, door_y2), (45, 60, 85), -1)
        # Door knob
        cv2.circle(canvas, (door_x2 - 8, int((door_y1 + door_y2) / 2)), 3, (160, 190, 210), -1)

        # Window on building
        win_x1, win_x2 = int(self.width * 0.10), int(self.width * 0.25)
        win_y1, win_y2 = int(self.height * 0.28), int(self.height * 0.45)
        cv2.rectangle(canvas, (win_x1, win_y1), (win_x2, win_y2), (180, 160, 130), -1)
        cv2.rectangle(canvas, (win_x1, win_y1), (win_x2, win_y2), (50, 50, 50), 2)
        # Window panes cross
        cv2.line(canvas, (int((win_x1 + win_x2) / 2), win_y1), (int((win_x1 + win_x2) / 2), win_y2), (50, 50, 50), 1)
        cv2.line(canvas, (win_x1, int((win_y1 + win_y2) / 2)), (win_x2, int((win_y1 + win_y2) / 2)), (50, 50, 50), 1)

        # 3. Ground / Driveway asphalt (bottom 35%)
        driveway_top = building_bot
        canvas[driveway_top:, :] = (55, 55, 55)

        # Curb / Grass strip between building and driveway
        cv2.line(canvas, (0, driveway_top), (self.width, driveway_top), (90, 90, 90), 2)

        # Distant vegetation/trees on the right side
        tree_x1 = building_w
        cv2.rectangle(canvas, (tree_x1, int(self.height * 0.25)), (self.width, driveway_top), (40, 75, 45), -1)

        return canvas

    def _synthesize_frame(
        self,
        dt: float,
        timestamp: float,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Procedurally render the frame based on the active scenario."""
        frame = self._base_background.copy()
        ground_truth: List[Dict[str, Any]] = []

        scenario = self.current_scenario

        # 1. Handle Lighting Shift Scenario
        if scenario == ScenarioType.LIGHTING_SHIFT.value:
            # Oscillate lighting offset
            self._lighting_offset += self._lighting_direction * (35.0 * dt)
            if self._lighting_offset > 50.0:
                self._lighting_direction = -1.0
            elif self._lighting_offset < -30.0:
                self._lighting_direction = 1.0

            # Apply global brightness delta
            int_offset = int(self._lighting_offset)
            if int_offset != 0:
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=int_offset)

        # 2. Handle Moving Person Scenario
        elif scenario in (ScenarioType.MOVING_PERSON.value, ScenarioType.PERSON.value):
            self._person_x += self._person_speed * dt
            if self._person_x > self.width + 50:
                self._person_x = -50.0

            px = int(self._person_x)
            py = int(self._person_y)
            pw, ph = 36, 92

            # Render person sprite
            self._draw_person(frame, px, py, pw, ph)

            # Calculate clipped bounding box
            x1 = max(0, px)
            y1 = max(0, py)
            x2 = min(self.width, px + pw)
            y2 = min(self.height, py + ph)
            bw = x2 - x1
            bh = y2 - y1

            if bw > 8 and bh > 15:
                ground_truth.append({
                    "class_name": "person",
                    "confidence": 1.0,
                    "bbox": (x1, y1, bw, bh),
                    "normalized_bbox": (
                        x1 / self.width,
                        y1 / self.height,
                        bw / self.width,
                        bh / self.height,
                    ),
                    "in_roi": True,
                })

        # 3. Handle Moving Car Scenario
        elif scenario in (
            ScenarioType.MOVING_CAR.value,
            ScenarioType.CAR.value,
            ScenarioType.VEHICLE.value,
        ):
            self._car_x += self._car_speed * dt
            if self._car_x > self.width + 150:
                self._car_x = -140.0

            cx = int(self._car_x)
            cy = int(self._car_y)
            cw, ch = 130, 58

            # Render vehicle sprite
            self._draw_car(frame, cx, cy, cw, ch)

            # Calculate clipped bounding box
            x1 = max(0, cx)
            y1 = max(0, cy)
            x2 = min(self.width, cx + cw)
            y2 = min(self.height, cy + ch)
            bw = x2 - x1
            bh = y2 - y1

            if bw > 15 and bh > 10:
                ground_truth.append({
                    "class_name": "car",
                    "confidence": 1.0,
                    "bbox": (x1, y1, bw, bh),
                    "normalized_bbox": (
                        x1 / self.width,
                        y1 / self.height,
                        bw / self.width,
                        bh / self.height,
                    ),
                    "in_roi": True,
                })

        # 4. Handle Out-Of-ROI Motion Scenario (Foliage/bird in sky zone)
        elif scenario == ScenarioType.OUT_OF_ROI_MOTION.value:
            ow, oh = 32, 18
            self._out_of_roi_x += self._out_of_roi_speed * dt
            if self._out_of_roi_x > self.width - 40:
                self._out_of_roi_speed = -abs(self._out_of_roi_speed)
                self._out_of_roi_x = min(self._out_of_roi_x, float(self.width - 40))
            elif self._out_of_roi_x < 20:
                self._out_of_roi_speed = abs(self._out_of_roi_speed)
                self._out_of_roi_x = max(self._out_of_roi_x, 20.0)

            # Defensive clamp to ensure coordinates never stray outside valid canvas
            self._out_of_roi_x = max(0.0, min(float(self.width - ow), self._out_of_roi_x))

            ox = int(self._out_of_roi_x)
            oy = int(self._out_of_roi_y)

            # Render bird/drone sprite in upper sky area
            cv2.ellipse(frame, (ox + 16, oy + 9), (14, 6), 0, 0, 360, (20, 20, 20), -1)
            cv2.line(frame, (ox + 2, oy + 2), (ox + 16, oy + 9), (15, 15, 15), 2)
            cv2.line(frame, (ox + 30, oy + 2), (ox + 16, oy + 9), (15, 15, 15), 2)

            # Calculate clipped bounding box strictly within frame boundaries [0, width] and [0, height]
            x1 = max(0, ox)
            y1 = max(0, oy)
            x2 = min(self.width, ox + ow)
            y2 = min(self.height, oy + oh)
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)

            ground_truth.append({
                "class_name": "out_of_roi",
                "confidence": 0.85,
                "bbox": (x1, y1, bw, bh),
                "normalized_bbox": (
                    x1 / self.width,
                    y1 / self.height,
                    bw / self.width,
                    bh / self.height,
                ),
                "in_roi": False,
            })

        # 5. Add subtle realistic sensor noise
        noise = np.random.normal(0, 1.2, frame.shape).astype(np.float32)
        frame_noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 6. Render real-time timestamp and camera OSD (On-Screen Display)
        time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        osd_text = f"CAM: {self.camera_id} [{scenario.upper()}] | {time_str}"

        # Draw semi-transparent banner for readability
        cv2.rectangle(frame_noisy, (10, 8), (480, 32), (0, 0, 0), -1)
        cv2.putText(
            frame_noisy,
            osd_text,
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        return frame_noisy, ground_truth

    def _draw_person(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """Draw a recognizable humanoid figure with head, torso, legs and shadow."""
        center_x = x + w // 2

        # 1. Soft ground shadow under feet
        shadow_center = (center_x, min(self.height - 3, y + h))
        shadow_axes = (w // 2 + 6, 7)
        overlay = frame.copy()
        cv2.ellipse(overlay, shadow_center, shadow_axes, 0, 0, 360, (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 2. Walking leg animation (alternating based on position)
        stride = int(5 * np.sin(x * 0.15))
        leg_top_y = y + int(h * 0.60)
        leg_bot_y = y + h
        # Left leg
        cv2.line(frame, (center_x - 5, leg_top_y), (center_x - 7 + stride, leg_bot_y), (35, 45, 55), 4)
        # Right leg
        cv2.line(frame, (center_x + 5, leg_top_y), (center_x + 7 - stride, leg_bot_y), (35, 45, 55), 4)

        # 3. Torso (Jacket)
        torso_top_y = y + int(h * 0.22)
        cv2.rectangle(frame, (center_x - 12, torso_top_y), (center_x + 12, leg_top_y), (140, 50, 40), -1)

        # 4. Arms
        cv2.line(frame, (center_x - 12, torso_top_y + 4), (center_x - 16 - stride, torso_top_y + 24), (130, 45, 35), 3)
        cv2.line(frame, (center_x + 12, torso_top_y + 4), (center_x + 16 + stride, torso_top_y + 24), (130, 45, 35), 3)

        # 5. Head
        head_radius = int(h * 0.11)
        head_center = (center_x, y + head_radius)
        cv2.circle(frame, head_center, head_radius, (170, 195, 220), -1)

    def _draw_car(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """Draw a recognizable vehicle/car with chassis, cabin, wheels, and lights."""
        # 1. Soft shadow on asphalt
        shadow_center = (x + w // 2, min(self.height - 4, y + h - 2))
        shadow_axes = (w // 2 + 10, 8)
        overlay = frame.copy()
        cv2.ellipse(overlay, shadow_center, shadow_axes, 0, 0, 360, (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # 2. Main chassis body
        chassis_top = y + int(h * 0.40)
        chassis_bot = y + int(h * 0.85)
        # Navy blue body
        cv2.rectangle(frame, (x, chassis_top), (x + w, chassis_bot), (160, 65, 35), -1)
        # Bumpers
        cv2.rectangle(frame, (x - 3, chassis_bot - 8), (x + 3, chassis_bot), (80, 80, 80), -1)
        cv2.rectangle(frame, (x + w - 3, chassis_bot - 8), (x + w + 3, chassis_bot), (80, 80, 80), -1)

        # 3. Cabin (roof & windows)
        cabin_pts = np.array([
            [x + int(w * 0.20), chassis_top],
            [x + int(w * 0.32), y + 2],
            [x + int(w * 0.75), y + 2],
            [x + int(w * 0.88), chassis_top],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [cabin_pts], (140, 55, 30))

        # Windows
        window_pts = np.array([
            [x + int(w * 0.24), chassis_top - 2],
            [x + int(w * 0.34), y + 6],
            [x + int(w * 0.73), y + 6],
            [x + int(w * 0.84), chassis_top - 2],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [window_pts], (40, 40, 40))

        # 4. Headlights and Taillights
        # Front headlight (right side)
        cv2.circle(frame, (x + w - 4, chassis_top + 8), 5, (120, 255, 255), -1)
        # Rear taillight (left side)
        cv2.circle(frame, (x + 3, chassis_top + 8), 4, (30, 30, 220), -1)

        # 5. Wheels
        wheel_radius = int(h * 0.17)
        wheel_y = chassis_bot
        # Rear wheel
        cv2.circle(frame, (x + int(w * 0.22), wheel_y), wheel_radius, (25, 25, 25), -1)
        cv2.circle(frame, (x + int(w * 0.22), wheel_y), wheel_radius // 2, (160, 160, 160), -1)
        # Front wheel
        cv2.circle(frame, (x + int(w * 0.78), wheel_y), wheel_radius, (25, 25, 25), -1)
        cv2.circle(frame, (x + int(w * 0.78), wheel_y), wheel_radius // 2, (160, 160, 160), -1)

    def _generator_worker(self) -> None:
        """Background thread generating frames at target FPS rate."""
        target_interval = 1.0 / self.fps_target
        last_tick = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                now_perf = time.perf_counter()
                dt = max(0.001, now_perf - last_tick)
                last_tick = now_perf

                self.generate_next_frame(dt=dt)

                elapsed = time.perf_counter() - now_perf
                sleep_time = target_interval - elapsed
                if sleep_time > 0.001:
                    self._stop_event.wait(sleep_time)

        except Exception as err:
            logger.exception("Unexpected error in SyntheticCameraStream %s: %s", self.camera_id, err)
        finally:
            logger.debug("Exited SyntheticCameraStream worker loop for %s", self.camera_id)
