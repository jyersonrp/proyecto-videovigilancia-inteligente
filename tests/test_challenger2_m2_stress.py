import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from smart_nvr.detection.inference import (
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


class TestFrameFloodRateLimiting:
    def test_frame_flood_50fps_continuous_motion_rate_limiting(self) -> None:
        mock_det = MockDetector()
        mock_det.set_detections([
            DetectionBox('person', 0.92, (20, 20, 40, 80), (0.1, 0.1, 0.2, 0.4))
        ])

        motion_det = MOG2MotionDetector()
        motion_det.detect = lambda frame, roi_mask=None, learning_rate=-1.0: (True, [(20, 20, 40, 80)])

        pipeline = HybridDetectionPipeline(
            camera_id='cam_flood_50fps',
            motion_detector=motion_det,
            ai_detector=mock_det,
            ai_fps=5.0,
        )

        dummy_frame = np.zeros((180, 320, 3), dtype=np.uint8)
        base_time = 1700000000.0

        ai_triggers: List[float] = []
        for i in range(100):
            ts = base_time + (i * 0.02)
            res = pipeline.process_frame(dummy_frame, timestamp=ts)
            if res.ai_triggered:
                ai_triggers.append(ts)

        assert len(ai_triggers) == 10, f'Expected 10 triggers over 2.0s at 5 FPS, got {len(ai_triggers)}'

        intervals = [ai_triggers[j + 1] - ai_triggers[j] for j in range(len(ai_triggers) - 1)]
        for idx, inv in enumerate(intervals):
            assert 0.16 <= inv <= 0.25, f'Interval {idx} ({inv:.4f}s) violated 4-6 FPS interval bounds'

        effective_fps = len(ai_triggers) / 2.0
        assert 4.0 <= effective_fps <= 6.0, f'Effective FPS ({effective_fps}) outside [4, 6] FPS'
        assert pipeline.telemetry['ai_inferences'] == 10
        assert pipeline.telemetry['motion_frames'] == 100

    def test_frame_flood_high_rate_60fps_and_120fps(self) -> None:
        mock_det = MockDetector()
        mock_det.set_detections([DetectionBox('car', 0.90, (10, 10, 50, 50), (0.1, 0.1, 0.2, 0.2))])

        motion_det = MOG2MotionDetector()
        motion_det.detect = lambda frame, roi_mask=None, learning_rate=-1.0: (True, [(10, 10, 50, 50)])

        pipeline = HybridDetectionPipeline(
            camera_id='cam_flood_extreme',
            motion_detector=motion_det,
            ai_detector=mock_det,
            ai_fps=5.0,
        )

        dummy_frame = np.zeros((180, 320, 3), dtype=np.uint8)
        base_time = 1700000000.0

        triggers_60: List[float] = []
        for i in range(120):
            ts = base_time + (i * (1.0 / 60.0))
            res = pipeline.process_frame(dummy_frame, timestamp=ts)
            if res.ai_triggered:
                triggers_60.append(ts)

        assert 9 <= len(triggers_60) <= 11

        pipeline.reset()
        triggers_120: List[float] = []
        base_time += 10.0
        for i in range(240):
            ts = base_time + (i * (1.0 / 120.0))
            res = pipeline.process_frame(dummy_frame, timestamp=ts)
            if res.ai_triggered:
                triggers_120.append(ts)

        assert 9 <= len(triggers_120) <= 11

    def test_frame_flood_jittery_timestamps(self) -> None:
        mock_det = MockDetector()
        mock_det.set_detections([DetectionBox('person', 0.88, (10, 10, 30, 60), (0.1, 0.1, 0.2, 0.3))])

        motion_det = MOG2MotionDetector()
        motion_det.detect = lambda frame, roi_mask=None, learning_rate=-1.0: (True, [(10, 10, 30, 60)])

        pipeline = HybridDetectionPipeline(
            camera_id='cam_jitter',
            motion_detector=motion_det,
            ai_detector=mock_det,
            ai_fps=5.0,
        )

        dummy_frame = np.zeros((180, 320, 3), dtype=np.uint8)
        t = 1700000000.0
        jitter_deltas = [0.012, 0.038, 0.015, 0.029, 0.045, 0.008, 0.022, 0.031] * 20

        ai_triggers: List[float] = []
        for dt in jitter_deltas:
            t += dt
            res = pipeline.process_frame(dummy_frame, timestamp=t)
            if res.ai_triggered:
                ai_triggers.append(t)

        for j in range(len(ai_triggers) - 1):
            interval = ai_triggers[j + 1] - ai_triggers[j]
            assert interval >= 0.20, f'Jitter trigger interval ({interval:.4f}s) violated min interval 0.20s'

    def test_ai_fps_boundary_rates_4fps_and_6fps(self) -> None:
        mock_det = MockDetector()
        mock_det.set_detections([DetectionBox('car', 0.91, (10, 10, 40, 40), (0.1, 0.1, 0.1, 0.1))])

        motion_det = MOG2MotionDetector()
        motion_det.detect = lambda frame, roi_mask=None, learning_rate=-1.0: (True, [(10, 10, 40, 40)])

        p4 = HybridDetectionPipeline(camera_id='c4', motion_detector=motion_det, ai_detector=mock_det, ai_fps=4.0)
        assert p4._min_ai_interval == 0.25
        triggers_4: List[float] = []
        base_t = 1700000000.0
        for i in range(100):
            ts = base_t + (i * 0.02)
            if p4.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=ts).ai_triggered:
                triggers_4.append(ts)
        assert len(triggers_4) == 8

        p6 = HybridDetectionPipeline(camera_id='c6', motion_detector=motion_det, ai_detector=mock_det, ai_fps=6.0)
        assert abs(p6._min_ai_interval - (1.0 / 6.0)) < 1e-5
        triggers_6: List[float] = []
        for i in range(100):
            ts = base_t + (i * 0.02)
            if p6.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=ts).ai_triggered:
                triggers_6.append(ts)
        assert 12 <= len(triggers_6) <= 13

    def test_intermittent_motion_resumption_immediate_trigger(self) -> None:
        mock_det = MockDetector()
        mock_det.set_detections([DetectionBox('person', 0.95, (10, 10, 30, 80), (0.1, 0.1, 0.1, 0.3))])

        motion_state = [True]
        motion_det = MOG2MotionDetector()
        motion_det.detect = lambda frame, roi_mask=None, learning_rate=-1.0: (
            motion_state[0],
            [(10, 10, 30, 80)] if motion_state[0] else [],
        )

        pipeline = HybridDetectionPipeline(
            camera_id='cam_burst',
            motion_detector=motion_det,
            ai_detector=mock_det,
            ai_fps=5.0,
        )

        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        t = 1700000000.0

        r1 = pipeline.process_frame(dummy, timestamp=t)
        assert r1.motion_detected and r1.ai_triggered

        r2 = pipeline.process_frame(dummy, timestamp=t + 0.02)
        assert r2.motion_detected and not r2.ai_triggered

        motion_state[0] = False
        r_idle = pipeline.process_frame(dummy, timestamp=t + 1.5)
        assert not r_idle.motion_detected and not r_idle.ai_triggered

        motion_state[0] = True
        r_resume = pipeline.process_frame(dummy, timestamp=t + 3.0)
        assert r_resume.motion_detected
        assert r_resume.ai_triggered, 'AI must trigger immediately on new motion after idle period'

    def test_intermediate_frames_detection_persistence(self) -> None:
        mock_det = MockDetector()
        mock_det.set_detections([DetectionBox('car', 0.93, (20, 20, 60, 40), (0.1, 0.1, 0.3, 0.2))])

        motion_det = MOG2MotionDetector()
        motion_det.detect = lambda frame, roi_mask=None, learning_rate=-1.0: (True, [(20, 20, 60, 40)])

        pipeline = HybridDetectionPipeline(
            camera_id='cam_persist',
            motion_detector=motion_det,
            ai_detector=mock_det,
            ai_fps=5.0,
            annotate=True,
        )

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        t = 1700000000.0

        f0 = pipeline.process_frame(frame, timestamp=t)
        assert f0.ai_triggered
        assert f0.has_detections
        assert len(f0.confirmed_detections) == 1
        assert f0.annotated_frame is not None

        f1 = pipeline.process_frame(frame, timestamp=t + 0.02)
        assert not f1.ai_triggered
        assert f1.has_detections, 'Intermediate frame must retain detections to prevent UI flicker'
        assert f1.primary_class == 'car'
        assert f1.annotated_frame is not None
        assert np.any(f1.annotated_frame > 0)

    def test_zero_timestamp_sentinel_edge_case(self) -> None:
        mock_det = MockDetector()
        mock_det.set_detections([DetectionBox('person', 0.90, (10, 10, 20, 20), (0.1, 0.1, 0.1, 0.1))])

        motion_det = MOG2MotionDetector()
        motion_det.detect = lambda frame, roi_mask=None, learning_rate=-1.0: (True, [(10, 10, 20, 20)])

        p = HybridDetectionPipeline(camera_id='cam_zero_ts', motion_detector=motion_det, ai_detector=mock_det, ai_fps=5.0)

        res0 = p.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=0.0)
        res1 = p.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=0.02)
        res2 = p.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=0.04)

        assert res0.ai_triggered is True
        assert res1.ai_triggered is True
        assert res2.ai_triggered is False


class TestMultiTierDetectorFallback:
    def test_fallback_missing_model_file(self) -> None:
        det = create_detector(
            model_path=Path('models/strictly_non_existent_model_12345.onnx'),
            preferred_tier='onnx',
            confidence_threshold=0.6,
            target_classes=['person', 'car'],
        )
        assert isinstance(det, MockDetector)
        assert det.confidence_threshold == 0.6
        assert 'person' in det.target_classes

    def test_fallback_directory_path_instead_of_file(self, tmp_path) -> None:
        dummy_dir = tmp_path / 'fake_model_dir.onnx'
        dummy_dir.mkdir()
        det = create_detector(model_path=dummy_dir, preferred_tier='onnx')
        assert isinstance(det, MockDetector)

    def test_fallback_zero_byte_empty_file(self, tmp_path) -> None:
        empty_file = tmp_path / 'empty_model.onnx'
        empty_file.touch()
        det = create_detector(model_path=empty_file, preferred_tier='onnx')
        assert isinstance(det, MockDetector)

    def test_fallback_corrupted_binary_file(self, tmp_path) -> None:
        corrupt_bin = tmp_path / 'corrupt_model.onnx'
        corrupt_bin.write_bytes(b'\x00\xff\xfe\xfd\x80\x10' * 512)
        det = create_detector(model_path=corrupt_bin, preferred_tier='onnx')
        assert isinstance(det, MockDetector)

    def test_fallback_corrupted_plaintext_file(self, tmp_path) -> None:
        corrupt_txt = tmp_path / 'text_masquerade.onnx'
        corrupt_txt.write_text('ONNX\x00NOT_A_REAL_MODEL_HEADER')
        det = create_detector(model_path=corrupt_txt, preferred_tier='onnx')
        assert isinstance(det, MockDetector)

    def test_fallback_tier2_opencv_dnn_on_corrupted_file(self, tmp_path) -> None:
        corrupt_file = tmp_path / 'bad_opencv.onnx'
        corrupt_file.write_text('bad content')
        det = create_detector(model_path=corrupt_file, preferred_tier='opencv_dnn')
        assert isinstance(det, MockDetector)

    def test_fallback_cascade_tier1_to_tier2_to_tier3(self, tmp_path) -> None:
        model_file = tmp_path / 'model.onnx'
        model_file.write_bytes(b'test_binary')

        with patch('smart_nvr.detection.inference.ONNXRuntimeDetector', side_effect=RuntimeError('ONNX error')):
            with patch('smart_nvr.detection.inference.OpenCVDNNDetector') as mock_t2:
                mock_t2.return_value = 'mocked_tier2'
                result = create_detector(model_path=model_file, preferred_tier='onnx')
                assert result == 'mocked_tier2'

        with patch('smart_nvr.detection.inference.ONNXRuntimeDetector', side_effect=RuntimeError('ONNX error')):
            with patch('smart_nvr.detection.inference.OpenCVDNNDetector', side_effect=RuntimeError('DNN error')):
                result = create_detector(model_path=model_file, preferred_tier='onnx')
                assert isinstance(result, MockDetector)

    def test_direct_detector_constructors_raise_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ONNXRuntimeDetector('missing_model.onnx')

        with pytest.raises(FileNotFoundError):
            OpenCVDNNDetector('missing_model.onnx')


class TestConfidenceBoundaryBehavior:
    def test_mock_detector_exact_confidence_boundary(self) -> None:
        threshold = 0.50
        detector = MockDetector(
            confidence_threshold=threshold,
            target_classes=['person', 'car'],
        )

        injected = [
            DetectionBox('person', 0.4900, (10, 10, 40, 40), (0.1, 0.1, 0.2, 0.2)),
            DetectionBox('person', 0.4999, (20, 20, 40, 40), (0.2, 0.2, 0.2, 0.2)),
            DetectionBox('person', 0.5000, (30, 30, 40, 40), (0.3, 0.3, 0.2, 0.2)),
            DetectionBox('person', 0.5001, (40, 40, 40, 40), (0.4, 0.4, 0.2, 0.2)),
            DetectionBox('car', 0.5100, (50, 50, 40, 40), (0.5, 0.5, 0.2, 0.2)),
        ]
        detector.set_detections(injected)

        results = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        confs = [r.confidence for r in results]

        assert 0.4900 not in confs
        assert 0.4999 not in confs

        assert 0.5000 in confs
        assert 0.5001 in confs
        assert 0.5100 in confs
        assert len(results) == 3

    def test_filter_predictions_confidence_boundary_and_nms_discrepancy(self) -> None:
        class ConcreteDetector(BaseDetector):
            def detect(self, frame: np.ndarray) -> List[DetectionBox]:
                return []

        det = ConcreteDetector(confidence_threshold=0.50, target_classes=['person'])

        boxes = [
            (10, 10, 20, 20),
            (40, 10, 20, 20),
            (70, 10, 20, 20),
            (100, 10, 20, 20),
        ]
        scores = [0.4900, 0.5000, 0.500001, 0.5100]
        class_ids = [0, 0, 0, 0]

        filtered = det.filter_predictions(boxes, scores, class_ids, (200, 200))
        accepted_scores = [f.confidence for f in filtered]

        assert 0.4900 not in accepted_scores
        assert 0.5000 not in accepted_scores
        assert 0.500001 in accepted_scores
        assert 0.5100 in accepted_scores

    def test_target_class_boundary_rejection_at_high_confidence(self) -> None:
        detector = MockDetector(
            confidence_threshold=0.50,
            target_classes=['person', 'car'],
        )

        injected = [
            DetectionBox('person', 0.51, (10, 10, 30, 30), (0.1, 0.1, 0.1, 0.1)),
            DetectionBox('dog', 0.99, (20, 20, 30, 30), (0.2, 0.2, 0.1, 0.1)),
            DetectionBox('chair', 0.95, (30, 30, 30, 30), (0.3, 0.3, 0.1, 0.1)),
            DetectionBox('bicycle', 0.85, (40, 40, 30, 30), (0.4, 0.4, 0.1, 0.1)),
        ]
        detector.set_detections(injected)

        results = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        classes = [r.class_name for r in results]

        assert classes == ['person']
        assert 'dog' not in classes
        assert 'chair' not in classes
        assert 'bicycle' not in classes

    def test_confidence_boundary_dynamic_threshold_adjustments(self) -> None:
        detector = MockDetector(confidence_threshold=0.50, target_classes=['person'])

        injected = [
            DetectionBox('person', 0.65, (10, 10, 30, 30), (0.1, 0.1, 0.1, 0.1)),
            DetectionBox('person', 0.85, (20, 20, 30, 30), (0.2, 0.2, 0.1, 0.1)),
        ]
        detector.set_detections(injected)

        r1 = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(r1) == 2

        detector.confidence_threshold = 0.70
        detector.set_detections(injected)
        r2 = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(r2) == 1
        assert r2[0].confidence == 0.85

        detector.confidence_threshold = 0.90
        detector.set_detections(injected)
        r3 = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(r3) == 0

    def test_negative_and_excessive_confidence_scores(self) -> None:
        detector = MockDetector(confidence_threshold=0.50, target_classes=['person'])

        injected = [
            DetectionBox('person', -0.5, (10, 10, 20, 20), (0.1, 0.1, 0.1, 0.1)),
            DetectionBox('person', 1.5, (20, 20, 20, 20), (0.2, 0.2, 0.1, 0.1)),
        ]
        detector.set_detections(injected)

        results = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert not any(r.confidence < 0.0 for r in results)
        assert any(r.confidence > 1.0 for r in results)
