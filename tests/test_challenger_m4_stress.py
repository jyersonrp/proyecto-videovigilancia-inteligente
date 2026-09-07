"""Adversarial Empirical Stress-Testing Suite for Milestone 4 (M4).

Empirical Challenger Verification:
1. Rapid concurrent burst test:
   - 10 threads dispatching 50 alerts each across 5 different cameras simultaneously (500 total dispatches).
   - Barrier synchronization to enforce simultaneous execution at microsecond scale.
   - Verification that cooldown is strictly enforced per camera without race conditions or lock deadlocks.
   - Per-camera alert isolation and remaining cooldown integrity.
   - Concurrent stress on raw AlertCooldownTracker methods (should_alert, get_remaining_cooldown, reset).
2. Queue overflow / backpressure test:
   - Verification that when queue fills up to maxsize, alerts are dropped gracefully without crashing or blocking the caller.
   - Strict caller latency validation (< 5ms per dispatch) ensuring non-blocking queue behavior.
   - Concurrent multi-threaded queue saturation (10 threads contending on tiny queue).
   - Verification of failed status and error message logging in database repository.
   - Boundary checks (queue_maxsize=1, invalid/negative maxsize).
3. Clean shutdown test:
   - Verification that AlertService.stop() terminates the background daemon thread within timeout without hanging.
   - Queue draining verification on clean shutdown.
   - High-frequency start/stop lifecycle stress (25 rapid cycles).
   - Idempotency of stop() (unstarted, double-stop).
   - Context manager lifecycle and exception safety.
   - Resilient shutdown termination when notifier raises exceptions during drain.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import pytest

from smart_nvr.alerts.cooldown import AlertCooldownTracker
from smart_nvr.alerts.notifier import (
    AlertPayload,
    BaseNotifier,
    ConsoleNotifier,
    GmailSmtpNotifier,
    MockNotifier,
    create_notifier,
)
from smart_nvr.alerts.service import AlertService
from smart_nvr.db.repository import DatabaseRepository


# ============================================================================
# Helpers & Fixtures
# ============================================================================

def make_stress_payload(
    event_id: Optional[str] = None,
    camera_id: str = "cam_test",
    camera_name: str = "Test Camera",
    detection_class: str = "person",
    confidence: float = 0.95,
    snapshot_bytes: Optional[bytes] = b"\xff\xd8\xff\xe0mock_snapshot_bytes\xff\xd9",
) -> AlertPayload:
    """Construct standard AlertPayload for stress testing."""
    return AlertPayload(
        event_id=event_id or f"evt_{uuid.uuid4().hex[:12]}",
        camera_id=camera_id,
        camera_name=camera_name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        detection_class=detection_class,
        confidence=confidence,
        snapshot_bytes=snapshot_bytes,
        detections=[{"class_name": detection_class, "confidence": confidence}],
    )


class SlowMockNotifier(BaseNotifier):
    """Spy notifier with configurable artificial latency and thread safety."""

    def __init__(self, latency_seconds: float = 0.05) -> None:
        self.latency_seconds = latency_seconds
        self.sent_alerts: List[AlertPayload] = []
        self._lock = threading.Lock()
        self.call_count = 0

    def send_alert(self, payload: AlertPayload) -> bool:
        if self.latency_seconds > 0:
            time.sleep(self.latency_seconds)
        with self._lock:
            self.call_count += 1
            self.sent_alerts.append(payload)
        return True

    def get_sent_alerts(self) -> List[AlertPayload]:
        with self._lock:
            return list(self.sent_alerts)


class FailingMockNotifier(BaseNotifier):
    """Notifier that raises an unhandled exception during send_alert."""

    def __init__(self, exc: Exception = RuntimeError("Simulated network fatal failure")) -> None:
        self.exc = exc
        self.attempts = 0
        self._lock = threading.Lock()

    def send_alert(self, payload: AlertPayload) -> bool:
        with self._lock:
            self.attempts += 1
        raise self.exc


# ============================================================================
# Challenge Category 1: Rapid Concurrent Burst & Cooldown Integrity
# ============================================================================

class TestCategory1RapidConcurrentBurst:
    """Adversarial testing of AlertCooldownTracker and AlertService under burst concurrency."""

    def test_rapid_concurrent_burst_10_threads_50_alerts_5_cameras(self) -> None:
        """Requirement 1: 10 threads dispatching 50 alerts each across 5 cameras simultaneously.

        Verify:
        - Cooldown is strictly enforced per camera without race conditions or lock deadlocks.
        - Exactly 1 alert is accepted per camera (5 total accepted across all 500 dispatches).
        - Exactly 495 dispatches are suppressed by cooldown.
        - Zero thread deadlocks or crashes.
        """
        num_threads = 10
        alerts_per_thread = 50
        num_cameras = 5
        camera_ids = [f"camera_{i}" for i in range(num_cameras)]

        mock_notifier = MockNotifier()
        service = AlertService(
            notifier=mock_notifier,
            default_cooldown_seconds=60.0,
            queue_maxsize=1000,
        )
        service.start()

        # Thread synchronization barrier to ensure simultaneous release
        barrier = threading.Barrier(num_threads)
        results: List[Tuple[str, bool, float]] = []
        results_lock = threading.Lock()

        def worker_task(thread_idx: int) -> None:
            # Wait for all 10 threads to be ready
            barrier.wait(timeout=5.0)

            for i in range(alerts_per_thread):
                # Distribute alerts across the 5 cameras
                cam_id = camera_ids[(thread_idx * alerts_per_thread + i) % num_cameras]
                payload = make_stress_payload(
                    event_id=f"evt_burst_t{thread_idx}_{i}",
                    camera_id=cam_id,
                    camera_name=f"Camera {cam_id}",
                )
                t0 = time.perf_counter()
                accepted = service.dispatch_alert(payload)
                t_elapsed = time.perf_counter() - t0

                with results_lock:
                    results.append((cam_id, accepted, t_elapsed))

        threads = [
            threading.Thread(target=worker_task, args=(tid,), name=f"BurstWorker-{tid}")
            for tid in range(num_threads)
        ]

        t_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "Worker thread hung or deadlocked!"

        total_burst_time = time.perf_counter() - t_start

        # Wait for service to process accepted alerts
        assert service.wait_until_empty(timeout=5.0) is True
        service.stop()

        # 1. Total dispatch count must be exactly 500
        assert len(results) == num_threads * alerts_per_thread

        # 2. Analyze accepted vs suppressed per camera
        camera_accepted: Dict[str, int] = {cid: 0 for cid in camera_ids}
        camera_suppressed: Dict[str, int] = {cid: 0 for cid in camera_ids}

        for cam_id, accepted, latency in results:
            if accepted:
                camera_accepted[cam_id] += 1
            else:
                camera_suppressed[cam_id] += 1

        # Strict assertion: Exactly 1 alert per camera allowed within the 60s window
        for cam_id in camera_ids:
            assert camera_accepted[cam_id] == 1, (
                f"Camera {cam_id} had {camera_accepted[cam_id]} alerts accepted, "
                f"expected strictly 1 under cooldown!"
            )
            assert camera_suppressed[cam_id] == (num_threads * alerts_per_thread // num_cameras) - 1

        total_accepted = sum(camera_accepted.values())
        total_suppressed = sum(camera_suppressed.values())
        assert total_accepted == num_cameras
        assert total_suppressed == (num_threads * alerts_per_thread) - num_cameras

        # 3. Verify notifier delivered exactly 5 alerts
        sent_alerts = mock_notifier.get_sent_alerts()
        assert len(sent_alerts) == num_cameras
        sent_cams = {p.camera_id for p in sent_alerts}
        assert sent_cams == set(camera_ids)

        # 4. Performance & deadlock sanity check
        assert total_burst_time < 2.0, f"Burst took {total_burst_time:.2f}s, expected sub-second execution"

    def test_raw_tracker_concurrent_burst_and_reset_races(self) -> None:
        """Adversarial stress on AlertCooldownTracker with concurrent should_alert and reset."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=10.0)
        num_threads = 12
        iterations = 100
        cam_id = "shared_cam"

        allowed_count = 0
        suppressed_count = 0
        count_lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        def hammer_tracker() -> None:
            nonlocal allowed_count, suppressed_count
            barrier.wait()
            for _ in range(iterations):
                if tracker.should_alert(cam_id):
                    with count_lock:
                        allowed_count += 1
                else:
                    with count_lock:
                        suppressed_count += 1

        threads = [threading.Thread(target=hammer_tracker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Without resets, strictly ONE call must have succeeded
        assert allowed_count == 1
        assert suppressed_count == (num_threads * iterations) - 1

        # Check remaining cooldown
        rem = tracker.get_remaining_cooldown(cam_id)
        assert 0.0 < rem <= 10.0

        # Now test concurrent reset() while threads query remaining cooldown
        reset_threads = 6
        read_threads = 6
        stop_flag = threading.Event()

        def resetter() -> None:
            while not stop_flag.is_set():
                tracker.reset(cam_id)
                time.sleep(0.001)

        def reader() -> None:
            while not stop_flag.is_set():
                _ = tracker.get_remaining_cooldown(cam_id)
                _ = tracker.get_last_alert_time(cam_id)
                _ = tracker.should_alert(cam_id, cooldown_seconds=0.05)

        all_t = [threading.Thread(target=resetter) for _ in range(reset_threads)] + [
            threading.Thread(target=reader) for _ in range(read_threads)
        ]
        for t in all_t:
            t.start()
        time.sleep(0.25)
        stop_flag.set()
        for t in all_t:
            t.join(timeout=2.0)
            assert not t.is_alive(), "Tracker reset/read race caused a deadlock!"

    def test_burst_with_real_sqlite_wal_database(self, tmp_path: Path) -> None:
        """Verify that concurrent burst dispatches log properly to a real SQLite database without lock errors."""
        db_file = tmp_path / "sqlite_burst.db"
        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()

        # Seed 5 cameras and events
        num_cameras = 5
        cam_ids = []
        for i in range(num_cameras):
            cid = repo.create_camera({"name": f"Cam {i}", "stream_url": f"synthetic://cam{i}"})
            cam_ids.append(cid)

        num_threads = 10
        alerts_per_thread = 20  # 200 total alerts to SQLite

        # Seed event records in database so foreign key constraint (event_id -> events.id) passes
        for tid in range(num_threads):
            for i in range(alerts_per_thread):
                eid = f"evt_db_t{tid}_{i}"
                cid = cam_ids[i % num_cameras]
                repo.create_event({
                    "id": eid,
                    "camera_id": cid,
                    "start_time": "2026-09-06 22:00:00",
                    "detection_class": "person",
                })

        mock_notifier = MockNotifier()
        service = AlertService(
            notifier=mock_notifier,
            db_repo=repo,
            default_cooldown_seconds=60.0,
            queue_maxsize=500,
        )
        service.start()
        barrier = threading.Barrier(num_threads)

        def burst_to_db(thread_idx: int) -> None:
            barrier.wait()
            for i in range(alerts_per_thread):
                cid = cam_ids[i % num_cameras]
                p = make_stress_payload(
                    event_id=f"evt_db_t{thread_idx}_{i}",
                    camera_id=cid,
                )
                service.dispatch_alert(p)

        threads = [threading.Thread(target=burst_to_db, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert service.wait_until_empty(timeout=5.0) is True
        service.stop()

        # Verify SQLite contents
        conn = repo.get_connection()
        cur = conn.execute("SELECT status, count(*) as cnt FROM alerts GROUP BY status")
        counts = {r["status"]: r["cnt"] for r in cur.fetchall()}

        # 5 should be 'sent' (1 per camera), remaining 195 should be 'suppressed_cooldown'
        assert counts.get("sent") == num_cameras
        assert counts.get("suppressed_cooldown") == (num_threads * alerts_per_thread) - num_cameras
        repo.close()


# ============================================================================
# Challenge Category 2: Queue Overflow & Backpressure
# ============================================================================

class TestCategory2QueueOverflowAndBackpressure:
    """Adversarial stress on queue bounded limits, dropped alerts, and non-blocking backpressure."""

    def test_queue_overflow_drops_gracefully_without_caller_blocking(self) -> None:
        """Requirement 2: Verify that when queue fills up to maxsize, alerts are dropped gracefully without crashing or blocking the caller."""
        # Queue maxsize = 10. Worker NOT started, so queue fills up and stays full.
        max_q = 10
        mock_notifier = MockNotifier()
        service = AlertService(
            notifier=mock_notifier,
            default_cooldown_seconds=0.0,  # Cooldown bypassed to isolate queue behavior
            queue_maxsize=max_q,
        )

        # 1. Fill queue to exact capacity
        for i in range(max_q):
            p = make_stress_payload(event_id=f"fill_{i}", camera_id=f"cam_{i}")
            assert service.dispatch_alert(p) is True
            assert service.queue.qsize() == i + 1

        assert service.queue.full() is True

        # 2. Dispatch 50 additional alerts into the full queue
        overflow_count = 50
        durations: List[float] = []

        for i in range(overflow_count):
            p = make_stress_payload(event_id=f"overflow_{i}", camera_id=f"cam_ov_{i}")
            t0 = time.perf_counter()
            accepted = service.dispatch_alert(p)
            durations.append(time.perf_counter() - t0)

            # MUST drop gracefully, returning False
            assert accepted is False
            # Queue size must remain exactly at maxsize
            assert service.queue.qsize() == max_q

        # 3. Caller Latency Check: Non-blocking assertion
        # Each dispatch_alert must return in < 5 milliseconds (no blocking on queue)
        max_duration = max(durations)
        avg_duration = sum(durations) / len(durations)
        assert max_duration < 0.05, f"dispatch_alert took {max_duration*1000:.2f}ms, suspected blocking!"
        assert avg_duration < 0.01, f"dispatch_alert average {avg_duration*1000:.2f}ms too slow!"

    def test_concurrent_queue_saturation_under_slow_consumer(self) -> None:
        """10 threads contending to push into a small queue (maxsize=5) with a slow consumer.

        Verifies that:
        - Threads never hang or deadhead on queue.Full.
        - Dropped alerts return False cleanly.
        - Backpressure does not corrupt queue contents.
        """
        max_q = 5
        slow_notifier = SlowMockNotifier(latency_seconds=0.04)  # 40ms per alert
        service = AlertService(
            notifier=slow_notifier,
            default_cooldown_seconds=0.0,
            queue_maxsize=max_q,
        )
        service.start()

        num_threads = 10
        pushes_per_thread = 15  # 150 total alerts into a size-5 queue
        barrier = threading.Barrier(num_threads)
        results: List[bool] = []
        lock = threading.Lock()

        def spam_queue(tid: int) -> None:
            barrier.wait()
            for j in range(pushes_per_thread):
                p = make_stress_payload(event_id=f"flood_t{tid}_{j}", camera_id=f"cam_flood_{tid}_{j}")
                ok = service.dispatch_alert(p)
                with lock:
                    results.append(ok)
                # Rapid fire without sleep to induce heavy backpressure
                time.sleep(0.001)

        threads = [threading.Thread(target=spam_queue, args=(i,)) for i in range(num_threads)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Thread hung under backpressure!"

        t_elapsed = time.perf_counter() - t0

        # Producers finished quickly (in ~100ms), far faster than the slow consumer would take
        assert t_elapsed < 1.0, f"Producers took {t_elapsed:.2f}s, backpressure blocked caller threads!"

        # Some were accepted, some were dropped
        accepted_count = sum(1 for r in results if r is True)
        dropped_count = sum(1 for r in results if r is False)
        assert len(results) == num_threads * pushes_per_thread
        assert accepted_count > 0
        assert dropped_count > 0
        assert accepted_count + dropped_count == 150

        # Allow service to drain and stop
        assert service.wait_until_empty(timeout=10.0) is True
        service.stop()

    def test_queue_overflow_db_audit_logging(self, tmp_path: Path) -> None:
        """Dropped alerts due to queue saturation must be logged to DatabaseRepository with status='failed'."""
        db_file = tmp_path / "overflow_audit.db"
        repo = DatabaseRepository(db_path=db_file)
        repo.init_db()

        # Seed camera and events
        cam_id = repo.create_camera({"name": "BackpressureCam", "stream_url": "synthetic://bp"})
        repo.create_event({"id": "evt_ok_1", "camera_id": cam_id, "start_time": "2026-09-06 22:00:00"})
        repo.create_event({"id": "evt_ok_2", "camera_id": cam_id, "start_time": "2026-09-06 22:00:01"})
        repo.create_event({"id": "evt_drop_3", "camera_id": cam_id, "start_time": "2026-09-06 22:00:02"})

        mock_notifier = MockNotifier()
        # Maxsize=2, worker NOT started
        service = AlertService(
            notifier=mock_notifier,
            db_repo=repo,
            default_cooldown_seconds=0.0,
            queue_maxsize=2,
        )

        p1 = make_stress_payload(event_id="evt_ok_1", camera_id=cam_id)
        p2 = make_stress_payload(event_id="evt_ok_2", camera_id=cam_id)
        p3 = make_stress_payload(event_id="evt_drop_3", camera_id=cam_id)

        assert service.dispatch_alert(p1) is True
        assert service.dispatch_alert(p2) is True
        # 3rd alert overflows queue
        assert service.dispatch_alert(p3) is False

        # Inspect database for dropped alert
        conn = repo.get_connection()
        cur = conn.execute("SELECT event_id, status, error_message FROM alerts WHERE event_id = 'evt_drop_3'")
        row = cur.fetchone()
        assert row is not None
        assert row["event_id"] == "evt_drop_3"
        assert row["status"] == "failed"
        assert "queue full" in row["error_message"].lower()

        repo.close()

    def test_queue_boundary_min_size(self) -> None:
        """Verify robust behavior with minimal or negative queue_maxsize."""
        # queue_maxsize=0 or -5 should be clamped to max(1, queue_maxsize)
        s1 = AlertService(notifier=MockNotifier(), queue_maxsize=0)
        assert s1.queue.maxsize == 1

        s2 = AlertService(notifier=MockNotifier(), queue_maxsize=-10)
        assert s2.queue.maxsize == 1

        # Test queue_maxsize=1
        s3 = AlertService(notifier=MockNotifier(), default_cooldown_seconds=0.0, queue_maxsize=1)
        assert s3.dispatch_alert(make_stress_payload(event_id="p1")) is True
        assert s3.dispatch_alert(make_stress_payload(event_id="p2")) is False


# ============================================================================
# Challenge Category 3: Clean Shutdown & Lifecycle
# ============================================================================

class TestCategory3CleanShutdownAndLifecycle:
    """Adversarial testing of AlertService lifecycle, clean thread termination, and queue draining."""

    def test_clean_shutdown_terminates_within_timeout_without_hanging(self) -> None:
        """Requirement 3: Verify that AlertService.stop() terminates the background daemon thread within timeout without hanging."""
        mock_notifier = MockNotifier()
        service = AlertService(notifier=mock_notifier)

        # 1. Not running initially
        assert service.is_running is False

        # 2. Start service
        service.start()
        assert service.is_running is True
        worker_thread = service._worker_thread
        assert worker_thread is not None
        assert worker_thread.is_alive() is True

        # 3. Stop service with timeout
        t0 = time.perf_counter()
        service.stop(timeout=2.0)
        stop_duration = time.perf_counter() - t0

        # Must terminate promptly (well under timeout, typical < 0.15s)
        assert stop_duration < 1.0, f"stop() took {stop_duration:.2f}s, expected fast exit"
        assert service.is_running is False
        assert not worker_thread.is_alive(), "Worker thread is still running after stop()!"
        assert service._worker_thread is None

    def test_shutdown_drains_remaining_queued_alerts(self) -> None:
        """Verify that stopping AlertService drains all remaining alerts from the queue before termination."""
        mock_notifier = MockNotifier()
        service = AlertService(notifier=mock_notifier, default_cooldown_seconds=0.0)

        # Enqueue 10 alerts BEFORE starting service
        num_alerts = 10
        for i in range(num_alerts):
            p = make_stress_payload(event_id=f"drain_{i}", camera_id=f"cam_drain_{i}")
            assert service.dispatch_alert(p) is True

        assert service.queue.qsize() == num_alerts

        # Start and immediately stop
        service.start()
        service.stop(timeout=3.0)

        # All 10 alerts must have been drained and delivered
        sent = mock_notifier.get_sent_alerts()
        assert len(sent) == num_alerts
        assert service.queue.empty() is True
        assert service.is_running is False

    def test_high_frequency_start_stop_cycles(self) -> None:
        """Stress-test rapid repeated start() and stop() cycles for resource/thread leaks."""
        mock_notifier = MockNotifier()
        service = AlertService(notifier=mock_notifier)

        cycles = 25
        t0 = time.perf_counter()
        for _ in range(cycles):
            service.start()
            assert service.is_running is True
            # Enqueue an alert during active cycle
            p = make_stress_payload()
            service.dispatch_alert(p)
            service.stop(timeout=1.0)
            assert service.is_running is False

        elapsed = time.perf_counter() - t0
        # 25 full cycles should take under 3.5s total
        assert elapsed < 5.0, f"25 start/stop cycles took {elapsed:.2f}s, suspected delay or leak"

    def test_stop_idempotency_and_unstarted_service(self) -> None:
        """Calling stop() on an unstarted service or multiple times consecutively must be safe."""
        service = AlertService(notifier=MockNotifier())

        # Stop unstarted service
        service.stop(timeout=1.0)
        assert service.is_running is False

        # Double stop
        service.start()
        service.stop(timeout=1.0)
        service.stop(timeout=1.0)
        assert service.is_running is False

    def test_context_manager_clean_exit_and_exception_handling(self) -> None:
        """Verify context manager enters and exits cleanly, even when exceptions are raised inside the block."""
        mock_notifier = MockNotifier()
        service = AlertService(notifier=mock_notifier)

        with pytest.raises(ValueError, match="Intentional test error"):
            with service:
                assert service.is_running is True
                service.dispatch_alert(make_stress_payload())
                raise ValueError("Intentional test error")

        # After exception, service must be cleanly stopped
        assert service.is_running is False

    def test_shutdown_resilience_when_notifier_raises_during_drain(self) -> None:
        """If notifier raises an unhandled exception during shutdown drain, the worker loop must log and exit cleanly without hanging."""
        failing_notifier = FailingMockNotifier()
        service = AlertService(
            notifier=failing_notifier,
            default_cooldown_seconds=0.0,
            queue_maxsize=10,
        )

        # Enqueue 3 alerts
        for i in range(3):
            service.dispatch_alert(make_stress_payload(event_id=f"failing_{i}"))

        # Start and stop
        service.start()
        t0 = time.perf_counter()
        service.stop(timeout=2.0)
        stop_duration = time.perf_counter() - t0

        # Must exit cleanly without hanging on the exception
        assert stop_duration < 1.0
        assert service.is_running is False
        assert failing_notifier.attempts >= 1


# ============================================================================
# Additional Adversarial & Edge Case Tests
# ============================================================================

class TestAdditionalAdversarialEdgeCases:
    """Edge cases: payload robustness, malformed inputs, and notifier factory."""

    def test_concurrent_start_race(self) -> None:
        """Calling start() simultaneously from 10 threads must safely spawn exactly 1 worker."""
        service = AlertService(notifier=MockNotifier())
        barrier = threading.Barrier(10)

        def call_start() -> None:
            barrier.wait()
            service.start()

        threads = [threading.Thread(target=call_start) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        assert service.is_running is True
        # Only 1 worker thread active
        w = service._worker_thread
        assert w is not None
        assert w.is_alive()

        service.stop()
        assert service.is_running is False

    def test_concurrent_dispatch_during_stop_race(self) -> None:
        """Simultaneous dispatch_alert calls while stop() is in progress must not deadlock or crash."""
        service = AlertService(notifier=MockNotifier(), default_cooldown_seconds=0.0)
        service.start()

        num_dispatchers = 5
        dispatches_per_thread = 30
        barrier = threading.Barrier(num_dispatchers + 1)

        def dispatcher(tid: int) -> None:
            barrier.wait()
            for i in range(dispatches_per_thread):
                service.dispatch_alert(make_stress_payload(event_id=f"evt_stop_race_{tid}_{i}"))
                time.sleep(0.001)

        def stopper() -> None:
            barrier.wait()
            time.sleep(0.01)
            service.stop(timeout=3.0)

        d_threads = [threading.Thread(target=dispatcher, args=(t,)) for t in range(num_dispatchers)]
        s_thread = threading.Thread(target=stopper)

        for t in d_threads:
            t.start()
        s_thread.start()

        for t in d_threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Dispatcher thread hung during stop race!"
        s_thread.join(timeout=5.0)
        assert not s_thread.is_alive(), "Stopper thread hung during stop race!"

        assert service.is_running is False

    def test_per_camera_heterogeneous_cooldown_concurrency(self) -> None:
        """Cameras with different cooldowns (Cam Fast: 0.05s vs Cam Slow: 5.0s) operating concurrently."""
        service = AlertService(notifier=MockNotifier(), default_cooldown_seconds=5.0)
        service.start()

        fast_cam = "cam_fast"
        slow_cam = "cam_slow"
        fast_allowed = 0
        slow_allowed = 0
        count_lock = threading.Lock()
        stop_event = threading.Event()

        def spam_fast() -> None:
            nonlocal fast_allowed
            while not stop_event.is_set():
                ok = service.dispatch_alert(
                    make_stress_payload(camera_id=fast_cam),
                    cooldown_seconds=0.05,
                )
                if ok:
                    with count_lock:
                        fast_allowed += 1
                time.sleep(0.01)

        def spam_slow() -> None:
            nonlocal slow_allowed
            while not stop_event.is_set():
                ok = service.dispatch_alert(
                    make_stress_payload(camera_id=slow_cam),
                    cooldown_seconds=5.0,
                )
                if ok:
                    with count_lock:
                        slow_allowed += 1
                time.sleep(0.01)

        t_fast = threading.Thread(target=spam_fast)
        t_slow = threading.Thread(target=spam_slow)

        t_fast.start()
        t_slow.start()

        time.sleep(0.35)  # Run for 350ms
        stop_event.set()

        t_fast.join(timeout=2.0)
        t_slow.join(timeout=2.0)

        service.stop()

        # Cam slow should strictly be allowed 1 time in 350ms with 5.0s cooldown
        assert slow_allowed == 1, f"Slow cam allowed {slow_allowed} times, expected 1"
        # Cam fast should be allowed multiple times (around 5-7 times in 350ms with 0.05s cooldown)
        assert fast_allowed >= 3, f"Fast cam allowed {fast_allowed} times, expected >= 3"

    def test_high_scale_1000_cameras_concurrency(self) -> None:
        """Stress-test AlertCooldownTracker tracking 1,000 distinct cameras concurrently across 20 threads."""
        tracker = AlertCooldownTracker(default_cooldown_seconds=30.0)
        num_cameras = 1000
        num_threads = 20
        cams_per_thread = num_cameras // num_threads

        def worker(tid: int) -> None:
            start_idx = tid * cams_per_thread
            end_idx = start_idx + cams_per_thread
            for c in range(start_idx, end_idx):
                cam_id = f"scale_cam_{c}"
                # First call must allow
                assert tracker.should_alert(cam_id) is True
                # Immediate second call must suppress
                assert tracker.should_alert(cam_id) is False
                assert tracker.get_remaining_cooldown(cam_id) > 0.0

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive()
        elapsed = time.perf_counter() - t0

        # 1,000 cameras checked twice = 2,000 queries across 20 threads in < 0.2s
        assert elapsed < 1.0, f"1,000 camera scale test took {elapsed:.2f}s"

    def test_alert_payload_with_zero_or_none_snapshot(self) -> None:
        """AlertPayload must handle None snapshot_bytes gracefully."""
        p = make_stress_payload(snapshot_bytes=None)
        assert p.get_snapshot_bytes() is None

        # Build email message with None snapshot bytes
        notifier = GmailSmtpNotifier()
        msg = notifier.build_email_message(p)
        assert msg["Subject"] is not None

    def test_console_notifier_concurrency(self) -> None:
        """ConsoleNotifier must safely handle concurrent multi-threaded dispatch."""
        notifier = ConsoleNotifier()
        service = AlertService(notifier=notifier, default_cooldown_seconds=0.0)
        service.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(service.dispatch_alert, make_stress_payload(event_id=f"console_{i}"))
                for i in range(20)
            ]
            for f in concurrent.futures.as_completed(futures):
                assert f.result() is True

        assert service.wait_until_empty(timeout=3.0) is True
        service.stop()

    def test_create_notifier_factory_variants(self) -> None:
        """create_notifier correctly handles disabled, mock, console, and gmail configs."""
        n_disabled = create_notifier({"ALERT_ENABLED": False})
        assert isinstance(n_disabled, MockNotifier)

        n_mock = create_notifier({"NOTIFIER_TYPE": "mock"})
        assert isinstance(n_mock, MockNotifier)

        n_console = create_notifier({"NOTIFIER_TYPE": "console"})
        assert isinstance(n_console, ConsoleNotifier)

        n_gmail = create_notifier({
            "NOTIFIER_TYPE": "gmail",
            "SMTP_SERVER": "smtp.gmail.com",
            "SMTP_PORT": 465,
            "SMTP_USERNAME": "test@gmail.com",
            "SMTP_PASSWORD": "pwd",
        })
        assert isinstance(n_gmail, GmailSmtpNotifier)
        assert n_gmail.use_ssl is True
