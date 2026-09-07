# Smart NVR (Sistema de Videovigilancia Inteligente)
# Test Infrastructure & Requirements Verification Specification (`TEST_INFRA.md`)

**Author**: E2E Test Writer  
**Status**: Authoritative Reference  
**Scope**: Opaque-box Test Design, Test Infrastructure, Hermetic Fixtures, and Requirements Coverage Matrix  
**Target Standard**: 100% Deterministic Pass, Zero Hardware Dependencies, Zero Thread/File-Handle Leaks  

---

## 1. Test Philosophy & Guiding Principles

The testing infrastructure of the **Smart NVR** project is designed according to four core pillars derived directly from `ORIGINAL_REQUEST.md`:

1. **Opaque-Box (Black-Box) Verification**:
   - Tests validate system behavior, observable side-effects, contract compliance, API responses, generated media artifacts (MP4 clips, JPEG snapshots), database state, and outgoing alert payloads.
   - Tests do NOT rely on private variables, internal thread names, or arbitrary implementation details.
   - If an internal component is refactored (e.g. switching between ONNX Runtime CPU and OpenCV DNN, or replacing deque with an optimized ring buffer), tests must remain valid without modification as long as external contracts are preserved.

2. **Strict Traceability to Requirements**:
   - Every test case directly links to requirements **R1–R6** and explicit Acceptance Criteria defined in `ORIGINAL_REQUEST.md`.
   - All 31 project features (**F01–F31**) from `PROJECT.md` are accounted for in a verifiable verification matrix.

3. **Hermeticity & Environmental Independence**:
   - **Zero Physical Camera Required**: Real video streams (RTSP/USB) are substituted with procedural synthetic generators (`SyntheticCameraStream` / `synthetic_video_feed`) generating controllable frame sequences with geometric motion, lighting changes, and noise.
   - **Zero Live Network Calls to Gmail**: External SMTP servers are intercepted using a local mock context (`mock_smtp_server` / `MockNotifier`) that captures and inspects full MIME messages, multipart headers, and inline CID attachments.
   - **Zero Global State Pollution**: SQLite databases and media storage directories operate in isolated temporary directories (`tmp_path` / `test_db` / `tmp_storage_dir`).

4. **Windows Resource Safety & Teardown Rigor**:
   - Given the Windows host OS (Windows 11 x64, NTFS), unclosed file handles (such as SQLite connections or OpenCV `VideoWriter` instances) result in fatal `[WinError 32]` sharing violations during pytest teardown.
   - All fixtures implement strict context-managed cleanup, thread signaling (`threading.Event.set()`), and explicit `.release()` / `.close()` calls in `finally:` blocks.

---

## 2. Test Design Methodology

To ensure exhaustive coverage without combinatorial explosion, the test suite leverages four formal testing techniques:

### 2.1 Category-Partition Equivalence Classes
The input domain is partitioned into mutually exclusive categories:
- **Video Resolutions**: Sub-VGA ($320\times240$), VGA ($640\times480$), HD 720p ($1280\times720$, standard baseline), Full HD 1080p ($1920\times1080$), 4K ($3840\times2160$).
- **Frame Rates**: Low (5 FPS), Nominal Surveillance (15 FPS), High (30 FPS), Accelerated Test Mode ($>100$ FPS).
- **Motion Patterns**:
  - $P_0$: Static background with sensor Gaussian noise.
  - $P_1$: Illumination changes and moving shadows ($MOG2$ shadow value $= 127$).
  - $P_2$: Motion outside designated Region of Interest (ROI).
  - $P_3$: Motion inside ROI with low AI confidence ($< \text{threshold}$).
  - $P_4$: Valid human target moving through ROI ($\ge \text{threshold}$).
  - $P_5$: Valid vehicle target moving through ROI ($\ge \text{threshold}$).
  - $P_6$: Continuous intermittent motion (bursts spaced $< \text{post\_roll}$ seconds apart).
- **Alert Dispatch States**:
  - $S_0$: Cold state (no prior alerts).
  - $S_1$: Active alert within cooldown window ($\Delta t < T_{\text{cooldown}}$).
  - $S_2$: Post-cooldown trigger ($\Delta t \ge T_{\text{cooldown}}$).

### 2.2 Boundary Value Analysis (BVA)
Boundary limits evaluated across critical parameters:
- **Frame Buffer Capacity**: $N = 0$, $N = 1$, $N = \text{target\_fps} \times \text{pre\_roll} - 1$, $N = \text{target\_fps} \times \text{pre\_roll}$, $N = \text{target\_fps} \times \text{pre\_roll} + 50$.
- **Detection Confidence**: $C = 0.0$, $C = \text{threshold} - 0.01$, $C = \text{threshold}$, $C = \text{threshold} + 0.01$, $C = 1.0$.
- **MOG2 Contour Area**: $A = 0$, $A = \text{min\_area} - 1$, $A = \text{min\_area}$, $A = \text{min\_area} + 1$.
- **Cooldown Interval**: $t = 0.0\text{s}$, $t = T_{\text{cooldown}} - 0.1\text{s}$, $t = T_{\text{cooldown}}$, $t = T_{\text{cooldown}} + 0.1\text{s}$.
- **Storage Limits**: Disk usage below quota, disk usage at $99.9\%$, disk usage exceeding quota trigger.

### 2.3 Pairwise Combinations (All-Pairs Testing)
Selected cross-cutting dimensions tested pairwise:
- `Source Type` $\times$ `Detection Mode` $\times$ `Storage Format`:
  - `{Synthetic, File, RTSP Mock} \times {MOG2-Only, MOG2+MockAI, MOG2+ONNX} \times {H.264 MP4, Fallback}`
- `Client Concurrency` $\times$ `Streaming Channel` $\times$ `System Load`:
  - `{1 client, 5 clients, 20 clients} \times {MJPEG multipart} \times {Idle, Active Detection Recording}`

### 2.4 Real-World Workload Testing
Realistic mission profiles simulating real deployment environments for 10–30 simulated seconds:
- **Residential Break-in**: Quiet yard $\rightarrow$ intruder traverses garden inside ROI $\rightarrow$ pre-roll preserves entry $\rightarrow$ post-roll captures exit $\rightarrow$ email delivered with high-contrast snapshot.
- **SMB Parking Lot**: Normal vehicle movement $\rightarrow$ recording triggered $\rightarrow$ repeated movement inside cooldown logs event without email spam.
- **Adverse Weather / Shadows**: Wind blowing tree branches outside ROI and shifting sun shadows ignored by MOG2.

---

## 3. Feature Inventory Traceability Matrix (F01–F31)

Every feature defined in `PROJECT.md` is mapped to its primary verification tier and test location:

| Feature ID | Feature Name | Requirement | Primary Test File | Verification Tier | Success Verification Criteria |
|---|---|---|---|---|---|
| **F01** | Decoupled Video Ingestion | R1 | `tests/test_ingestion.py` | Tier 1, Tier 2 | Threaded capture yields frames without blocking caller; automatic reconnect on source drop |
| **F02** | FrameBroadcaster Fanout | R1, R2 | `tests/test_ingestion.py` | Tier 1, Tier 3 | Single-encode JPEG; maxsize=1 queue drops stale frames; multiple consumers receive distinct frames |
| **F03** | MOG2 Motion Detection | R1 | `tests/test_detection.py` | Tier 1, Tier 2 | Downscaled 320x180 processing; shadow value 127 ignored; $<10\%$ idle CPU utilization |
| **F04** | Configurable ROIs | R1 | `tests/test_detection.py` | Tier 1, Tier 2 | Motion outside polygon mask discarded; motion intersecting mask triggers Phase 2 |
| **F05** | Lightweight DL Inference | R1 | `tests/test_detection.py` | Tier 1, Tier 3 | Correct classification of 'person', 'car'; bounding boxes normalized to $[0, 1]$ |
| **F06** | Inference Rate Limiting | R1 | `tests/test_detection.py` | Tier 1, Tier 2 | Inferences capped at 4–6 FPS during active motion; confidence threshold filter rejects $< \text{threshold}$ |
| **F07** | Synthetic Camera Simulator | R6 | `tests/test_ingestion.py` | Tier 1, Tier 4 | Generates valid BGR frames with programmable motion shapes; zero physical hardware required |
| **F08** | In-Memory Circular Buffer | R4 | `tests/test_storage.py` | Tier 1, Tier 2 | Holds 3–5s pre-roll; thread-safe FIFO eviction; array cloning prevents memory aliasing |
| **F09** | Continuous Event Fusion | R4 | `tests/test_storage.py` | Tier 1, Tier 3 | Intermittent motion within post-roll window extends recording into a single cohesive clip |
| **F10** | Universal Browser MP4 Writer | R4 | `tests/test_storage.py` | Tier 1, Tier 2 | Generates valid H.264 (avc1/H264) MP4 playable in HTML5 `<video>` without transcoding |
| **F11** | SQLite WAL Relational DB | R4 | `tests/test_storage.py` | Tier 1, Tier 3 | WAL mode active; `busy_timeout=5000`; composite index on `(camera_id, timestamp)` queries fast |
| **F12** | Storage Retention Manager | R4 | `tests/test_storage.py` | Tier 1, Tier 2 | Relative paths used; files and DB records purged when max days or disk quota exceeded |
| **F13** | Non-blocking Gmail SMTP | R3 | `tests/test_alerts.py` | Tier 1, Tier 3 | Email dispatch queued asynchronously; detection/ingestion thread never blocked by SMTP handshake |
| **F14** | Structured Rich Email Alert | R3 | `tests/test_alerts.py` | Tier 1, Tier 2 | MIME multipart/related with HTML template and CID inline JPEG snapshot attachment with bounding boxes |
| **F15** | Per-Camera Cooldown Throttling | R3 | `tests/test_alerts.py` | Tier 1, Tier 3 | Suppresses emails within cooldown period; event still logged in DB with `suppressed_cooldown` status |
| **F16** | Modular Notifier Interface | R3 | `tests/test_alerts.py` | Tier 1, Tier 3 | `BaseNotifier` polymorphism; `MockNotifier` captures payloads in-memory for testing |
| **F17** | FastAPI Server Core | R2 | `tests/test_api.py` | Tier 1, Tier 2 | Application boots cleanly; lifespan manages ingestion threads, alerting queue, and database pool |
| **F18** | Camera Management Endpoints | R2 | `tests/test_api.py` | Tier 1, Tier 2 | CRUD `/api/cameras`; connection probe `/api/cameras/test-connection`; snapshot endpoint |
| **F19** | Dynamic Thresholds Config | R2 | `tests/test_api.py` | Tier 1, Tier 2 | Hot-reloading sensitivity, confidence, and ROI without restarting video ingestion |
| **F20** | Paginated Event History | R2 | `tests/test_api.py` | Tier 1, Tier 3 | Filtering by `camera_id`, `start_date`, `end_date`, `detection_class`; pagination with total count |
| **F21** | Low-Latency Live Streaming | R2 | `tests/test_api.py` | Tier 1, Tier 3 | `/api/cameras/{id}/stream` yields `multipart/x-mixed-replace`; sub-500ms latency profile |
| **F22** | HTTP Range Video Streaming | R2 | `tests/test_api.py` | Tier 1, Tier 2 | `/api/events/{id}/video` returns HTTP 206 Partial Content with `Content-Range` headers for HTML5 scrubbing |
| **F23** | System Settings Endpoints | R2, R3 | `tests/test_api.py` | Tier 1, Tier 2 | GET/PUT `/api/settings`; POST `/api/settings/test-email` verifies SMTP connectivity |
| **F24** | Interactive OpenAPI Docs | R2 | `tests/test_api.py` | Tier 1 | Swagger UI accessible at `/docs` (HTTP 200); OpenAPI schema valid JSON at `/openapi.json` |
| **F25** | Zero-Build Web Dashboard | R5 | `tests/test_dashboard.py` | Tier 1 | GET `/` returns HTML5 document containing Tailwind CSS links and dashboard layout elements |
| **F26** | Multi-Stream Live Grid | R5 | `tests/test_dashboard.py` | Tier 1, Tier 4 | Grid containers render camera feeds with live status indicators and alert badges |
| **F27** | Event History Gallery | R5 | `tests/test_dashboard.py` | Tier 1, Tier 4 | Gallery view presents thumbnails, metadata badges, and modal HTML5 video player |
| **F28** | Interactive Canvas ROI Editor | R5 | `tests/test_dashboard.py` | Tier 1 | Canvas element rendered with coordinate persistence hooks |
| **F29** | Alert & SMTP Config View | R5 | `tests/test_dashboard.py` | Tier 1 | Configuration form with password masking and test email trigger UI |
| **F30** | Pytest 4-Tier Automated Suite | R6 | `tests/test_e2e.py` | Tier 1–4 | Complete test runner passes 100% of tests deterministically in CI/local environments |
| **F31** | Project Packaging & Docs | R6 | `tests/test_e2e.py` | Tier 1 | `requirements.txt`, `.env.example`, `run_server.py`, `README.md` verified complete |

---

## 4. The 4-Tier Test Architecture

```
                      ┌───────────────────────────────────────────────┐
                      │    Tier 4: Real-World Application Scenarios   │
                      │  (Residential Break-in, Parking Lot, Shadows) │
                      └───────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────▼───────────────────────┐
                      │  Tier 3: Cross-Feature Pairwise Interactions  │
                      │   (Ingestion+AI+Record, Event+SQLite+Alert)   │
                      └───────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────▼───────────────────────┐
                      │   Tier 2: Boundary, Concurrency & Stress      │
                      │  (Extreme Resolutions, Timeouts, Burst Motion)│
                      └───────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────▼───────────────────────┐
                      │   Tier 1: Feature Isolation & Contract Tests  │
                      │  (≥5 tests per feature area: Units & Modals)  │
                      └───────────────────────────────────────────────┘
```

### 4.1 Tier 1: Feature Coverage (Isolation)
Tests each functional unit in isolation with mocked inputs and deterministic outputs (at least 5 tests per major subsystem):
- **Ingestion & Simulator (`test_ingestion.py`)**:
  1. Synthetic frame generator dimensions and color format (BGR uint8).
  2. Frame sequence delivery and timestamp monotonicity.
  3. FrameBroadcaster single-subscriber delivery.
  4. FrameBroadcaster multi-subscriber fan-out.
  5. Slow consumer frame drop behavior (preventing buffer bloat).
- **Detection & Motion (`test_detection.py`)**:
  1. MOG2 background initialization on static frames (0 motion contours).
  2. Shadow contour elimination (gray value 127 ignored).
  3. ROI polygon inclusion (motion inside ROI detected).
  4. ROI polygon exclusion (motion outside ROI discarded).
  5. AI confidence threshold filtering ($< 0.5$ rejected, $\ge 0.5$ accepted).
- **Storage & Circular Buffer (`test_storage.py`)**:
  1. Circular buffer FIFO eviction at maximum frame limit.
  2. Pre-roll frame extraction maintains exact chronological order.
  3. Deep frame cloning prevents memory corruption during live capture.
  4. SQLite schema table creation and WAL mode verification.
  5. Storage manager relative path formatting for video and snapshot artifacts.
- **Alerting & Cooldown (`test_alerts.py`)**:
  1. Alert payload creation with metadata and bounding boxes.
  2. Structured MIME email building (MIMEMultipart related + alternative + inline image).
  3. Inline CID image attachment header verification.
  4. Per-camera cooldown allows initial alert.
  5. Per-camera cooldown suppresses subsequent alert within window.
- **FastAPI REST API (`test_api.py`)**:
  1. Root documentation availability (`GET /docs` $\rightarrow$ 200).
  2. Camera list retrieval (`GET /api/cameras` $\rightarrow$ 200).
  3. Camera creation with valid configuration (`POST /api/cameras` $\rightarrow$ 201).
  4. Event history pagination and total count calculation.
  5. Settings inspection and dynamic threshold update (`PUT /api/settings`).
- **Dashboard & Presentation (`test_dashboard.py`)**:
  1. Index page loads HTML5 with title and container hierarchy.
  2. Static CSS files served with correct `text/css` MIME type.
  3. Static JS client files accessible.
  4. Live stream placeholder img tags configured with streaming routes.
  5. Video modal element present for event playback.

### 4.2 Tier 2: Boundary & Corner Cases
Stress and boundary conditions that trigger failure in naive implementations:
1. **Zero / Empty Inputs**:
   - Querying events on empty SQLite database returns empty list and count 0.
   - Pushing 0 frames to circular buffer does not throw on `get_pre_roll_frames()`.
2. **Extreme Resolutions**:
   - Ultra-low resolution ($160\times120$) downscaling handles gracefully without dividing by zero.
   - High resolution ($3840\times2160$ 4K) processes through MOG2 downscale without OOM.
3. **Rapid Bursts & High FPS**:
   - Processing 100 frames in accelerated simulation mode does not leak thread handles or crash queue.
4. **Invalid Credentials & Network Errors**:
   - Invalid SMTP host or port triggers graceful error logging without crashing the FastAPI lifespan or alerting worker.
5. **Windows File Locks & Resource Cleanup**:
   - Rapidly creating, writing, and closing 10 consecutive MP4 files cleans up file handles immediately.

### 4.3 Tier 3: Cross-Feature Interactions (Pairwise)
Validates the communication boundaries between independent subsystems:
- **Pair 1: Ingestion $\rightarrow$ MOG2 $\rightarrow$ AI Trigger**:
  Frames emitted by `SyntheticCameraStream` pass into `MOG2MotionDetector`. When motion is detected, the pipeline automatically schedules inference and receives classified `DetectionBox` objects.
- **Pair 2: Motion Detection $\rightarrow$ Circular Buffer $\rightarrow$ Video Recording**:
  On detection trigger, `EventRecorder` drains 3–5s of pre-roll frames from `CircularFrameBuffer`, switches to `RECORDING` state, continues capturing live frames, and completes recording with 5s post-roll.
- **Pair 3: Event Finalization $\rightarrow$ SQLite Persistence $\rightarrow$ Alert Queue**:
  Once the MP4 file is written, an event record is inserted into SQLite WAL database, and an `AlertPayload` is dispatched to the background notifier queue.
- **Pair 4: Live MJPEG Streaming $\rightarrow$ Multiple Concurrent Clients**:
  Two concurrent HTTP clients subscribe to `/api/cameras/{id}/stream`. Both receive independent JPEG multipart frame streams without interfering with detection or recording.

### 4.4 Tier 4: Real-World Application Scenarios
Full end-to-end integration workflows exercising the complete stack:
- **Scenario 4.1: Residential Break-In Simulation**:
  1. System initializes with 1 synthetic camera running 5s pre-roll buffer.
  2. For $t \in [0.0, 3.0\text{s}]$, scene is quiet (background only).
  3. At $t = 3.0\text{s}$, a synthetic humanoid target enters the ROI and moves across the frame until $t = 6.0\text{s}$.
  4. Pipeline flags motion, AI confirms 'person' (confidence $> 0.85$).
  5. Post-roll extends recording until $t = 11.0\text{s}$.
  6. Assertions:
     - Exactly 1 MP4 clip created on disk, containing pre-roll ($t < 3\text{s}$) and post-roll ($t > 6\text{s}$).
     - SQLite contains 1 event row with valid relative paths to video and snapshot.
     - Mock SMTP captures 1 alert email with subject containing camera name and class 'person', plus an inline snapshot.
- **Scenario 4.2: SMB Parking Lot Vehicle Monitoring & Cooldown**:
  1. Vehicle target enters ROI at $t = 2.0\text{s}$.
  2. First alert is dispatched via email.
  3. Second vehicle enters at $t = 5.0\text{s}$ (within 60s cooldown window).
  4. Assertions:
     - Exactly 2 event records persisted in SQLite.
     - First event has `alert_status = 'sent'`.
     - Second event has `alert_status = 'suppressed_cooldown'`.
     - Mock SMTP captures exactly 1 email (preventing spamming).
- **Scenario 4.3: Night / Shadow False-Positive Rejection**:
  1. Synthetic feed introduces gradual luminance shifts and moving shadow shapes (pixel value 127).
  2. Assertions:
     - MOG2 shadow thresholding filters out all shadow pixels.
     - Zero AI inferences are dispatched.
     - Zero MP4 clips recorded, zero emails sent.
- **Scenario 4.4: Continuous Motion with Dynamic Post-Roll Extension**:
  1. Target moves, stops for 2 seconds (less than post-roll window), and moves again.
  2. Assertions:
     - Recording state machine resets post-roll deadline instead of finalizing early.
     - Results in 1 single continuous MP4 file rather than fragmented micro-clips.

---

## 5. Coverage Thresholds & Quality Metrics

| Metric | Target Standard | Enforcement Mechanism |
|---|---|---|
| **Automated Test Pass Rate** | **100%** (0 failures, 0 errors) | CI / `pytest` exit code 0 |
| **Requirements Coverage (R1–R6)** | **100%** mapped and verified | Traceability matrix in `TEST_INFRA.md` |
| **Feature Coverage (F01–F31)** | **100%** mapped to test suites | Pytest suite markers (`@pytest.mark.tier1` to `tier4`) |
| **Windows Resource Leaks** | **0** file locks (`[WinError 32]`) | Explicit fixture teardown and process cleanup |
| **Thread Concurrency Leaks** | **0** unjoined daemon/worker threads | Cooperative shutdown signals (`stop_event`) |
| **Hardware Independence** | **100%** hermetic execution | Synthetic cameras & Mock SMTP |

---

## 6. Test Execution Instructions

### 6.1 Running the Complete Automated Test Suite
From the project root:
```powershell
# Run all tests with standard summary
.\.venv\Scripts\pytest -v

# Run with verbose output and duration profiling
.\.venv\Scripts\pytest -v --durations=10
```

### 6.2 Targeted Tier Execution via Markers
```powershell
# Run only Tier 1 Feature Isolation tests
.\.venv\Scripts\pytest -v -m tier1

# Run only Tier 2 Boundary & Corner Case tests
.\.venv\Scripts\pytest -v -m tier2

# Run only Tier 3 Cross-Feature Interaction tests
.\.venv\Scripts\pytest -v -m tier3

# Run only Tier 4 Real-World E2E Scenario tests
.\.venv\Scripts\pytest -v -m tier4

# Run all End-to-End tests
.\.venv\Scripts\pytest -v -m e2e
```

### 6.3 Windows Troubleshooting Notes
- If `PermissionError: [WinError 32]` occurs during directory cleanup, ensure that any `cv2.VideoCapture` or `sqlite3.Connection` instances opened during tests are explicitly closed before exiting the test function.
- All temporary directories generated during test runs are placed in system temporary paths managed by pytest (`tmp_path`).
