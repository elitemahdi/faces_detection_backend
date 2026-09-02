# 👁️ AI Vision & Security Intelligence Backend

An enterprise-grade, asynchronous **Face Recognition, Multi-Object Tracking (MOT), and Spatial Surveillance Intelligence Microservice** built with **FastAPI**, **YOLOv8-Face / OpenCV YuNet**, **AdaFace ONNX**, **Supervision ByteTrack**, and **SQLAlchemy 2.0 (PostgreSQL / asyncpg)**.

---

## 🧠 AI Vision & Recognition Pipeline Architecture

The face recognition system implements a high-throughput, multi-stage pipeline designed for low-latency surveillance video analysis:

```mermaid
flowchart TD
    subgraph Ingestion["1. Video & Stream Ingestion"]
        A["Raw Input: Image, RTSP Stream, or Video File"]
    end

    subgraph Detection["2. Face Detection & 5-Point Landmarks"]
        A --> B["YOLOv8-Face (YoloFaceDetector) / OpenCV YuNet Fallback"]
        B -->|"Bounding Box (x, y, w, h)"| BB["Face Bounds & Score"]
        B -->|"5 Facial Landmarks"| LM["Eyes (2), Nose (1), Mouth Corners (2)"]
    end

    subgraph Tracking["3. Multi-Object Tracking & Resolution Gating"]
        BB & LM --> K["ByteTrack MOT: Kalman Filter & Track ID"]
        K --> GAT{"Resolution Check: min(w, h) >= 40px?"}
        GAT -- "Too Small (< 40px)" --> DIST["Track Person as Distant Visitor"]
        GAT -- "Close Enough (>= 40px)" --> C["cv2.estimateAffinePartial2D vs Reference 5 Points"]
    end

    subgraph Alignment["4. 2D Similarity Affine Transform"]
        C --> D["cv2.warpAffine: 112x112 Aligned RGB Crop"]
    end

    subgraph Recognition["5. Deep Feature Extraction"]
        D --> E["AdaFace IR-50 (ONNX Runtime) / SFace CPU Fallback"]
        E --> F["512-Dimensional L2-Normalized Embedding Vector"]
    end

    subgraph Search["6. Vector Search & Identity Verification"]
        F --> G["Vectorized Cosine Similarity Matrix Product"]
        G --> H{"Similarity Threshold Check"}
        H -- "Match Found" --> I["Identified User Profile"]
        H -- "Below Threshold" --> J["Unknown Visitor"]
    end

    subgraph Security["7. Spatial Security & Heatmap Analytics"]
        I & J & DIST --> L["Spatial Security Zones: Point-in-Polygon Check"]
        L --> M["Spatial Heatmap Density & Breach Event Logging"]
    end
```

### Stage-by-Stage Breakdown

#### 1. Face Detection & 5-Point Landmark Extraction: **YOLOv8-Face (`YoloFaceDetector`)**
- **Model**: `models/yolov8n_face.onnx` (with automatic fallback to `models/face_detection_yunet.onnx`).
- **Surveillance Advantages**:
  - Multi-scale Feature Pyramid Network (FPN) detects small, distant surveillance faces ($20 \times 20$ to $40 \times 40$ px).
  - Handles severe off-axis camera pitch angles ($> 30^\circ$) and head tilts.
  - Deep backbone remains robust against glare, motion blur, and low-light CCTV noise.
- **Output**: Bounding box $[x, y, w, h]$, confidence score, and **5 facial landmarks** (Right Eye, Left Eye, Nose Tip, Right Mouth, Left Mouth).

#### 2. Multi-Object Tracking & Distance Gating: **ByteTrack + Resolution Filter**
- **Kalman Filtering**: Maintains persistent `track_id` across occlusions and fast movement.
- **Distance Resolution Gating**: Minimum bounding box threshold ($\ge 40 \times 40$ px) prevents running expensive AdaFace feature extraction on distant, blurry faces, eliminating false positive biometric matches.

#### 3. Normalization & Affine Alignment: **OpenCV Affine Transform**
- **Standard Reference Coordinates**:
  ```python
  import numpy as np

  REFERENCE_5_POINTS = np.array([
      [38.2946, 51.6963],  # Right Eye
      [73.5318, 51.5014],  # Left Eye
      [56.0252, 71.7366],  # Nose Tip
      [41.5493, 92.3655],  # Right Mouth
      [70.7299, 92.2041],  # Left Mouth
  ], dtype=np.float32)
  ```
- **Transformation**: `cv2.estimateAffinePartial2D` calculates scale, rotation, and translation, followed by `cv2.warpAffine` producing a standard **$112 \times 112$** face crop.

#### 4. Deep Feature Extraction: **AdaFace IR-50 (`adaface_ir50.onnx`)**
- **Inference Engine**: ONNX Runtime with multi-threaded intra/inter-op execution.
- **Normalization**: Float32 tensor scaled to $[-1.0, 1.0]$ in $(B, 3, 112, 112)$ CHW format.
- **Output**: **512-dimensional $L_2$-normalized vector** ($\|v\|_2 = 1.0$).
- **CPU Fallback**: SFace ONNX (`models/face_recognition_sface.onnx`) for low-power edge CPU deployments.

#### 5. Vectorized Similarity Matching: **Cosine Matrix Multiplication ($Q \cdot E^T$)**
- Efficient dot-product matrix multiplication against all enrolled database vectors:
  $$\text{sim}(Q, E_i) = Q \cdot E_i \quad (\text{for } \|Q\|_2 = 1, \|E_i\|_2 = 1)$$
- **Default Threshold**: $\tau \approx 0.36 - 0.38$ for high-confidence biometric identification.

#### 6. Spatial Security Zones & Heatmap Engine
- **Spatial Security Zones**: Fast sub-millisecond point-in-polygon checks (`cv2.pointPolygonTest`) to enforce access policies (`BANNED`, `ADMIN`, `REGISTERED`, `DETECTED`).
- **Heatmap Accumulator**: Sub-second discrete coordinate accumulation with continuous Gaussian smoothing.

---

## 🌟 Key Features

### 1. Dual-Stage Deep Face Pipeline
- **YOLOv8-Face / YuNet Detectors**: Flexible detector architecture with automatic model discovery and 5-point landmark regression.
- **AdaFace ONNX Embedding**: Vectorized 512-dimensional facial feature extraction from $112 \times 112$ aligned crops.
- **Vectorized Similarity Search**: Fast dot-product matrix multiplication $(Q \cdot E^T)$ against L2-normalized enrolled facial vectors.

### 2. Multi-Object Tracking (MOT) with ByteTrack
- **Kalman Filtering**: Persistent `track_id` maintained across frame occlusions and rapid movements.
- **Identity Caching**: Deep embeddings are extracted once per track (and refreshed periodically every $N$ frames), eliminating 80%+ redundant ONNX compute.
- **Unique Visitor Analytics**: Computes true distinct visitor metrics in offline video jobs.

### 3. Spatial Security Zone Engine
- **Vectorized Containment**: High-speed sub-millisecond point-in-polygon verification (`cv2.pointPolygonTest`).
- **4-Tier Access Policies**:
  - 🚫 **BANNED**: Zero tolerance. Any human presence triggers an immediate breach alert.
  - 🔐 **ADMIN**: Restricted. Only recognized users with administrator roles are permitted.
  - 👤 **REGISTERED**: Authorized personnel only. Unregistered or unknown visitors trigger alerts.
  - 👁️ **DETECTED**: Public or monitored zone. Any detected face is verified and highlighted.
- **Crisp HUD Overlay Rendering**: Multi-pass rendering with semi-transparent alpha fills, 100% opaque solid outline borders, and centroid-positioned status tags.

### 4. Spatial Heatmap & Visual Intelligence
- **Additive Density Accumulation**: Discrete coordinate accumulation with continuous Gaussian kernel smoothing.
- **Multi-Dimensional Filtering**: Filter by date range, daily time-of-day clock window (e.g. `10:00` to `16:00`), specific user ID / name, recognition status, and zone ID.
- **Analytics & Export**: 24-hour activity curve, user leaderboard, zone traffic breakdown, and CSV data export.

### 5. High-Throughput Streaming & Video Studio
- **Dual Streaming Modes**: Standard MJPEG HTTP stream (`/feed`) and ultra-low-latency WebSocket feed (`/ws`) with bidirectional threshold control.
- **Offline Video Analysis**: Batch video processing with frame sampling, bounding box annotation, zone breach tracking, and thread-safe metadata indexing.

### 6. Production-Ready Async Architecture
- **SQLAlchemy 2.0 & asyncpg**: Native connection pooling (`pool_size=20`, `max_overflow=10`, `pool_recycle=3600`), `scalar_one_or_none()` unique lookups, and soft-delete filters.
- **High-Speed Direct Inserts**: Direct table inserts bypassing ORM unit-of-work tracking for high-throughput spatial event logging.
- **Decoupled Frame Cache**: Non-blocking in-memory frame cache (`FrameCache`) for instant snapshot rendering without camera contention.

---

## 📡 API Endpoints Overview

| Category | Method & Endpoint | Description |
|---|---|---|
| **System** | `GET /health` | Service health status check |
| **Cameras** | `GET /api/v1/cameras/` | List all registered cameras |
| **Cameras** | `POST /api/v1/cameras/` | Register a new camera device or RTSP stream |
| **Cameras** | `GET /api/v1/cameras/{id}` | Get camera configuration details |
| **Cameras** | `PUT /api/v1/cameras/{id}` | Update camera configuration details |
| **Cameras** | `DELETE /api/v1/cameras/{id}` | Soft-delete camera |
| **Cameras** | `GET /api/v1/cameras/{id}/feed` | Live MJPEG annotated video stream |
| **Cameras** | `WS /api/v1/cameras/{id}/ws` | Bidirectional WebSocket stream & threshold control |
| **Cameras** | `GET /api/v1/cameras/{id}/snapshot` | Single annotated JPEG snapshot |
| **Zones** | `GET /api/v1/cameras/{id}/zones` | List security zones for a camera |
| **Zones** | `POST /api/v1/cameras/{id}/zones` | Create a normalized polygon security zone |
| **Zones** | `PUT /api/v1/zones/{id}` | Update zone coordinates or security policy |
| **Zones** | `DELETE /api/v1/zones/{id}` | Soft-delete security zone |
| **Heatmap** | `GET /api/v1/cameras/{id}/heatmap` | Query raw density matrix and KPI statistics |
| **Heatmap** | `GET /api/v1/cameras/{id}/heatmap/image` | Render high-resolution JPEG heatmap image overlay |
| **Heatmap** | `GET /api/v1/cameras/{id}/heatmap/export` | Export spatial detection events to CSV |
| **Heatmap** | `POST /api/v1/cameras/{id}/heatmap/generate-sample-data` | Seed sample heatmap detection data |
| **Users** | `GET /api/v1/users/` | List registered user profiles |
| **Users** | `POST /api/v1/users/` | Enroll user with portrait photo and extract 512-dim embedding |
| **Users** | `GET /api/v1/users/{id}` | Get user profile details |
| **Users** | `PUT /api/v1/users/{id}` | Update user profile details or replace photo |
| **Users** | `DELETE /api/v1/users/{id}` | Soft-delete user profile |
| **Videos** | `POST /api/v1/videos/annotate` | Upload and process video with MOT & zone breach checks |
| **Videos** | `GET /api/v1/videos/` | List processed video jobs and statistics |
| **Videos** | `GET /api/v1/videos/{id}` | Download annotated video file |
| **Videos** | `DELETE /api/v1/videos/{id}` | Delete processed video file |

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root:

```env
# Database Configuration (PostgreSQL with asyncpg driver)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/faces_detection

# Storage Configuration
STORAGE_PATH=storage
VIDEOS_STORAGE_PATH=storage/videos
PHOTOS_STORAGE_PATH=storage/photos

# AI Model Paths & Selection
DETECTOR_BACKBONE=auto                  # "yolo", "yunet", or "auto"
YOLO_MODEL_PATH=models/yolov8n_face.onnx
YUNET_MODEL_PATH=models/face_detection_yunet.onnx
ADAFACE_MODEL_PATH=models/adaface_ir50.onnx
SFACE_MODEL_PATH=models/face_recognition_sface.onnx

# Surveillance Quality & Distance Gating
MIN_FACE_SIZE_FOR_RECOGNITION=40        # Minimum px before running AdaFace
MATCH_THRESHOLD=0.38
TRACK_ACTIVATION_THRESHOLD=0.35
LOST_TRACK_BUFFER=30
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or `pip`
- PostgreSQL 14+

### Setup & Run
```bash
# 1. Clone the repository
git clone https://github.com/your-username/faces_detection.git
cd faces_detection

# 2. Set up virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# 3. Start PostgreSQL and verify tables
# Tables will be automatically created on startup via FastAPI lifespan

# 4. Start the development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🧪 Testing

Run backend tests using Python:

```bash
# Run unit and integration tests
pytest
```
