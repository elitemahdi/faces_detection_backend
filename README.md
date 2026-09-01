# 👁️ AI Vision & Security Intelligence Backend

An enterprise-grade, asynchronous **Face Recognition, Multi-Object Tracking (MOT), and Spatial Surveillance Intelligence Microservice** built with **FastAPI**, **OpenCV YuNet**, **AdaFace ONNX**, **Supervision ByteTrack**, and **SQLAlchemy 2.0 (PostgreSQL / asyncpg)**.

---

## 🌟 Key Features

### 1. Dual-Stage Deep Face Pipeline
- **Stage 1 — YuNet Detector**: Fast, lightweight OpenCV DNN model running on C++ engine for face detection and 5-point landmark regression (eyes, nose tip, mouth corners).
- **Stage 2 — AdaFace ONNX Embedding**: Vectorized 512-dimensional facial feature extraction from $112 \times 112$ aligned crops with automatic affine transformation and SFace fallback.
- **Vectorized Similarity Search**: Fast dot-product matrix multiplication $(Q \cdot E^T)$ against L2-normalized enrolled facial vectors.

### 2. Multi-Object Tracking (MOT) with ByteTrack
- **Kalman Filtering**: Persistent `track_id` maintained across frame occlusions and rapid movements via `supervision.ByteTrack`.
- **Identity Caching**: Deep embeddings are extracted once per track (and refreshed periodically every $N$ frames) rather than on every frame, eliminating 80%+ redundant ONNX inference compute.
- **Unique Visitor Analytics**: Computes true distinct visitor metrics (`len(unique_track_ids)`) in offline video jobs.

### 3. Spatial Security Zone Engine
- **Vectorized Containment**: High-speed sub-millisecond point-in-polygon verification (`cv2.pointPolygonTest`).
- **4-Tier Access Policies**:
  - 🚫 **BANNED**: Zero tolerance. Any human presence triggers an immediate breach alert.
  - 🔐 **ADMIN**: Restricted. Only recognized users with administrator roles are permitted.
  - 👤 **REGISTERED**: Authorized personnel only. Unregistered or unknown visitors trigger alerts.
  - 👁️ **DETECTED**: Public/Monitored zone. Any detected face is verified and highlighted.
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
- **High-Speed Direct Inserts**: `insert(DetectionEvent).values(...)` bypassing ORM unit-of-work tracking for high-throughput spatial event logging.
- **Decoupled Frame Cache**: Non-blocking in-memory frame cache (`FrameCache`) for instant snapshot rendering without camera contention.

---

## 🏗️ Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Capture["Video / Stream Ingestion"]
        CAM[Camera Feed / RTSP Stream]
        VID[Offline Video Upload]
    end

    subgraph Pipeline["AI Vision & Tracking Pipeline"]
        YUNET["YuNet Face Detector (5-pt Landmarks)"]
        ALIGN["112x112 Affine Alignment & Crop"]
        BYTETRACK["ByteTrack MOT (Kalman Filter)"]
        CACHE["Track Identity Cache"]
        ADAFACE["AdaFace ONNX (512-dim Embedding)"]
    end

    subgraph Security["Security & Analytics Engine"]
        ZONE["Zone Policy Engine (Point-in-Polygon)"]
        HEATMAP["Spatial Heatmap Accumulator"]
        DB[(PostgreSQL / asyncpg)]
    end

    subgraph Egress["Clients & UI Dashboard"]
        WS["WebSocket Stream (/ws)"]
        MJPEG["MJPEG Feed (/feed)"]
        REST["REST API & CSV Export"]
    end

    CAM --> YUNET
    VID --> YUNET
    YUNET --> BYTETRACK
    BYTETRACK --> CACHE
    CACHE -- Miss / Recheck --> ALIGN --> ADAFACE --> CACHE
    CACHE --> ZONE
    ZONE --> HEATMAP
    HEATMAP --> DB
    ZONE --> WS
    ZONE --> MJPEG
    HEATMAP --> REST
```

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status |
| **Cameras** | | |
| `GET` | `/api/v1/cameras/` | List all registered cameras |
| `POST` | `/api/v1/cameras/` | Register a new camera device / RTSP stream |
| `GET` | `/api/v1/cameras/{id}` | Get camera details |
| `PUT` | `/api/v1/cameras/{id}` | Update camera details |
| `DELETE`| `/api/v1/cameras/{id}` | Soft-delete camera |
| `GET` | `/api/v1/cameras/{id}/feed` | Live MJPEG annotated video stream |
| `WS` | `/api/v1/cameras/{id}/ws` | Bidirectional WebSocket stream & threshold control |
| `GET` | `/api/v1/cameras/{id}/snapshot` | Single annotated JPEG snapshot |
| **Zones** | | |
| `GET` | `/api/v1/cameras/{id}/zones` | List security zones for a camera |
| `POST` | `/api/v1/cameras/{id}/zones` | Create a normalized polygon zone |
| `PUT` | `/api/v1/zones/{id}` | Update zone coordinates or policy |
| `DELETE`| `/api/v1/zones/{id}` | Soft-delete security zone |
| **Heatmap & Analytics** | | |
| `GET` | `/api/v1/cameras/{id}/heatmap` | Query raw density matrix & KPI statistics |
| `GET` | `/api/v1/cameras/{id}/heatmap/image` | Render high-res JPEG heatmap image overlay |
| `GET` | `/api/v1/cameras/{id}/heatmap/export`| Export spatial detection events to CSV |
| `POST` | `/api/v1/cameras/{id}/heatmap/generate-sample-data` | Seed sample heatmap detection data |
| **Users & Biometrics** | | |
| `GET` | `/api/v1/users/` | List registered users |
| `POST` | `/api/v1/users/` | Enroll user with portrait photo & extract 512-dim embedding |
| `GET` | `/api/v1/users/{id}` | Get user details |
| `PUT` | `/api/v1/users/{id}` | Update user details / replace photo |
| `DELETE`| `/api/v1/users/{id}` | Soft-delete user |
| **Video Processing** | | |
| `POST` | `/api/v1/videos/annotate` | Upload and process video with MOT & zone breach checks |
| `GET` | `/api/v1/videos/` | List processed video jobs & stats |
| `GET` | `/api/v1/videos/{id}` | Download annotated video file |
| `DELETE`| `/api/v1/videos/{id}` | Delete processed video file |

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root:

```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/faces_detection

# Storage Configuration
STORAGE_PATH=storage
VIDEOS_STORAGE_PATH=storage/videos
PHOTOS_STORAGE_PATH=storage/photos

# AI Model Files
FACE_DETECTION_MODEL=models/face_detection_yunet_2023mar.onnx
FACE_RECOGNITION_MODEL=models/adaface_ir50_webface4m.onnx

# Match & Threshold Parameters
MATCH_THRESHOLD=0.38
TRACK_ACTIVATION_THRESHOLD=0.40
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
