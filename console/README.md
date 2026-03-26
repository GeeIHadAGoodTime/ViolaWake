# ViolaWake Console

Web-based Console for training custom wake word models. Record 10 voice samples (augmented to 110+ training samples), train a model, and download the .onnx file -- all from your browser.

> **Note:** 10 recordings work well for personal, single-speaker wake word detection. For multi-speaker or production deployment, collect 50+ samples from different speakers and environments.

## Quick Start

```bash
# 1. Install everything
python console/launch.py --install

# 2. Start the Console
python console/launch.py

# 3. Open your browser
#    Frontend: http://localhost:5173
#    Backend API docs: http://localhost:8000/docs
```

## Architecture

```
Browser (React)  ──→  FastAPI Backend  ──→  ViolaWake SDK Training
   port 5173            port 8000           (CPU or GPU)
       │                    │
   RecordRTC           SQLite DB
   16kHz WAV          JWT Auth
   wavesurfer.js      File Storage
```

## User Flow

1. **Register** — Create an account (email + password)
2. **Login** — Get a JWT token
3. **Record** — Record 10 samples of your wake word (1.5s each, augmented to 110+ for training)
4. **Train** — Click "Start Training" — backend runs the ViolaWake SDK pipeline
5. **Monitor** — Watch training progress in real-time (SSE)
6. **Download** — Get your custom `.onnx` model file
7. **Use** — Load the model with the ViolaWake SDK:

```python
from violawake_sdk import WakeDetector

detector = WakeDetector(model="path/to/your_model.onnx", threshold=0.80)
for chunk in detector.stream_mic():
    if detector.detect(chunk):
        print("Wake word detected!")
```

## Development

### Backend

```bash
cd console/backend
pip install -r requirements.txt
python run.py
# Runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Frontend

```bash
cd console/frontend
npm install
npm run dev
# Runs at http://localhost:5173
```

### Testing

```bash
# Backend unit tests (no server needed)
pytest console/tests/test_backend.py -v

# API E2E tests (starts backend automatically)
python console/run_e2e.py --api-only

# Browser E2E tests (starts both servers, requires Playwright)
python console/run_e2e.py --install  # one-time: installs Playwright
python console/run_e2e.py

# Quality gates
python tools/quality_gate.py --all
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/auth/register` | - | Create account |
| POST | `/api/auth/login` | - | Get JWT token |
| GET | `/api/auth/me` | JWT | Get user profile |
| POST | `/api/recordings/upload` | JWT | Upload WAV file |
| GET | `/api/recordings` | JWT | List recordings |
| POST | `/api/training/start` | JWT | Start training job |
| GET | `/api/training/status/:id` | JWT | Get job status |
| GET | `/api/training/stream/:id` | JWT | SSE progress stream |
| GET | `/api/models` | JWT | List trained models |
| GET | `/api/models/:id/download` | JWT | Download .onnx |
| GET | `/api/models/:id/config` | JWT | Get model config |
