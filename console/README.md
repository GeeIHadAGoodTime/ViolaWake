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

### Trusted Proxy Configuration

Auth rate limiting uses the direct connection IP by default. If you deploy behind
trusted reverse proxies, set `VIOLAWAKE_TRUSTED_PROXY_COUNT` to the number of
trusted proxies in front of the app. When set to a value greater than `0`, the
backend uses the Nth-from-right `X-Forwarded-For` entry for rate limiting; when
set to `0`, `X-Forwarded-For` is ignored entirely.

Examples:

- `VIOLAWAKE_TRUSTED_PROXY_COUNT=0`: trust only the socket peer IP
- `VIOLAWAKE_TRUSTED_PROXY_COUNT=1`: trust one reverse proxy and use the rightmost forwarded IP
- `VIOLAWAKE_TRUSTED_PROXY_COUNT=2`: trust two reverse proxies and use the second IP from the right

## API Endpoints

### Auth

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/auth/register` | - | Create account |
| POST | `/api/auth/login` | - | Get JWT token |
| GET | `/api/auth/me` | JWT | Get user profile |
| POST | `/api/auth/verify-email` | - | Verify email from signed token |
| POST | `/api/auth/forgot-password` | - | Send password reset email |
| POST | `/api/auth/reset-password` | - | Reset password from signed token |
| POST | `/api/auth/change-password` | JWT | Change authenticated user's password |
| POST | `/api/auth/download-token` | JWT | Issue short-lived download/SSE token |
| DELETE | `/api/auth/account` | JWT | Delete account (requires password confirmation) |

### Recordings

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/recordings/upload` | JWT | Upload WAV file |
| GET | `/api/recordings` | JWT | List recordings |
| DELETE | `/api/recordings/:id` | JWT | Delete a recording |

### Training

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/training/start` | JWT | Start training job |
| GET | `/api/training/status/:id` | JWT | Get job status |
| GET | `/api/training/stream/:id` | JWT/Token | SSE progress stream |

### Models

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/models` | JWT | List trained models |
| GET | `/api/models/:id/download` | JWT/Token | Download .onnx file |
| GET | `/api/models/:id/config` | JWT | Get model config and metrics |
| GET | `/api/models/:id/performance` | JWT | Get model performance details |
| DELETE | `/api/models/:id` | JWT | Delete a trained model |

### Jobs

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/jobs` | JWT | Submit a new training job |
| GET | `/api/jobs` | JWT | List user's training jobs |
| GET | `/api/jobs/:id` | JWT | Get a single training job |
| DELETE | `/api/jobs/:id` | JWT | Cancel a pending/running job |
| POST | `/api/jobs/resume` | JWT | Resume paused queue after circuit breaker |
| GET | `/api/jobs/circuit-breaker/state` | JWT | Get circuit breaker state |

### Billing

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/billing/checkout` | JWT | Create Stripe checkout session |
| POST | `/api/billing/webhook` | Stripe | Handle Stripe webhook events |
| GET | `/api/billing/subscription` | JWT | Get subscription and usage |
| POST | `/api/billing/portal` | JWT | Create Stripe billing portal session |
| GET | `/api/billing/usage` | JWT | Get current month's usage vs limit |

### Teams

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/teams` | JWT | Create a new team |
| GET | `/api/teams` | JWT | List user's teams |
| GET | `/api/teams/:id` | JWT | Get team details |
| POST | `/api/teams/:id/invite` | JWT | Invite a user by email |
| POST | `/api/teams/:id/join` | JWT | Accept invite via signed token |
| DELETE | `/api/teams/:id/members/:user_id` | JWT | Remove a team member |
| PATCH | `/api/teams/:id/members/:user_id` | JWT | Change a member's role |
| POST | `/api/teams/:id/models/:model_id/share` | JWT | Share a model with the team |
| GET | `/api/teams/:id/models` | JWT | List team's shared models |
| DELETE | `/api/teams/:id` | JWT | Delete a team (owner only) |

### Files

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/files/:key` | JWT/Token | Serve locally stored file |

### Health

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/health` | - | Health summary |
| GET | `/api/health/live` | - | Liveness check |
| GET | `/api/health/ready` | - | Readiness check |
| GET | `/api/health/details` | Admin | Detailed health (requires X-Admin-Token) |

### Admin

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/admin/cleanup` | Admin | Trigger retention cleanup (requires X-Admin-Token) |
