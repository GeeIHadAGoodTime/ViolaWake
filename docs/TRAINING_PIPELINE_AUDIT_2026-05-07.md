# Training Pipeline Audit - 2026-05-07

Scope: `src/violawake_sdk/` and `console/backend/`, focused on latent fresh-deploy failures in the synchronous training path and adjacent SDK evaluation/runtime paths.

Summary counts: CRITICAL 2, HIGH 4, MEDIUM 4, LOW 0.

One-line fixes applied:
- `0c3923b` - pinned remaining SDK `OWWModel` evaluation/contamination helpers to ONNX.
- `9cbb8e6` - made backend startup fail after failed Alembic retries and made OpenWakeWord resource download failures visible to `set -e`.

## Findings

### 1. CRITICAL - Alembic retry loop could start the app after every migration attempt failed

File/lines: `console/backend/entrypoint.sh:8-15`

Evidence:
```sh
for i in 1 2 3 4 5; do
    if alembic upgrade head; then
        break
    fi
    echo "[entrypoint] alembic try $i failed; retrying in 3s..."
    [ "$i" = "5" ] && exit 1
    sleep 3
done
```

Before `9cbb8e6`, the loop had no final failure exit, so the fifth failed try fell through and the backend could start against an unmigrated or unavailable database. Proposed fix: applied in `9cbb8e6`; keep this fail-fast behavior and consider logging a clearer final "migration failed" line. Estimated effort: done; optional polish 15 minutes.

### 2. CRITICAL - SDK training helpers can call `sys.exit(1)` inside the backend worker path

File/lines: `src/violawake_sdk/tools/train.py:1060-1063`, `src/violawake_sdk/tools/train.py:1159-1165`, `src/violawake_sdk/tools/train.py:1203-1208`, `console/backend/app/job_queue.py:834`

Evidence:
```py
except ImportError as e:
    print(f"ERROR: PyTorch required for training: {e}", file=sys.stderr)
    print("Install with: pip install 'violawake[training]'", file=sys.stderr)
    sys.exit(1)

if len(pos_embs) < 5:
    ...
    sys.exit(1)

if len(all_neg_embs) < 5:
    ...
    sys.exit(1)
```
```py
except Exception as exc:
    ...
```

`SystemExit` inherits from `BaseException`, not `Exception`, so these library-path exits bypass the job failure handler. A fresh deploy with missing training deps, bad embeddings, or no usable negatives can leave the task exception unhandled instead of marking the job failed cleanly. Proposed fix: split CLI exits from library behavior. Make `_train_temporal_cnn` and other callable helpers raise `RuntimeError`/typed exceptions, and let `main()` convert them to exit codes. As a short-term guard, catch `SystemExit` at the backend boundary and mark the job failed without swallowing `asyncio.CancelledError`. Estimated effort: 0.5-1 day with tests.

### 3. HIGH - Remaining `OWWModel()` defaults would still select the broken TFLite path

File/lines: `src/violawake_sdk/training/evaluate.py:264`, `src/violawake_sdk/training/evaluate.py:327`, `src/violawake_sdk/tools/contamination_check.py:114`

Evidence:
```py
oww = OWWModel(inference_framework="onnx")
```

Pre-fix grep found default constructors at those sites. They are adjacent to training/evaluation workflows and would have failed the same way as `tools/train.py` when the container has a `tflite_runtime` that cannot read the current OpenWakeWord `.tflite` schema. Proposed fix: applied in `0c3923b`; keep `grep -R "OWWModel()" src/violawake_sdk console/backend` as a regression check. Estimated effort: done.

### 4. HIGH - OpenWakeWord resource pre-download was masked by shell pipeline behavior

File/lines: `console/backend/entrypoint.sh:22-23`

Evidence:
```sh
echo "[entrypoint] ensuring openwakeword backbone models are downloaded"
python -c "from openwakeword.utils import download_models; download_models()"
```

Before `9cbb8e6`, this command was piped through `tail -3 || true`. In POSIX `sh`, pipeline status comes from `tail`, so a failing Python download could still return success and the app would become healthy until first training. Proposed fix: applied in `9cbb8e6`; if output trimming is needed later, use a shell that supports `pipefail` or capture logs without hiding exit status. Estimated effort: done.

### 5. HIGH - Synchronous training still depends on external Edge TTS, and the offline fallback is not deployed

File/lines: `src/violawake_sdk/tools/train.py:278`, `src/violawake_sdk/tools/train.py:547`, `src/violawake_sdk/tools/train.py:587`, `src/violawake_sdk/tools/train.py:418-434`, `console/Dockerfile.backend:20`, `pyproject.toml:87-100`

Evidence:
```py
communicate = edge_tts.Communicate(text, voice)
ok = _edge_tts_synthesize(word, voice, out_path)
ok = _edge_tts_synthesize(phrase, voice, out_path)
```
```dockerfile
RUN pip install --no-cache-dir "/sdk[training]"
```

The production image installs `violawake[training]`, which includes `edge-tts` but not the `tts` extra (`kokoro-onnx`). `_generate_tts_positives()` has a Kokoro fallback probe, but `TTSEngine` only loads `kokoro-onnx` and its large model files on first synthesis. Confusable and speech negatives do not try Kokoro at all. On a customer machine without internet, behind a proxy, or during Edge rate limiting, quality drops and training can fail if the mounted corpus is absent or too small. Proposed fix: make offline training explicit. Either ship/pre-download an offline TTS backend and its model assets into the image/cache, or remove online TTS from the synchronous path and require a mounted negative/positive corpus with a startup readiness check. Estimated effort: 1-2 days.

### 6. HIGH - Production job queue persists to local SQLite even when the app database is Postgres

File/lines: `console/backend/app/database.py:17-23`, `console/backend/app/job_queue.py:151`, `docker-compose.production.yml:36`

Evidence:
```py
DATABASE_URL = settings.db_url.strip() if settings.db_url and settings.db_url.strip() else ...
self._db_path = db_path or (settings.data_dir / "job_queue.db")
```

The app data model can use `VIOLAWAKE_DB_URL=postgresql+asyncpg://...`, but training job state is always an `aiosqlite` file under `/app/data`. The production compose file has a volume, so single-node Docker is survivable. Fresh deploys without a persistent `/app/data` volume, horizontal replicas, or platform restarts can lose pending/running job state while the main database remains intact. Proposed fix: move the queue tables into Postgres or add an explicit deployment guard that refuses multi-replica/ephemeral-volume production mode. Estimated effort: 1-2 days.

### 7. MEDIUM - Universal negative corpus is optional in the console path but quality-critical

File/lines: `console/backend/app/services/training_service.py:230-270`, `src/violawake_sdk/tools/train.py:2472-2485`, `docker-compose.production.yml:25-32`

Evidence:
```py
_CORPUS_SEARCH_PATHS = [
    Path(__file__).resolve().parent.parent.parent.parent / "corpus",
    Path.home() / ".violawake" / "corpus",
    Path("corpus"),
]
if total_neg < 5:
    raise RuntimeError("Only %s negative files generated. edge-tts may not be installed or network unavailable." % total_neg)
```

The CLI warns when no universal corpus exists, but the backend only hard-fails when total negatives fall below five. If Edge TTS produces enough synthetic negatives, training can succeed without LibriSpeech/MUSAN and create high false-positive models. Production compose mounts `./corpus:/app/corpus:ro`, but Docker will create an empty host directory if it is missing. Proposed fix: add a backend warning/progress event and a startup/health check for minimum corpus counts, or require `VIOLAWAKE_NEGATIVES_CORPUS_DIR`/`/app/corpus` in production. Estimated effort: 0.5 day.

### 8. MEDIUM - Quota enforcement is clean at limit but not atomic with job submission

File/lines: `console/backend/app/routes/billing.py:252-308`, `console/backend/app/routes/jobs.py:101-120`

Evidence:
```py
if used >= limit:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, ...)
...
job_id = await queue.submit_job(...)
await record_usage(db, current_user.id, action="training_job")
```

If a user is already at quota, the route returns a clean 403, not a 500. The latent issue is concurrency and partial failure: two requests at `limit - 1` can both pass `check_training_quota()`, and a queue insert can succeed before `record_usage()` fails or increments. Proposed fix: reserve usage atomically before queue submission, or put queue submission and usage reservation behind one database transaction/outbox. Estimated effort: 0.5-1 day.

### 9. MEDIUM - Health/readiness does not validate training prerequisites

File/lines: `console/backend/app/health.py:122-155`, `console/backend/app/config.py:195-198`

Evidence:
```py
upload_dir = _check_directory(settings.upload_dir)
models_dir = _check_directory(settings.models_dir)
component_status = _combine_statuses(upload_dir["status"], models_dir["status"])
```

Runtime dirs are created at import, and health checks database, queue, uploads, and models. It does not check `tmp_dir` writability, OpenWakeWord resource presence, universal corpus counts, or importability of training-only dependencies. After the entrypoint fixes, OWW download is a startup prerequisite in Docker, but local/fresh SDK starts and non-Docker runs can still report healthy before first training fails. Proposed fix: add a training readiness component with cheap checks and a more expensive admin-only preflight. Estimated effort: 0.5 day.

### 10. MEDIUM - SDK model cache defaults to `Path.home()` and is not bound to the backend data volume

File/lines: `src/violawake_sdk/models.py:27`, `src/violawake_sdk/models.py:152-158`, `src/violawake_sdk/wake_detector.py:631-637`, `src/violawake_sdk/tts.py:128-129`

Evidence:
```py
DEFAULT_MODEL_DIR = Path.home() / ".violawake" / "models"
model_dir = Path(os.environ.get("VIOLAWAKE_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
model_dir.mkdir(parents=True, exist_ok=True)
```

SDK inference and Kokoro TTS auto-download into the process home directory unless `VIOLAWAKE_MODEL_DIR` is set. The backend Dockerfile/compose do not set that env var to `/app/data/models`, so downloaded SDK assets can be ephemeral or unwritable depending on platform user/home behavior. Proposed fix: set `VIOLAWAKE_MODEL_DIR=/app/data/sdk-models` in backend production config and document the same for fresh SDK deploys that disable internet access. Estimated effort: 15-30 minutes plus deploy validation.

## Lazy Import Inventory

`src/violawake_sdk/tools/train.py`: lazy imports at lines 68, 227, 267-272, 290, 304, 322, 343-344, 363-366, 400-418, 472-473, 522, 604-606, 634-637, 686-689, 721-724, 820, 865-871, 950, 1056-1071, 1087-1088, 1396-1397, 1447, 1538-1543, 1728-1750, 2066-2068, 2451, 2586. Risky runtime-only deps here: `openwakeword`, `edge_tts`, `pydub`, `torchaudio`, `audiomentations`, `torch`, `onnx`, `onnxruntime`, `scipy`, `violawake_sdk.tts`/Kokoro.

`src/violawake_sdk/oww_backbone.py`: lazy imports at lines 126 (`openwakeword.utils.download_models`) and 187 (`MODEL_REGISTRY`). The resource download is now enforced in Docker entrypoint, but SDK/local usage still auto-downloads on first missing-resource access.

Backend services/platform lazy imports: `console/backend/app/services/training_service.py:100` imports SDK training helpers only when a job runs; `app/storage.py:151-152` and `198` import `boto3`/`botocore` only when R2 is configured or queried; `app/routes/billing.py:69`, `273`, `277` import Stripe/email/asyncio lazily; `app/routes/recordings.py:128-156` imports `io`, `numpy`, and `scipy` during upload validation/resampling; `app/middleware.py:111-112` imports Sentry only when configured; `app/main.py:42`, `76`, `170` imports retention/email services at startup/admin execution; `app/job_queue.py:117`, `791`, `1170`, `1195` imports subscription/email/retention helpers at priority, completion, and cleanup time.

## External Network Dependencies

Synchronous training path: Edge TTS at `src/violawake_sdk/tools/train.py:278`, OpenWakeWord `download_models()` at `src/violawake_sdk/oww_backbone.py:126-132` and `console/backend/entrypoint.sh:23`, optional Kokoro model downloads via `get_model_path()` in `src/violawake_sdk/tts.py:128-129`, and optional object storage calls through R2 in `console/backend/app/storage.py:151-217`.

SDK/download path: model auto-download via `urllib.request.urlopen()` in `src/violawake_sdk/models.py:197-267`, explicit downloads via `requests.get()` in `src/violawake_sdk/models.py:483`, corpus expansion via `requests.get()` in `src/violawake_sdk/tools/expand_corpus.py:89`, and certificate-pinned downloads in `src/violawake_sdk/security/cert_pinning.py:664-706`.

## Filesystem Expectations

Required backend paths: `console/backend/app/config.py:30-35` defines `data_dir`, `db_path`, `upload_dir`, `models_dir`, and `tmp_dir`; `config.py:195-198` creates them at import. `job_queue.py:151` requires a writable local `data/job_queue.db`; `training_service.py:72`, `107`, and `job_queue.py:707` require `settings.tmp_dir` for temporary training work.

Optional but quality-critical corpus paths: `training_service.py:231-234`, `tools/train.py:2424-2428`, and `console/backend/scripts/train_full_pipeline.py:198-202` search repo `corpus`, `~/.violawake/corpus`, and CWD `corpus`. Absence is not fatal unless too few negatives remain.

SDK model cache path: `src/violawake_sdk/models.py:27` and `152-158` use `~/.violawake/models` unless `VIOLAWAKE_MODEL_DIR` is set.
