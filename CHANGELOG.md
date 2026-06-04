# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.9] - 2026-06-04

### Console Backend
- Sweep all `.get()` calls on Stripe `_StripeObject` payloads in `console/backend/app/routes/billing.py` (the 0.2.8 fix only patched the webhook entry point; 11 more sites inside `_handle_checkout_completed` and `_handle_subscription_updated` had the same dict-API assumption). Added a `_stripe_get` helper that handles both `dict` and `_StripeObject`.

### Security
- `.dockerignore`: dropped the `!.env.production` negation that forced the production secrets file into the Docker build context (CLOUDFLARE_TUNNEL_TOKEN, VIOLAWAKE_ADMIN_TOKEN, Stripe keys). No current Dockerfile copied it, but a future `COPY . .` would silently bake secrets into the published image. Runtime env still comes from `docker compose --env-file .env.production`.

### Docs / ADRs
- ADR-006 supersedes ADR-001: multi-runtime inference (ONNX + TFLite) instead of ONNX-only.
- ADR-007 supersedes ADR-002: temporal-CNN wake head on frozen OWW backbone instead of MLP-on-OWW.
- `docs/REGISTRY.md` marks ADR-001/002 as Superseded; adds ADR-006/007 as Accepted.
- `benchmark_v2/README.md` documents the three-layer reproducibility model (score-CSV reproducer → corpus → TTS rebuild) so Lane 5's "claim from clean clone" path is explicit.
- `scripts/generate_docs.py` API-public-surface ratchet: pdoc emits `id="<name>"`, not `id="violawake_sdk.<name>"`. Fixed the anchor pattern; 25 SDK symbols now verified.

## [0.2.8] - 2026-06-04

### Console Backend
- Fix Stripe webhook handler `event.get("id")` AttributeError — `construct_event` returns a `_StripeObject` not a dict; use subscript access. Reachable on every real Stripe webhook delivery after the library version bump.

### Packaging
- Release workflow: install `violawake_sdk` editable in the `update-docs` job so `tools/update_model_registry.py` can import the package (was `ModuleNotFoundError`).

## [0.2.7] - 2026-06-04

### SDK
- Add the documented `ModelCache` wrapper in `violawake_sdk.models` while preserving the existing `get_model_path()`, `download_model()`, and `list_cached_models()` functions.
- Restore the `violawake` compatibility package in built wheels so `import violawake` matches the README contract. **The 0.2.6 wheel shipped without it; users hitting `ModuleNotFoundError: violawake` on `pip install violawake==0.2.6` should upgrade to 0.2.7.**
- Include the documented `WakewordDetector` compatibility alias in `violawake_sdk.__all__` so `from violawake_sdk import *` exposes the full documented top-level surface.
- Wake detector audio sources now fail closed on contract drift (non-16 kHz mono, wrong frame stride). OWW backbone integrity check fails closed on hash mismatch.
- VoicePipeline surfaces no-op STT, TTS-misconfiguration, and always-on VAD failures instead of silently returning to idle. STT engine preserves the real `faster-whisper` transitive import cause rather than reporting "package missing."
- Training pipeline enforces the 16 kHz mono audio contract at the loader; mis-rated audio fails fast before embedding extraction. Fixes `_train_mlp_on_oww` `training_start` NameError reachable on every successful MLP-on-OWW training path.

### Packaging
- Add release-wheel smoke coverage that installs the built wheel, imports `violawake` and `violawake_sdk`, verifies all `violawake-*` console-script targets are present, and runs `violawake-download --help` from the wheel install.
- Exclude `_diag/` audit artifacts from source distributions.

### Browser / WASM
- Fix WASM/Python score parity bug: `Math.trunc` instead of `Math.round` for float→int16 conversion (was producing score drift across the same audio).
- Demo page (`/wasm/demo/`): force single-threaded ORT to avoid CSP `blob:` script-source rejection on the deployed site.

### Console (SaaS) & Frontend
- Add public-claim reproducer (`benchmark_v2/reproduce_claims.py`) and Ratchet gate so every headline benchmark number traces to a checked-in script + corpus at a pinned model SHA.
- Live backend oracle now uses the actual API contract (`/api/billing/checkout`, `resource_id` in `DownloadTokenRequest`) — earlier oracle pointed at nonexistent routes/fields.
- SPA route rewrite preserves `/login`, `/register`, `/dashboard` against Cloudflare clean-URL canonicalization (was collapsing back to `/`).
- Public copy aligns with reproducible claims; unreproducible operator benchmark numbers removed.

### Operations
- R2 Postgres backup restore drill (`scripts/backup_restore_drill.py`) with stale-backup detection + non-prod scratch-container restore + `--inspect-only` SQL sanity asserting both CREATE TABLE and COPY public.* sections are present.
- Stale deploy-launch helper retired in favor of the documented `docs/DEPLOYMENT.md` flow.
- Quality-gate framework: `quality/gates.yaml` registry, mechanical pre-commit hooks (`scripts/check_no_direct_main_commits.py`, `scripts/check_ratchet_rule.py`), CI workflow.

### Governance
- Rewrite `CLAUDE.md` and add `docs/LANE_LEDGER.md` for the 2026-06-03 audit cycle (12 disjoint lanes following PMBOK WBS + DDD bounded contexts + business capability mapping).

## [0.2.6] - 2026-05-08

### Console (SaaS)
- Surface A/B/C/F model `quality_grade` values in the console model list so trained-model quality is visible without opening the detailed performance view.
- Add multi-file recording bulk upload plus the `violawake-generate` sample-generation workflow, with per-user upload rate limiting on recording ingest.
- Add recording storage caps, WAV/FLAC magic-byte validation, canonical 16 kHz mono PCM_16 re-encoding, and append-only upload audit logging for accepted and rejected uploads.

## [0.2.5] - 2026-05-07

### SDK
- `violawake_sdk.__version__` now reads from `importlib.metadata.version("violawake")` instead of a hardcoded constant. v0.2.4 was published to PyPI as `0.2.4` but reported `__version__ == "0.2.2"` because the constant had not been bumped — this kind of drift is now impossible.
- Quality gate (`tools.train`) — silence subgrade now uses a near-silence (1e-4 RMS gaussian) fallback when pure-silence audio is rejected by the OWW backbone. Previously, `silence_window_count == 0` left `silence_max_score` defaulted to 0.0, which silently passed the silence subgrade for Grade A/B even though no silence test had run. Models now actually have to be checked against a low-energy input. If both pure silence and near-silence produce zero embeddings, the gate forces Grade F as a safety floor. Adds `silence_source` field to the model `config_json` (`"silence" | "near_silence" | "none"`) so downstream tools can audit which path ran.

### Console (SaaS)
- Per-event progress dispatch in the training job queue (`console/backend/app/job_queue.py`) is now tolerant of transient stalls: a single progress write that takes >10s (e.g. during backend restart warmup) used to abort the entire training job with `error_reason=timeout`. Bumped to 60s and TimeoutError is now caught — a stalled progress event drops the event but keeps the job running. Job 51 on 2026-05-07 was killed this way after a backend recreate; future restarts no longer have this blast radius.

### CI
- `release.yml` (introduced in v0.2.4 fix) now actually verified end-to-end by this release (v0.2.4 had to be manually uploaded via twine because the workflow was untested).

## [0.2.4] - 2026-05-07

### SDK
- Pin `torch.onnx.export` to the legacy (non-dynamo) exporter in `export_temporal_onnx`. torch 2.10+ defaults to the dynamo exporter, which has no ONNX dispatcher for `aten.adaptive_max_pool2d` (the lowering of `nn.AdaptiveMaxPool1d(1)` used by `TemporalCNN`). Without this pin, training jobs failed at the export step with `DispatchError: No ONNX function found for aten.adaptive_max_pool2d`. The legacy path supports it and matches how the production reference model was originally exported.
- Add `onnxscript` to the Console backend requirements so the new exporter has its dependency satisfied even when the legacy path is used.

### Tests
- `tests/live/full_pipeline_e2e.py`: use the literal wake word string (`"viola"`) instead of a unique-per-run identifier. The training pipeline synthesizes additional TTS positives by literally speaking the configured `wake_word` across multiple voices, so a unique identifier produced gibberish positives and degraded the trained model.
- Extend the live training-status polling deadline from 15 minutes to 60 minutes to match real CPU training time.

## [0.2.3] - 2026-05-07

### SDK
- Auto-download OpenWakeWord backbone files on first use. Previously, `pip install violawake[oww]` left users with `ModelNotFoundError` until they manually ran `openwakeword.utils.download_models()`. Now the SDK calls it automatically when the backbone files are missing on first `WakeDetector(...)` call. Existing installations are unaffected.
- Update `oww_backbone` SHA-256 in MODEL_REGISTRY to the current upstream openwakeword release. Eliminates the spurious "OWW backbone hash mismatch" warning that appeared on freshly-installed environments.

### Console (SaaS)
- Stripe LIVE mode activated for production. Test mode remains supported via env-var swap.
- Postgres backups: daily `pg_dump` to Cloudflare R2 bucket `violawake-backups`, 30-day retention via R2 lifecycle rule.
- WASM browser-SDK demo at https://violawake.com/wasm/demo/ now ships with the OpenWakeWord backbone + ViolaWake temporal_cnn ONNX files pre-hosted; no user input required.
- Backend Content-Security-Policy header now set on all API responses (defence-in-depth).
- Cloudflare Pages frontend now serves CSP, HSTS, Permissions-Policy, COEP/COOP via `_headers`.
- Rate-limit keying switched from `request.client.host` (Cloudflare edge IP — collapsed all users to a few IPs) to `CF-Connecting-IP` (real per-user IP). `REGISTER_LIMIT` raised from 10/hour to 100/hour.
- Marketing landing page accuracy claim now explicitly framed as "upper bound on production reference model; your custom model varies with sample quantity/quality".

### Added
- VAD `process_frame()` and `is_speech()` now accept `np.ndarray` (float32, float64, int16) in addition to bytes
- `_coerce_to_bytes()` shared helper for input normalization across VAD backends
- API docs generation setup with pdoc (`docs` optional dependency)

### Security
- Add login timing oracle protection (dummy bcrypt on non-existent accounts)
- Make password reset tokens single-use via JTI tracking
- Require password confirmation for account deletion
- Handle chunked transfer encoding in body size middleware (ASGI receive wrapper)
- Add Stripe webhook idempotency (bounded event ID cache, 1000 entries)
- Use bounded OrderedDict for download token JTIs (10K cap + TTL pruning)
- Make usage counter atomic (SQL-level `SET count = count + 1`, no read-modify-write race)
- Cancel Stripe subscription on account deletion
- Replace abandoned `python-jose[cryptography]` dependency with `PyJWT[crypto]>=2.8`

### Fixed
- Fix SSE training stream (use `addEventListener("training", ...)` instead of `onmessage`)
- Fix nginx port mismatch (`EXPOSE 80`, compose port `80:80`)
- Correct Docker env var names to use `VIOLAWAKE_` prefix
- Add Alembic migration for `failed_login_count`, `locked_until`, `deleted_at` columns
- Add production frontend service to `docker-compose.production.yml`
- Wire `send_training_complete()` email on training job completion
- Wire `send_quota_warning()` email at 80% usage
- Add trial status display to Billing page
- Fix `UploadResponse` type (add `wake_word` field)
- Fix double `await` in ModelCard download handler
- Enable team members to download shared models (not just view)
- Use UUID in recording filenames (eliminate concurrent upload collision)
- Simplify CORS origin configuration to 3 clear branches
- Preserve return path in protected route redirects (`?return=` param)

## [0.2.2] - 2026-04-05

### Fixed
- Silence quality gate bug: zero-energy audio correctly rejected by OWW backbone now scores 0.0 instead of 1.0 (was causing false Grade F on perfectly good models)
- Training pipeline consistency: patience=15 everywhere (CLI, SDK, Console — was 10 in some paths)
- Console training service: added `augment_source_files` parameter and repo-root corpus search path to match CLI pipeline
- Standalone `train_full_pipeline.py`: same fixes as Console for full parity

## [0.2.1] - 2026-03-30

### Added
- Kokoro TTS fallback when Edge TTS is unavailable
- `temporal_convgru` reserve model in registry
- Registry integrity checking (`check_registry_integrity()`)

### Changed
- `r3_10x_s42` MLP model marked DEPRECATED in registry (fails live mic test, max score 0.50)
- Removed `viola_mlp_oww` and `viola_cnn_v4` from registry (never uploaded to GitHub Releases)

## [0.2.0] - 2026-03-28

### Added
- **TemporalCNN production model** (`temporal_cnn`): 9-frame sliding window over OWW embeddings, d'=8.577, EER 0.8%, AUC 0.9993 — replaces MLP as default
- 8-phase training pipeline: user positives → TTS (20 voices x 3 phrases) → audiomentations augmentation → confusable negatives R1 (30 words) → R2 (16 words) → speech negatives (104 phrases) → universal corpus (LibriSpeech, MUSAN) → TemporalCNN training
- Post-training quality gate with A/B/C/F grading (Grade F blocks ONNX export)
- FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05) with AdamW + CosineAnnealingLR + EMA
- Group-aware stratified train/val split preventing augmentation data leakage
- Auto-evaluation on held-out 20% test set
- `docs/PROVEN_TRAINING_RECIPE.md` — canonical pipeline documentation

### Changed
- Default production model: `temporal_cnn` (was `r3_10x_s42` MLP)
- Model alias `"viola"` now resolves to `temporal_cnn`

## [0.1.0] - 2026-03-27

### Added
- Wake word detection with 4-gate decision policy (silence guard, threshold, cooldown, playback suppression) plus optional multi-window confirmation
- 3-of-3 multi-window confirmation reducing FAPH by 87%
- Production default model: temporal_cnn (EER 5.49%, best live recall + lowest FP)
- VAD engine with 3-backend fallback (WebRTC -> Silero -> RMS)
- STT integration via faster-whisper (5 model sizes)
- TTS integration via Kokoro-82M with sentence-chunked streaming
- VoicePipeline orchestrating Wake->VAD->STT->TTS
- Training CLI (violawake-train) with augmentation pipeline (gain, time stretch, pitch shift, additive noise, time shift)
- Evaluation CLI (violawake-eval) with EER, ROC AUC, FAPH, FRR@FAR operating points
- `violawake-generate` CLI for headless TTS sample generation
- `violawake-expand-corpus` CLI for LibriSpeech/MUSAN corpus download
- `[generate]` optional extra for sample generation without the full training stack
- Quality gate with A/B/C/F grading post-training
- Auto-evaluation with a held-out 20% test set
- Safe tarball extraction with zip-slip protection
- Model registry with SHA-256 verification and auto-download
- Confusable word generation for adversarial testing

### Fixed
- SDK inference path rewritten to use correct OWW 2-model pipeline
- Critical normalization fix: mel model expects int16-range float32, output needs mel/10+2 transform
- Without normalization fix, FAPH was 65x worse (783/h vs 12/h)

### Security
- SHA-256 model integrity verification
- HTTPS-only model downloads
