# Security Policy

## Reporting Vulnerabilities
Report security vulnerabilities to security@useviola.com or via GitHub Security Advisories.

## Security Defaults
- Model downloads use HTTPS with SHA-256 integrity verification
- Network audio sources bind to localhost (127.0.0.1) by default
- No pickle serialization — speaker profiles use JSON + numpy .npz
- Certificate pinning infrastructure available (see src/violawake_sdk/security/)

## Model Integrity
Models are verified against SHA-256 hashes in the model registry. If a hash mismatch is detected, the download is rejected and the corrupted file is deleted.

## Console Security

The ViolaWake Console (web dashboard for training and managing wake word models) implements the following security measures:

- **JWT authentication with bcrypt password hashing** — all passwords stored as bcrypt hashes; JWTs issued on login with configurable expiry
- **Login timing oracle protection** — failed logins against non-existent accounts perform a dummy bcrypt hash to prevent timing-based user enumeration
- **Account lockout** — 5 consecutive failed logins trigger a 15-minute lockout period
- **Single-use password reset tokens** — reset tokens tracked by JTI (JWT ID); each token can only be used once
- **Account deletion requires password confirmation** — prevents unauthorized deletion via stolen session tokens; also cancels any active Stripe subscription
- **Rate limiting on all auth and billing endpoints** — prevents brute-force and abuse
- **Body size enforcement including chunked transfers** — 15 MB limit enforced via ASGI receive wrapper, covering both Content-Length and chunked Transfer-Encoding
- **Stripe webhook signature verification + idempotency** — webhook payloads verified against Stripe signing secret; bounded event ID cache (1,000 entries) prevents duplicate processing
- **Atomic usage counters** — quota tracking uses SQL-level `SET count = count + 1` to eliminate read-modify-write race conditions
- **Bounded in-memory caches** — download token JTIs (10K cap + TTL pruning) and webhook event IDs (1K cap) use bounded OrderedDicts to prevent memory exhaustion
- **Security headers in production** — HSTS, X-Content-Type-Options, X-Frame-Options enforced
- **Non-root Docker containers** — production containers drop to a non-root user via gosu entrypoint
- **Per-user SSE connection limits** — prevents a single user from exhausting server-sent event connections
- **Training job queue limits** — per-user caps on concurrent training jobs to prevent resource monopolization
