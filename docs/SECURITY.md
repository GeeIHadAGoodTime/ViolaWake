# Security Notes

## Recording Upload Hardening

Recording uploads are accepted only as WAV or FLAC, verified by extension and
magic bytes before decode. Accepted audio is decoded with `soundfile` only,
then stored as canonical 16 kHz mono PCM_16 WAV with a server-generated UUID
filename. Original upload bytes are discarded.

Upload storage caps are stacked:

- Per file: 5 MB.
- Per user per rolling 24 hours: 50 MB on `free`; 500 MB on `developer` and `business`.
- Per user lifetime: 200 MB on `free`; 2 GB on `developer` and `business`.
- Global `/app/data` volume: reject uploads if the app data directory is above
  50 GB used or the backing filesystem has less than 5 GB free.

Decode runs behind a 30 second watchdog. The current watchdog is thread-based:
it bounds request wall-clock time, but a tighter memory cap requires subprocess
isolation and OS resource limits. That process sandbox is deferred to Tier 3.

Every upload attempt writes an append-only JSON line to
`/app/data/logs/uploads.jsonl`; the backend rotates it to `uploads.jsonl.1`
when it reaches 100 MB. Fields include `ts`, `user_id`, Cloudflare
`X-Forwarded-For` client IP fallback to `client.host`, claimed filename,
stored UUID filename when accepted, claimed MIME type, first 12 magic bytes,
decode status, recording id, and wake word. Storage and rate cap denials that
occur before decode are recorded with `decode_status="cap_rejected"`.

## Tier 3 Deferred

These are deliberately not part of the current patch and should be treated as
the next hardening tier:

- Move decode and canonical re-encode into a subprocess with explicit memory,
  CPU, file descriptor, and wall-clock limits.
- Run the backend container with `--read-only` plus explicit writable mounts for
  `/app/data` and temporary decode space.
- Drop unnecessary Linux capabilities and use `no-new-privileges`.
- Consider a dedicated decoder sidecar so hostile media parsing is isolated
  from the authenticated API process and database credentials.

## Cloudflare WAF Rules

Two of three target rules are deployed on the `violawake.com` zone (Free plan).
The third needs a paid plan; backend cap covers it. Original spec preserved
below for posterity / paid-plan upgrade path.

**Deployed (live)**
- Geo-block `/api/*` outside US/CA/EU/EEA — `http_request_firewall_custom`,
  ruleset `bf95b91734b44cd9a2f7f9324a90285a`.
- Rate-limit `/api/recordings/*` at 17 req per 10s per IP+colo (≈100/min),
  10s mitigation block — `http_ratelimit`, ruleset
  `bb8c390ce5da4feca02dbf375b0d286e`. Free plan only allows `period=10s` and
  `mitigation_timeout=10s`.

**Not deployed**
- Block oversized recording-upload bodies at edge: requires WAF Advanced plan
  (Free plan rejects `http.request.body.size` filter with `not entitled`). The
  backend's 5 MB per-file cap (returns 413, audit-logged) covers this without
  Cloudflare. Upgrade to Pro+ to push enforcement to the edge.

### Original ruleset spec (apply by hand or via API on paid plan)

1. Block oversized recording uploads.

   Expression:

   ```text
   (http.request.uri.path matches "^/api/recordings/(upload|bulk-upload)" and http.request.body.size > 6291456)
   ```

   Action: `Block`

2. Rate-limit recording endpoints at the edge.

   Expression:

   ```text
   (http.request.uri.path matches "^/api/recordings/")
   ```

   Setting: `100 requests / 60 seconds`, characteristic `IP`, action `Block`
   or managed challenge.

3. Geo-block countries the service does not intentionally serve.

   Default suggestion: allow US, Canada, and the EU/EEA only, then adjust this
   list in the dashboard for real customer geography.

   Expression:

   ```text
   (http.request.uri.path matches "^/api/" and not ip.geoip.country in {"US" "CA" "AT" "BE" "BG" "HR" "CY" "CZ" "DK" "EE" "FI" "FR" "DE" "GR" "HU" "IE" "IT" "LV" "LT" "LU" "MT" "NL" "PL" "PT" "RO" "SK" "SI" "ES" "SE" "IS" "LI" "NO"})
   ```

   Action: `Block`
