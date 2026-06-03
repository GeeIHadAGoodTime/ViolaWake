# Lane 9 audit report - SaaS Console Frontend

Verdict: **FAIL for the currently deployed live site**.

The branch contains a frontend build-chain fix for the live auth/console routing failure, but I did not deploy it. Per `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md:65` and `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md:69`, production deploys are out of scope for this audit.

## Scope read

- Lane question: `docs/LANE_LEDGER.md:536` asks whether `violawake.com` renders sign-up, console, comparison, pricing, and docs pages and talks to the live backend.
- Lane 9 owns `console/frontend/`, generated `console/frontend/dist/`, `tests/live/test_live_website.py`, the accessibility baseline, deploy-pages workflow, and SEO audit material (`docs/LANE_LEDGER.md:544`).
- Required frontend success criteria include production API URL baking, render checks, and accessibility baseline non-regression (`docs/LANE_LEDGER.md:561`, `docs/LANE_LEDGER.md:569`).
- Required negative probes must be constructed and run, not only reviewed (`_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md:19`).

## MUST-FIX found

### MUST-FIX 1 - deployed auth and console routes collapse to `/`

Live `/login`, `/register`, `/dashboard`, and pricing plan register links do not render their intended auth/console pages. They return a console shell, but the browser ends on `/` and renders the marketing homepage, so the sign-up and console surfaces are not reachable.

Evidence:

```powershell
python _diag/2026-06-03/lane9_live_probe.py
```

Excerpt:

```text
"pagesChecked": 20,
"failures": 6,
"url": "https://violawake.com/register",
"finalUrl": "https://violawake.com/",
"expectedTextMissing": ["Create account"],
"expectedSelectorMissing": ["#name", "#email", "#password"]
...
"url": "https://violawake.com/login",
"finalUrl": "https://violawake.com/",
"expectedTextMissing": ["Welcome back"],
"expectedSelectorMissing": ["#email", "#password"]
```

Direct redirect evidence:

```powershell
curl.exe -L -I https://violawake.com/login
```

Excerpt:

```text
HTTP/1.1 308 Permanent Redirect
Location: /app
HTTP/1.1 200 OK
```

The frontend generator previously wrote SPA route rewrites through `app.html`; the branch fix now uses a directory index app shell. Relevant fixed code:

- `console/frontend/scripts/generate-marketing.mjs:16`
- `console/frontend/scripts/generate-marketing.mjs:646`
- `console/frontend/scripts/generate-marketing.mjs:662`

Post-fix generated rewrite evidence:

```powershell
Select-String -Path 'console/frontend/dist/_redirects' -Pattern '/login|/register|/dashboard'
```

Output:

```text
console\frontend\dist\_redirects:15:/login /app/index.html 200
console\frontend\dist\_redirects:16:/register /app/index.html 200
console\frontend\dist\_redirects:20:/dashboard /app/index.html 200
```

## Build and backend evidence

Relevant source:

- `console/frontend/package.json:8` runs `tsc && vite build && node scripts/generate-marketing.mjs`.
- `console/frontend/src/api.ts:24` sets `BASE_URL` from `import.meta.env.VITE_API_URL || "/api"`, so an absent env build is the known broken shape.

Production build:

```powershell
$env:VITE_API_URL='https://api.violawake.com/api'; cmd /c npm run build
```

Excerpt:

```text
vite v5.4.21 building for production...
81 modules transformed.
dist/assets/index-DVsxgtKt.js   313.24 kB
Generated 15 marketing routes and markdown mirrors in .../console/frontend/dist
```

Production bundle probe:

```powershell
python _diag/2026-06-03/lane9_bundle_api_probe.py console/frontend/dist --expect production
```

Output:

```text
PASS: production API base is baked into JS assets: console\frontend\dist\assets\index-DVsxgtKt.js
```

Absent-env negative probe:

```powershell
cmd /c npx.cmd vite build --mode lane9-absent --outDir ../../_diag/2026-06-03/lane9_dist_absent --emptyOutDir
python _diag/2026-06-03/lane9_bundle_api_probe.py _diag/2026-06-03/lane9_dist_absent --expect absent-env-detected
```

Output:

```text
vite v5.4.21 building for lane9-absent...
dist/assets/index-CATSciWZ.js   313.22 kB
PASS: absent VITE_API_URL fallback detected in JS assets: _diag\2026-06-03\lane9_dist_absent\assets\index-CATSciWZ.js
```

Local built-bundle backend target probe:

```powershell
python _diag/2026-06-03/lane9_local_bundle_backend_probe.py
```

Output excerpt:

```text
"expectedApiLogin": "https://api.violawake.com/api/auth/login",
"apiRequests": [
  "POST https://api.violawake.com/api/auth/login"
]
```

The local probe sees expected CORS errors because the production API does not allow `127.0.0.1` as an origin. That is not a deployed-site failure; it still proves the built bundle targets the production backend host.

## Live render evidence

The strengthened live probe covered 20 same-origin pages discovered from the required set. Public marketing/docs pages rendered without console errors or 5xx. Auth/console routes failed by landing on `/`.

Passing examples from the same run:

```text
"url": "https://violawake.com/pricing",
"finalUrl": "https://violawake.com/pricing/",
"status": 200,
"consoleErrors": [],
"expectedTextMissing": []
...
"url": "https://violawake.com/docs",
"finalUrl": "https://violawake.com/docs/",
"status": 200,
"consoleErrors": [],
"expectedTextMissing": []
```

Backend live login could not be exercised because the live login form is absent:

```powershell
python _diag/2026-06-03/lane9_live_backend_probe.py
```

Output excerpt:

```text
"loginUrl": "https://violawake.com/login",
"finalUrl": "https://violawake.com/",
"error": "Page.fill: Timeout 5000ms exceeded..."
```

## Accessibility baseline

Baseline source: `tests/live/ACCESSIBILITY_AUDIT_2026-05-07.md:9` through `tests/live/ACCESSIBILITY_AUDIT_2026-05-07.md:14` records scores of `/` 95, `/pricing` 93, `/register` 95, `/login` 95, `/privacy` 90, `/terms` 90. It also says the baseline was not WCAG-AA clean (`tests/live/ACCESSIBILITY_AUDIT_2026-05-07.md:5`) and documents known color/link/heading failures (`tests/live/ACCESSIBILITY_AUDIT_2026-05-07.md:16`).

LHCI failed on this Windows host after each Lighthouse run due Chrome profile cleanup `EPERM`, but direct Lighthouse wrote JSON before that cleanup failure. I parsed those JSON files and did not retain the bulky generated artifacts in this branch. Parsed results:

```powershell
cmd /c npx.cmd --yes lighthouse <url> --only-categories=accessibility --output=json --output-path=_diag/2026-06-03/lighthouse-<slug>.json --chrome-flags="--headless=new --disable-gpu --no-sandbox --disable-dev-shm-usage"
```

Parsed output:

```text
root: score=0.95 baseline=0.95 failed=['color-contrast']
pricing: score=0.95 baseline=0.93 failed=['color-contrast']
privacy: score=0.95 baseline=0.90 failed=['color-contrast']
terms: score=0.94 baseline=0.90 failed=['color-contrast']
register: score=1.00 baseline=0.95 failed=[]
login: score=1.00 baseline=0.95 failed=[]
```

No numeric accessibility-score regression was observed. The `/register` and `/login` accessibility scores are not meaningful pass evidence because those live routes render the wrong page; the render MUST-FIX supersedes them.

## Public claims

I did not find an incorrect public numeric or factual claim while doing the render/build audit. I did not attempt a Lane 5 reproducibility audit of the public numbers; any correction there should route to Lane 5 or Lane 11.

## Planned gates

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: frontend-api-url-baked
contract: Production frontend JS bundles must contain https://api.violawake.com/api, and an absent VITE_API_URL build must be detected by the same-origin "/api" fallback.
detector: _diag/2026-06-03/lane9_bundle_api_probe.py
own_tests:
  - _diag/2026-06-03/lane9_bundle_api_probe.py
  - _diag/2026-06-03/lane9_bundle_api_probe.py
```

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: frontend-spa-route-render
contract: Auth and console routes must preserve their requested route and render expected SPA selectors instead of falling through to the marketing homepage.
detector: _diag/2026-06-03/lane9_live_probe.py
own_tests:
  - _diag/2026-06-03/lane9_live_probe.py
  - _diag/2026-06-03/lane9_local_bundle_backend_probe.py
```

## Self-audit gate

- I did not deploy the fixed bundle to production; production-destructive deploy actions are forbidden in this audit, so the live site remains FAIL until an authorized deploy.
- I did not run Firefox or Safari. The dispatch prompt narrowed the browser requirement to latest stable Chrome/headless fallback for this audit, and Chrome found a blocking render issue.
- I did not complete real billing or Stripe checkout. Charging or triggering real billing is forbidden by the common corrections, and Lane 9's MUST-FIX bar does not require live card flow.
- I did not exhaustively reproduce public numeric claims. Lane 9 additional rules route incorrect public claims to Lane 5 or Lane 11, and this audit was a render/build/backend-target audit.
- I did not perform a full manual keyboard or screen-reader pass. I ran Lighthouse accessibility baseline checks and route-specific browser probes; full manual assistive-tech certification remains outside this lane pass.
