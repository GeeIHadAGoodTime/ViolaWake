# Lane 3 Audit Report - Browser Wake Detection (WASM)

## Verdict

**MUST-FIX.**

Branch source now has a passing local WASM/Python parity harness and catches the two Lane 3 negative probes, but the currently deployed `https://violawake.com/wasm/demo/` still fails before model load because the live CSP blocks ONNX Runtime Web's blob-loaded WASM worker/module. I did not deploy, push, or merge.

Also: no parity tolerance was documented in-repo, so this audit used the dispatch default: score-space `L_inf <= 1e-3`.

## Implemented fixes

1. **JS audio contract is now enforced.**
   `wasm/src/detector.ts:22`, `wasm/src/detector.ts:215`, and `wasm/src/detector.ts:361` validate every public `getScore()`/`detect()` frame as exactly 320 finite `Float32Array` samples in `[-1, 1]`.

2. **Wrong classifier models now fail fast.**
   `wasm/src/detector.ts:147`, `wasm/src/detector.ts:151`, and `wasm/src/detector.ts:329` validate classifier input shape against `[1, 96]` or `[1, seq_len, 96]`; `wasm/src/detector.ts:346` validates scalar output shape when metadata is available.

3. **JS float-to-int16 conversion now matches Python.**
   Python casts `float * 32767` to `int16`, truncating toward zero. `wasm/src/features.ts:177` now uses `Math.trunc()` instead of `Math.round()`.

4. **Live smoke now probes the deployed WASM route.**
   `tests/live/test_live_wasm.py:25` adds `/wasm/demo/`, `tests/live/test_live_wasm.py:47` fixes the `asyncio.run()`-inside-running-loop issue, and `tests/live/test_live_wasm.py:109` reports ONNX URLs plus console errors on failure.

5. **Demo source avoids the live CSP blob-worker failure mode.**
   `wasm/demo/index.html:300` and `console/frontend/public/wasm/demo/index.html:300` force `ort.env.wasm.numThreads = 1`; line 301 in both files sets `ort.env.wasm.proxy = false`. Rebuilt `wasm/dist/*` and synced the tracked public WASM dist files.

6. **Added durable audit scripts.**
   `wasm/scripts/wasm_parity_audit.py` orchestrates Python-reference scoring, Node/WASM scoring, bundle metrics, and negative probes. `wasm/scripts/run_wasm_scores.mjs` runs the built bundle in Node. `wasm/scripts/live_wasm_smoke.py` runs a bounded browser smoke for deployed or local demo URLs.

## Evidence

### Parity and negative probes

Command:

```bat
cmd /c python wasm/scripts/wasm_parity_audit.py --json
```

Output excerpt:

```json
{
  "tolerance": 0.001,
  "parity": {
    "max_linf": 1.621246337890625e-05,
    "pass": true
  },
  "corpus": {
    "name": "deterministic_speech_like",
    "sample_count": 10,
    "sample_frames": 19200,
    "sample_rate": 16000
  },
  "negative_probes": [
    {
      "name": "wrong_frame_stride_160_samples",
      "caught": true,
      "excerpt": [
        "RangeError: audioBuffer must contain exactly 320 samples (20ms at 16000Hz); got 160."
      ]
    },
    {
      "name": "wrong_classifier_model_embedding_model",
      "caught": true,
      "excerpt": [
        "Error: Classifier model has unsupported input shape [\"unk__314\",76,32,1]; expected [1, 96] or [1, seq_len, 96]."
      ]
    }
  ]
}
```

Important limitation: this is not a real-human-audio PASS. The repo has no checked-in WAV/FLAC/MP3 corpus, and `--prefer-edge-tts --json` fell back with:

```text
edge-tts failed (No audio was received. Please verify that your parameters are correct.)
```

The harness accepts `--corpus-dir` and should be rerun against a real 10-sample corpus before marking Lane 3 PASS.

### Bundle and latency metrics

Same command output excerpt:

```json
{
  "bundle_metrics": {
    "bundle_js_bytes": 22705,
    "melspectrogram_onnx_bytes": 1087958,
    "embedding_model_onnx_bytes": 1326578,
    "temporal_cnn_onnx_bytes": 102378,
    "model_asset_total_bytes": 2516914
  },
  "node": {
    "load_ms": 640.761,
    "first_frame_score_ms": 0.4934000000000651,
    "first_temporal_score_audio_ms": 720,
    "first_temporal_score_call_ms": 24.797199999999975,
    "frame_latency": {
      "count": 600,
      "p50_ms": 0.019099999999980355,
      "p95_ms": 10.10089999999991,
      "p99_ms": 27.36460000000011,
      "max_ms": 48.677599999999984
    }
  }
}
```

The wake head is 102,378 bytes as documented, but I found no explicit WASM bundle-size or first-detection-latency bar. If the Python SDK p99 frame target (`<= 15ms`) is treated as the WASM bar, this run is not green (`p99_ms=27.36`) on this memory-constrained Windows host.

### Model identity

Command:

```bat
Get-FileHash -Algorithm SHA256 console\frontend\public\wasm\models\*.onnx
```

Relevant output:

```text
melspectrogram.onnx  BA2B0E0F8B7B875369A2C89CB13360FF53BAC436F2895CCED9F479FA65EB176F
embedding_model.onnx 70D164290C1D095D1D4EE149BC5E00543250A7316B59F31D056CFF7BD3075C1F
temporal_cnn.onnx    9C0B12C68593CFDB3D320A3B34667913B18D63E89EB01247D6332D7839AC9EFE
```

Command:

```bat
set PYTHONPATH=src
python -c "from violawake_sdk.oww_backbone import get_openwakeword_backbone_hashes; print(get_openwakeword_backbone_hashes('onnx'))"
```

Output:

```text
{'oww_mel_sha256': 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f', 'oww_emb_sha256': '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f'}
```

### Deployed live smoke

Command:

```bat
cmd /c set VIOLAWAKE_LIVE=1&& python -m pytest tests/live/test_live_wasm.py::test_wasm_demo_requests_onnx_models --no-cov -ra -vv
```

Current deployed output excerpt:

```text
FAILED tests/live/test_live_wasm.py::test_wasm_demo_requests_onnx_models
AssertionError: {
  'console_errors': [
    'Loading the script \'blob:https://violawake.com/...\' violates the following Content Security Policy directive: "script-src \'self\' \'unsafe-inline\' https://js.stripe.com https://cdn.jsdelivr.net https://static.cloudflareinsights.com https://plausible.io"...'
  ],
  'onnx_urls': []
}
```

Command:

```bat
cmd /c python wasm/scripts/live_wasm_smoke.py
```

Current deployed output excerpt:

```json
{
  "url": "https://violawake.com/wasm/demo/",
  "status_text": "Error \u2014 see log",
  "onnx_request_count": 0,
  "request_failures": [
    "GET blob:https://violawake.com/...: csp"
  ],
  "pass": false,
  "log_text": "Ready.\\n\\n[...] Loading models...\\n[...] Error: no available backend found. ERR: [wasm] TypeError: Failed to fetch dynamically imported module: blob:https://violawake.com/..."
}
```

### Patched local smoke

Command:

```powershell
$proc = Start-Process -FilePath python -ArgumentList @('-m','http.server','8765','--bind','127.0.0.1','--directory','console/frontend/public') -PassThru -WindowStyle Hidden
try {
  Start-Sleep -Seconds 2
  python wasm/scripts/live_wasm_smoke.py --url http://127.0.0.1:8765/wasm/demo/ --origin http://127.0.0.1:8765
} finally {
  Stop-Process -Id $proc.Id -Force
}
```

Output excerpt:

```json
{
  "pass": true,
  "status_text": "Listening\u2026 say \"Viola\"",
  "onnx_request_count": 3,
  "console_errors": [],
  "request_failures": [],
  "log_text": "Ready.\\n\\n[...] Loading models...\\n[...] Models loaded.\\n[...] Microphone access granted.\\n[...] Detection running.\\n[...] Wake word detected! score=0.853  (detection #1)"
}
```

### Build/typecheck

Command:

```bat
cmd /c npm.cmd run typecheck
```

Output:

```text
> violawake@0.1.0 typecheck
> tsc --noEmit
```

Command:

```bat
cmd /c npm.cmd run typecheck && npm.cmd run build
```

Output excerpt:

```text
> violawake@0.1.0 build
> rollup -c rollup.config.mjs
created dist/violawake.js
created dist/violawake.cjs
```

Command:

```bat
python -m py_compile wasm\scripts\wasm_parity_audit.py wasm\scripts\live_wasm_smoke.py
```

Output: no output, exit code 0.

## Planned Gates

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: wasm-python-score-parity
contract: Built WASM and Python SDK must score the same 10 shared 16k mono 20ms-frame samples within score-space L_inf tolerance, and must catch wrong frame stride plus wrong classifier model probes.
detector: wasm/scripts/wasm_parity_audit.py
own_tests:
  - python wasm/scripts/wasm_parity_audit.py --json
  - python wasm/scripts/wasm_parity_audit.py --corpus-dir <real-10-sample-corpus> --json
```

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: wasm-live-demo-model-load
contract: The deployed WASM demo must reach listening state, request all three ONNX models, and emit no browser console or request failures.
detector: wasm/scripts/live_wasm_smoke.py
own_tests:
  - python wasm/scripts/live_wasm_smoke.py
  - python wasm/scripts/live_wasm_smoke.py --url http://127.0.0.1:8765/wasm/demo/ --origin http://127.0.0.1:8765
```

## Remaining MUST-FIX items

1. **Current production is still red.** Deploying this branch, or equivalent ORT single-thread/no-proxy demo settings, is required before the live smoke can pass.
2. **Real-audio parity is not established.** The repo has no checked-in real audio corpus; `edge-tts` produced no audio in this environment. Rerun `wasm/scripts/wasm_parity_audit.py --corpus-dir <real-10-sample-corpus> --json`.
3. **WASM-specific size/latency bars are not explicitly documented.** The report includes actual bytes and latency. A lane PASS needs a clear WASM bundle and first-detection threshold, or an explicit decision that the Python SDK bars apply.
4. **If Python SDK p99 frame latency (`<=15ms`) is applied to WASM, this host run fails.** Node/WASM p99 was `27.36ms` on this constrained Windows session.

## Mandatory self-audit gate

- I did not exhaustively probe natural human speech because no real audio corpus is present in the repo and `edge-tts` returned no audio; the parity harness now accepts a real corpus directory for the required rerun.
- I did not deploy to `violawake.com`; production-destructive and publish actions are forbidden by the dispatch correction, so live proof remains against the old deployed bundle.
- I did not exhaustively benchmark across browsers; Chromium was used for the smoke because the existing live harness is Chromium-based and the lane asked for a deployed-page smoke, not cross-browser certification.
- I did not modify global CSP headers to allow `blob:` in `script-src`; the branch instead avoids the blob-worker path by forcing single-thread/no-proxy ORT in the demo.
- I did not audit the full frontend app outside `/wasm/demo/`; Lane 3 ownership is the WASM detector/demo path, and broader frontend routing/marketing behavior belongs to the frontend lane.
