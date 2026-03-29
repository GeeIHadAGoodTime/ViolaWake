# ViolaWake WASM

Browser/WASM port of the ViolaWake wake word SDK.
Runs the full three-model inference pipeline entirely in the browser using
[ONNX Runtime Web](https://github.com/microsoft/onnxruntime).

## What it does

Detects the wake word "Viola" (or any custom wake word trained with the
ViolaWake CLI) in real-time microphone audio, producing a score (0.0–1.0)
per 20ms frame — the same pipeline as the Python SDK.

## Models required

Three ONNX files are needed. They are the **same files** as the Python SDK:

| File | Source |
|------|--------|
| `melspectrogram.onnx` | Ships with the `openwakeword` Python package (`openwakeword/resources/models/`) |
| `embedding_model.onnx` | Ships with the `openwakeword` Python package |
| `temporal_cnn.onnx` | [ViolaWake GitHub Releases](https://github.com/GeeIHadAGoodTime/ViolaWake/releases) — `violawake-download --model temporal_cnn` |

Copy all three files into a directory served alongside your web app
(e.g., `public/models/`). The browser fetches them at runtime.

## Install

```bash
npm install violawake onnxruntime-web
```

## Usage

```typescript
import { WakeDetector } from "violawake";
// Required: also load onnxruntime-web (it ships its own WASM assets)
// In a Vite/webpack project, import ort separately and configure wasmPaths:
import * as ort from "onnxruntime-web";
ort.env.wasm.wasmPaths = "/node_modules/onnxruntime-web/dist/";

const detector = new WakeDetector({
  threshold: 0.80,          // 0.0–1.0  (default 0.80)
  cooldownS: 2.0,           // seconds between detections (default 2.0)
  confirmCount: 1,          // consecutive frames required (default 1; use 3 to reduce FP)
  melspecModelUrl:    "/models/melspectrogram.onnx",
  embeddingModelUrl:  "/models/embedding_model.onnx",
  classifierModelUrl: "/models/temporal_cnn.onnx",
});

await detector.load(); // Fetches and initialises all three ONNX sessions

// Feed 20ms audio frames (320 float32 samples at 16 kHz, range [-1, 1]):
const detected: boolean = await detector.detect(float32Frame);
const score: number     = await detector.getScore(float32Frame); // 0.0–1.0, no gates

// Clean up
detector.dispose();
```

## Real-time microphone example

```typescript
const stream = await navigator.mediaDevices.getUserMedia({
  audio: { sampleRate: 16000, channelCount: 1 },
});
const ctx = new AudioContext({ sampleRate: 16000 });
const source = ctx.createMediaStreamSource(stream);
const processor = ctx.createScriptProcessor(512, 1, 1);

let buf = new Float32Array(0);
const FRAME = 320;

processor.onaudioprocess = async (e) => {
  const input = e.inputBuffer.getChannelData(0);
  const merged = new Float32Array(buf.length + input.length);
  merged.set(buf);
  merged.set(input, buf.length);
  buf = merged;

  while (buf.length >= FRAME) {
    const frame = buf.slice(0, FRAME);
    buf = buf.slice(FRAME);
    if (await detector.detect(frame)) {
      console.log("Wake word detected!");
    }
  }
};

source.connect(processor);
processor.connect(ctx.destination);
```

## API

### `new WakeDetector(options?)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `threshold` | `number` | `0.80` | Detection confidence threshold (0.0–1.0). Higher = fewer false positives. |
| `cooldownS` | `number` | `2.0` | Minimum seconds between consecutive detections. |
| `confirmCount` | `number` | `1` | Consecutive above-threshold frames required before detection fires. Set to 3 to reduce false positives at the cost of ~60ms latency. |
| `melspecModelUrl` | `string` | `./models/melspectrogram.onnx` | URL for the melspectrogram ONNX model. |
| `embeddingModelUrl` | `string` | `./models/embedding_model.onnx` | URL for the embedding ONNX model. |
| `classifierModelUrl` | `string` | `./models/temporal_cnn.onnx` | URL for the ViolaWake classifier ONNX model. |
| `ortOptions` | `ort.InferenceSession.SessionOptions` | `{ executionProviders: ["wasm"] }` | Forwarded to all three ONNX Runtime sessions. |

### `detector.load(): Promise<void>`

Fetch and initialise all three ONNX Runtime sessions. Must be called once before
`detect()` or `getScore()`. Rejects on network or model-load error.

### `detector.detect(audioBuffer: Float32Array): Promise<boolean>`

Process one 20ms frame (320 float32 samples at 16 kHz, range [-1, 1]).
Returns `true` if all four decision gates pass (RMS floor, threshold,
confirmation count, cooldown). The frame is also scored internally.

### `detector.getScore(audioBuffer: Float32Array): Promise<number>`

Same as `detect()` but returns the raw classifier score (0.0–1.0) without
applying any decision gates. Use for visualisation or custom thresholding.

### `detector.reset(): void`

Reset internal streaming state (embedding buffer, cooldown, scores) without
unloading ONNX sessions.

### `detector.resetCooldown(): void`

Allow immediate re-detection by resetting the cooldown window.

### `detector.dispose(): void`

Release ONNX inference sessions and reset state. Call when done to free WASM memory.

## Demo

```bash
cd wasm
npm install
npm run build
npm run demo    # serves demo/ on http://localhost:5000
```

Open `http://localhost:5000` and allow microphone access. Place the three `.onnx`
model files in `wasm/demo/models/` before running.

## Browser compatibility

| Browser | Status |
|---------|--------|
| Chrome 90+ | Supported |
| Edge 90+ | Supported |
| Firefox 89+ | Supported |
| Safari 15.2+ | Supported (SharedArrayBuffer required — needs COOP/COEP headers) |
| Mobile Chrome | Supported |
| Mobile Safari | Supported (iOS 15.2+) |

**Safari / SharedArrayBuffer note**: ONNX Runtime Web's multithreaded WASM backend
requires `SharedArrayBuffer`, which needs the
[COOP/COEP HTTP headers](https://web.dev/coop-coep/) set on your server.
The single-threaded WASM backend (`executionProviders: ["wasm"]`) works without
these headers and is the default. Performance is slightly lower but fully functional.

## Threshold tuning

| Threshold | Behaviour |
|-----------|-----------|
| 0.70 | Sensitive — more detections, more false positives |
| **0.80** | **Balanced (default)** — recommended starting point |
| 0.85 | Conservative — fewer false positives, may miss some |
| 0.90+ | Very conservative — lowest FP rate |

Start at 0.80. If false positives are too frequent, increase to 0.85 or add
`confirmCount: 3` (requires 3 consecutive frames ≥ threshold, adding ~60ms latency).

## Building from source

```bash
cd wasm
npm install
npm run build   # outputs to wasm/dist/
npm run typecheck
```

## License

Apache 2.0 — same as the ViolaWake SDK.
