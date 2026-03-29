/**
 * ViolaWake WASM — browser wake word detection
 *
 * Usage:
 *   import { WakeDetector } from "violawake";
 *
 *   const detector = new WakeDetector({ threshold: 0.80 });
 *   await detector.load();  // fetches the three ONNX models
 *
 *   // then feed 20ms frames (320 samples at 16kHz):
 *   const detected = await detector.detect(float32Frame);
 */

export { WakeDetector } from "./detector.js";
export type { WakeDetectorOptions } from "./detector.js";
export {
  OWWBackbone,
  SAMPLE_RATE,
  EMBEDDING_DIM,
  MEL_FRAMES_PER_EMBEDDING,
  MEL_STRIDE,
  OWW_CHUNK_SAMPLES,
} from "./features.js";
