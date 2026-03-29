/**
 * detector.ts
 *
 * WakeDetector — browser/WASM port of violawake_sdk.WakeDetector.
 *
 * Pipeline (matches wake_detector.py):
 *   audio frame (Float32Array, 16kHz mono)
 *     → OWWBackbone (melspec + embedding ONNX)
 *     → Temporal CNN ONNX (or plain MLP, depending on model input shape)
 *     → score (0.0 – 1.0)
 *     → decision gate (RMS, threshold, cooldown)
 *
 * Model files required (same .onnx files used by the Python SDK):
 *   melspectrogram.onnx   — from openwakeword Python package
 *   embedding_model.onnx  — from openwakeword Python package
 *   temporal_cnn.onnx     — ViolaWake classifier (GitHub Releases)
 */

import * as ort from "onnxruntime-web";
import { OWWBackbone, EMBEDDING_DIM, SAMPLE_RATE } from "./features.js";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface WakeDetectorOptions {
  /**
   * Detection confidence threshold (0.0–1.0).
   * Default: 0.80  (same as Python SDK default).
   */
  threshold?: number;

  /**
   * Minimum seconds between consecutive detections.
   * Default: 2.0
   */
  cooldownS?: number;

  /**
   * Consecutive above-threshold frames required before firing.
   * Default: 1  (set to 3 to reduce false positives).
   */
  confirmCount?: number;

  /**
   * URL for the melspectrogram backbone ONNX.
   * Default: "./models/melspectrogram.onnx"
   */
  melspecModelUrl?: string;

  /**
   * URL for the embedding backbone ONNX.
   * Default: "./models/embedding_model.onnx"
   */
  embeddingModelUrl?: string;

  /**
   * URL for the ViolaWake classifier ONNX.
   * Default: "./models/temporal_cnn.onnx"
   */
  classifierModelUrl?: string;

  /**
   * ONNX Runtime Web session options forwarded to all three sessions.
   */
  ortOptions?: ort.InferenceSession.SessionOptions;
}

// ---------------------------------------------------------------------------
// WakeDetector
// ---------------------------------------------------------------------------

export class WakeDetector {
  private readonly threshold: number;
  private readonly cooldownS: number;
  private readonly confirmCount: number;

  // Loaded after load()
  private backbone!: OWWBackbone;
  private classifierSession!: ort.InferenceSession;
  private classifierInputName!: string;

  // Temporal model state
  private isTemporal = false;
  private temporalSeqLen = 9;
  private embeddingBuffer: Float32Array[] = [];

  // Decision state
  private _lastScore = 0.0;

  /** The raw score from the most recent `detect()` or `getScore()` call. */
  get lastScore(): number {
    return this._lastScore;
  }
  private lastDetectionTime = 0; // performance.now() ms
  private confirmCounter = 0;

  // Lazy-load URLs (stored for load())
  private readonly melspecModelUrl: string;
  private readonly embeddingModelUrl: string;
  private readonly classifierModelUrl: string;
  private readonly ortOptions: ort.InferenceSession.SessionOptions;

  constructor(options: WakeDetectorOptions = {}) {
    this.threshold = options.threshold ?? 0.80;
    this.cooldownS = options.cooldownS ?? 2.0;
    this.confirmCount = options.confirmCount ?? 1;
    this.melspecModelUrl = options.melspecModelUrl ?? "./models/melspectrogram.onnx";
    this.embeddingModelUrl = options.embeddingModelUrl ?? "./models/embedding_model.onnx";
    this.classifierModelUrl = options.classifierModelUrl ?? "./models/temporal_cnn.onnx";
    this.ortOptions = options.ortOptions ?? { executionProviders: ["wasm"] };

    if (this.threshold < 0.0 || this.threshold > 1.0) {
      throw new RangeError(`threshold must be in [0.0, 1.0], got ${this.threshold}`);
    }
    if (this.cooldownS < 0) {
      throw new RangeError(`cooldownS must be >= 0, got ${this.cooldownS}`);
    }
    if (this.confirmCount < 1) {
      throw new RangeError(`confirmCount must be >= 1, got ${this.confirmCount}`);
    }
  }

  /**
   * Load all three ONNX models. Must be called before detect() / getScore().
   */
  async load(): Promise<void> {
    this.backbone = await OWWBackbone.create(
      this.melspecModelUrl,
      this.embeddingModelUrl,
      this.ortOptions,
    );

    this.classifierSession = await ort.InferenceSession.create(
      this.classifierModelUrl,
      this.ortOptions,
    );
    this.classifierInputName = this.classifierSession.inputNames[0];

    // Detect temporal vs MLP from input shape
    const inputMeta = this.classifierSession.inputNames[0];
    // onnxruntime-web exposes inputMetadata via classifierSession.inputMetadata
    // We detect shape from a dry run is unnecessary — use the declared shape.
    // The temporal_cnn model has a 3-D input: (batch, seq_len, embedding_dim).
    // We probe by inspecting the model metadata if available, otherwise default.
    try {
      const meta = (this.classifierSession as any).inputMetadata ?? {};
      const shape = meta[inputMeta]?.dimensions ?? null;
      if (shape && shape.length === 3) {
        this.isTemporal = true;
        this.temporalSeqLen = typeof shape[1] === "number" && shape[1] > 0 ? shape[1] : 9;
      }
    } catch {
      // Fallback: assume temporal (temporal_cnn is the production model)
      this.isTemporal = true;
      this.temporalSeqLen = 9;
    }
  }

  /**
   * Process a 20ms audio frame (320 samples at 16kHz, float32 in [-1, 1]).
   * Returns true if wake word detected, false otherwise.
   *
   * Applies the 4-gate decision policy (RMS, threshold, cooldown, confirm).
   *
   * @param audioBuffer Float32Array of exactly 320 samples (20ms at 16kHz)
   */
  async detect(audioBuffer: Float32Array): Promise<boolean> {
    const score = await this.getScore(audioBuffer);

    // Gate 1: RMS floor (silence / DC offset guard)
    const rms = this._computeRms(audioBuffer);
    if (rms < 1.0 / 32768.0) {
      // RMS below silence floor (1.0 in int16 scale → 1/32768 in float32 scale)
      return false;
    }

    // Gate 2: Threshold
    if (score < this.threshold) {
      this.confirmCounter = 0;
      return false;
    }

    // K2: Confirmation gate
    this.confirmCounter++;
    if (this.confirmCounter < this.confirmCount) {
      return false;
    }
    this.confirmCounter = 0;

    // Gate 3: Cooldown
    const now = performance.now();
    if (now - this.lastDetectionTime < this.cooldownS * 1000) {
      return false;
    }
    this.lastDetectionTime = now;
    return true;
  }

  /**
   * Process a 20ms audio frame and return the raw classifier score (0.0–1.0).
   * Bypasses all decision gates — useful for visualisation and custom logic.
   *
   * @param audioBuffer Float32Array of exactly 320 samples (20ms at 16kHz)
   */
  async getScore(audioBuffer: Float32Array): Promise<number> {
    if (!this.backbone || !this.classifierSession) {
      throw new Error("WakeDetector not loaded. Call load() first.");
    }

    const { produced, embedding } = await this.backbone.pushAudio(audioBuffer);

    let score: number;

    if (embedding === null) {
      score = this._lastScore;
    } else if (this.isTemporal) {
      if (produced) {
        this.embeddingBuffer.push(embedding.slice());
        if (this.embeddingBuffer.length > this.temporalSeqLen) {
          this.embeddingBuffer.shift();
        }
        if (this.embeddingBuffer.length >= this.temporalSeqLen) {
          score = await this._runTemporalClassifier();
        } else {
          score = 0.0;
        }
      } else {
        score = this._lastScore;
      }
    } else {
      if (produced) {
        score = await this._runMlpClassifier(embedding);
      } else {
        score = this._lastScore;
      }
    }

    this._lastScore = score;
    return score;
  }

  /**
   * Reset internal streaming state (embedding buffer, cooldown, scores).
   * Does NOT unload the ONNX sessions.
   */
  reset(): void {
    this.backbone?.reset();
    this.embeddingBuffer = [];
    this._lastScore = 0.0;
    this.lastDetectionTime = 0;
    this.confirmCounter = 0;
  }

  /**
   * Reset the cooldown window, allowing immediate re-detection.
   */
  resetCooldown(): void {
    this.lastDetectionTime = 0;
  }

  /**
   * Release ONNX inference sessions. After calling dispose() the detector
   * cannot be used for inference without calling load() again.
   */
  dispose(): void {
    this.reset();
    // onnxruntime-web InferenceSession.release() is present in some versions
    (this.classifierSession as any)?.release?.();
  }

  // --- Private helpers ---

  private async _runTemporalClassifier(): Promise<number> {
    // Input shape: (1, seq_len, EMBEDDING_DIM)
    const flat = new Float32Array(this.temporalSeqLen * EMBEDDING_DIM);
    for (let i = 0; i < this.temporalSeqLen; i++) {
      flat.set(this.embeddingBuffer[i], i * EMBEDDING_DIM);
    }
    const tensor = new ort.Tensor("float32", flat, [1, this.temporalSeqLen, EMBEDDING_DIM]);
    const feeds: Record<string, ort.Tensor> = { [this.classifierInputName]: tensor };
    const results = await this.classifierSession.run(feeds);
    const output = results[this.classifierSession.outputNames[0]];
    return (output.data as Float32Array)[0];
  }

  private async _runMlpClassifier(embedding: Float32Array): Promise<number> {
    // Input shape: (1, EMBEDDING_DIM)
    const tensor = new ort.Tensor("float32", embedding.slice(), [1, EMBEDDING_DIM]);
    const feeds: Record<string, ort.Tensor> = { [this.classifierInputName]: tensor };
    const results = await this.classifierSession.run(feeds);
    const output = results[this.classifierSession.outputNames[0]];
    return (output.data as Float32Array)[0];
  }

  private _computeRms(audioBuffer: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < audioBuffer.length; i++) {
      sum += audioBuffer[i] * audioBuffer[i];
    }
    return Math.sqrt(sum / audioBuffer.length);
  }
}
