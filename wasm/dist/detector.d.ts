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
export declare class WakeDetector {
    private readonly threshold;
    private readonly cooldownS;
    private readonly confirmCount;
    private backbone;
    private classifierSession;
    private classifierInputName;
    private isTemporal;
    private temporalSeqLen;
    private embeddingBuffer;
    private _lastScore;
    /** The raw score from the most recent `detect()` or `getScore()` call. */
    get lastScore(): number;
    private lastDetectionTime;
    private confirmCounter;
    private readonly melspecModelUrl;
    private readonly embeddingModelUrl;
    private readonly classifierModelUrl;
    private readonly ortOptions;
    constructor(options?: WakeDetectorOptions);
    /**
     * Load all three ONNX models. Must be called before detect() / getScore().
     */
    load(): Promise<void>;
    /**
     * Process a 20ms audio frame (320 samples at 16kHz, float32 in [-1, 1]).
     * Returns true if wake word detected, false otherwise.
     *
     * Applies the 4-gate decision policy (RMS, threshold, cooldown, confirm).
     *
     * @param audioBuffer Float32Array of exactly 320 samples (20ms at 16kHz)
     */
    detect(audioBuffer: Float32Array): Promise<boolean>;
    /**
     * Process a 20ms audio frame and return the raw classifier score (0.0–1.0).
     * Bypasses all decision gates — useful for visualisation and custom logic.
     *
     * @param audioBuffer Float32Array of exactly 320 samples (20ms at 16kHz)
     */
    getScore(audioBuffer: Float32Array): Promise<number>;
    /**
     * Reset internal streaming state (embedding buffer, cooldown, scores).
     * Does NOT unload the ONNX sessions.
     */
    reset(): void;
    /**
     * Reset the cooldown window, allowing immediate re-detection.
     */
    resetCooldown(): void;
    /**
     * Release ONNX inference sessions. After calling dispose() the detector
     * cannot be used for inference without calling load() again.
     */
    dispose(): void;
    private _runTemporalClassifier;
    private _runMlpClassifier;
    private _getClassifierInputShape;
    private _getClassifierOutputShape;
    private _validateClassifierInputShape;
    private _validateClassifierOutputShape;
    private _validateAudioFrame;
    private _classifierLooksTemporal;
    private _computeRms;
}
