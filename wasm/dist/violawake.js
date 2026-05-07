import * as ort from 'onnxruntime-web';

/**
 * features.ts
 *
 * Streaming audio buffer and mel-spectrogram + embedding extraction
 * that mirrors OpenWakeWordBackbone from oww_backbone.py.
 *
 * Pipeline (identical to Python SDK):
 *   int16 PCM  →  melspectrogram ONNX  →  / 10.0 + 2.0  →  76-frame window
 *               →  embedding ONNX  →  96-d float32 vector
 *
 * Key constants from oww_backbone.py:
 *   SAMPLE_RATE              = 16_000
 *   MEL_FRAMES_PER_EMBEDDING = 76
 *   MEL_STRIDE               = 8
 *   EMBEDDING_DIM            = 96
 *   OWW_CHUNK_SAMPLES        = 1_280   (80ms at 16 kHz)
 *   MELSPEC_CONTEXT_SAMPLES  = 160 * 3 = 480
 */
// --- Constants (must match oww_backbone.py) ---
const SAMPLE_RATE = 16000;
const MEL_FRAMES_PER_EMBEDDING = 76;
const MEL_STRIDE = 8;
const EMBEDDING_DIM = 96;
const OWW_CHUNK_SAMPLES = 1280; // 80ms at 16 kHz
const MELSPEC_CONTEXT_SAMPLES = 160 * 3; // 480 samples context overlap
const MAX_RAW_SAMPLES = SAMPLE_RATE * 10; // 10s ring buffer
const MAX_MELSPEC_FRAMES = 10 * 97; // ~10s of mel frames at ~97 frames/s
// ---------------------------------------------------------------------------
// Ring buffer (mirrors _RingBuffer in oww_backbone.py)
// Stores int16 samples in a fixed-capacity Float32Array (we use float32 for
// WebAudio compatibility; int16 range is preserved until normalization).
// ---------------------------------------------------------------------------
class RingBuffer {
    constructor(capacity) {
        this.writePos = 0;
        this.count = 0;
        this.capacity = capacity;
        this.buf = new Int16Array(capacity);
    }
    get length() {
        return this.count;
    }
    /** Append int16 samples. */
    extend(data) {
        const n = data.length;
        if (n === 0)
            return;
        if (n >= this.capacity) {
            // Keep only the tail
            this.buf.set(data.subarray(data.length - this.capacity));
            this.writePos = 0;
            this.count = this.capacity;
            return;
        }
        const end = this.writePos + n;
        if (end <= this.capacity) {
            this.buf.set(data, this.writePos);
        }
        else {
            const first = this.capacity - this.writePos;
            this.buf.set(data.subarray(0, first), this.writePos);
            this.buf.set(data.subarray(first), 0);
        }
        this.writePos = end % this.capacity;
        this.count = Math.min(this.count + n, this.capacity);
    }
    /** Return the last n samples in chronological order. */
    tail(n) {
        n = Math.min(n, this.count);
        if (n === 0)
            return new Int16Array(0);
        const start = (this.writePos - n + this.capacity) % this.capacity;
        if (start + n <= this.capacity) {
            return this.buf.slice(start, start + n);
        }
        // Wraps around — two slices
        const result = new Int16Array(n);
        const firstLen = this.capacity - start;
        result.set(this.buf.subarray(start), 0);
        result.set(this.buf.subarray(0, this.writePos), firstLen);
        return result;
    }
}
class OWWBackbone {
    constructor(melspecSession, embeddingSession) {
        this.melspecRows = 0; // number of mel frames in buffer
        this.accumulatedSamples = 0;
        this.remainder = new Int16Array(0);
        this.lastEmbedding = null;
        this.melspecSession = melspecSession;
        this.embeddingSession = embeddingSession;
        this.melspecInputName = melspecSession.inputNames[0];
        this.embeddingInputName = embeddingSession.inputNames[0];
        // Pre-fill melspec buffer with 1.0 (matches Python: np.ones((76, 32)))
        this.rawBuffer = new RingBuffer(MAX_RAW_SAMPLES);
        this.melspecBuffer = new Float32Array(MAX_MELSPEC_FRAMES * 32).fill(1.0);
        this.melspecRows = MEL_FRAMES_PER_EMBEDDING; // pre-warmed context
    }
    static async create(melspecModelUrl, embeddingModelUrl, ortOptions) {
        const opts = {
            executionProviders: ["wasm"],
            ...ortOptions,
        };
        const [mel, emb] = await Promise.all([
            ort.InferenceSession.create(melspecModelUrl, opts),
            ort.InferenceSession.create(embeddingModelUrl, opts),
        ]);
        return new OWWBackbone(mel, emb);
    }
    reset() {
        this.rawBuffer = new RingBuffer(MAX_RAW_SAMPLES);
        this.melspecBuffer = new Float32Array(MAX_MELSPEC_FRAMES * 32).fill(1.0);
        this.melspecRows = MEL_FRAMES_PER_EMBEDDING;
        this.accumulatedSamples = 0;
        this.remainder = new Int16Array(0);
        this.lastEmbedding = null;
    }
    /**
     * Push an audio frame (int16 PCM or float32 normalised to [-1, 1]).
     * Returns {produced, embedding} matching Python push_audio().
     */
    async pushAudio(audioFrame) {
        // Convert to int16 (mirrors _to_pcm_int16)
        let pcmI16;
        if (audioFrame instanceof Int16Array) {
            pcmI16 = audioFrame;
        }
        else {
            // float32 in [-1, 1] → int16
            pcmI16 = new Int16Array(audioFrame.length);
            for (let i = 0; i < audioFrame.length; i++) {
                const s = Math.max(-1, Math.min(1, audioFrame[i]));
                pcmI16[i] = Math.round(s * 32767);
            }
        }
        // Prepend remainder from previous call
        if (this.remainder.length > 0) {
            const merged = new Int16Array(this.remainder.length + pcmI16.length);
            merged.set(this.remainder);
            merged.set(pcmI16, this.remainder.length);
            pcmI16 = merged;
            this.remainder = new Int16Array(0);
        }
        const total = this.accumulatedSamples + pcmI16.length;
        const remainder = total % OWW_CHUNK_SAMPLES;
        const toBuffer = remainder > 0 ? pcmI16.subarray(0, pcmI16.length - remainder) : pcmI16;
        if (remainder > 0) {
            this.remainder = pcmI16.slice(pcmI16.length - remainder);
        }
        this.rawBuffer.extend(toBuffer);
        this.accumulatedSamples += toBuffer.length;
        const newEmbeddings = [];
        if (this.accumulatedSamples >= OWW_CHUNK_SAMPLES &&
            this.accumulatedSamples % OWW_CHUNK_SAMPLES === 0) {
            await this._streamingMelspectrogram(this.accumulatedSamples);
            const nChunks = this.accumulatedSamples / OWW_CHUNK_SAMPLES;
            // Iterate newest-first (matches Python loop: range(n_chunks-1, -1, -1))
            for (let chunkIdx = nChunks - 1; chunkIdx >= 0; chunkIdx--) {
                const offset = MEL_STRIDE * chunkIdx; // frames from end
                const endRow = this.melspecRows - offset;
                const startRow = endRow - MEL_FRAMES_PER_EMBEDDING;
                if (startRow >= 0 && endRow <= this.melspecRows) {
                    const window = this._getMelWindow(startRow, endRow); // (76, 32)
                    const embedding = await this._predictEmbedding(window);
                    this.lastEmbedding = embedding;
                    newEmbeddings.push(embedding);
                }
            }
            this.accumulatedSamples = 0;
        }
        if (newEmbeddings.length > 0) {
            return { produced: true, embedding: newEmbeddings[newEmbeddings.length - 1] };
        }
        return { produced: false, embedding: this.lastEmbedding };
    }
    // --- Private helpers ---
    async _streamingMelspectrogram(nSamples) {
        const windowSamples = nSamples + MELSPEC_CONTEXT_SAMPLES;
        const raw = this.rawBuffer.tail(windowSamples);
        const newFrames = await this._predictMelspectrogram(raw);
        // Append new frames and trim to MAX_MELSPEC_FRAMES
        const newRows = newFrames.length / 32;
        const combined = this._appendMelFrames(newFrames, newRows);
        this.melspecBuffer = combined.buffer;
        this.melspecRows = combined.rows;
    }
    _appendMelFrames(newFrames, newRows) {
        const totalRows = this.melspecRows + newRows;
        if (totalRows <= MAX_MELSPEC_FRAMES) {
            const buf = new Float32Array(totalRows * 32);
            buf.set(this.melspecBuffer.subarray(0, this.melspecRows * 32));
            buf.set(newFrames, this.melspecRows * 32);
            return { buffer: buf, rows: totalRows };
        }
        // Trim oldest frames
        const keepRows = Math.min(MAX_MELSPEC_FRAMES, totalRows);
        const buf = new Float32Array(keepRows * 32);
        const dropRows = totalRows - keepRows;
        // Copy tail of old buffer (after dropRows) plus new frames
        const oldKeepRows = this.melspecRows - dropRows;
        if (oldKeepRows > 0) {
            buf.set(this.melspecBuffer.subarray(dropRows * 32, this.melspecRows * 32));
            buf.set(newFrames, oldKeepRows * 32);
        }
        else {
            // New frames alone exceed the limit — keep their tail
            const dropNew = -oldKeepRows;
            buf.set(newFrames.subarray(dropNew * 32));
        }
        return { buffer: buf, rows: keepRows };
    }
    _getMelWindow(startRow, endRow) {
        return this.melspecBuffer.slice(startRow * 32, endRow * 32);
    }
    async _predictMelspectrogram(pcmI16) {
        // Input: float32 batch (1, N) — model expects raw float32 PCM in int16 range
        const f32 = new Float32Array(pcmI16.length);
        for (let i = 0; i < pcmI16.length; i++) {
            f32[i] = pcmI16[i]; // Keep int16 magnitude, cast to float32
        }
        const tensor = new ort.Tensor("float32", f32, [1, f32.length]);
        const feeds = { [this.melspecInputName]: tensor };
        const results = await this.melspecSession.run(feeds);
        const output = results[this.melspecSession.outputNames[0]];
        // Shape: (1, N_frames, 32) — squeeze batch dim
        const raw = output.data;
        // Apply OWW normalization: / 10.0 + 2.0
        const out = new Float32Array(raw.length);
        for (let i = 0; i < raw.length; i++) {
            out[i] = raw[i] / 10.0 + 2.0;
        }
        return out; // flat (N_frames * 32)
    }
    async _predictEmbedding(melWindow) {
        // Input shape: (1, 76, 32, 1) — batch, frames, bins, channel
        const rows = MEL_FRAMES_PER_EMBEDDING;
        const cols = 32;
        const input = new Float32Array(rows * cols); // batch+channel squeezed
        input.set(melWindow.subarray(0, rows * cols));
        const tensor = new ort.Tensor("float32", input, [1, rows, cols, 1]);
        const feeds = { [this.embeddingInputName]: tensor };
        const results = await this.embeddingSession.run(feeds);
        const output = results[this.embeddingSession.outputNames[0]];
        const raw = output.data;
        // Flatten to 96-d
        const emb = new Float32Array(EMBEDDING_DIM);
        emb.set(raw.subarray(0, EMBEDDING_DIM));
        return emb;
    }
}

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
// ---------------------------------------------------------------------------
// WakeDetector
// ---------------------------------------------------------------------------
class WakeDetector {
    /** The raw score from the most recent `detect()` or `getScore()` call. */
    get lastScore() {
        return this._lastScore;
    }
    constructor(options = {}) {
        // Temporal model state
        this.isTemporal = false;
        this.temporalSeqLen = 9;
        this.embeddingBuffer = [];
        // Decision state
        this._lastScore = 0.0;
        this.lastDetectionTime = 0; // performance.now() ms
        this.confirmCounter = 0;
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
    async load() {
        this.backbone = await OWWBackbone.create(this.melspecModelUrl, this.embeddingModelUrl, this.ortOptions);
        this.classifierSession = await ort.InferenceSession.create(this.classifierModelUrl, this.ortOptions);
        this.classifierInputName = this.classifierSession.inputNames[0];
        // Detect temporal vs MLP from input shape. onnxruntime-web has exposed
        // inputMetadata as both an object keyed by input name and an array of
        // { name, shape } entries across releases, so support both forms.
        const shape = this._getClassifierInputShape(this.classifierInputName);
        if (shape?.length === 3) {
            this.isTemporal = true;
            this.temporalSeqLen = typeof shape[1] === "number" && shape[1] > 0 ? shape[1] : 9;
        }
        else if (shape?.length === 2) {
            this.isTemporal = false;
        }
        else if (this._classifierLooksTemporal()) {
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
    async detect(audioBuffer) {
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
    async getScore(audioBuffer) {
        if (!this.backbone || !this.classifierSession) {
            throw new Error("WakeDetector not loaded. Call load() first.");
        }
        const { produced, embedding } = await this.backbone.pushAudio(audioBuffer);
        let score;
        if (embedding === null) {
            score = this._lastScore;
        }
        else if (this.isTemporal) {
            if (produced) {
                this.embeddingBuffer.push(embedding.slice());
                if (this.embeddingBuffer.length > this.temporalSeqLen) {
                    this.embeddingBuffer.shift();
                }
                if (this.embeddingBuffer.length >= this.temporalSeqLen) {
                    score = await this._runTemporalClassifier();
                }
                else {
                    score = 0.0;
                }
            }
            else {
                score = this._lastScore;
            }
        }
        else {
            if (produced) {
                score = await this._runMlpClassifier(embedding);
            }
            else {
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
    reset() {
        this.backbone?.reset();
        this.embeddingBuffer = [];
        this._lastScore = 0.0;
        this.lastDetectionTime = 0;
        this.confirmCounter = 0;
    }
    /**
     * Reset the cooldown window, allowing immediate re-detection.
     */
    resetCooldown() {
        this.lastDetectionTime = 0;
    }
    /**
     * Release ONNX inference sessions. After calling dispose() the detector
     * cannot be used for inference without calling load() again.
     */
    dispose() {
        this.reset();
        // onnxruntime-web InferenceSession.release() is present in some versions
        this.classifierSession?.release?.();
    }
    // --- Private helpers ---
    async _runTemporalClassifier() {
        // Input shape: (1, seq_len, EMBEDDING_DIM)
        const flat = new Float32Array(this.temporalSeqLen * EMBEDDING_DIM);
        for (let i = 0; i < this.temporalSeqLen; i++) {
            flat.set(this.embeddingBuffer[i], i * EMBEDDING_DIM);
        }
        const tensor = new ort.Tensor("float32", flat, [1, this.temporalSeqLen, EMBEDDING_DIM]);
        const feeds = { [this.classifierInputName]: tensor };
        const results = await this.classifierSession.run(feeds);
        const output = results[this.classifierSession.outputNames[0]];
        return output.data[0];
    }
    async _runMlpClassifier(embedding) {
        // Input shape: (1, EMBEDDING_DIM)
        const tensor = new ort.Tensor("float32", embedding.slice(), [1, EMBEDDING_DIM]);
        const feeds = { [this.classifierInputName]: tensor };
        const results = await this.classifierSession.run(feeds);
        const output = results[this.classifierSession.outputNames[0]];
        return output.data[0];
    }
    _getClassifierInputShape(inputName) {
        const metadata = this.classifierSession?.inputMetadata;
        if (!metadata) {
            return null;
        }
        const inputMeta = Array.isArray(metadata)
            ? metadata.find((entry) => entry?.name === inputName) ?? metadata[0]
            : metadata[inputName];
        const shape = inputMeta?.dimensions ?? inputMeta?.shape;
        return Array.isArray(shape) ? shape : null;
    }
    _classifierLooksTemporal() {
        return /temporal|convgru|gru/i.test(this.classifierModelUrl);
    }
    _computeRms(audioBuffer) {
        let sum = 0;
        for (let i = 0; i < audioBuffer.length; i++) {
            sum += audioBuffer[i] * audioBuffer[i];
        }
        return Math.sqrt(sum / audioBuffer.length);
    }
}

export { EMBEDDING_DIM, MEL_FRAMES_PER_EMBEDDING, MEL_STRIDE, OWWBackbone, OWW_CHUNK_SAMPLES, SAMPLE_RATE, WakeDetector };
//# sourceMappingURL=violawake.js.map
