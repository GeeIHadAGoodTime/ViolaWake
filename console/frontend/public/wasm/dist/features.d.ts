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
import * as ort from "onnxruntime-web";
export declare const SAMPLE_RATE = 16000;
export declare const MEL_FRAMES_PER_EMBEDDING = 76;
export declare const MEL_STRIDE = 8;
export declare const EMBEDDING_DIM = 96;
export declare const OWW_CHUNK_SAMPLES = 1280;
export interface BackboneResult {
    /** Whether a new embedding was produced this call. */
    produced: boolean;
    /** The latest 96-d embedding, or null if no embedding yet. */
    embedding: Float32Array | null;
}
export declare class OWWBackbone {
    private readonly melspecSession;
    private readonly embeddingSession;
    private readonly melspecInputName;
    private readonly embeddingInputName;
    private rawBuffer;
    private melspecBuffer;
    private melspecRows;
    private accumulatedSamples;
    private remainder;
    private lastEmbedding;
    private constructor();
    static create(melspecModelUrl: string, embeddingModelUrl: string, ortOptions?: ort.InferenceSession.SessionOptions): Promise<OWWBackbone>;
    reset(): void;
    /**
     * Push an audio frame (int16 PCM or float32 normalised to [-1, 1]).
     * Returns {produced, embedding} matching Python push_audio().
     */
    pushAudio(audioFrame: Int16Array | Float32Array): Promise<BackboneResult>;
    private _streamingMelspectrogram;
    private _appendMelFrames;
    private _getMelWindow;
    private _predictMelspectrogram;
    private _predictEmbedding;
}
