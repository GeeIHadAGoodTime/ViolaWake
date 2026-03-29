/**
 * Client-side recording quality analysis.
 *
 * Uses raw Float32Array PCM data (range -1..1) to detect silence,
 * clipping, bad duration, and low SNR before upload.
 */

export type IssueSeverity = "error" | "warning";

export interface QualityIssue {
  severity: IssueSeverity;
  message: string;
}

export interface AudioQualityResult {
  rms: number;
  clippedPercent: number;
  duration: number;
  snrDb: number;
  issues: QualityIssue[];
  /** True if any issue is an error (upload should be blocked). */
  hasErrors: boolean;
  /** True if any issue is a warning (upload allowed but user informed). */
  hasWarnings: boolean;
}

// ── Thresholds ──────────────────────────────────────────────────────────────

const RMS_SILENCE_THRESHOLD = 0.01;
const CLIP_SAMPLE_THRESHOLD = 0.99;
const CLIP_PERCENT_THRESHOLD = 5;
const MIN_DURATION_S = 0.5;
const MAX_DURATION_S = 5.0;
const SNR_DB_THRESHOLD = 6;
const SNR_FRAME_SECONDS = 0.1; // 100ms frames for SNR estimation

/**
 * Analyse a recording for quality issues.
 *
 * @param samples  Mono Float32 PCM in the range -1..1
 * @param sampleRate  Sample rate of the buffer (e.g. 16000)
 */
export function analyzeAudioQuality(
  samples: Float32Array,
  sampleRate: number,
): AudioQualityResult {
  const duration = samples.length / sampleRate;

  // ── RMS ────────────────────────────────────────────────────────────────
  let sumSquares = 0;
  let clippedCount = 0;

  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    sumSquares += v * v;
    if (Math.abs(v) > CLIP_SAMPLE_THRESHOLD) {
      clippedCount++;
    }
  }

  const rms = samples.length > 0 ? Math.sqrt(sumSquares / samples.length) : 0;
  const clippedPercent =
    samples.length > 0 ? (clippedCount / samples.length) * 100 : 0;

  // ── SNR (simple frame-based estimate) ──────────────────────────────────
  const frameSize = Math.floor(sampleRate * SNR_FRAME_SECONDS);
  let maxFrameRms = 0;
  let minFrameRms = Infinity;

  if (frameSize > 0 && samples.length >= frameSize) {
    const frameCount = Math.floor(samples.length / frameSize);
    for (let f = 0; f < frameCount; f++) {
      let frameSum = 0;
      const offset = f * frameSize;
      for (let i = 0; i < frameSize; i++) {
        const v = samples[offset + i];
        frameSum += v * v;
      }
      const frameRms = Math.sqrt(frameSum / frameSize);
      if (frameRms > maxFrameRms) maxFrameRms = frameRms;
      if (frameRms < minFrameRms) minFrameRms = frameRms;
    }
  }

  // Avoid log(0) and division by zero
  const snrDb =
    minFrameRms > 0 && maxFrameRms > 0
      ? 20 * Math.log10(maxFrameRms / minFrameRms)
      : minFrameRms === 0 && maxFrameRms > 0
        ? Infinity // noise floor is zero — perfect
        : 0;

  // ── Build issues list ──────────────────────────────────────────────────
  const issues: QualityIssue[] = [];

  // Duration checks (errors — block upload)
  if (duration < MIN_DURATION_S) {
    issues.push({
      severity: "error",
      message: "Recording too short. Please say the full wake word.",
    });
  } else if (duration > MAX_DURATION_S) {
    issues.push({
      severity: "error",
      message: "Recording too long. Please record just the wake word.",
    });
  }

  // Silence check (error — block upload)
  if (rms < RMS_SILENCE_THRESHOLD) {
    issues.push({
      severity: "error",
      message:
        "Recording appears to be silent. Please check your microphone.",
    });
  }

  // Clipping check (warning — allow upload)
  if (clippedPercent > CLIP_PERCENT_THRESHOLD) {
    issues.push({
      severity: "warning",
      message:
        "Recording may be clipped (too loud). Try moving further from the microphone.",
    });
  }

  // SNR check (warning — allow upload)
  if (snrDb < SNR_DB_THRESHOLD && rms >= RMS_SILENCE_THRESHOLD) {
    issues.push({
      severity: "warning",
      message:
        "Background noise is high. Try recording in a quieter environment.",
    });
  }

  return {
    rms,
    clippedPercent,
    duration,
    snrDb,
    issues,
    hasErrors: issues.some((i) => i.severity === "error"),
    hasWarnings: issues.some((i) => i.severity === "warning"),
  };
}
