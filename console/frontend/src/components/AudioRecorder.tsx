import { useState, useRef, useCallback, useEffect } from "react";
import { encodeWAV, resample } from "../utils/wavEncoder";
import {
  analyzeAudioQuality,
  type AudioQualityResult,
  type QualityIssue,
} from "../utils/audioQuality";

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob, duration: number) => void;
  maxDuration?: number;
}

type RecorderState = "idle" | "countdown" | "recording" | "done";

const TARGET_SAMPLE_RATE = 16000;

/**
 * Translate a getUserMedia error into a human-readable message.
 */
function getMicErrorMessage(err: unknown): string {
  if (!(err instanceof DOMException)) {
    return "Microphone access failed. Please try again.";
  }

  switch (err.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Microphone permission was denied. Please allow microphone access in your browser settings and try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No microphone found. Please connect a microphone and try again.";
    case "NotReadableError":
    case "TrackStartError":
      return "Your microphone is in use by another application. Close other apps using the mic and try again.";
    case "OverconstrainedError":
      return "Your microphone does not support the required audio settings.";
    case "SecurityError":
      return "Microphone access is blocked. This page must be served over HTTPS.";
    default:
      return `Microphone error: ${err.message}`;
  }
}

export default function AudioRecorder({
  onRecordingComplete,
  maxDuration = 3,
}: AudioRecorderProps) {
  const [state, setState] = useState<RecorderState>("idle");
  const [countdown, setCountdown] = useState(3);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [micError, setMicError] = useState<string | null>(null);
  const [qualityResult, setQualityResult] =
    useState<AudioQualityResult | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const samplesRef = useRef<Float32Array[]>([]);
  const animFrameRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const timerRef = useRef<number>(0);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(animFrameRef.current);
      clearInterval(timerRef.current);
      cleanup();
    };
  }, []);

  // Check for browser support on mount
  useEffect(() => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (window.location.protocol === "http:" && window.location.hostname !== "localhost") {
        setMicError(
          "Microphone access requires HTTPS. Please access this page over a secure connection.",
        );
      } else {
        setMicError(
          "Your browser does not support audio recording. Please use a recent version of Chrome, Firefox, Edge, or Safari.",
        );
      }
    }
  }, []);

  function cleanup() {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    analyserRef.current = null;
  }

  const updateLevel = useCallback(() => {
    if (!analyserRef.current) return;
    const data = new Uint8Array(analyserRef.current.fftSize);
    analyserRef.current.getByteTimeDomainData(data);

    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const val = (data[i] - 128) / 128;
      sum += val * val;
    }
    const rms = Math.sqrt(sum / data.length);
    setLevel(Math.min(1, rms * 4));

    if (state === "recording") {
      animFrameRef.current = requestAnimationFrame(updateLevel);
    }
  }, [state]);

  const stopRecording = useCallback(() => {
    clearInterval(timerRef.current);
    cancelAnimationFrame(animFrameRef.current);
    setQualityResult(null);

    const allSamples = samplesRef.current;
    if (allSamples.length === 0) {
      cleanup();
      setState("idle");
      return;
    }

    // Concatenate all chunks
    const totalLength = allSamples.reduce(
      (acc, chunk) => acc + chunk.length,
      0,
    );
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of allSamples) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    // Resample to target rate
    const nativeSampleRate =
      audioContextRef.current?.sampleRate ?? 44100;
    const resampled = resample(
      merged,
      nativeSampleRate,
      TARGET_SAMPLE_RATE,
    );

    const duration = resampled.length / TARGET_SAMPLE_RATE;

    // ── Quality gate ──────────────────────────────────────────────
    const quality = analyzeAudioQuality(resampled, TARGET_SAMPLE_RATE);
    setQualityResult(quality);

    if (quality.hasErrors) {
      // Reject: keep state so user can re-record, don't call onRecordingComplete
      cleanup();
      setState("idle");
      return;
    }

    // Passed (possibly with warnings) — encode and deliver
    const wavBlob = encodeWAV(resampled, TARGET_SAMPLE_RATE);

    cleanup();
    setState("done");
    onRecordingComplete(wavBlob, duration);
  }, [onRecordingComplete]);

  async function startRecording() {
    setMicError(null);
    setQualityResult(null);
    samplesRef.current = [];
    setState("countdown");

    // Countdown 3-2-1
    for (let i = 3; i >= 1; i--) {
      setCountdown(i);
      await new Promise((r) => setTimeout(r, 1000));
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      mediaStreamRef.current = stream;

      const ctx = new AudioContext();
      audioContextRef.current = ctx;

      // The AudioContext may start suspended if created outside a user-gesture
      // microtask (the 3-second countdown breaks the gesture chain). Explicitly
      // resume to ensure audio actually flows through the processing graph.
      if (ctx.state === "suspended") {
        await ctx.resume();
      }

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;
      source.connect(analyser);

      // Use ScriptProcessor for raw PCM capture
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        samplesRef.current.push(new Float32Array(input));
      };

      source.connect(processor);
      processor.connect(ctx.destination);

      setState("recording");
      startTimeRef.current = Date.now();
      setElapsed(0);

      // Update elapsed timer
      timerRef.current = window.setInterval(() => {
        const secs = (Date.now() - startTimeRef.current) / 1000;
        setElapsed(secs);
        if (secs >= maxDuration) {
          stopRecording();
        }
      }, 50);

      // Start level meter
      animFrameRef.current = requestAnimationFrame(updateLevel);
    } catch (err) {
      console.error("Microphone access failed:", err);
      setMicError(getMicErrorMessage(err));
      setState("idle");
    }
  }

  function handleRecord() {
    if (state === "idle" || state === "done") {
      startRecording();
    } else if (state === "recording") {
      stopRecording();
    }
  }

  const progressPct = Math.min(100, (elapsed / maxDuration) * 100);
  const isDisabled = micError !== null && state === "idle";

  return (
    <div className="audio-recorder">
      {micError && (
        <div className="recorder-error" role="alert">
          <span className="recorder-error-icon">{"\u26A0"}</span>
          <span className="recorder-error-text">{micError}</span>
        </div>
      )}

      {qualityResult && qualityResult.issues.length > 0 && (
        <div className="quality-issues" role="alert">
          {qualityResult.issues.map((issue: QualityIssue, i: number) => (
            <div
              key={i}
              className={`quality-issue quality-issue-${issue.severity}`}
            >
              <span className="quality-issue-icon">
                {issue.severity === "error" ? "\u2716" : "\u26A0"}
              </span>
              <span className="quality-issue-text">{issue.message}</span>
            </div>
          ))}
          {qualityResult.hasErrors && (
            <p className="quality-retry-hint">
              Please click Record to try again.
            </p>
          )}
        </div>
      )}

      {state === "countdown" && (
        <div className="recorder-countdown">
          <div className="countdown-number">{countdown}</div>
          <p className="countdown-text">Get ready...</p>
        </div>
      )}

      {state !== "countdown" && (
        <>
          <div className="recorder-level-container">
            <div
              className="recorder-level-bar"
              style={{
                height: `${Math.max(4, level * 100)}%`,
                opacity: state === "recording" ? 1 : 0.3,
              }}
            />
          </div>

          {state === "recording" && (
            <div className="recorder-timer">
              <div className="recorder-progress-track">
                <div
                  className="recorder-progress-fill"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              <span className="recorder-elapsed">
                {elapsed.toFixed(1)}s / {maxDuration}s
              </span>
            </div>
          )}

          <button
            className={`recorder-button ${
              state === "recording" ? "recording" : ""
            }`}
            onClick={handleRecord}
            disabled={isDisabled}
          >
            {state === "recording" && (
              <span className="recorder-pulse" />
            )}
            <span className="recorder-button-inner">
              {state === "recording" ? "Stop" : "Record"}
            </span>
          </button>

          {state === "idle" && !micError && (
            <p className="recorder-hint">
              Click to start recording
            </p>
          )}
          {state === "done" && (
            <p className="recorder-hint">
              Recording complete. Click to re-record.
            </p>
          )}
        </>
      )}
    </div>
  );
}
