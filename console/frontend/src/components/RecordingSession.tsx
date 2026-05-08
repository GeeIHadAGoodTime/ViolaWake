import { useCallback, useEffect, useRef, useState } from "react";
import AudioRecorder from "./AudioRecorder";
import MicMonitor from "./MicMonitor";
import { drawWaveformFromBlob } from "../utils/waveform";

interface RecordingSessionProps {
  wakeWord: string;
  targetCount: number;
  onComplete: (recordings: Blob[]) => void;
}

interface RecordingSlot {
  blob: Blob;
  duration: number;
  url: string;
}

const DEVICE_STORAGE_KEY = "violawake.preferred_input_device";

function readPersistedDeviceId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(DEVICE_STORAGE_KEY);
    return raw && raw.length > 0 ? raw : null;
  } catch {
    return null;
  }
}

function persistDeviceId(deviceId: string | null): void {
  try {
    if (deviceId) {
      window.localStorage.setItem(DEVICE_STORAGE_KEY, deviceId);
    } else {
      window.localStorage.removeItem(DEVICE_STORAGE_KEY);
    }
  } catch {
    // localStorage may be disabled (incognito on some browsers); just skip.
  }
}

/**
 * Per-slot waveform — re-rendered when the slot's blob changes.
 */
function SlotWaveform({ blob }: { blob: Blob }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawWaveformFromBlob(blob, canvas, { color: "#7e6cff" }).catch((err) => {
      if (!cancelled) {
        console.warn("waveform render failed:", err);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [blob]);

  return (
    <canvas
      ref={canvasRef}
      className="slot-waveform"
      width={120}
      height={32}
      aria-label="Recording waveform"
    />
  );
}

export default function RecordingSession({
  wakeWord,
  targetCount,
  onComplete,
}: RecordingSessionProps) {
  const [slots, setSlots] = useState<(RecordingSlot | null)[]>(
    () => Array(targetCount).fill(null),
  );
  const [currentIndex, setCurrentIndex] = useState(0);
  const [reRecordIndex, setReRecordIndex] = useState<number | null>(
    null,
  );
  const [phase, setPhase] = useState<"recording" | "review">(
    "recording",
  );
  const [deviceId, setDeviceId] = useState<string | null>(() =>
    readPersistedDeviceId(),
  );
  const audioRefs = useRef<Map<number, HTMLAudioElement>>(new Map());

  const completedCount = slots.filter((s) => s !== null).length;
  const activeIndex =
    reRecordIndex !== null ? reRecordIndex : currentIndex;

  const handleRecordingComplete = useCallback(
    (blob: Blob, duration: number) => {
      const url = URL.createObjectURL(blob);
      const slot: RecordingSlot = { blob, duration, url };

      setSlots((prev) => {
        const next = [...prev];
        // Revoke old URL if re-recording
        if (next[activeIndex]?.url) {
          URL.revokeObjectURL(next[activeIndex]!.url);
        }
        next[activeIndex] = slot;
        return next;
      });

      if (reRecordIndex !== null) {
        setReRecordIndex(null);
      } else {
        const nextIndex = currentIndex + 1;
        if (nextIndex >= targetCount) {
          setPhase("review");
        } else {
          setCurrentIndex(nextIndex);
        }
      }
    },
    [activeIndex, currentIndex, reRecordIndex, targetCount],
  );

  const handleDeviceChange = useCallback((next: string | null) => {
    setDeviceId(next);
    persistDeviceId(next);
  }, []);

  function handleReRecord(index: number) {
    setReRecordIndex(index);
    setPhase("recording");
  }

  function handleSubmit() {
    const blobs = slots
      .filter((s): s is RecordingSlot => s !== null)
      .map((s) => s.blob);
    onComplete(blobs);
  }

  function playRecording(index: number) {
    const slot = slots[index];
    if (!slot) return;

    let audio = audioRefs.current.get(index);
    if (!audio) {
      audio = new Audio(slot.url);
      audioRefs.current.set(index, audio);
    }
    audio.currentTime = 0;
    audio.play();
  }

  const progressPct = (completedCount / targetCount) * 100;

  return (
    <div className="recording-session">
      {/* Always-on mic preview: device picker + live level meter */}
      <MicMonitor
        selectedDeviceId={deviceId}
        onDeviceChange={handleDeviceChange}
      />

      {/* Progress bar */}
      <div className="session-progress">
        <div className="session-progress-header">
          <span className="session-progress-label">
            {phase === "review"
              ? "Review your recordings"
              : reRecordIndex !== null
                ? `Re-recording sample ${reRecordIndex + 1}`
                : `Recording ${Math.min(currentIndex + 1, targetCount)} of ${targetCount}`}
          </span>
          <span className="session-progress-count">
            {completedCount}/{targetCount}
          </span>
        </div>
        <div
          className="session-progress-track"
          role="progressbar"
          aria-valuenow={Math.round(progressPct)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`Recording progress: ${completedCount} of ${targetCount} samples`}
        >
          <div
            className="session-progress-fill"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {/* Wake word display */}
      <div className="session-wake-word">
        Say: <strong>&ldquo;{wakeWord}&rdquo;</strong>
      </div>

      {/* Recording phase */}
      {phase === "recording" && (
        <div className="session-recorder-area">
          <AudioRecorder
            key={`recorder-${activeIndex}-${Date.now()}`}
            onRecordingComplete={handleRecordingComplete}
            maxDuration={3}
            deviceId={deviceId}
          />
        </div>
      )}

      {/* Slot grid */}
      <div className="session-slots">
        {slots.map((slot, i) => (
          <div
            key={i}
            className={`session-slot ${
              slot ? "filled" : ""
            } ${
              phase === "recording" && i === activeIndex
                ? "active"
                : ""
            }`}
          >
            <span className="slot-number">{i + 1}</span>
            {slot ? (
              <div className="slot-controls">
                <SlotWaveform blob={slot.blob} />
                <button
                  className="slot-play"
                  type="button"
                  onClick={() => playRecording(i)}
                  aria-label={`Play recording ${i + 1}`}
                >
                  &#9654;
                </button>
                <span className="slot-duration">
                  {slot.duration.toFixed(1)}s
                </span>
                {phase === "review" && (
                  <button
                    className="slot-redo"
                    type="button"
                    onClick={() => handleReRecord(i)}
                    aria-label={`Re-record sample ${i + 1}`}
                  >
                    &#8634;
                  </button>
                )}
              </div>
            ) : (
              <span className="slot-empty">--</span>
            )}
          </div>
        ))}
      </div>

      {/* Submit button (review phase only, all slots filled) */}
      {phase === "review" && completedCount === targetCount && (
        <button className="session-submit" onClick={handleSubmit}>
          Start Training
        </button>
      )}
    </div>
  );
}
