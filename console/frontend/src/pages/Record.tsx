import { useState } from "react";
import { useNavigate } from "react-router-dom";
import RecordingSession from "../components/RecordingSession";
import { uploadRecording, startTraining } from "../api";

type RecordPhase = "setup" | "recording" | "uploading" | "error";

const TARGET_RECORDINGS = 10;

export default function RecordPage() {
  const [phase, setPhase] = useState<RecordPhase>("setup");
  const [wakeWord, setWakeWord] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  function handleStartRecording() {
    const trimmed = wakeWord.trim().toLowerCase();
    if (trimmed.length < 2) return;
    setWakeWord(trimmed);
    setPhase("recording");
  }

  async function handleRecordingsComplete(blobs: Blob[]) {
    setPhase("uploading");
    setUploadProgress(0);
    setError(null);

    try {
      const recordingIds: number[] = [];

      for (let i = 0; i < blobs.length; i++) {
        const result = await uploadRecording(
          blobs[i],
          wakeWord,
          i,
        );
        recordingIds.push(result.recording_id);
        setUploadProgress(
          Math.round(((i + 1) / blobs.length) * 50),
        );
      }

      setUploadProgress(60);

      const { job_id } = await startTraining(
        wakeWord,
        recordingIds,
      );

      navigate(`/training/${job_id}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Upload failed",
      );
      setPhase("error");
    }
  }

  return (
    <div className="record-page">
      {phase === "setup" && (
        <div className="record-setup">
          <h1 className="page-title">Train a Wake Word</h1>
          <p className="page-subtitle">
            Choose a wake word, then record 10 voice samples. The
            model will learn to detect your voice saying this word.
          </p>

          <div className="setup-form">
            <div className="form-group">
              <label htmlFor="wakeword" className="form-label">
                Wake Word
              </label>
              <input
                id="wakeword"
                type="text"
                className="form-input form-input-large"
                placeholder='e.g., "jarvis", "hey computer"'
                value={wakeWord}
                onChange={(e) => setWakeWord(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleStartRecording();
                }}
                autoFocus
              />
              <span className="form-hint">
                Choose a 1-3 word phrase. Longer phrases tend to
                work better.
              </span>
            </div>

            <button
              className="btn btn-primary btn-large"
              onClick={handleStartRecording}
              disabled={wakeWord.trim().length < 2}
            >
              Start Recording
            </button>
          </div>
        </div>
      )}

      {phase === "recording" && (
        <RecordingSession
          wakeWord={wakeWord}
          targetCount={TARGET_RECORDINGS}
          onComplete={handleRecordingsComplete}
        />
      )}

      {phase === "uploading" && (
        <div className="record-uploading">
          <div className="spinner" />
          <h2>Uploading recordings...</h2>
          <div className="upload-progress-track">
            <div
              className="upload-progress-fill"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <p className="upload-progress-text">
            {uploadProgress < 50
              ? `Uploading samples... ${uploadProgress * 2}%`
              : uploadProgress < 100
                ? "Starting training job..."
                : "Redirecting..."}
          </p>
        </div>
      )}

      {phase === "error" && (
        <div className="record-error">
          <h2>Something went wrong</h2>
          <p className="error-message">{error}</p>
          <div className="error-actions">
            <button
              className="btn btn-primary"
              onClick={() => setPhase("recording")}
            >
              Try Again
            </button>
            <button
              className="btn btn-ghost"
              onClick={() => setPhase("setup")}
            >
              Start Over
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
