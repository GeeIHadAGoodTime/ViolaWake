import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  ApiError,
  bulkUploadRecordings,
  getRecordings,
  startTraining,
} from "../api";
import AudioRecorder from "../components/AudioRecorder";
import MicMonitor from "../components/MicMonitor";
import { useToast } from "../contexts/ToastContext";
import type { Recording } from "../types";
import "./AddSamples.css";

type ActiveTab = "record" | "upload";
type UploadStatus = "queued" | "uploading" | "success" | "error";

interface RecordedClip {
  id: number;
  blob: Blob;
  duration: number;
  url: string;
}

interface UploadFileRow {
  id: number;
  file: File;
  status: UploadStatus;
  progress: number;
  error: string | null;
  recordingId: number | null;
}

const DEVICE_STORAGE_KEY = "violawake.preferred_input_device";
const RECOMMENDED_SAMPLE_COUNT = 30;
const MIN_TRAINING_SAMPLE_COUNT = 5;
const MAX_BULK_FILES = 50;
const MAX_FILE_BYTES = 5 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = [".wav", ".flac"];
const ACCEPTED_AUDIO = ACCEPTED_EXTENSIONS.join(",");

function readPersistedDeviceId(): string | null {
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
    // localStorage may be disabled; recording still works.
  }
}

function formatRetryAfter(seconds: number | null): string {
  if (!seconds) return "Please wait a few minutes and try again.";
  if (seconds < 60) return `Please wait ${seconds}s and try again.`;
  return `Please wait ${Math.ceil(seconds / 60)} minutes and try again.`;
}

function fileExtension(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

function validateFile(file: File): string | null {
  if (!ACCEPTED_EXTENSIONS.includes(fileExtension(file.name))) {
    return `Unsupported format. Use ${ACCEPTED_EXTENSIONS.join(", ")}.`;
  }
  if (file.size > MAX_FILE_BYTES) {
    return "File is larger than 5 MB.";
  }
  return null;
}

export default function AddSamplesPage() {
  const { wakeWord: wakeWordParam } = useParams();
  const wakeWord = (wakeWordParam ?? "").trim().toLowerCase();
  const navigate = useNavigate();
  const { addToast } = useToast();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const nextIdRef = useRef(1);
  const recordedClipsRef = useRef<RecordedClip[]>([]);

  const [activeTab, setActiveTab] = useState<ActiveTab>("record");
  const [existingRecordings, setExistingRecordings] = useState<Recording[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [recordedClips, setRecordedClips] = useState<RecordedClip[]>([]);
  const [uploadFiles, setUploadFiles] = useState<UploadFileRow[]>([]);
  const [deviceId, setDeviceId] = useState<string | null>(() =>
    readPersistedDeviceId(),
  );
  const [recorderKey, setRecorderKey] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    getRecordings(wakeWord)
      .then((recordings) => {
        if (!cancelled) {
          setExistingRecordings(recordings);
          setLoadError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : "Could not load samples");
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [wakeWord]);

  useEffect(() => {
    recordedClipsRef.current = recordedClips;
  }, [recordedClips]);

  useEffect(() => {
    return () => {
      recordedClipsRef.current.forEach((clip) => URL.revokeObjectURL(clip.url));
    };
  }, []);

  const queuedUploadFiles = uploadFiles.filter((row) => row.status === "queued");
  const successfulUploadFiles = uploadFiles.filter((row) => row.status === "success");
  const newSampleCount =
    recordedClips.length + queuedUploadFiles.length + successfulUploadFiles.length;
  const totalSampleCount =
    existingRecordings.length + recordedClips.length + queuedUploadFiles.length;
  const progressPct = Math.min(
    100,
    Math.round((totalSampleCount / RECOMMENDED_SAMPLE_COUNT) * 100),
  );
  const canSubmit =
    !isSubmitting &&
    newSampleCount >= 1 &&
    totalSampleCount >= MIN_TRAINING_SAMPLE_COUNT;

  const sampleHint = useMemo(() => {
    if (existingRecordings.length === 0) {
      return "Previous training samples were deleted for privacy. Record or upload fresh samples to retrain.";
    }
    return `${existingRecordings.length} saved sample${
      existingRecordings.length === 1 ? "" : "s"
    } will be included with the new clips.`;
  }, [existingRecordings.length]);

  const handleDeviceChange = useCallback((next: string | null) => {
    setDeviceId(next);
    persistDeviceId(next);
  }, []);

  const handleRecordingComplete = useCallback((blob: Blob, duration: number) => {
    const url = URL.createObjectURL(blob);
    setRecordedClips((prev) => [
      ...prev,
      {
        id: nextIdRef.current++,
        blob,
        duration,
        url,
      },
    ]);
    setRecorderKey((prev) => prev + 1);
  }, []);

  function removeRecordedClip(id: number) {
    setRecordedClips((prev) => {
      const clip = prev.find((item) => item.id === id);
      if (clip) URL.revokeObjectURL(clip.url);
      return prev.filter((item) => item.id !== id);
    });
  }

  function playClip(url: string) {
    const audio = new Audio(url);
    audio.currentTime = 0;
    audio.play();
  }

  function addFiles(files: File[]) {
    const pendingFileRows = uploadFiles.filter((row) => row.status !== "success").length;
    const remainingSlots = MAX_BULK_FILES - recordedClips.length - pendingFileRows;
    if (remainingSlots <= 0) {
      addToast("warning", `Upload batches are limited to ${MAX_BULK_FILES} files.`);
      return;
    }

    const accepted = files.slice(0, remainingSlots);
    if (accepted.length < files.length) {
      addToast("warning", `Only ${remainingSlots} more files fit in this upload batch.`);
    }

    setUploadFiles((prev) => [
      ...prev,
      ...accepted.map((file) => {
        const error = validateFile(file);
        return {
          id: nextIdRef.current++,
          file,
          status: error ? "error" as const : "queued" as const,
          progress: 0,
          error,
          recordingId: null,
        };
      }),
    ]);
  }

  function handleFilePick(event: ChangeEvent<HTMLInputElement>) {
    addFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  }

  function removeUploadFile(id: number) {
    setUploadFiles((prev) => prev.filter((row) => row.id !== id));
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    addFiles(Array.from(event.dataTransfer.files));
  }

  async function handleSubmit() {
    if (!canSubmit) return;

    setIsSubmitting(true);
    setSubmitError(null);

    const pendingRows = uploadFiles.filter((row) => row.status === "queued");
    const filesToUpload: Array<File | Blob> = [
      ...recordedClips.map((clip) => clip.blob),
      ...pendingRows.map((row) => row.file),
    ];

    if (filesToUpload.length === 0) {
      try {
        const { job_id } = await startTraining(
          wakeWord,
          existingRecordings.map((recording) => recording.id),
        );
        navigate(`/training/${job_id}`);
      } catch (err) {
        let message = err instanceof Error ? err.message : "Could not retrain";
        if (err instanceof ApiError && err.status === 429) {
          message = `Training limit reached. ${formatRetryAfter(err.retryAfter)}`;
          addToast("warning", message);
        } else {
          addToast("error", message);
        }
        setSubmitError(message);
        setIsSubmitting(false);
      }
      return;
    }

    setUploadFiles((prev) =>
      prev.map((row) =>
        row.status === "queued"
          ? { ...row, status: "uploading", progress: 50, error: null }
          : row,
      ),
    );

    try {
      const uploadResult = await bulkUploadRecordings(filesToUpload, wakeWord);
      const recordedResults = uploadResult.results.slice(0, recordedClips.length);
      const fileResults = uploadResult.results.slice(recordedClips.length);

      const newRecordingIds = uploadResult.results
        .filter((result) => result.status === "success" && result.recording_id !== null)
        .map((result) => result.recording_id as number);

      const successfulRecords = uploadResult.results.filter(
        (result) => result.status === "success" && result.recording_id !== null,
      );
      if (successfulRecords.length > 0) {
        const createdAt = new Date().toISOString();
        setExistingRecordings((prev) => [
          ...successfulRecords.map((result) => ({
            id: result.recording_id as number,
            wake_word: result.wake_word ?? wakeWord,
            filename: result.filename,
            duration_s: result.duration_s ?? 0,
            created_at: createdAt,
          })),
          ...prev,
        ]);
      }

      setRecordedClips((prev) => {
        const kept = prev.filter((clip, index) => {
          const result = recordedResults[index];
          if (result?.status === "error") return true;
          URL.revokeObjectURL(clip.url);
          return false;
        });
        return kept;
      });

      setUploadFiles((prev) => {
        const pendingById = new Map(pendingRows.map((row, index) => [row.id, fileResults[index]]));
        return prev.map((row) => {
          const result = pendingById.get(row.id);
          if (!result) return row;
          return {
            ...row,
            status: result.status,
            progress: result.status === "success" ? 100 : 0,
            error: result.error,
            recordingId: result.recording_id,
          };
        });
      });

      const recordedFailures = recordedResults.filter((result) => result.status === "error");
      if (recordedFailures.length > 0) {
        addToast("warning", `${recordedFailures.length} recorded clip(s) could not be uploaded.`);
      }

      const combinedIds = [
        ...existingRecordings.map((recording) => recording.id),
        ...newRecordingIds,
      ];

      if (newRecordingIds.length === 0) {
        setSubmitError("No new samples uploaded successfully.");
        setIsSubmitting(false);
        return;
      }

      if (combinedIds.length < MIN_TRAINING_SAMPLE_COUNT) {
        setSubmitError(
          `Need at least ${MIN_TRAINING_SAMPLE_COUNT} valid samples to retrain. ${combinedIds.length} are ready.`,
        );
        setIsSubmitting(false);
        return;
      }

      if (uploadResult.failed > 0) {
        addToast("warning", "Some files failed; retraining with the successful samples.");
      }

      const { job_id } = await startTraining(wakeWord, combinedIds);
      navigate(`/training/${job_id}`);
    } catch (err) {
      let message = err instanceof Error ? err.message : "Could not retrain";
      if (err instanceof ApiError && err.status === 429) {
        message = `Upload rate limit reached. ${formatRetryAfter(err.retryAfter)}`;
        addToast("warning", message);
      } else {
        addToast("error", message);
      }
      setSubmitError(message);
      setUploadFiles((prev) =>
        prev.map((row) =>
          row.status === "uploading"
            ? { ...row, status: "queued", progress: 0 }
            : row,
        ),
      );
      setIsSubmitting(false);
    }
  }

  if (!wakeWord) {
    return (
      <div className="add-samples-page">
        <div className="add-samples-empty">
          <h1 className="page-title">Add Samples</h1>
          <p className="page-subtitle">Choose a wake word before adding samples.</p>
          <button className="btn btn-primary" onClick={() => navigate("/record")}>
            Start a Recording Session
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="add-samples-page">
      <div className="add-samples-header">
        <div>
          <h1 className="page-title">Add Samples</h1>
          <p className="page-subtitle">
            Improve <strong>&ldquo;{wakeWord}&rdquo;</strong> with more voice data.
          </p>
        </div>
        <button
          className="btn btn-ghost"
          type="button"
          onClick={() => navigate("/dashboard")}
        >
          Back to Dashboard
        </button>
      </div>

      <section className="sample-summary" aria-live="polite">
        <div className="sample-summary-stat">
          <span className="sample-summary-number">
            {isLoading ? "--" : existingRecordings.length}
          </span>
          <span className="sample-summary-label">current samples</span>
        </div>
        <div className="sample-summary-body">
          <div className="sample-summary-row">
            <span>Total after new samples</span>
            <strong>{totalSampleCount}</strong>
          </div>
          <div className="sample-target-track">
            <div
              className="sample-target-fill"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <p>
            Recommended target: {RECOMMENDED_SAMPLE_COUNT}+ samples. Minimum to
            retrain: {MIN_TRAINING_SAMPLE_COUNT}.
          </p>
          <p className="sample-summary-note">{sampleHint}</p>
          {loadError && <p className="sample-summary-error">{loadError}</p>}
        </div>
      </section>

      <div className="add-samples-tabs" role="tablist" aria-label="Add sample method">
        <button
          className={activeTab === "record" ? "active" : ""}
          type="button"
          role="tab"
          aria-selected={activeTab === "record"}
          onClick={() => setActiveTab("record")}
        >
          Record more
        </button>
        <button
          className={activeTab === "upload" ? "active" : ""}
          type="button"
          role="tab"
          aria-selected={activeTab === "upload"}
          onClick={() => setActiveTab("upload")}
        >
          Upload files
        </button>
      </div>

      {activeTab === "record" && (
        <section className="add-samples-panel">
          <MicMonitor
            selectedDeviceId={deviceId}
            onDeviceChange={handleDeviceChange}
          />

          <div className="add-recorder-area">
            <div className="session-wake-word">
              Say: <strong>&ldquo;{wakeWord}&rdquo;</strong>
            </div>
            <AudioRecorder
              key={recorderKey}
              onRecordingComplete={handleRecordingComplete}
              maxDuration={3}
              deviceId={deviceId}
            />
          </div>

          <div className="add-sample-grid">
            {recordedClips.length === 0 ? (
              <div className="add-sample-placeholder">No new recordings yet.</div>
            ) : (
              recordedClips.map((clip, index) => (
                <div className="add-sample-tile" key={clip.id}>
                  <span className="add-sample-index">{index + 1}</span>
                  <span>{clip.duration.toFixed(1)}s</span>
                  <div className="add-sample-actions">
                    <button
                      type="button"
                      onClick={() => playClip(clip.url)}
                      aria-label={`Play recorded sample ${index + 1}`}
                    >
                      Play
                    </button>
                    <button
                      type="button"
                      onClick={() => removeRecordedClip(clip.id)}
                      aria-label={`Remove recorded sample ${index + 1}`}
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>
      )}

      {activeTab === "upload" && (
        <section className="add-samples-panel">
          <div
            className={`upload-dropzone ${isDragging ? "dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_AUDIO}
              multiple
              onChange={handleFilePick}
              hidden
            />
            <strong>Drop audio files here</strong>
            <span>WAV or FLAC. 5 MB each, 50 files per batch.</span>
            <button
              className="btn btn-ghost"
              type="button"
              onClick={() => fileInputRef.current?.click()}
            >
              Choose Files
            </button>
          </div>

          <div className="upload-file-list">
            {uploadFiles.length === 0 ? (
              <div className="upload-file-empty">No files selected.</div>
            ) : (
              uploadFiles.map((row) => (
                <div className={`upload-file-row ${row.status}`} key={row.id}>
                  <div className="upload-file-main">
                    <strong>{row.file.name}</strong>
                    <span>{(row.file.size / 1024 / 1024).toFixed(2)} MB</span>
                    {row.error && <em>{row.error}</em>}
                  </div>
                  <div className="upload-file-progress" aria-hidden="true">
                    <div style={{ width: `${row.progress}%` }} />
                  </div>
                  <span className="upload-file-status">{row.status}</span>
                  {row.status !== "uploading" && row.status !== "success" && (
                    <button
                      type="button"
                      onClick={() => removeUploadFile(row.id)}
                      aria-label={`Remove ${row.file.name}`}
                    >
                      Remove
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
        </section>
      )}

      {submitError && <div className="add-samples-submit-error">{submitError}</div>}

      <div className="add-samples-submit">
        <span>
          {newSampleCount < 1
            ? "Add at least 1 new sample."
            : totalSampleCount < MIN_TRAINING_SAMPLE_COUNT
              ? `Need ${MIN_TRAINING_SAMPLE_COUNT - totalSampleCount} more valid sample(s).`
              : `${newSampleCount} new sample(s) ready.`}
        </span>
        <button
          className="btn btn-primary btn-large"
          type="button"
          disabled={!canSubmit}
          onClick={handleSubmit}
        >
          {isSubmitting
            ? "Starting retrain..."
            : `Retrain with ${totalSampleCount} total samples`}
        </button>
      </div>
    </div>
  );
}
