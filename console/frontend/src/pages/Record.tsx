import { useCallback, useRef, useState } from "react";
import type { ChangeEvent, DragEvent, KeyboardEvent } from "react";
import { useNavigate } from "react-router-dom";
import RecordingSession from "../components/RecordingSession";
import { ApiError, bulkUploadRecordings, startTraining } from "../api";
import "./AddSamples.css";

type RecordPhase = "setup" | "recording";
type ActiveTab = "record" | "upload";
type UploadStatus = "queued" | "uploading" | "success" | "error";

interface UploadFileRow {
  id: number;
  file: File;
  status: UploadStatus;
  progress: number;
  error: string | null;
  recordingId: number | null;
}

const TARGET_RECORDINGS = 10;
const MAX_BULK_FILES = 50;
const MAX_FILE_BYTES = 5 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = [".wav", ".flac"];
const ACCEPTED_AUDIO = ACCEPTED_EXTENSIONS.join(",");

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

function formatRetryAfter(seconds: number | null): string {
  if (!seconds) return "Please wait a few minutes and try again.";
  if (seconds < 60) return `Please wait ${seconds}s and try again.`;
  return `Please wait ${Math.ceil(seconds / 60)} minutes and try again.`;
}

function errorMessage(err: unknown, fallback: string): string {
  if (err instanceof ApiError && err.status === 429) {
    return `${fallback}. ${formatRetryAfter(err.retryAfter)}`;
  }
  return err instanceof Error ? err.message : fallback;
}

export default function RecordPage() {
  const [phase, setPhase] = useState<RecordPhase>("setup");
  const [wakeWord, setWakeWord] = useState("");
  const [activeTab, setActiveTab] = useState<ActiveTab>("record");
  const [recordedBlobs, setRecordedBlobs] = useState<Blob[]>([]);
  const [uploadFiles, setUploadFiles] = useState<UploadFileRow[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadingBatches, setUploadingBatches] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const nextIdRef = useRef(1);
  const navigate = useNavigate();

  const successfulUploadFiles = uploadFiles.filter(
    (row) => row.status === "success" && row.recordingId !== null,
  );
  const queuedUploadFiles = uploadFiles.filter((row) => row.status === "queued");
  const uploadingFiles = uploadFiles.filter((row) => row.status === "uploading");
  const recordedCount = recordedBlobs.length;
  const uploadedCount = successfulUploadFiles.length;
  const readySampleCount = recordedCount + uploadedCount;
  const remainingSamples = Math.max(0, TARGET_RECORDINGS - readySampleCount);
  const isUploadingFiles = uploadingBatches > 0 || uploadingFiles.length > 0;
  const canTrain =
    readySampleCount >= TARGET_RECORDINGS &&
    !isSubmitting &&
    !isUploadingFiles;

  function resetSessionState() {
    setRecordedBlobs([]);
    setUploadFiles([]);
    setUploadError(null);
    setSubmitError(null);
    setUploadProgress(0);
    setActiveTab("record");
  }

  function handleStartRecording() {
    const trimmed = wakeWord.trim().toLowerCase();
    if (trimmed.length < 2) return;
    resetSessionState();
    setWakeWord(trimmed);
    setPhase("recording");
  }

  const handleRecordingsChange = useCallback((blobs: Blob[]) => {
    setRecordedBlobs(blobs);
  }, []);

  async function uploadRows(rows: UploadFileRow[]) {
    if (rows.length === 0) return;

    setUploadingBatches((count) => count + 1);
    setUploadError(null);
    const rowIds = new Set(rows.map((row) => row.id));

    setUploadFiles((prev) =>
      prev.map((row) =>
        rowIds.has(row.id)
          ? { ...row, status: "uploading", progress: 50, error: null }
          : row,
      ),
    );

    try {
      const uploadResult = await bulkUploadRecordings(
        rows.map((row) => row.file),
        wakeWord,
      );
      const resultsById = new Map(
        rows.map((row, index) => [row.id, uploadResult.results[index]]),
      );

      setUploadFiles((prev) =>
        prev.map((row) => {
          const result = resultsById.get(row.id);
          if (!result) return row;
          return {
            ...row,
            status: result.status,
            progress: result.status === "success" ? 100 : 0,
            error: result.error,
            recordingId: result.recording_id,
          };
        }),
      );

      if (uploadResult.failed > 0) {
        setUploadError("Some files could not be uploaded. Check the rows below.");
      }
    } catch (err) {
      const message = errorMessage(err, "Could not upload files");
      setUploadError(message);
      setUploadFiles((prev) =>
        prev.map((row) =>
          rowIds.has(row.id) && row.status === "uploading"
            ? { ...row, status: "error", progress: 0, error: message }
            : row,
        ),
      );
    } finally {
      setUploadingBatches((count) => Math.max(0, count - 1));
    }
  }

  function addFiles(files: File[]) {
    const existingRows = uploadFiles.length;
    const remainingSlots = MAX_BULK_FILES - existingRows;
    if (remainingSlots <= 0) {
      setUploadError(`Upload batches are limited to ${MAX_BULK_FILES} files.`);
      return;
    }

    const accepted = files.slice(0, remainingSlots);
    if (accepted.length < files.length) {
      setUploadError(`Only ${remainingSlots} more files fit in this upload batch.`);
    } else {
      setUploadError(null);
    }

    const rows = accepted.map((file) => {
      const error = validateFile(file);
      return {
        id: nextIdRef.current++,
        file,
        status: error ? "error" as const : "queued" as const,
        progress: 0,
        error,
        recordingId: null,
      };
    });

    setUploadFiles((prev) => [...prev, ...rows]);
    void uploadRows(rows.filter((row) => row.status === "queued"));
  }

  function handleFilePick(event: ChangeEvent<HTMLInputElement>) {
    addFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    addFiles(Array.from(event.dataTransfer.files));
  }

  function openFilePicker() {
    if (!isSubmitting) fileInputRef.current?.click();
  }

  function handleDropzoneKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  }

  function removeUploadFile(id: number) {
    setUploadFiles((prev) => prev.filter((row) => row.id !== id));
  }

  async function handleTrain(blobsOverride?: Blob[]) {
    const blobsToUpload = blobsOverride ?? recordedBlobs;
    const uploadedRecordingIds = uploadFiles
      .filter((row) => row.status === "success" && row.recordingId !== null)
      .map((row) => row.recordingId as number);

    if (blobsToUpload.length + uploadedRecordingIds.length < TARGET_RECORDINGS) {
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);
    setUploadProgress(0);

    try {
      let recordingIds = [...uploadedRecordingIds];

      if (blobsToUpload.length > 0) {
        setUploadProgress(35);
        const uploadResult = await bulkUploadRecordings(blobsToUpload, wakeWord);
        const recordedIds = uploadResult.results
          .filter((result) => result.status === "success" && result.recording_id !== null)
          .map((result) => result.recording_id as number);

        recordingIds = [...recordingIds, ...recordedIds];

        if (uploadResult.failed > 0) {
          const firstError = uploadResult.results.find(
            (result) => result.status === "error" && result.error,
          )?.error;
          setSubmitError(
            `${uploadResult.failed} recorded sample(s) could not be uploaded.${
              firstError ? ` ${firstError}` : ""
            }`,
          );
        }
      }

      if (recordingIds.length < TARGET_RECORDINGS) {
        setSubmitError(
          `Need at least ${TARGET_RECORDINGS} valid samples to train. ${recordingIds.length} are ready.`,
        );
        setIsSubmitting(false);
        return;
      }

      setUploadProgress(70);
      const { job_id } = await startTraining(wakeWord, recordingIds);
      setUploadProgress(100);
      navigate(`/training/${job_id}`);
    } catch (err) {
      setSubmitError(errorMessage(err, "Could not start training"));
      setIsSubmitting(false);
    }
  }

  return (
    <div className="record-page">
      {phase === "setup" && (
        <div className="record-setup">
          <h1 className="page-title">Train a Wake Word</h1>
          <p className="page-subtitle">
            Choose a wake word, then record samples or upload your own WAV/FLAC
            recordings. The model will learn to detect your voice saying this word.
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
                Choose a 1-3 word phrase. Longer phrases tend to work better.
              </span>
            </div>

            <button
              className="btn btn-primary btn-large"
              onClick={handleStartRecording}
              disabled={wakeWord.trim().length < 2}
            >
              Continue
            </button>
          </div>

          <details className="tips-card">
            <summary className="tips-card-summary">
              Tips for a good wake-word recording
            </summary>
            <ul className="tips-card-content">
              <li>
                Pick a wake word with 2 to 4 syllables, like &quot;hey buddy&quot; or a
                short first-and-last-name combo. Avoid words you say often in normal
                conversation.
              </li>
              <li>
                Speak at a normal indoor volume. Say the whole phrase in about a second
                and a half, right after you start recording.
              </li>
              <li>Avoid recording next to a fan, TV, or music playing in the background.</li>
              <li>
                Record all {TARGET_RECORDINGS} samples with a little variation in pace and
                distance from the microphone. The same delivery every time trains a
                narrower model.
              </li>
              <li>
                If training fails after a couple of tries, the quickest fix is usually to
                train again with the same recordings, not to re-record from scratch.
              </li>
            </ul>
          </details>
        </div>
      )}

      {phase === "recording" && (
        <div className="add-samples-page">
          <div className="add-samples-header">
            <div>
              <h1 className="page-title">Train a Wake Word</h1>
              <p className="page-subtitle">
                Provide samples for <strong>&ldquo;{wakeWord}&rdquo;</strong>.
              </p>
            </div>
            <button
              className="btn btn-ghost"
              type="button"
              onClick={() => setPhase("setup")}
              disabled={isSubmitting}
            >
              Change Wake Word
            </button>
          </div>

          <section className="sample-summary" aria-live="polite">
            <div className="sample-summary-stat">
              <span className="sample-summary-number">
                {readySampleCount} / {TARGET_RECORDINGS}
              </span>
              <span className="sample-summary-label">samples ready</span>
            </div>
            <div className="sample-summary-body">
              <div className="sample-summary-row">
                <span>
                  {readySampleCount} / {TARGET_RECORDINGS} samples ready (
                  {recordedCount} recorded, {uploadedCount} uploaded)
                </span>
                <strong>{remainingSamples === 0 ? "Ready" : `${remainingSamples} to go`}</strong>
              </div>
              <div className="sample-target-track">
                <div
                  className="sample-target-fill"
                  style={{
                    width: `${Math.min(
                      100,
                      Math.round((readySampleCount / TARGET_RECORDINGS) * 100),
                    )}%`,
                  }}
                />
              </div>
              <p className="sample-summary-note">
                Minimum to train: {TARGET_RECORDINGS} valid samples. Recordings and
                successfully uploaded files both count.
              </p>
              {queuedUploadFiles.length > 0 && (
                <p className="sample-summary-note">
                  {queuedUploadFiles.length} selected file(s) are waiting to upload.
                </p>
              )}
            </div>
          </section>

          <div className="add-samples-tabs" role="tablist" aria-label="Sample method">
            <button
              className={activeTab === "record" ? "active" : ""}
              type="button"
              role="tab"
              aria-selected={activeTab === "record"}
              onClick={() => setActiveTab("record")}
            >
              Record
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

          <section className="add-samples-panel" hidden={activeTab !== "record"}>
            <RecordingSession
              wakeWord={wakeWord}
              targetCount={TARGET_RECORDINGS}
              onComplete={handleTrain}
              onRecordingsChange={handleRecordingsChange}
              showSubmitButton={false}
            />
          </section>

          <section className="add-samples-panel" hidden={activeTab !== "upload"}>
            <div
              className={`upload-dropzone ${isDragging ? "dragging" : ""}`}
              onClick={openFilePicker}
              onKeyDown={handleDropzoneKeyDown}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
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
                onClick={(event) => {
                  event.stopPropagation();
                  openFilePicker();
                }}
                disabled={isSubmitting}
              >
                Choose Files
              </button>
            </div>

            {uploadError && (
              <div className="add-samples-submit-error">{uploadError}</div>
            )}

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

          {submitError && (
            <div className="add-samples-submit-error">{submitError}</div>
          )}

          <div className="add-samples-submit">
            <span>
              {remainingSamples > 0
                ? `Need ${remainingSamples} more sample(s)`
                : isSubmitting
                  ? uploadProgress < 70
                    ? "Uploading recorded samples..."
                    : "Starting training job..."
                  : `${readySampleCount} samples ready (${recordedCount} recorded, ${uploadedCount} uploaded)`}
            </span>
            <button
              className="btn btn-primary btn-large"
              type="button"
              disabled={!canTrain}
              onClick={() => void handleTrain()}
            >
              {isSubmitting
                ? "Starting training..."
                : `Train with ${readySampleCount} sample${readySampleCount === 1 ? "" : "s"}`}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
