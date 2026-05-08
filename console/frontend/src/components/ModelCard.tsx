import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { Model, ModelConfig, QualityGrade } from "../types";
import { getModelDownloadUrl, getModelConfig, deleteModel } from "../api";
import { useToast } from "../contexts/ToastContext";

interface ModelCardProps {
  model: Model;
  onDeleted?: (modelId: number) => void;
}

const GRADE_LABELS: Record<QualityGrade, string> = {
  A: "Excellent",
  B: "Good",
  C: "Acceptable",
  F: "Failed",
};

function normalizeQualityGrade(value: unknown): QualityGrade | null {
  if (typeof value !== "string") {
    return null;
  }

  const grade = value.trim().toUpperCase();
  if (grade === "A" || grade === "B" || grade === "C" || grade === "F") {
    return grade;
  }

  return null;
}

function getConfigQualityGrade(config: ModelConfig | null): QualityGrade | null {
  if (!config) {
    return null;
  }

  return (
    normalizeQualityGrade(config.training_config.quality_grade) ??
    normalizeQualityGrade(config.training_config.quality_gate?.grade)
  );
}

function getGradePill(grade: QualityGrade | null): {
  className: string;
  label: string;
} {
  if (grade === null) {
    return {
      className: "grade-pill-unknown",
      label: "Grade unavailable",
    };
  }

  return {
    className: `grade-pill-${grade.toLowerCase()}`,
    label: `Grade ${grade} - ${GRADE_LABELS[grade]}`,
  };
}

function formatGradeValue(grade: QualityGrade | null): string {
  if (grade === null) {
    return "Unavailable";
  }
  return `${grade} - ${GRADE_LABELS[grade]}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024)
    return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function formatNullableMetric(
  value: number | null,
  digits: number,
  suffix = "",
): string {
  if (value === null || Number.isNaN(value)) {
    return "Unavailable";
  }
  return `${value.toFixed(digits)}${suffix}`;
}

function formatFarPerHour(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Unavailable";
  }

  const digits = value < 0.01 ? 3 : 2;
  return `~${value.toFixed(digits)}/hr`;
}

function formatRecall(frr: number | null | undefined): string {
  if (frr === null || frr === undefined || Number.isNaN(frr)) {
    return "Unavailable";
  }

  const recall = Math.min(100, Math.max(0, (1 - frr) * 100));
  return `${recall.toFixed(recall === Math.round(recall) ? 0 : 1)}%`;
}

export default function ModelCard({ model, onDeleted }: ModelCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [config, setConfig] = useState<ModelConfig | null>(null);
  const [loadingConfig, setLoadingConfig] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const { addToast } = useToast();
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const cancelButtonRef = useRef<HTMLButtonElement | null>(null);

  const grade = normalizeQualityGrade(model.quality_grade) ?? getConfigQualityGrade(config);
  const gradePill = getGradePill(grade);

  useEffect(() => {
    if (!showDeleteConfirm) {
      return;
    }

    const dialog = dialogRef.current;
    const previousActiveElement = document.activeElement as HTMLElement | null;
    cancelButtonRef.current?.focus();

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !deleting) {
        setShowDeleteConfirm(false);
        return;
      }

      if (event.key !== "Tab" || !dialog) {
        return;
      }

      const focusableElements = Array.from(
        dialog.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      );

      if (focusableElements.length === 0) {
        return;
      }

      const first = focusableElements[0];
      const last = focusableElements[focusableElements.length - 1];

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previousActiveElement?.focus();
    };
  }, [deleting, showDeleteConfirm]);

  useEffect(() => {
    let cancelled = false;

    async function loadConfig() {
      setLoadingConfig(true);
      try {
        const cfg = await getModelConfig(model.id);
        if (!cancelled) {
          setConfig(cfg);
        }
      } catch {
        if (!cancelled) {
          setConfig(null);
        }
      } finally {
        if (!cancelled) {
          setLoadingConfig(false);
        }
      }
    }

    loadConfig();

    return () => {
      cancelled = true;
    };
  }, [model.id]);

  function toggleExpand() {
    setExpanded(!expanded);
  }

  async function handleDownload() {
    const url = await getModelDownloadUrl(model.id);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${model.wake_word}.onnx`;
    a.click();
  }

  async function handleDelete() {
    setDeleting(true);
    try {
      await deleteModel(model.id);
      addToast("success", `Model "${model.wake_word}" deleted.`);
      onDeleted?.(model.id);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to delete model";
      addToast("error", message);
    }
    setDeleting(false);
    setShowDeleteConfirm(false);
  }

  return (
    <div className="model-card">
      <div className="model-card-header">
        <h3 className="model-name">{model.wake_word}</h3>
        <span className={`grade-pill ${gradePill.className}`}>
          {gradePill.label}
        </span>
      </div>

      <div className="model-card-meta">
        <span className="model-date">
          {formatDate(model.created_at)}
        </span>
        <span className="model-size">
          {formatBytes(model.size_bytes)}
        </span>
      </div>

      <div className="model-card-stat-row">
        <span>
          False alarms:{" "}
          {loadingConfig && !config
            ? "Loading..."
            : formatFarPerHour(config?.far_per_hour)}
        </span>
        <span aria-hidden="true">&middot;</span>
        <span>
          Recall:{" "}
          {loadingConfig && !config
            ? "Loading..."
            : formatRecall(config?.frr)}
        </span>
      </div>

      <div className="model-card-actions">
        <Link
          className="btn btn-ghost"
          to={`/model/${model.id}/performance`}
        >
          View Performance
        </Link>
        <button
          className="btn btn-primary"
          onClick={handleDownload}
        >
          Download .onnx
        </button>
        <Link
          className="btn btn-ghost"
          to={`/record/${encodeURIComponent(model.wake_word)}/add`}
        >
          Add samples &amp; retrain
        </Link>
        <button className="btn btn-ghost" onClick={toggleExpand}>
          {expanded ? "Hide details" : "View details"}
        </button>
        <button
          className="btn btn-danger"
          onClick={() => setShowDeleteConfirm(true)}
          disabled={deleting}
        >
          Delete
        </button>
      </div>

      {showDeleteConfirm && (
        <div
          className="delete-confirm-overlay"
          onClick={() => {
            if (!deleting) {
              setShowDeleteConfirm(false);
            }
          }}
        >
          <div
            className="delete-confirm"
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={`delete-model-title-${model.id}`}
            aria-describedby={`delete-model-description-${model.id}`}
            onClick={(event) => event.stopPropagation()}
          >
            <p
              className="delete-confirm-text"
              id={`delete-model-title-${model.id}`}
            >
              Permanently delete the <strong>{model.wake_word}</strong> model?
            </p>
            <p
              className="delete-confirm-subtext"
              id={`delete-model-description-${model.id}`}
            >
              This cannot be undone.
            </p>
            <div className="delete-confirm-actions">
              <button
                className="btn btn-danger"
                onClick={handleDelete}
                disabled={deleting}
              >
                {deleting ? "Deleting..." : "Yes, delete"}
              </button>
              <button
                className="btn btn-ghost"
                ref={cancelButtonRef}
                onClick={() => setShowDeleteConfirm(false)}
                disabled={deleting}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {expanded && (
        <div className="model-card-details">
          {loadingConfig ? (
            <p className="loading-text">Loading config...</p>
          ) : config ? (
            <div className="config-grid">
              <div className="config-item">
                <span className="config-label">SDK grade</span>
                <span className="config-value">
                  {formatGradeValue(getConfigQualityGrade(config))}
                </span>
              </div>
              <div className="config-item">
                <span className="config-label">
                  False alarms/hr
                </span>
                <span className="config-value">
                  {formatFarPerHour(config.far_per_hour)}
                </span>
              </div>
              <div className="config-item">
                <span className="config-label">
                  Recall
                </span>
                <span className="config-value">
                  {formatRecall(config.frr)}
                </span>
              </div>
              <div className="config-item">
                <span className="config-label">d-prime</span>
                <span className="config-value">
                  {formatNullableMetric(config.d_prime, 2)}
                </span>
              </div>
            </div>
          ) : (
            <p className="loading-text">
              Could not load config details.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
