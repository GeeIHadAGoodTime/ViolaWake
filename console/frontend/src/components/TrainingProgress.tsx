import { useEffect, useRef } from "react";
import { useTraining } from "../hooks/useTraining";

interface TrainingProgressProps {
  jobId: number;
  onComplete: (modelId: number, dPrime: number) => void;
}

function getDPrimeGrade(dPrime: number): {
  label: string;
  className: string;
} {
  if (dPrime >= 15) return { label: "Excellent", className: "grade-excellent" };
  if (dPrime >= 10) return { label: "Good", className: "grade-good" };
  if (dPrime >= 5) return { label: "Fair", className: "grade-fair" };
  return { label: "Poor", className: "grade-poor" };
}

function getStatusLabel(status: string): string {
  switch (status) {
    case "queued":
      return "Queued";
    case "training":
      return "Training";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    default:
      return status;
  }
}

export default function TrainingProgress({
  jobId,
  onComplete,
}: TrainingProgressProps) {
  const state = useTraining(jobId);
  const completeFiredRef = useRef(false);

  // Trigger onComplete when training finishes successfully
  useEffect(() => {
    if (
      state.status === "completed" &&
      state.modelId !== null &&
      state.dPrime !== null &&
      !completeFiredRef.current
    ) {
      completeFiredRef.current = true;
      onComplete(state.modelId, state.dPrime);
    }
  }, [state.status, state.modelId, state.dPrime, onComplete]);

  return (
    <div className="training-progress">
      {/* Status badge */}
      <div className={`training-status-badge status-${state.status}`}>
        {getStatusLabel(state.status)}
        {!state.connected && state.status === "training" && (
          <span className="status-reconnecting">
            {" "}
            (reconnecting...)
          </span>
        )}
      </div>

      {/* Progress bar */}
      <div
        className="training-progress-bar"
        role="progressbar"
        aria-valuenow={Math.round(state.progress)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Training progress: ${Math.round(state.progress)}%`}
      >
        <div className="training-progress-track">
          <div
            className="training-progress-fill"
            style={{ width: `${state.progress}%` }}
          />
        </div>
        <span className="training-progress-pct">
          {Math.round(state.progress)}%
        </span>
      </div>

      {/* Epoch counter */}
      {state.totalEpochs > 0 && (
        <div className="training-epochs">
          Epoch {state.epoch} / {state.totalEpochs}
        </div>
      )}

      {/* Loss values */}
      {(state.trainLoss !== null || state.valLoss !== null) && (
        <div className="training-losses">
          {state.trainLoss !== null && (
            <div className="loss-item">
              <span className="loss-label">Train Loss</span>
              <span className="loss-value">
                {state.trainLoss.toFixed(4)}
              </span>
            </div>
          )}
          {state.valLoss !== null && (
            <div className="loss-item">
              <span className="loss-label">Val Loss</span>
              <span className="loss-value">
                {state.valLoss.toFixed(4)}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Message */}
      <p className="training-message">{state.message}</p>

      {/* Completion result */}
      {state.status === "completed" && state.dPrime !== null && (
        <div className="training-result">
          <div className="result-dprime">
            <span className="dprime-label">d-prime</span>
            <span className="dprime-value">
              {state.dPrime.toFixed(2)}
            </span>
            <span
              className={`dprime-grade ${getDPrimeGrade(state.dPrime).className}`}
            >
              {getDPrimeGrade(state.dPrime).label}
            </span>
          </div>
        </div>
      )}

      {/* Error state */}
      {state.status === "failed" && state.error && (
        <div className="training-error">
          <p className="error-message">{state.error}</p>
        </div>
      )}
    </div>
  );
}
