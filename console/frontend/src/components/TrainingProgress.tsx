import { useEffect, useRef } from "react";
import { useTraining } from "../hooks/useTraining";
import { trackEvent } from "../analytics";

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
    case "running":
      return "Running";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
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
      trackEvent("training_completed");
      onComplete(state.modelId, state.dPrime);
    }
  }, [state.status, state.modelId, state.dPrime, onComplete]);

  return (
    <div className="training-progress">
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {`Training ${getStatusLabel(state.status)}. ${Math.round(state.progress)}% complete.${state.totalEpochs > 0 ? ` Epoch ${state.epoch} of ${state.totalEpochs}.` : ""} ${state.error ?? state.message}`}
      </div>

      {/* Status badge */}
      <div className={`training-status-badge status-${state.status}`}>
        {getStatusLabel(state.status)}
        {!state.connected && state.status === "running" && (
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
      {(state.status === "failed" || state.status === "cancelled") &&
        state.error && (
        <div className="training-error">
          <p className="error-message">{state.error}</p>
        </div>
      )}

      {/* Recovery tips: shown on any failed run so the next attempt is more
          likely to pass, not just a repeat of the same one that just failed.
          Training has real run-to-run variance (see the quality-gate message
          above), so "try again" is genuinely the right first move before
          anyone re-records from scratch. */}
      {state.status === "failed" && (
        <div className="training-tips">
          <p className="training-tips-title">Training failed. Here is what to try next</p>
          <ul>
            <li>
              Training results vary a little from run to run. Try training again with the
              same recordings before you record anything new.
            </li>
            <li>
              If it keeps failing for the same wake word, record a few more samples with a
              little variation in pace and distance from the microphone, rather than
              repeating the exact same delivery each time.
            </li>
            <li>
              Speak at a normal indoor volume, away from a fan, TV, or music, and say the
              whole wake word in one go right after you start recording.
            </li>
            <li>
              Still stuck after a few tries? <a href="/contact">Contact support</a> and we
              will look into it.
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
