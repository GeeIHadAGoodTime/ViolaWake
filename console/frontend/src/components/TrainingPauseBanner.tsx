import { useCallback, useEffect, useState } from "react";
import type { CircuitBreakerState } from "../types";
import { getCircuitBreakerState, resumeTraining } from "../api";

/**
 * Surfaces a paused training queue in-product and lets the user clear it.
 *
 * Before #4207 the console had no failure-state channel at all: a customer whose
 * account was paused after repeated failures saw nothing in-product and had no
 * way out (the resume endpoint had no caller). This banner reads the breaker
 * state and exposes the one action that clears a pause.
 *
 * It renders nothing while loading, on error, or when the queue is not paused,
 * so it is safe to drop onto any page.
 */
export default function TrainingPauseBanner({
  onResumed,
}: {
  onResumed?: () => void;
}) {
  const [state, setState] = useState<CircuitBreakerState | null>(null);
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState<string | null>(null);

  const load = useCallback(() => {
    getCircuitBreakerState()
      .then(setState)
      .catch(() => {
        // Breaker state not available (e.g. billing/queue not configured) —
        // silently render nothing rather than break the page.
        setState(null);
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleResume = useCallback(async () => {
    setResuming(true);
    setResumeError(null);
    try {
      await resumeTraining();
      setState((prev) => (prev ? { ...prev, paused: false } : prev));
      onResumed?.();
    } catch (err) {
      setResumeError(
        err instanceof Error ? err.message : "Failed to resume training",
      );
    } finally {
      setResuming(false);
    }
  }, [onResumed]);

  if (!state || !state.paused) {
    return null;
  }

  return (
    <div className="dashboard-error" role="alert">
      <div>
        <strong>Training is paused.</strong> Your account was paused after
        repeated training failures, so new runs will not start until you resume.
        {state.pause_reason ? (
          <>
            {" "}
            Last reason: {state.pause_reason}
          </>
        ) : null}
      </div>
      <div className="dashboard-upgrade-actions">
        <button
          className="btn btn-primary"
          onClick={handleResume}
          disabled={resuming}
        >
          {resuming ? "Resuming…" : "Resume training"}
        </button>
      </div>
      {resumeError && (
        <p role="status" className="dashboard-empty-note">
          {resumeError}
        </p>
      )}
    </div>
  );
}
