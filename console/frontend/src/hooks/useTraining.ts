import { useState, useEffect, useCallback, useRef } from "react";
import type { TrainingEvent, TrainingStatus } from "../types";
import * as api from "../api";

interface TrainingState {
  status: TrainingStatus;
  progress: number;
  epoch: number;
  totalEpochs: number;
  trainLoss: number | null;
  valLoss: number | null;
  dPrime: number | null;
  modelId: number | null;
  message: string;
  error: string | null;
  connected: boolean;
}

const INITIAL_STATE: TrainingState = {
  status: "queued",
  progress: 0,
  epoch: 0,
  totalEpochs: 0,
  trainLoss: null,
  valLoss: null,
  dPrime: null,
  modelId: null,
  message: "Waiting for training to start...",
  error: null,
  connected: false,
};

const POLL_INTERVAL_MS = 3000;

function getStatusMessage(status: TrainingStatus, error: string | null): string {
  switch (status) {
    case "queued":
      return "Waiting for training to start...";
    case "running":
      return "Training is in progress.";
    case "completed":
      return "Training complete.";
    case "failed":
      return error || "Training failed.";
    case "cancelled":
      return error || "Training cancelled.";
    default:
      return "Training status updated.";
  }
}

function isTerminal(status: TrainingStatus): boolean {
  return (
    status === "completed" ||
    status === "failed" ||
    status === "cancelled"
  );
}

export function useTraining(jobId: number) {
  const [state, setState] = useState<TrainingState>(INITIAL_STATE);
  const eventSourceRef = useRef<EventSource | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearPolling = useCallback(() => {
    if (pollIntervalRef.current !== null) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const pollStatus = useCallback(async () => {
    try {
      const job = await api.getTrainingStatus(jobId);
      setState((prev) => ({
        ...prev,
        status: job.status,
        progress: job.progress,
        dPrime: job.d_prime ?? prev.dPrime,
        modelId: job.model_id ?? prev.modelId,
        error: job.error,
        message: getStatusMessage(job.status, job.error),
      }));

      // Stop polling on terminal states
      if (isTerminal(job.status)) {
        clearPolling();
      }
    } catch {
      // polling error, will retry on next interval
    }
  }, [jobId, clearPolling]);

  const startPolling = useCallback(() => {
    clearPolling();
    // Poll immediately, then every POLL_INTERVAL_MS
    pollStatus();
    pollIntervalRef.current = setInterval(pollStatus, POLL_INTERVAL_MS);
  }, [pollStatus, clearPolling]);

  useEffect(() => {
    let disposed = false;

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    void api
      .createTrainingStream(jobId)
      .then((es) => {
        if (disposed) {
          es.close();
          return;
        }

        eventSourceRef.current = es;

        es.onopen = () => {
          setState((prev) => ({ ...prev, connected: true }));
        };

        es.onmessage = (event) => {
          try {
            const data: TrainingEvent = JSON.parse(event.data);
            setState((prev) => ({
              ...prev,
              status: data.status,
              progress: data.progress,
              epoch: data.epoch,
              totalEpochs: data.total_epochs,
              trainLoss: data.train_loss ?? prev.trainLoss,
              valLoss: data.val_loss ?? prev.valLoss,
              dPrime: data.d_prime ?? prev.dPrime,
              modelId: data.model_id ?? prev.modelId,
              message: data.message || prev.message,
              error: data.error,
              connected: true,
            }));

            if (isTerminal(data.status)) {
              es.close();
            }
          } catch {
            // ignore malformed events
          }
        };

        es.onerror = () => {
          setState((prev) => ({ ...prev, connected: false }));
          es.close();
          startPolling();
        };
      })
      .catch(() => {
        startPolling();
      });

    return () => {
      disposed = true;
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      clearPolling();
    };
  }, [jobId, startPolling, clearPolling]);

  return state;
}
