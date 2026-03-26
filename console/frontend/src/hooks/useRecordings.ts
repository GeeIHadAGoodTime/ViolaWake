import { useState, useCallback } from "react";
import * as api from "../api";

interface RecordingEntry {
  id: number;
  blob: Blob;
  duration: number;
  uploaded: boolean;
}

interface UseRecordingsReturn {
  recordings: RecordingEntry[];
  uploading: boolean;
  error: string | null;
  addRecording: (index: number, blob: Blob, duration: number) => void;
  replaceRecording: (
    index: number,
    blob: Blob,
    duration: number,
  ) => void;
  uploadAll: (
    wakeWord: string,
  ) => Promise<number[]>;
  reset: () => void;
}

export function useRecordings(targetCount: number): UseRecordingsReturn {
  const [recordings, setRecordings] = useState<RecordingEntry[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addRecording = useCallback(
    (index: number, blob: Blob, duration: number) => {
      setRecordings((prev) => {
        const next = [...prev];
        next[index] = { id: -1, blob, duration, uploaded: false };
        return next;
      });
    },
    [],
  );

  const replaceRecording = useCallback(
    (index: number, blob: Blob, duration: number) => {
      setRecordings((prev) => {
        const next = [...prev];
        next[index] = { id: -1, blob, duration, uploaded: false };
        return next;
      });
    },
    [],
  );

  const uploadAll = useCallback(
    async (wakeWord: string): Promise<number[]> => {
      setUploading(true);
      setError(null);

      const validRecordings = recordings.filter(
        (r) => r !== undefined,
      );
      if (validRecordings.length < targetCount) {
        setError(
          `Need ${targetCount} recordings, have ${validRecordings.length}`,
        );
        setUploading(false);
        return [];
      }

      try {
        const ids: number[] = [];
        for (let i = 0; i < recordings.length; i++) {
          const rec = recordings[i];
          if (!rec) continue;

          const result = await api.uploadRecording(
            rec.blob,
            wakeWord,
            i,
          );
          ids.push(result.recording_id);

          setRecordings((prev) => {
            const next = [...prev];
            if (next[i]) {
              next[i] = { ...next[i], id: result.recording_id, uploaded: true };
            }
            return next;
          });
        }

        setUploading(false);
        return ids;
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Upload failed";
        setError(message);
        setUploading(false);
        return [];
      }
    },
    [recordings, targetCount],
  );

  const reset = useCallback(() => {
    setRecordings([]);
    setError(null);
  }, []);

  return {
    recordings,
    uploading,
    error,
    addRecording,
    replaceRecording,
    uploadAll,
    reset,
  };
}
