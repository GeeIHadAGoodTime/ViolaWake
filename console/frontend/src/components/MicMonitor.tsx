import { useCallback, useEffect, useRef, useState } from "react";

interface MicMonitorProps {
  /** Persisted-deviceId callback; parent stores in localStorage and forwards to AudioRecorder. */
  onDeviceChange: (deviceId: string | null) => void;
  /** Currently-selected device. `null` means "browser default". */
  selectedDeviceId: string | null;
}

interface AudioInputDevice {
  deviceId: string;
  label: string;
}

/**
 * Always-on microphone preview: lists available audio inputs, lets the user
 * pick one, and shows a live level meter so they know the mic is hot before
 * the recording session starts.
 *
 * Uses its own getUserMedia stream so it can run independently of the
 * AudioRecorder's per-recording capture stream. Browsers are happy to share
 * the mic across multiple streams from the same origin; the OS-level
 * recording indicator just lights up once.
 *
 * Stops the stream on unmount so the recording indicator clears when the
 * user navigates away.
 */
export default function MicMonitor({
  onDeviceChange,
  selectedDeviceId,
}: MicMonitorProps) {
  const [devices, setDevices] = useState<AudioInputDevice[]>([]);
  const [permissionState, setPermissionState] = useState<
    "pending" | "granted" | "denied" | "error"
  >("pending");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activeLabel, setActiveLabel] = useState<string>("");
  const [level, setLevel] = useState<number>(0); // 0..1
  const [peak, setPeak] = useState<number>(0); // 0..1, decays
  const [clipping, setClipping] = useState<boolean>(false);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>(0);
  const peakRef = useRef<number>(0);
  const peakDecayRef = useRef<number>(0);

  const stopStream = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
  }, []);

  const startStream = useCallback(
    async (deviceId: string | null) => {
      stopStream();
      setErrorMessage(null);
      try {
        const constraints: MediaStreamConstraints = {
          audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
          },
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        setPermissionState("granted");

        // Now that we have permission, enumerate again — labels become
        // populated only after the user grants mic access.
        const enumerated = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = enumerated
          .filter((d) => d.kind === "audioinput")
          .map((d) => ({
            deviceId: d.deviceId,
            label: d.label || `Microphone (${d.deviceId.slice(0, 8)})`,
          }));
        setDevices(audioInputs);

        const track = stream.getAudioTracks()[0];
        const settings = track?.getSettings?.();
        const settledId = settings?.deviceId ?? null;
        const labelMatch = audioInputs.find((d) => d.deviceId === settledId);
        setActiveLabel(
          labelMatch?.label ?? track?.label ?? "Default microphone",
        );

        const ctx = new AudioContext();
        audioContextRef.current = ctx;
        if (ctx.state === "suspended") {
          await ctx.resume();
        }
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.4;
        analyserRef.current = analyser;
        source.connect(analyser);

        const buffer = new Float32Array(analyser.fftSize);
        const tick = () => {
          if (!analyserRef.current) return;
          analyserRef.current.getFloatTimeDomainData(buffer);
          let sumSq = 0;
          let absMax = 0;
          let didClip = false;
          for (let i = 0; i < buffer.length; i++) {
            const v = buffer[i];
            sumSq += v * v;
            const a = Math.abs(v);
            if (a > absMax) absMax = a;
            if (a >= 0.98) didClip = true;
          }
          const rms = Math.sqrt(sumSq / buffer.length);
          // Map RMS (-60 dBFS .. 0 dBFS) onto 0..1 for the meter.
          const dbFs = 20 * Math.log10(Math.max(rms, 1e-7));
          const normalized = Math.max(0, Math.min(1, (dbFs + 60) / 60));
          setLevel(normalized);

          // Peak hold with slow decay (~250ms time constant).
          const now = performance.now();
          if (absMax > peakRef.current) {
            peakRef.current = absMax;
            peakDecayRef.current = now;
          } else {
            const elapsed = now - peakDecayRef.current;
            if (elapsed > 250) {
              peakRef.current = Math.max(0, peakRef.current - 0.01);
            }
          }
          setPeak(peakRef.current);
          setClipping(didClip);

          animFrameRef.current = requestAnimationFrame(tick);
        };
        animFrameRef.current = requestAnimationFrame(tick);
      } catch (err) {
        console.error("MicMonitor getUserMedia failed:", err);
        if (err instanceof DOMException) {
          if (
            err.name === "NotAllowedError" ||
            err.name === "PermissionDeniedError"
          ) {
            setPermissionState("denied");
            setErrorMessage(
              "Microphone permission denied. Allow access in your browser to continue.",
            );
            return;
          }
        }
        setPermissionState("error");
        setErrorMessage(
          err instanceof Error ? err.message : "Could not access microphone",
        );
      }
    },
    [stopStream],
  );

  // Mount: start with the persisted/selected device.
  useEffect(() => {
    startStream(selectedDeviceId);
    return () => {
      stopStream();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-open the stream when the parent changes the selected device.
  useEffect(() => {
    if (permissionState === "granted") {
      startStream(selectedDeviceId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDeviceId]);

  const handlePick = (deviceId: string) => {
    onDeviceChange(deviceId === "" ? null : deviceId);
  };

  return (
    <div className="mic-monitor">
      <div className="mic-monitor-header">
        <span className="mic-monitor-label">Microphone</span>
        {permissionState === "granted" && devices.length > 0 ? (
          <select
            className="mic-monitor-select"
            value={selectedDeviceId ?? ""}
            onChange={(e) => handlePick(e.target.value)}
            aria-label="Select microphone"
          >
            <option value="">System default</option>
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label}
              </option>
            ))}
          </select>
        ) : (
          <span className="mic-monitor-active">{activeLabel || "…"}</span>
        )}
      </div>

      {errorMessage && (
        <div className="mic-monitor-error" role="alert">
          {errorMessage}
        </div>
      )}

      <div
        className={`mic-monitor-meter ${clipping ? "clipping" : ""}`}
        role="meter"
        aria-label="Live microphone level"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(level * 100)}
      >
        <div
          className="mic-monitor-meter-fill"
          style={{ width: `${level * 100}%` }}
        />
        <div
          className="mic-monitor-meter-peak"
          style={{ left: `${Math.min(100, peak * 100)}%` }}
        />
      </div>

      <div className="mic-monitor-hint">
        {permissionState === "pending" && "Waiting for microphone permission…"}
        {permissionState === "granted" &&
          (level < 0.05
            ? "Very quiet — try speaking closer to the mic."
            : clipping
              ? "Signal is clipping — move further away or lower input gain."
              : "Mic is live. Speak to see the meter respond.")}
        {permissionState === "denied" &&
          "Allow microphone access in your browser to enable recording."}
        {permissionState === "error" &&
          "Could not start microphone. Check that no other app is using it."}
      </div>
    </div>
  );
}
