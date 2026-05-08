/**
 * Render a min/max envelope waveform from a 16-bit PCM mono WAV blob onto a canvas.
 *
 * Reads samples directly from the WAV body (skipping the 44-byte header that
 * `encodeWAV` writes), downsamples to one bucket per pixel, and draws each
 * bucket as a vertical line from min to max. This produces a recognisable
 * "envelope" view of the recording at any width.
 */
export async function drawWaveformFromBlob(
  blob: Blob,
  canvas: HTMLCanvasElement,
  options: { color?: string; background?: string } = {},
): Promise<void> {
  const buffer = await blob.arrayBuffer();
  const view = new DataView(buffer);

  // The fmt sub-chunk in our encoder always sits at offset 12-35; the data
  // header at 36-43; PCM samples start at offset 44. If we ever round-trip
  // through a different encoder we'd need a real WAV parser, but the only
  // producer in this app is wavEncoder.ts which writes a fixed layout.
  const HEADER_LENGTH = 44;
  if (buffer.byteLength <= HEADER_LENGTH) {
    return;
  }

  const sampleCount = (buffer.byteLength - HEADER_LENGTH) / 2;
  const samples = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    const int16 = view.getInt16(HEADER_LENGTH + i * 2, true);
    samples[i] = int16 / 0x8000;
  }

  drawWaveformFromSamples(samples, canvas, options);
}

/**
 * Same as `drawWaveformFromBlob` but takes already-decoded float samples.
 * Useful when the caller already has Float32 in hand (avoids re-parse).
 */
export function drawWaveformFromSamples(
  samples: Float32Array,
  canvas: HTMLCanvasElement,
  options: { color?: string; background?: string } = {},
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const { color = "#7e6cff", background = "transparent" } = options;
  const width = canvas.width;
  const height = canvas.height;
  const mid = height / 2;

  ctx.clearRect(0, 0, width, height);
  if (background !== "transparent") {
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();

  if (samples.length === 0 || width === 0) return;

  const samplesPerBucket = Math.max(1, Math.floor(samples.length / width));
  for (let x = 0; x < width; x++) {
    let min = 1.0;
    let max = -1.0;
    const start = x * samplesPerBucket;
    const end = Math.min(samples.length, start + samplesPerBucket);
    for (let i = start; i < end; i++) {
      const s = samples[i];
      if (s < min) min = s;
      if (s > max) max = s;
    }
    // Map [-1, 1] to [height, 0]
    const yMax = mid - max * mid;
    const yMin = mid - min * mid;
    ctx.moveTo(x + 0.5, yMax);
    ctx.lineTo(x + 0.5, yMin);
  }

  ctx.stroke();
}
