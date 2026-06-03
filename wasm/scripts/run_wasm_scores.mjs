#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { pathToFileURL } from "node:url";

import * as ort from "onnxruntime-web";

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      throw new Error(`Unexpected argument: ${arg}`);
    }
    const key = arg.slice(2);
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = value;
      i++;
    }
  }
  return args;
}

function requireArg(args, name) {
  const value = args[name];
  if (value === undefined || value === true) {
    throw new Error(`Missing required --${name}`);
  }
  return value;
}

function percentile(values, pct) {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil((pct / 100) * sorted.length) - 1);
  return sorted[index];
}

function summarize(values) {
  return {
    count: values.length,
    p50_ms: percentile(values, 50),
    p95_ms: percentile(values, 95),
    p99_ms: percentile(values, 99),
    max_ms: values.length ? Math.max(...values) : 0,
  };
}

function readCorpus(corpusPath, sampleCount, sampleFrames) {
  const bytes = fs.readFileSync(corpusPath);
  const expectedBytes = sampleCount * sampleFrames * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.byteLength < expectedBytes) {
    throw new Error(
      `Corpus byte length mismatch: expected at least ${expectedBytes}, got ${bytes.byteLength}`,
    );
  }
  const arrayBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + expectedBytes);
  return new Float32Array(arrayBuffer);
}

function frameFromSample(samples, sampleIndex, sampleFrames, offset, frameSize) {
  const start = sampleIndex * sampleFrames + offset;
  const end = start + frameSize;
  return samples.slice(start, end);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const corpusPath = path.resolve(requireArg(args, "corpus"));
  const modelDir = path.resolve(requireArg(args, "model-dir"));
  const bundlePath = path.resolve(requireArg(args, "bundle"));
  const ortWasmDir = path.resolve(requireArg(args, "ort-wasm-dir"));
  const sampleCount = Number.parseInt(requireArg(args, "sample-count"), 10);
  const sampleFrames = Number.parseInt(requireArg(args, "sample-frames"), 10);
  const frameSize = Number.parseInt(args["frame-size"] ?? "320", 10);
  const classifierName = args.classifier ?? "temporal_cnn.onnx";

  if (!Number.isInteger(sampleCount) || sampleCount < 1) {
    throw new Error(`Invalid sample-count: ${args["sample-count"]}`);
  }
  if (!Number.isInteger(sampleFrames) || sampleFrames < 1) {
    throw new Error(`Invalid sample-frames: ${args["sample-frames"]}`);
  }
  if (!Number.isInteger(frameSize) || frameSize < 1) {
    throw new Error(`Invalid frame-size: ${args["frame-size"]}`);
  }

  ort.env.wasm.wasmPaths = pathToFileURL(ortWasmDir + path.sep).href;

  const { WakeDetector } = await import(pathToFileURL(bundlePath).href);
  const detector = new WakeDetector({
    melspecModelUrl: path.join(modelDir, "melspectrogram.onnx"),
    embeddingModelUrl: path.join(modelDir, "embedding_model.onnx"),
    classifierModelUrl: path.join(modelDir, classifierName),
    cooldownS: 0,
  });

  const loadStart = performance.now();
  await detector.load();
  const loadMs = performance.now() - loadStart;

  const samples = readCorpus(corpusPath, sampleCount, sampleFrames);
  const allScores = [];
  const latencies = [];
  let firstFrameScoreMs = null;
  let firstTemporalScoreWallMs = null;
  let firstTemporalScoreAudioMs = null;

  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    detector.reset();
    const sampleScores = [];
    for (let offset = 0; offset + frameSize <= sampleFrames; offset += frameSize) {
      const frame = frameFromSample(samples, sampleIndex, sampleFrames, offset, frameSize);
      const scoreStart = performance.now();
      const score = await detector.getScore(frame);
      const scoreMs = performance.now() - scoreStart;
      latencies.push(scoreMs);
      sampleScores.push(score);

      if (firstFrameScoreMs === null) {
        firstFrameScoreMs = scoreMs;
      }
      if (firstTemporalScoreWallMs === null && score !== 0) {
        firstTemporalScoreWallMs = scoreMs;
        firstTemporalScoreAudioMs = offset + frameSize;
      }
    }
    allScores.push(sampleScores);
  }

  detector.dispose();

  process.stdout.write(
    `${JSON.stringify(
      {
        bundle: bundlePath,
        model_dir: modelDir,
        classifier: classifierName,
        sample_count: sampleCount,
        sample_frames: sampleFrames,
        frame_size: frameSize,
        load_ms: loadMs,
        first_frame_score_ms: firstFrameScoreMs ?? 0,
        first_temporal_score_call_ms: firstTemporalScoreWallMs ?? 0,
        first_temporal_score_audio_ms:
          firstTemporalScoreAudioMs === null ? null : (firstTemporalScoreAudioMs / 16000) * 1000,
        frame_latency: summarize(latencies),
        scores: allScores,
      },
      null,
      2,
    )}\n`,
  );
}

main().catch((error) => {
  process.stderr.write(`${error?.stack || error?.message || String(error)}\n`);
  process.exit(1);
});
