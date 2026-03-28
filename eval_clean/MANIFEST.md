# Clean Evaluation Set Manifest

**Generated**: 2026-03-26 18:06 UTC
**Script**: `tools/build_clean_eval_set.py`
**Seed**: 42
**Generation time**: 247.1s

## Summary

| Category | Count |
|----------|-------|
| Positives (original) | 96 |
| Positives (reverb variant) | 96 |
| Positives (noisy variant) | 96 |
| **Total positives** | **288** |
| Negatives (adversarial TTS) | 260 |
| Negatives (generic speech) | 270 |
| Negatives (LibriSpeech) | 0 |
| Negatives (noise/silence) | 16 |
| **Total negatives** | **546** |
| **Grand total** | **834** |

## Voices Used

| Engine | Unique Voices |
|--------|---------------|
| Edge TTS | 22 |
| pyttsx3 (SAPI5) | 2 |
| Kokoro | 0 |

## Methodology

### Zero Training Overlap Guarantee

This eval set has **zero overlap** with training data:
- All positives are freshly synthesized via TTS (Edge TTS, pyttsx3, Kokoro)
- No recorded audio from any training corpus is included
- LibriSpeech test-clean is a held-out partition never used in ViolaWake training
- Adversarial negatives are TTS-generated confusable words
- Noise samples are procedurally generated with a fixed seed

### Positive Samples

**Phrases**: "viola", "hey viola", "ok viola", "viola wake up"

**Augmentation variants** (2 per original):
- **Reverb**: Convolution with a synthetic exponentially-decaying IR (RT60=0.3s),
  70% dry / 30% wet mix. Simulates small room acoustics.
- **Noisy**: Pink noise added at 20 dB SNR. Simulates moderate background noise.

All samples are 16kHz mono, 16-bit PCM WAV, padded/trimmed to 1.5s (24000 samples).

### Negative Samples

**Adversarial TTS**: Confusable words synthesized with 10 diverse voices.
Words include: "violet", "violin", "violence", "violent", "villa", "vanilla", "valley", "volume", "hola", "voila"... (26 total)

**Generic speech**: 30 common English phrases synthesized with
10 voices. These test that the model does not trigger on general speech.

**LibriSpeech test-clean**: Random 1.5s clips from the standard ASR held-out test set
(different speakers, read speech). Skipped (--skip-librispeech or torchaudio unavailable).

**Noise/silence**: Procedurally generated silence, white noise, pink noise, room tone,
hum, clicks, static, and Brownian noise. These confirm the model does not trigger on
non-speech audio.

## Audio Format

- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Bit depth: 16-bit signed integer PCM
- Duration: 1.5s (24000 samples)
- Container: WAV

## Reproducibility

All random operations use numpy with seed=42. Re-running with the same
seed and same TTS engine versions will produce identical noise/augmentation variants.
TTS outputs themselves may vary slightly across engine versions.

## Errors Encountered

- **Kokoro TTS (36 errors)**: All 9 Kokoro voices x 4 phrases failed because the
  Kokoro ONNX models are not downloaded on this machine. To add Kokoro samples,
  run `violawake-download --model kokoro_v1_0 && violawake-download --model kokoro_voices_v1_0`
  then re-run the script.
- **Edge TTS (8 failures)**: 2 Edge TTS voices (en-US-DavisNeural, en-US-AmberNeural)
  failed for all 4 phrases due to transient network errors. 22/24 voices succeeded (88/96 samples = 92%).
  Re-running the script will retry these voices.
