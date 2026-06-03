"""Basic wake word detection example.

Uses only detect() for the boolean decision. To inspect the raw score,
read detector.last_scores[-1] *after* detect() -- this avoids running
inference twice per frame (detect() already calls the scoring engine
internally).

Requires: pip install "violawake[oww]"
          python -c "from openwakeword.utils import download_models; download_models()"
Optional microphone path: pip install "violawake[audio]"
"""

import numpy as np

from violawake_sdk import WakeDetector
from violawake_sdk._exceptions import AudioCaptureError, ModelNotFoundError

try:
    detector = WakeDetector(model="temporal_cnn", threshold=0.80)
except (ModelNotFoundError, FileNotFoundError):
    print("Model not found. Run first: violawake-download --model temporal_cnn")
    raise SystemExit(1)

print("Listening for 'Viola'... (say it!)")
try:
    for chunk in detector.stream_mic():
        if detector.detect(chunk):
            score = detector.last_scores[-1] if detector.last_scores else 0.0
            print(f"Wake word detected! (score={score:.3f})")
            break
except (ImportError, AudioCaptureError) as exc:
    print(f"Microphone unavailable: {exc}")
    silence = np.zeros(320, dtype=np.float32)
    detector.detect(silence)
    score = detector.last_scores[-1] if detector.last_scores else 0.0
    print(f"Detector initialized. Synthetic silence score={score:.3f}")
except KeyboardInterrupt:
    print("\nStopped.")
