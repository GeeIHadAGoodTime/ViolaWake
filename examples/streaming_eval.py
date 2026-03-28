"""Evaluate false accepts per hour on a WAV file."""

import sys

from violawake_sdk import WakeDetector
from violawake_sdk.tools.streaming_eval import streaming_faph

if len(sys.argv) < 2:
    print("Usage: python streaming_eval.py <audio.wav>")
    sys.exit(1)

detector = WakeDetector(model="temporal_cnn", threshold=0.80, confirm_count=3)
result = streaming_faph(detector, sys.argv[1])
print(f"Duration: {result['total_hours']:.2f}h")
print(f"False accepts: {result['n_false_accepts']}")
print(f"FAPH: {result['faph']:.2f}")
