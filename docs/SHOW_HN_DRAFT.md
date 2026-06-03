# Show HN: ViolaWake - open-source wake word SDK with browser training

ViolaWake is an open-source wake word SDK that ships both the training pipeline and a web Console. Record samples in the browser, train a temporal CNN model, and download the ONNX artifact.

```bash
pip install "violawake[audio,download,oww]"
```

```python
from violawake_sdk import WakeDetector
detector = WakeDetector(model="temporal_cnn", threshold=0.80, confirm_count=3)
for chunk in detector.stream_mic():
    if detector.detect(chunk): print("Wake word detected!")
```

Train your own custom wake word from the terminal:

```bash
violawake-train --wake-word "hey jarvis" --positives ./recordings/ --output jarvis.onnx
```

Benchmark numbers: `benchmark_v2/run_benchmark.py` over `benchmark_v2/corpus` reports `temporal_cnn` at **5.49% EER** versus openWakeWord Alexa at **8.24% EER** on a shared **700-file negative corpus** and **180 synthetic positives per system**. This is the public synthetic Edge-TTS benchmark, not a real-speaker production-eval claim.

The training pipeline is the real story. It can generate TTS positives, phonetically-confusable negatives, and optional LibriSpeech or MUSAN negatives if those corpora are available. A post-training quality gate reports EER, FAR, FRR, and ROC AUC before a model ships.

All training code is open (Apache 2.0). Same pipeline runs via CLI, Python API, or the web Console with the same training entrypoints.

Links: GitHub (https://github.com/GeeIHadAGoodTime/ViolaWake) | PyPI (https://pypi.org/project/violawake/) | Console (https://violawake.com)

I'd love feedback on: real-world negative datasets for benchmarking, whether the Console workflow is simpler than current alternatives, and what use cases you'd want wake words for.
