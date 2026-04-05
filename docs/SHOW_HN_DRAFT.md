# Show HN: ViolaWake - open-source wake word SDK with browser training

ViolaWake is an open-source wake word SDK that ships both the training pipeline and a web Console. Record 10 samples in the browser, train a temporal CNN model, download the ONNX artifact. The whole thing runs on CPU in under 3 minutes.

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

Key numbers: our temporal CNN production model hits **d’=8.577, EER 0.8%, AUC 0.9993** with only **25K params and <5ms inference**. The full model is 102 KB + 1.33 MB shared OWW backbone.

The training pipeline is the real story. From just 10-20 voice recordings, it auto-generates a full training corpus: TTS positives across 20 diverse voices, two rounds of phonetically-confusable negatives, 100+ common speech phrases as negatives, plus LibriSpeech and MUSAN data if available. A post-training quality gate (A/B/C/F grading) blocks bad models from shipping. We proved this works by training a "big chungus" wake word from 20 recordings — Grade A, zero false positives on general speech.

All training code is open (Apache 2.0). Same pipeline runs via CLI, Python API, or the web Console — identical results.

Links: GitHub (https://github.com/GeeIHadAGoodTime/ViolaWake) | PyPI (https://pypi.org/project/violawake/) | Console (https://violawake.com)

I’d love feedback on: real-world negative datasets for benchmarking, whether the Console workflow is simpler than current alternatives, and what use cases you’d want wake words for.
