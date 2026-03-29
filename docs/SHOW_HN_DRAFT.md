# Show HN: ViolaWake - open-source wake words with a real browser training flow

ViolaWake is an open-source wake word stack that does something most OSS wake-word projects still do not: it ships both the SDK and a working web Console, so you can record samples in the browser, train a model, and download the ONNX artifact you actually ship.

```python
from violawake_sdk import WakeDetector
detector = WakeDetector(model="my_word.onnx")
for chunk in detector.stream_mic():
    if detector.detect(chunk): print("wake")
```

One number that matters: our default model hit **5.49% EER** on benchmark v2, and the wake-word runtime footprint is only **102 KB head + 1.33 MB shared backbone**.

Why now: in 2026, voice UX is back in everything, but the default choices are still either closed pricing walls or DIY training setups that break. Open, inspectable wake-word tooling should be easier to ship.

I’d love feedback on three things: real-world negative datasets we should benchmark against, whether the Console workflow is actually simpler than current OSS alternatives, and whether the hosted roadmap is pointed at the right features.
