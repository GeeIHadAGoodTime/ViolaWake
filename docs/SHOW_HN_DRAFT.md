# Show HN: ViolaWake – Open-source wake word engine (alternative to Porcupine)

I've been building ViolaWake as an open-source wake word engine for people who want something closer to Porcupine's UX/performance tradeoff without the closed training stack or commercial licensing wall.

The repo is Apache 2.0, the training code is open, and the models run with ONNX at inference time. The main thing I wanted to make inspectable was evaluation: the SDK includes tooling for Cohen's d, FAR, FRR, and ROC-style reporting instead of "trust me, it works." One honest caveat: the headline separability number in the repo is measured against synthetic negatives, not a large adversarial speech-negative benchmark. I think that's still useful, but I don't want to oversell what it means in the real world.

It also ships as more than just a detector. There is a bundled wake -> STT -> TTS pipeline in Python, so you can prototype a full local voice loop without stitching together three separate projects. If you want a lower-friction training flow, there's also a Console in the repo for browser-based training and model download.

Install is just:

```bash
pip install violawake
```

GitHub: https://github.com/GeeIHadAGoodTime/ViolaWake

What I'd really like feedback on:
- Accuracy testing on diverse real-world datasets, especially speech-heavy negatives
- Platform support priorities (macOS, Raspberry Pi, Android, browser/WASM, etc.)
- Whether the Console and training workflow are actually simpler than the current open-source alternatives
- Any obvious gaps in the evaluation methodology or claims
