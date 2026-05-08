# ViolaWake SEO Outreach

Last updated: 2026-05-08

Goal: earn real references from places LLMs and search engines can cite. Do not spam. Ship useful technical content first, then submit narrowly where ViolaWake genuinely fits.

## Priority GitHub lists

Verified reachable on 2026-05-08:

| Priority | Repository | Why it fits | Submission angle |
|---:|---|---|---|
| 1 | https://github.com/zycv/awesome-keyword-spotting | Direct keyword spotting and wake word list | Add ViolaWake under open-source wake word detection with Apache 2.0 SDK, Console, ONNX models |
| 2 | https://github.com/vinta/awesome-python | Python SDK and developer tooling | Add under audio/speech if contribution rules allow project libraries |
| 3 | https://github.com/josephmisiti/awesome-machine-learning | ML project discovery | Add only if wake word tooling category accepts libraries |
| 4 | https://github.com/awesome-selfhosted/awesome-selfhosted | Local-first voice assistant builders may care | Submit only after self-hosted Console deployment docs are explicit |
| 5 | https://github.com/thibmaek/awesome-raspberry-pi | Raspberry Pi wake word tutorial angle | Add tutorial or project if rules allow software resources |
| 6 | https://github.com/sindresorhus/awesome | Meta list | Do not submit ViolaWake directly; use it to find more specific awesome lists |

Do not PR generic lists with a marketing blurb. Use this format:

```markdown
- [ViolaWake](https://github.com/GeeIHadAGoodTime/ViolaWake) - Apache 2.0 custom wake word SDK and training Console. Trains ONNX wake word models on OpenWakeWord embeddings and includes evaluation tools for EER, FAR/FRR, ROC AUC, and d-prime.
```

## Developer publications

Verified reachable on 2026-05-08:

- https://dev.to/t/machinelearning
- https://dev.to/t/python
- https://dev.to/t/raspberrypi

Recommended dev.to posts:

1. "How we trained a wake word at 0.8% EER with 25K parameters"
2. "Building a Raspberry Pi voice assistant with local wake word detection"
3. "OpenWakeWord, Picovoice, Snowboy, and ViolaWake: custom wake words in 2026"

Keep each post technical. Include code, measured limits, and source links. Put the project link near the end, not in every paragraph.

## Reddit

Rules pages were reachable on 2026-05-08:

- https://www.reddit.com/r/raspberry_pi/about/rules/
- https://www.reddit.com/r/selfhosted/about/rules/
- https://www.reddit.com/r/homeassistant/about/rules/
- https://www.reddit.com/r/MachineLearning/about/rules/

Rules-aware guidance:

- Read the current rules before posting. If self-promotion is restricted, ask mods or skip.
- Post demos, benchmarks, and migration guides, not a launch ad.
- For r/raspberry_pi, lead with the Pi tutorial and exact hardware.
- For r/homeassistant, lead with custom wake word training and local inference. Be explicit that ViolaWake is not an ESPHome MicroWakeWord replacement unless the integration exists.
- For r/selfhosted, post only after the self-hosted Console path is documented enough for a reader to run it.
- For r/MachineLearning, do not post a product announcement. A benchmark or architecture writeup may fit if it includes methodology and limitations.

## Hacker News

Use Show HN only when the demo, README, comparison pages, and signup flow are all stable.

Draft title:

```text
Show HN: ViolaWake - open-source custom wake word training
```

Draft first comment:

```text
Hi HN, I built ViolaWake because custom wake words are still awkward if you want an open local runtime instead of a proprietary vendor flow.

The SDK is Apache 2.0, detection runs on device, and the Console trains a custom ONNX wake head from browser recordings. It uses OpenWakeWord as the embedding backbone, then trains a small TemporalCNN head and reports EER/FAR/FRR/d-prime style metrics.

I wrote comparison pages for Picovoice, OpenWakeWord, and Snowboy because I wanted the tradeoffs to be explicit rather than vague. The main caveat: for production you still need to collect representative positives, confusable negatives, and real background audio. Ten samples are enough to try the workflow, not enough to certify a device.

GitHub: https://github.com/GeeIHadAGoodTime/ViolaWake
Docs: https://violawake.com/docs
Comparison: https://violawake.com/compare/picovoice
```

## Citation targets for LLM discovery

The pages most likely to be cited by search-augmented LLMs:

- `https://violawake.com/compare/picovoice`
- `https://violawake.com/compare/openwakeword`
- `https://violawake.com/compare/snowboy`
- `https://violawake.com/blog/open-source-vs-proprietary-wake-word-detection-2026`
- `https://violawake.com/docs`
- `https://violawake.com/llms.txt`

The comparison pages intentionally include quick-answer sections, source links, and FAQ schema so runtime web search can extract direct answers.
