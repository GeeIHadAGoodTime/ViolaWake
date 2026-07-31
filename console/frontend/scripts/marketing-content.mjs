export const site = {
  name: "ViolaWake",
  origin: "https://violawake.com",
  github: "https://github.com/GeeIHadAGoodTime/ViolaWake",
  apiDocs: "https://geeihadagoodtime.github.io/ViolaWake/",
  contactEmail: "hello@violawake.com",
  description:
    "Open-source custom wake word training, on-device detection, and an Apache 2.0 Python SDK for voice activation.",
};

export const sources = {
  picovoiceProduct: "https://picovoice.ai/platform/porcupine/",
  picovoicePricing: "https://picovoice.ai/pricing/",
  picovoiceDocs: "https://picovoice.ai/docs/porcupine/",
  picovoiceQuickStart: "https://picovoice.ai/docs/quick-start/porcupine-python/",
  picovoiceAndroidQuickStart: "https://picovoice.ai/docs/quick-start/porcupine-android/",
  picovoiceFaq: "https://picovoice.ai/docs/faq/porcupine/",
  picovoiceTerms: "https://picovoice.ai/docs/terms-of-use/",
  openWakeWord: "https://github.com/dscripka/openWakeWord",
  snowboy: "https://github.com/Kitt-AI/snowboy",
  snowboyLicense: "https://github.com/Kitt-AI/snowboy/blob/master/LICENSE",
  googleKws:
    "https://github.com/google-research/google-research/tree/master/kws_streaming",
  llmsTxt: "https://llmstxt.org/",
};

const nav = [
  { label: "Docs", href: "/docs" },
  { label: "Compare", href: "/compare/picovoice" },
  { label: "Blog", href: "/blog" },
  { label: "FAQ", href: "/faq" },
];

const commonLinks = `
## Keep exploring

- [SDK docs](/docs)
- [Picovoice comparison](/compare/picovoice)
- [OpenWakeWord comparison](/compare/openwakeword)
- [Raspberry Pi tutorial](/blog/raspberry-pi-voice-assistant-violawake)
`;

const verifiedDate = "2026-05-08";

const comparisonTrademarkNotice = `Comparison information accurate as of 2026-05-08. Picovoice and Porcupine are trademarks of Picovoice Inc.; OpenWakeWord is a project of David Scripka; Snowboy was a trademark of Kitt.AI (acquired by Baidu, deprecated 2020). All trademarks belong to their respective owners and are used here for nominative comparison only. ViolaWake is not affiliated with, endorsed by, or sponsored by these projects or companies. Report inaccuracies to hello@violawake.com.`;

const comparisonFaqs = {
  picovoice: [
    {
      q: "Is ViolaWake a Picovoice alternative?",
      a:
        "Yes. ViolaWake is an open-source alternative for teams that want custom wake word training, ONNX models, an Apache 2.0 SDK, and runtime detection without API keys.",
    },
    {
      q: "Does ViolaWake require cloud inference?",
      a:
        "No. Detection runs on device. The hosted Console is only for account, recording, training, and model management workflows.",
    },
    {
      q: "Is the ViolaWake Console free?",
      a:
        "Yes. The Console is a free service: sign up, record samples, train models, and download ONNX files at no charge, subject to monthly usage limits.",
    },
  ],
  openwakeword: [
    {
      q: "Is ViolaWake built on OpenWakeWord?",
      a:
        "Yes. ViolaWake uses OpenWakeWord as a frozen embedding backbone, then adds ViolaWake temporal heads, evaluation tools, training workflow, Console UX, and deployment docs.",
    },
    {
      q: "Should I use OpenWakeWord directly?",
      a:
        "Use OpenWakeWord directly if you want a lower-level open wake word framework. Use ViolaWake when you want an Apache 2.0 SDK plus a hosted browser training flow and opinionated evaluation pipeline.",
    },
  ],
  snowboy: [
    {
      q: "Is Snowboy still maintained?",
      a:
        "No. The Snowboy README says KITT.AI planned to shut down official products and APIs by December 31, 2020, leaving GitHub repositories open with community support.",
    },
    {
      q: "Can ViolaWake replace Snowboy on Raspberry Pi?",
      a:
        "Yes for Python and ONNX-based wake word projects. Snowboy model files are not directly compatible, but the migration path is to collect samples and train a new ViolaWake ONNX model.",
    },
  ],
};

export const pages = [
  {
    path: "/",
    title: "ViolaWake - Open Source Custom Wake Word Training",
    description:
      "Train custom wake words, deploy ONNX models on device, and use an Apache 2.0 Python SDK. The open alternative to Picovoice Porcupine.",
    ogImage: "/og/violawake-og.png",
    priority: "1.0",
    changefreq: "weekly",
    nav,
    schema: ["Organization", "WebSite", "SoftwareApplication"],
    markdown: `
# Custom wake words. Open source. Yours forever.

Train a personal wake-word detector from your voice samples. Apache 2.0 SDK, ONNX models, no API keys. Works offline forever.

[Get started free](/register)
[View on GitHub](${site.github})

Apache 2.0 licensed. No runtime API keys. No phone-home.

## Quick answer: what is ViolaWake?

ViolaWake is an open-source wake word SDK and web Console for custom voice activation. Train from recordings, download an ONNX model, and run detection locally in Python. The hosted Console manages recording and training; runtime detection stays on device.

## Why developers compare ViolaWake to Picovoice

Picovoice Porcupine is a proprietary wake word SDK with broad platform documentation and an AccessKey workflow. ViolaWake is the open-source path: Apache 2.0 SDK, ONNX export, inspectable training code, and no runtime API key.

| Feature | ViolaWake | Picovoice Porcupine |
|---|---|---|
| SDK license | Apache 2.0 SDK | Proprietary SDK and service terms |
| Runtime key | No API key or phone-home | AccessKey required by Picovoice docs |
| Model format | ONNX wake head with OpenWakeWord backbone | Picovoice .ppn/.pv model assets |
| Training path | Browser Console or open CLI | Picovoice Console text-to-wake-word flow |
| Local inference | Yes | Yes |
| Source | [ViolaWake GitHub](${site.github}) | [Picovoice Porcupine docs](${sources.picovoiceDocs}) |

Comparison checked as of ${verifiedDate}. Picovoice AccessKey requirement verified from [Picovoice Porcupine Python Quick Start](${sources.picovoiceQuickStart}).

## Accuracy and deployment signals

ViolaWake's production reference model is a TemporalCNN(96, 9) wake head with 25,409 parameters and a 102 KB ONNX export. The reference evaluation reports 0.8% EER and d-prime 8.58. User-trained models from 10 samples are personal-demo quality; accuracy depends on sample quantity, microphones, rooms, negatives, and threshold tuning.

## Build path

1. Sign up.
2. Record 10+ samples.
3. Train a custom TemporalCNN head.
4. Review metrics.
5. Download the ONNX model.
6. Deploy with the Apache 2.0 SDK.

## Use it from Python

~~~python
from violawake_sdk import WakeDetector

detector = WakeDetector(model="my_word.onnx", threshold=0.80)
for frame in mic_stream():
    if detector.detect(frame):
        print("Wake word detected")
~~~

## What makes the project useful

- Apache 2.0 SDK.
- ONNX wake head export.
- OpenWakeWord embedding backbone.
- Browser recording and training.
- EER, FAR, FRR, ROC AUC, d-prime, and threshold tooling.
- No API keys or phone-home at runtime.

${commonLinks}
`,
  },
  {
    path: "/compare/picovoice",
    title: "ViolaWake vs Picovoice Porcupine - Open Source Alternative",
    description:
      "Compare ViolaWake and Picovoice Porcupine for custom wake words, cost, licensing, model format, accuracy claims, and deployment.",
    ogImage: "/og/violawake-vs-picovoice.png",
    priority: "0.95",
    changefreq: "weekly",
    nav,
    schema: ["FAQPage", "BreadcrumbList"],
    faqs: comparisonFaqs.picovoice,
    markdown: `
# ViolaWake vs Picovoice Porcupine

## Quick answer: is ViolaWake a Picovoice alternative?

Yes. Pick ViolaWake when you want an Apache 2.0 SDK, ONNX model output, no runtime API key, and training code you can inspect. Pick Picovoice Porcupine when you want a proprietary SDK with broad platform support, typed wake-word generation, and a vendor account workflow.

## Summary table

| Category | ViolaWake | Picovoice Porcupine |
|---|---|---|
| Best fit | Open-source custom wake words with portable model files | Proprietary SDK with vendor-managed Console workflow |
| SDK license | Apache 2.0 SDK and training code | Picovoice SDK and service terms |
| Runtime key | No AccessKey, no API key, no phone-home | AccessKey required by Picovoice docs |
| Model output | 102 KB ONNX wake head plus OpenWakeWord backbone | Picovoice .ppn/.pv assets |
| Custom wake words | Train from user recordings in Console or CLI | Type a phrase in Picovoice Console and download a model |
| Accuracy disclosure | 0.8% EER on production reference model; user-trained accuracy varies | FAQ claims 97%+ detection with less than 1 false alarm in 10 hours |
| Cost | Free Console and free Apache 2.0 SDK | Verify current commercial terms with Picovoice |

Comparison checked as of ${verifiedDate}. Competitor claims are linked in Verified claims.

## Decision guide

Pick ViolaWake when the runtime artifact matters. The SDK is Apache 2.0, the wake head exports as ONNX, and detection runs locally without a ViolaWake API call. The Console is a hosted convenience for recording, training, and model management; downloaded models continue to run locally.

Pick Picovoice when procurement favors a proprietary vendor and a fast text-to-wake-word flow. Picovoice documents a broad platform matrix, Console model generation, and SDK quick starts. That is useful when your team wants vendor support and is comfortable with the AccessKey workflow.

## AccessKey and runtime control

Picovoice Porcupine requires an AccessKey. Picovoice's introduction describes AccessKey as an authentication and authorization token that verifies usage within account limits. The Python quick start passes access_key into pvporcupine.create(), including for custom keyword files. The Android quick start adds INTERNET and RECORD_AUDIO permissions and passes the key through setAccessKey().

For offline-only deployments, confirm Picovoice license behavior directly with Picovoice. ViolaWake's local SDK has no AccessKey, no license validation call, and no runtime network dependency. Train or download the model, ship the ONNX file, and run detection on device.

## Training and model ownership

Porcupine's Console flow is optimized for speed: choose a language, type the phrase, train, and download a platform-specific model. ViolaWake asks for real voice samples because it trains a detector from your recordings. A first test can start with 10 samples; production work should add more speakers, rooms, microphones, distances, background speech, music, and hard negatives.

ViolaWake's ownership story is direct. You receive an ONNX wake head. You can version it, evaluate it, ship it inside your release artifact, and run it with the Apache 2.0 SDK. The training pipeline uses an OpenWakeWord embedding backbone and a small ViolaWake TemporalCNN head.

## Accuracy

Picovoice's FAQ says Porcupine achieves 97%+ detection with less than 1 false alarm in 10 hours, and that its standard model uses about 1 MB of memory and less than 4% of a single Raspberry Pi 3 core. Treat those as Picovoice's published claims and test on your own target hardware.

ViolaWake's production reference model reports 0.8% EER and d-prime 8.58 on a curated benchmark. That number belongs to the production reference model, not to every user-trained model. A model trained from 10 samples is a personal-demo baseline; production accuracy depends on the sample set and deployment audio.

## Procurement checklist

Write down the operating constraints before choosing:

- Need an Apache 2.0 SDK and no runtime API key? Pick ViolaWake.
- Need vendor support and text-to-wake-word generation? Evaluate Picovoice.
- Need to prove idle audio stays local? Test the runtime network path.
- Need commercial terms or indemnity? Ask the vendor before integrating.
- Need a confident decision? Train one ViolaWake model, generate one Porcupine model, and run both on the same positives, confusables, music, room noise, and target hardware.

## Verified claims

- Picovoice docs say AccessKey authenticates, authorizes, and verifies usage within account limits. Source: [Picovoice Porcupine docs](${sources.picovoiceDocs}). Verified ${verifiedDate}.
- Picovoice Python quick start requires access_key in pvporcupine.create(), including for custom keyword files. Source: [Picovoice Python Quick Start](${sources.picovoiceQuickStart}). Verified ${verifiedDate}.
- Picovoice Android quick start includes INTERNET permission and setAccessKey(). Source: [Picovoice Android Quick Start](${sources.picovoiceAndroidQuickStart}). Verified ${verifiedDate}.
- Picovoice FAQ claims 97%+ detection, less than 1 false alarm in 10 hours, about 1 MB memory, and less than 4% of a Raspberry Pi 3 core. Source: [Picovoice Porcupine FAQ](${sources.picovoiceFaq}). Verified ${verifiedDate}.
- Picovoice docs describe typed Console wake-word generation and platform-specific downloaded model files. Source: [Picovoice Porcupine docs](${sources.picovoiceDocs}). Verified ${verifiedDate}.
- Picovoice pricing URL was checked for this page; do not rely on unsourced third-party pricing snippets. Source: [Picovoice pricing](${sources.picovoicePricing}). Verified ${verifiedDate}.

## FAQ

### Is ViolaWake a Picovoice alternative?

Yes. ViolaWake is an open-source alternative for teams that want custom wake word training, ONNX models, an Apache 2.0 SDK, and runtime detection without API keys.

### Does ViolaWake require cloud inference?

No. Detection runs on device. The hosted Console is for account, recording, training, and model management.

### Is the ViolaWake Console free?

Yes. The Console is a free service: sign up, record samples, train models, and download ONNX files at no charge, subject to monthly usage limits.

${commonLinks}

${comparisonTrademarkNotice}
`,
  },
  {
    path: "/compare/openwakeword",
    title: "ViolaWake vs OpenWakeWord - Hosted Training on an Open Backbone",
    description:
      "Compare ViolaWake and OpenWakeWord, including the honest upstream relationship, training workflow, SDK scope, and deployment tradeoffs.",
    ogImage: "/og/violawake-vs-openwakeword.png",
    priority: "0.9",
    changefreq: "monthly",
    nav,
    schema: ["FAQPage", "BreadcrumbList"],
    faqs: comparisonFaqs.openwakeword,
    markdown: `
# ViolaWake vs OpenWakeWord

## Quick answer: how is ViolaWake different from OpenWakeWord?

ViolaWake builds on OpenWakeWord. Use OpenWakeWord directly when you want the lower-level framework; use ViolaWake when you want browser recording, managed training, ONNX export, evaluation guidance, and an Apache 2.0 SDK around that backbone.

## Summary table

| Category | ViolaWake | OpenWakeWord |
|---|---|---|
| Relationship | Uses OpenWakeWord as frozen embedding backbone | Upstream wake word framework |
| Code license | Apache 2.0 SDK and training code | Apache 2.0 code |
| Included models | User-trained ViolaWake ONNX wake heads | Pre-trained models have CC BY-NC-SA 4.0 terms |
| Training interface | Browser Console and CLI | Python package, notebooks, examples |
| Runtime shape | ONNX wake head plus OpenWakeWord backbone | Three-component OpenWakeWord pipeline |
| Evaluation | EER, FAR/FRR, ROC AUC, d-prime, streaming checks | Project guidance and lower-level testing |
| Best fit | Teams that want a product workflow | Developers who want direct framework control |

Comparison checked as of ${verifiedDate}. Competitor claims are linked in Verified claims.

## The honest upstream story

OpenWakeWord is upstream infrastructure for ViolaWake's wake path. ViolaWake uses the OpenWakeWord embedding model as a frozen feature extractor, then trains a small TemporalCNN wake head on top. That relationship is part of the product: the backbone handles general audio representation, while ViolaWake handles sample capture, training workflow, model export, and SDK ergonomics.

OpenWakeWord's README describes it as an open-source wake word library for creating voice-enabled applications and interfaces. It includes pre-trained models and tooling for training new models. Its code is Apache 2.0, while included pre-trained models are licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 because of training-data constraints.

## What ViolaWake adds

ViolaWake adds product shape. Sign up, record samples in the browser, train a custom wake head, inspect metrics, download ONNX output, and use the model through one Python SDK. That workflow is useful when teammates need repeatable model creation without assembling notebooks, storage, auth, and deployment docs from scratch.

The product layer does not make wake-word accuracy automatic. It makes the quality loop visible: collect positives, add representative negatives, mine confusables, train, inspect EER and thresholds, then test streaming false alarms on target audio.

## Training pipeline comparison

ViolaWake's current wake head is TemporalCNN(96, 9), 25,409 parameters, and about 102 KB as ONNX. It runs over OpenWakeWord embeddings through ONNX runtime inference. The production reference model reports 0.8% EER and d-prime 8.58 on a curated benchmark; user-trained models from 10 samples should be treated as personal-demo quality until validated on target audio.

OpenWakeWord gives lower-level control. Its README describes 80 ms frames, a score from 0 to 1, and a shared feature extraction backbone. It also says included models aim for less than 5% false rejects and less than 0.5 false accepts per hour with appropriate threshold tuning. Those are upstream project targets; test your own deployment before relying on them.

## When to use OpenWakeWord directly

Pick OpenWakeWord directly when you want a Python-first framework, already have data collection and evaluation in place, or want to stay close to upstream examples. Direct use is also reasonable when your application already has a training pipeline and only needs the wake-word library.

Pick ViolaWake when you need a product workflow: accounts, browser recording, managed training, model history, team features, support contact, and deployable docs. The goal is not to replace OpenWakeWord's value. The goal is to make a custom wake word easier to ship.

## Runtime pieces

The runtime relationship is straightforward. OpenWakeWord converts audio into embeddings. ViolaWake consumes those embeddings with a custom wake head and decision policy. This keeps the wake-specific model small and makes the exported artifact easy to evaluate and version.

## Data and evaluation checklist

For either path, evaluate with the audio you expect to hear:

- Real positive wake attempts.
- Normal speech negatives.
- Similar-sounding phrases.
- Music, television, fans, keyboards, and room noise.
- Target microphones and target hardware.
- Long idle streaming tests for false alarms per hour.

## Migration from OpenWakeWord

Existing 16 kHz mono WAV or FLAC positives can feed ViolaWake's CLI path. Keep your positives, add representative negatives, train a ViolaWake head, evaluate EER/FAR/FRR and streaming false alarms, then replace direct scoring with WakeDetector only if the SDK surface fits your product.

The files are not a universal drop-in replacement for every OpenWakeWord workflow. The data and evaluation discipline carry over.

## Verified claims

- OpenWakeWord README describes it as an open-source wake word library and lists an Apache-2.0 repository license. Source: [OpenWakeWord GitHub README](${sources.openWakeWord}). Verified ${verifiedDate}.
- OpenWakeWord README says repository code is Apache 2.0 and included pre-trained models are CC BY-NC-SA 4.0. Source: [OpenWakeWord GitHub README](${sources.openWakeWord}). Verified ${verifiedDate}.
- OpenWakeWord README describes 80 ms frames, score output from 0 to 1, and a shared feature extraction backbone. Source: [OpenWakeWord GitHub README](${sources.openWakeWord}). Verified ${verifiedDate}.
- OpenWakeWord README describes its models as three separate components: pre-processing, shared feature extraction backbone, and prediction model. Source: [OpenWakeWord GitHub README](${sources.openWakeWord}). Verified ${verifiedDate}.
- OpenWakeWord README states included models aim for less than 5% false rejects and less than 0.5 false accepts per hour with appropriate threshold tuning. Source: [OpenWakeWord GitHub README](${sources.openWakeWord}). Verified ${verifiedDate}.
- OpenWakeWord README says new models can be trained with 100% synthetic speech on top of the frozen shared feature extractor. Source: [OpenWakeWord GitHub README](${sources.openWakeWord}). Verified ${verifiedDate}.

## FAQ

### Is ViolaWake built on OpenWakeWord?

Yes. ViolaWake uses OpenWakeWord as a frozen embedding backbone, then adds ViolaWake temporal heads, evaluation tools, training workflow, Console UX, and deployment docs.

### Should I use OpenWakeWord directly?

Use OpenWakeWord directly if you want a lower-level open wake word framework. Use ViolaWake when you want an Apache 2.0 SDK plus a hosted browser training flow and opinionated evaluation pipeline.

${commonLinks}

${comparisonTrademarkNotice}
`,
  },
  {
    path: "/compare/snowboy",
    title: "ViolaWake vs Snowboy - Replacement for Deprecated Hotword Detection",
    description:
      "Snowboy was shut down in 2020. Compare ViolaWake as an Apache 2.0 Snowboy replacement with ONNX models and browser training.",
    ogImage: "/og/violawake-vs-snowboy.png",
    priority: "0.9",
    changefreq: "monthly",
    nav,
    schema: ["FAQPage", "BreadcrumbList"],
    faqs: comparisonFaqs.snowboy,
    markdown: `
# ViolaWake vs Snowboy

## Quick answer: what is the best Snowboy replacement?

Use ViolaWake for new Snowboy-style projects that need custom wake words, local inference, Python, ONNX, and an Apache 2.0 SDK. Snowboy remains useful historical context, but KITT.AI announced that official products and APIs would shut down by December 31, 2020.

## Summary table

| Category | ViolaWake | Snowboy |
|---|---|---|
| Maintenance state | Active product and SDK | Official products and APIs shut down after 2020 announcement |
| SDK license | Apache 2.0 SDK and training code | Apache 2.0 for source, libraries, resource files, and bundled snowboy.umdl; other hotword models have separate licenses |
| Model format | ONNX wake head plus OpenWakeWord backbone | Snowboy .pmdl/.umdl files |
| Training workflow | Browser Console and CLI | Legacy Hotword-as-a-Service API path |
| Runtime network | No runtime API key or phone-home | README says Snowboy did not use Internet or stream voice to cloud |
| Best fit today | New custom wake word projects | Maintaining legacy devices that already work |

Comparison checked as of ${verifiedDate}. Competitor claims are linked in Verified claims.

## Why Snowboy still appears in search

Snowboy solved a real developer problem: local hotword detection for Raspberry Pi, Python, and small-device projects. Old tutorials still rank because Snowboy was simple, practical, and available when local voice ML was harder to assemble.

That history creates a risk for new projects. The Snowboy README says KITT.AI planned to shut down Snowboy, NLU, and Chatflow by December 31, 2020, take down official websites and APIs, and leave GitHub repositories open with community support only. Treat Snowboy as a migration source, not a fresh dependency choice.

## What this means for new projects

If you are starting a device, kiosk, assistant, robot, or home automation project, use maintained wake-word tooling. You need reproducible training, current Python support, clear packaging, evaluation metrics, and a path to fix false alarms after deployment.

ViolaWake is built for that path. It trains ONNX wake heads, uses an OpenWakeWord embedding backbone, and exposes a Python SDK. The hosted Console captures samples and trains models; the local detector does not require runtime cloud inference.

## License and model compatibility

Snowboy's license surface is specific. The LICENSE file says it governs the source code, libraries, resource files, and bundled snowboy/resources/snowboy.umdl model. Other hotword models have their own licenses. If you inherited a custom .pmdl or .umdl, verify distribution rights before shipping it again.

ViolaWake does not convert Snowboy model files. Migration means retraining. The valuable assets are the wake phrase, known false-trigger phrases, sample collection process, and deployment audio. Reuse that knowledge, then produce a new ONNX model.

## Migration guide from Snowboy to ViolaWake

1. Identify the wake phrase and deployment hardware.
2. Collect fresh positive samples as 16 kHz mono WAV or FLAC.
3. Add negative audio: normal speech, music, device noise, room noise, and similar-sounding phrases.
4. Train with the ViolaWake Console or CLI.
5. Evaluate EER, FAR, FRR, recall, and streaming false alarms per hour.
6. Tune the threshold on the target device.
7. Replace Snowboy runtime calls with ViolaWake WakeDetector calls.

## Raspberry Pi considerations

Snowboy's README says it ran on Raspberry Pi and consumed less than 10% CPU on a single-core 700 MHz ARMv6 Pi. ViolaWake takes the modern route: ONNX runtime inference with a 102 KB wake head and shared OpenWakeWord backbone.

Test the new model on the exact Pi, microphone, enclosure, and room you plan to ship. Run idle listening near fans, speakers, keyboards, televisions, and HVAC noise. Then test wake attempts from normal speaking distance and count misses.

## Accuracy and false alarms

Do not migrate by matching a Snowboy sensitivity number to a ViolaWake threshold. The scales are different. Use behavior instead: long idle audio for false triggers, real wake attempts for misses, and separate thresholds for noisy and quiet deployments if needed.

ViolaWake provides EER, FAR, FRR, ROC AUC, d-prime, and streaming false-alarm checks. Those metrics are more useful than a single tutorial recording.

## Common migration traps

Avoid these mistakes:

- Do not assume the old Snowboy training API is available.
- Do not ship a model that heard one speaker in one room.
- Do not evaluate only on silence.
- Do not skip similar phrases.
- Do not carry forward a legacy model file without checking its license.

## When to keep Snowboy

Keep Snowboy only when an old offline device already works, has no update plan, and has acceptable support risk. Migrate when you need new model training, modern Python packaging, source transparency, repeatable evaluation, or a maintained SDK.

## Verified claims

- KITT.AI announced on March 18, 2020 that Snowboy and other products would shut down by December 31, 2020, with repositories remaining open for community support. Source: [Snowboy GitHub README](${sources.snowboy}). Verified ${verifiedDate}.
- Snowboy README says Snowboy did not use Internet or stream voice to the cloud. Source: [Snowboy GitHub README](${sources.snowboy}). Verified ${verifiedDate}.
- Snowboy README says it ran on Raspberry Pi and consumed less than 10% CPU on the weakest Pi, described as single-core 700 MHz ARMv6. Source: [Snowboy GitHub README](${sources.snowboy}). Verified ${verifiedDate}.
- Snowboy README lists Python2/Python3 wrappers and says Windows was not supported. Source: [Snowboy GitHub README](${sources.snowboy}). Verified ${verifiedDate}.
- Snowboy LICENSE says it covers source code, libraries, resource files, and snowboy/resources/snowboy.umdl; other hotword models have their own licenses. Source: [Snowboy LICENSE](${sources.snowboyLicense}). Verified ${verifiedDate}.

## FAQ

### Is Snowboy still maintained?

No. The Snowboy README says KITT.AI planned to shut down official products and APIs by December 31, 2020, leaving GitHub repositories open with community support.

### Can ViolaWake replace Snowboy on Raspberry Pi?

Yes for Python and ONNX-based wake word projects. Snowboy model files are not directly compatible, but the migration path is to collect samples and train a new ViolaWake ONNX model.

${commonLinks}

${comparisonTrademarkNotice}
`,
  },
  {
    path: "/docs",
    title: "ViolaWake Docs - SDK Quickstart, API Reference, and Training",
    description:
      "Start with the ViolaWake Python SDK, train custom wake words, read API docs, and deploy ONNX wake word models on device.",
    ogImage: "/og/violawake-og.png",
    priority: "0.85",
    changefreq: "weekly",
    nav,
    schema: ["BreadcrumbList"],
    markdown: `
# ViolaWake documentation

Start here if you want to use the SDK rather than the marketing site.

## Quickstart

~~~bash
pip install "violawake[oww]"
~~~

~~~python
from violawake_sdk import WakeDetector

with WakeDetector(model="temporal_cnn", threshold=0.80) as detector:
    for chunk in detector.stream_mic():
        if detector.detect(chunk):
            print("Wake word detected")
            break
~~~

## Important SDK facts

- The canonical package import is "violawake_sdk".
- Wake detection requires the OpenWakeWord runtime backbone. Install the "oww" extra unless your environment already provides it.
- The default wake head is a TemporalCNN ONNX model.
- Detection runs locally on device.
- The SDK includes wake detection, async detection, VAD, optional STT/TTS paths, confidence helpers, power management, and model discovery.

## Primary references

- [GitHub README](${site.github}#readme)
- [API docs](${site.apiDocs})
- [Python package quickstart](${site.github}#quick-start)
- [Training CLI reference](${site.github}#cli-tools-reference)
- [Web Console README](${site.github}/blob/master/console/README.md)
- [SDK __init__.py public API](${site.github}/blob/master/src/violawake_sdk/__init__.py)

## Training custom wake words

Use the Console if you want a browser workflow. Use the CLI if you want local control:

~~~bash
violawake-train --word "jarvis" --positives samples/jarvis --output models/jarvis.onnx
~~~

For production, collect more than the minimum. Include multiple speakers, microphones, rooms, and hard negatives.

## Evaluation tools

ViolaWake documents EER, FAR, FRR, ROC AUC, d-prime, and streaming false alarms per hour. Run evaluation on your own target audio before product deployment.

${commonLinks}
`,
  },
  {
    path: "/faq",
    title: "ViolaWake FAQ - Custom Wake Word SDK and Console",
    description:
      "Answers about offline wake word detection, Raspberry Pi support, training samples, d-prime, privacy, cost, and licensing.",
    ogImage: "/og/violawake-og.png",
    priority: "0.85",
    changefreq: "weekly",
    nav,
    schema: ["FAQPage", "BreadcrumbList"],
    faqs: [
      {
        q: "How is ViolaWake different from Alexa?",
        a:
          "Alexa is a consumer assistant ecosystem. ViolaWake is a developer SDK and Console for custom wake word detection. It gives you a model and local SDK path rather than a complete assistant service.",
      },
      {
        q: "Does the SDK work offline?",
        a:
          "Yes. Wake word detection runs locally on device. The Console is used for hosted recording and training workflows, not runtime cloud inference.",
      },
      {
        q: "Does ViolaWake work on Raspberry Pi?",
        a:
          "Yes, subject to normal device, microphone, Python, and ONNX runtime constraints. Benchmark on the exact Pi and microphone you plan to ship.",
      },
      {
        q: "How many training samples do I need?",
        a:
          "The Console can test with 10 recordings. Production deployments should collect more samples across speakers, rooms, microphones, distances, and background conditions.",
      },
      {
        q: "What is d-prime?",
        a:
          "d-prime is a signal detection measure that estimates separation between positive wake word scores and negative audio scores. Higher is generally better, but it should be read with FAR and recall.",
      },
      {
        q: "Can I commercially use the SDK?",
        a:
          "Yes. The SDK is Apache 2.0. The hosted Console is a free service with its own terms.",
      },
      {
        q: "Is my voice data private?",
        a:
          "The SDK does not send inference audio to ViolaWake. Console recordings are used to provide the training service and are covered by the Privacy Policy.",
      },
      {
        q: "What happens if I delete my account?",
        a:
          "You keep downloaded models. Deleting your account removes your hosted recordings and models; local SDK inference keeps working.",
      },
    ],
    markdown: `
# ViolaWake FAQ

## How is ViolaWake different from Alexa?

Alexa is a consumer assistant ecosystem with its own account, cloud services, hardware integrations, skills, and brand wake word. ViolaWake is not trying to be Alexa. ViolaWake is a developer SDK and Console for custom wake word detection. You train or download a wake word model, run detection locally, and decide what your product does after activation.

## Does the SDK work offline?

Yes. Wake word detection runs locally on device. The hosted Console is for account, recording, training, and model management. Once you have a model and the SDK installed, inference does not require a ViolaWake API call.

## Does ViolaWake work on Raspberry Pi?

Yes, with the normal caveat that wake word quality depends on the exact hardware and microphone. Test on the Pi model, OS image, microphone, enclosure, and room noise you plan to ship. The documented wake runtime footprint is small, but performance should always be measured on target hardware.

## How many training samples do I need?

The Console can run a first test from 10 recordings. That is useful for proving the workflow. Production work needs more data. Collect more speakers, more rooms, different distances, background speech, music, fans, TV, and phonetically similar words.

## What is d-prime?

d-prime is a signal detection metric. It estimates how separated the positive score distribution is from the negative score distribution. Higher d-prime usually means an easier thresholding problem. It is not enough by itself. Always pair it with false accept rate, false reject rate, recall, threshold, and real streaming tests.

## Can I commercially use the SDK?

Yes. The ViolaWake SDK is Apache 2.0. You can use it commercially under that license. The Console is a hosted service with Free, Developer, Business, and Enterprise plan terms.

## Is my voice data private?

The SDK does not send inference audio to ViolaWake. Console recordings are processed to provide the service you requested. Read the Privacy Policy for retention, storage, and email-provider details.

## What if I delete my account?

Downloaded ONNX models remain yours to use locally with the SDK. Deleting your account removes your hosted recordings and models.

## Is ViolaWake a Picovoice alternative?

Yes, for developers who want open training code, ONNX model output, Apache 2.0 SDK licensing, and transparent evaluation. Picovoice Porcupine remains a mature proprietary wake word product with strong enterprise positioning.

## Is ViolaWake an OpenWakeWord fork?

No. ViolaWake uses OpenWakeWord as an embedding backbone and builds a productized training and SDK layer around it. OpenWakeWord deserves explicit credit as upstream infrastructure.

## Can I migrate from Snowboy?

Yes, but not by converting Snowboy model files. Collect or reuse wake phrase samples, train a new ViolaWake model, and test it against your old deployment environment.

${commonLinks}
`,
  },
  {
    path: "/about",
    title: "About ViolaWake - Open Source Wake Word Training",
    description:
      "ViolaWake is built by developers shipping local voice AI: open SDK, browser training Console, and transparent wake word evaluation.",
    ogImage: "/og/violawake-og.png",
    priority: "0.75",
    changefreq: "monthly",
    nav,
    schema: ["Organization", "BreadcrumbList"],
    markdown: `
# About ViolaWake

ViolaWake exists because custom wake word training should not be locked behind opaque vendor workflows. The project grew out of practical voice assistant work: local wake detection, low false alarms, transparent evaluation, and a workflow normal developers can run.

## What we believe

- Wake word inference should run locally.
- Developers should be able to inspect the SDK they ship.
- Accuracy claims should link to methodology.
- Proprietary vendors can be useful, but openness should be a real option.
- A browser Console can save time without making the runtime closed.

## Who is building it

ViolaWake is maintained by the same builder behind the Viola voice assistant work and the public GitHub project at [GeeIHadAGoodTime/ViolaWake](${site.github}). The project is young, but it already has the pieces that matter for an end-to-end wake word workflow: SDK, training, Console, model downloads, privacy docs, and operational runbooks.

## Contact

Email [hello@violawake.com](mailto:hello@violawake.com) for product questions, [enterprise@violawake.com](mailto:enterprise@violawake.com) for larger deployments, and [security@violawake.com](mailto:security@violawake.com) for vulnerability reports.

${commonLinks}
`,
  },
  {
    path: "/privacy",
    title: "ViolaWake Privacy Policy",
    description:
      "Privacy policy for ViolaWake Console recordings, account data, SDK local inference, retention, and support requests.",
    ogImage: "/og/violawake-og.png",
    priority: "0.6",
    changefreq: "monthly",
    nav,
    schema: ["BreadcrumbList"],
    markdown: `
# Privacy Policy

Last updated: July 31, 2026

ViolaWake provides an open-source SDK and a hosted Console. The SDK performs wake word inference on your device. The Console stores account information, recordings, training artifacts, and support messages needed to provide the service.

## Information we collect

- Account name and email address.
- Voice recordings you upload or record in the Console.
- Training job metadata and trained model artifacts.
- Support and contact messages you send.

## How we use information

We use information to operate the Console, train requested models, send transactional email, prevent abuse, and respond to support requests. We do not sell personal information. We do not use your voice recordings for advertising.

## SDK local inference

The ViolaWake SDK does not send inference audio to ViolaWake servers. If you run the SDK locally with a downloaded model, detection happens on your device.

## Retention

Console recordings and trained model artifacts follow the retention rules shown in the application and terms. Download models you want to keep before deleting your account.

## Contact

Email [privacy@violawake.com](mailto:privacy@violawake.com) for privacy requests.
`,
  },
  {
    path: "/terms",
    title: "ViolaWake Terms of Service",
    description:
      "Terms for using the ViolaWake Console, including account use, acceptable use, SDK licensing, retention, and account deletion.",
    ogImage: "/og/violawake-og.png",
    priority: "0.6",
    changefreq: "monthly",
    nav,
    schema: ["BreadcrumbList"],
    markdown: `
# Terms of Service

Last updated: July 31, 2026

These terms govern use of the hosted ViolaWake Console. The SDK is licensed separately under Apache License 2.0.

## Service description

The Console lets users create accounts, record or upload wake word samples, submit training jobs, manage models, and use team features where enabled.

## Your content

You retain ownership of recordings you upload and models produced for your account. You grant ViolaWake the limited rights needed to store and process that content to provide the requested service.

## Acceptable use

Do not upload recordings without consent. Do not use the service for unlawful surveillance, abuse, harassment, or attempts to overload training infrastructure.

## Cost

The Console is a free service, subject to monthly usage limits that protect training capacity. Deleting your account does not remove your right to use downloaded models with the SDK.

## Contact

Email [legal@violawake.com](mailto:legal@violawake.com) for terms questions.
`,
  },
  {
    path: "/contact",
    title: "Contact ViolaWake",
    description:
      "Contact ViolaWake for product questions, enterprise custom wake word deployments, privacy requests, legal questions, and security reports.",
    ogImage: "/og/violawake-og.png",
    priority: "0.7",
    changefreq: "monthly",
    nav,
    schema: ["Organization", "BreadcrumbList"],
    markdown: `
# Contact

We are a small team. Email goes to a real inbox.

## General questions

[hello@violawake.com](mailto:hello@violawake.com)

## Enterprise

[enterprise@violawake.com](mailto:enterprise@violawake.com)

Use this for volume licensing, custom training requirements, on-prem deployment questions, and support planning.

## Security

[security@violawake.com](mailto:security@violawake.com)

Please include reproduction steps, affected URLs or package versions, impact, and whether the issue is already public.

## Privacy and legal

[privacy@violawake.com](mailto:privacy@violawake.com)

[legal@violawake.com](mailto:legal@violawake.com)

${commonLinks}
`,
  },
];

export const blogPosts = [
  {
    path: "/blog/how-we-trained-wake-word-08-eer-25k-parameters",
    title: "How We Trained a Wake Word at 0.8% EER with 25K Parameters",
    description:
      "A technical deep dive into ViolaWake's TemporalCNN wake word architecture, OpenWakeWord embeddings, confusable negatives, and evaluation metrics.",
    ogImage: "/og/violawake-og.png",
    date: "2026-05-08",
    priority: "0.8",
    changefreq: "monthly",
    tags: ["training", "TemporalCNN", "wake word"],
    schema: ["Article"],
    markdown: `
# How we trained a wake word at 0.8% EER with 25K parameters

The strongest ViolaWake reference model is intentionally small. The documented production recipe uses a TemporalCNN over OpenWakeWord embeddings: 96-dimensional embedding frames, a 9-frame window, two 1D convolution layers, batch normalization, dropout, adaptive max pooling, and a compact MLP head.

The result is a 25,409-parameter wake head that exports to about 102 KB as ONNX. That wake head pairs with the shared OpenWakeWord backbone, which the repository documents as about 1.33 MB, for roughly 1.43 MB total runtime footprint for wake detection.

## Architecture

The TemporalCNN keeps ordering across the wake word. Instead of flattening or mean-pooling all frames immediately, the model applies convolution across time so it can learn local temporal patterns in the phrase. The architecture documented in src/violawake_sdk/training/temporal_model.py is:

- Conv1d from 96 channels to 64 channels.
- BatchNorm, ReLU, and dropout.
- Conv1d from 64 channels to 32 channels.
- BatchNorm, ReLU, and adaptive max pooling.
- Linear 32 to 16.
- Linear 16 to 1 with sigmoid output.

## Training data

The proven recipe combines user positives, TTS positives, confusable negatives, speech negatives, and universal negative corpora where available. The goal is not simply to separate "wake word" from silence. The goal is to separate the wake word from normal speech, music, noise, and words that sound close enough to trigger a naive detector.

## Why confusables matter

False activations often come from near phrases. A wake word model for "viola" has to learn that "violin", "violent", "villa", and other similar sounds are not the wake word. ViolaWake's recipe uses two rounds of confusable negative mining: a broad round and a tighter hard-negative round.

## Metrics

The documented reference recipe reports d-prime 8.577, EER 0.8%, and AUC 0.9993. The public benchmark v2 is harsher and reports 5.49% EER on a shared adversarial corpus. Both numbers are useful. One describes the reference recipe. The other sets expectations for a more challenging comparison.

## Practical lesson

Small models can work when the embedding backbone is strong and the negative set is serious. The training process matters as much as the head architecture. If you only train on positives and easy silence, the model will probably fail in a living room.

## Sources

- [ViolaWake proven training recipe](${site.github}/blob/master/docs/PROVEN_TRAINING_RECIPE.md)
- [Temporal model source](${site.github}/blob/master/src/violawake_sdk/training/temporal_model.py)
- [Training CLI source](${site.github}/blob/master/src/violawake_sdk/tools/train.py)

${commonLinks}
`,
  },
  {
    path: "/blog/open-source-vs-proprietary-wake-word-detection-2026",
    title: "Open Source vs Proprietary Wake Word Detection: 2026 Landscape",
    description:
      "A 2026 survey of Picovoice, OpenWakeWord, Snowboy, Google KWS research, and ViolaWake for custom wake word detection.",
    ogImage: "/og/violawake-og.png",
    date: "2026-05-08",
    priority: "0.8",
    changefreq: "monthly",
    tags: ["landscape", "Picovoice", "OpenWakeWord"],
    schema: ["Article"],
    markdown: `
# Open source vs proprietary wake word detection: 2026 landscape

Wake word detection sits between ML research, embedded systems, privacy, and product UX. The best choice depends on whether you want a vendor relationship, an open framework, or a productized open workflow.

## Picovoice Porcupine

Picovoice Porcupine is the mature proprietary reference point. It markets fast on-device custom wake word detection, no training data requirement, broad deployment targets, and enterprise readiness. It is a strong choice when you want vendor support and a sales-led commercial path.

## OpenWakeWord

OpenWakeWord is the most important open framework in this space. It is Apache 2.0, Python-friendly, and widely cited in Home Assistant and maker ecosystems. It is ideal when you want direct access to an open wake word project and are comfortable assembling your own workflow.

## Snowboy

Snowboy is historically important but deprecated. KITT.AI announced the shutdown of official products and APIs by December 31, 2020. Snowboy still matters for migration searches because many old tutorials and Raspberry Pi projects used it.

## Google KWS research

Google's kws_streaming repository is an academic and engineering reference for streaming keyword spotting. It is not a hosted product, but it is useful background for understanding streaming-aware model design.

## ViolaWake

ViolaWake tries to occupy the gap between framework and proprietary service. It uses OpenWakeWord as a backbone, adds a TemporalCNN training path, publishes evaluation tooling, and provides a browser Console. The local SDK stays Apache 2.0.

## Recommendation

Choose proprietary when procurement, support, and platform breadth matter most. Choose lower-level open source when you want full control and can absorb ML workflow complexity. Choose ViolaWake when you want open runtime ownership plus a productized training workflow.

## Sources

- [Picovoice Porcupine](${sources.picovoiceProduct})
- [OpenWakeWord](${sources.openWakeWord})
- [Snowboy](${sources.snowboy})
- [Google kws_streaming](${sources.googleKws})

${commonLinks}
`,
  },
  {
    path: "/blog/raspberry-pi-voice-assistant-violawake",
    title: "Build a Raspberry Pi Voice Assistant in 30 Minutes with ViolaWake",
    description:
      "A practical Raspberry Pi wake word tutorial using ViolaWake, a USB microphone, local ONNX inference, and a simple Python command loop.",
    ogImage: "/og/violawake-og.png",
    date: "2026-05-08",
    priority: "0.8",
    changefreq: "monthly",
    tags: ["Raspberry Pi", "tutorial", "Python"],
    schema: ["Article", "HowTo"],
    markdown: `
# Building a Raspberry Pi voice assistant in 30 minutes with ViolaWake

This tutorial builds the wake word layer for a Raspberry Pi assistant. It uses local inference, a USB microphone, Python, and a ViolaWake ONNX model.

## What you need

- Raspberry Pi 4 or newer.
- USB microphone or a known-good audio HAT.
- Python 3.10 or newer.
- A trained ViolaWake model.
- Network access for installation, unless you pre-stage wheels and model files.

## Step 1: install the SDK

~~~bash
python -m venv .venv
. .venv/bin/activate
pip install "violawake[oww]"
~~~

## Step 2: copy your model

Download your ONNX model from the ViolaWake Console or train locally with the CLI. Copy it to the Pi:

~~~bash
scp my_word.onnx pi@raspberrypi.local:/home/pi/models/my_word.onnx
~~~

## Step 3: run detection

~~~python
from violawake_sdk import WakeDetector

def handle_wake():
    print("Wake word detected. Start your command pipeline here.")

with WakeDetector(model="/home/pi/models/my_word.onnx", threshold=0.80) as detector:
    for frame in detector.stream_mic():
        if detector.detect(frame):
            handle_wake()
~~~

## Step 4: tune threshold

Start at 0.80. If the assistant triggers while music or TV is playing, raise the threshold. If it misses normal wake attempts, lower the threshold slightly or collect more training data.

## Step 5: test like a device owner

Do not test only at your desk. Test near the kitchen, next to a fan, with music, with quiet speech, with loud speech, and from across the room. Count false alarms and misses separately.

## Step 6: add the rest of the assistant

After wake detection, you can run STT, a command parser, a local LLM, or a cloud assistant. Keep wake word detection local so idle audio does not leave the device.

${commonLinks}
`,
  },
];

export const blogIndex = {
  path: "/blog",
  title: "ViolaWake Blog - Wake Word Training and Voice AI",
  description:
    "Technical articles about custom wake words, open-source voice activation, Raspberry Pi assistants, Picovoice alternatives, and evaluation.",
  ogImage: "/og/violawake-og.png",
  priority: "0.75",
  changefreq: "weekly",
  nav,
  schema: ["BreadcrumbList"],
};

export const allMarketingPaths = [
  ...pages.map((page) => page.path),
  blogIndex.path,
  ...blogPosts.map((post) => post.path),
];
