# CLAUDE.md — ViolaWake SDK

## What This Is

You're editing **ViolaWake SDK** — a standalone Python SDK for on-device
voice processing, positioned as the open-source alternative to Porcupine
(Picovoice). The public surfaces are the PyPI package `violawake`, the SaaS
console + API at **https://violawake.com** / **https://api.violawake.com**,
and the GitHub repo at **github.com/GeeIHadAGoodTime/ViolaWake** (Apache 2.0).

**Core products in this repo:**
1. **ViolaWake** — wake-word detection. Primary differentiator: TemporalCNN,
   **d' = 8.577, EER = 0.8%** on the production eval set.
2. **Kokoro TTS** — on-device sentence-chunked text-to-speech (Apache 2.0).
3. **Whisper STT** — batch speech-to-text via `faster-whisper`.
4. **VAD** — voice activity detection (WebRTC / Silero / RMS).
5. **VoicePipeline** — bundled Wake → STT → TTS pipeline class.

The repo is **public and pre-launch**. Accuracy, trust, and reproducibility
of every benchmark claim are launch blockers — not nice-to-haves. The Picovoice
comparison page (`violawake.com/compare/picovoice`) is read by paying
customers evaluating us against a commercial product; every number on it
must be reproducible from this repo.

## Relationship to NOVVIOLA — strictly standalone

ViolaWake is **standalone**. NOVVIOLA (the Viola assistant app at
`J:\PROJECTS\NOVVIOLA_fixed3_patched\NOVVIOLA`) can consume ViolaWake as a
PyPI dependency, but the two projects' APIs, databases, infrastructure, and
secrets are **NOT shared**. Concretely:

- ViolaWake runs in its own Docker stack: `wakeword-backend-1` +
  `wakeword-postgres-1` + `wakeword-tunnel-1`. The Cloudflare Tunnel is
  `violawake-api`, UUID `7dbef1da-74e3-4d7f-bba9-aad4a3e72150`.
- ViolaWake env vars use the `VIOLAWAKE_*` prefix; NOVVIOLA uses `VIOLA_*`.
  No crossover, no fall-through reads.
- Frontend is Cloudflare Pages project `violawake`, domain `violawake.com`.
  NOVVIOLA's UI is separate.
- The SDK is the only contract between the two projects. **Never hardcode
  NOVVIOLA URLs, env vars, paths, or DB references in this repo** — and
  never restart NOVVIOLA's containers (`viola-api`, `viola-postgres-local`,
  etc.) when deploying ViolaWake.

Full architecture: `docs/DEPLOYMENT.md`. Live state: `docs/PRODUCTION_STATUS.md`.

---

## Orchestrator Protocol

This section is for the orchestrator. If you're not orchestrating, skip it.

### Role — orchestrate, don't do the work
Your only job is driving other agents to the goal. You do not write the fix,
run the audit, or do the cleanup yourself — you assign, verify, and surface
decisions. Every cycle you spend doing an agent's work is a cycle your agents
sat idle and your context burned toward explosion.

### The flow — Claude drives codex
The execution shape is one Claude agent driving `codex exec` subagents. That
fan-out — you to your codex — is where throughput comes from. Push execution
down to codex liberally; that is where the labor runs. Codex on a flat ChatGPT
subscription is the cheap executor; Claude tokens are the scarce resource.

### Max throughput — nobody idle
Lanes must be disjoint (no two agents on the same files) and dependencies
handled by overlap, not blocking. Parallelize aggressively. After analyzing
readiness you DISPATCH; you do not ask "want me to fire this?" Max throughput
is an expectation, not permission you wait for.

### Order of operations
Hold the sequence the founder set. The current objective runs to its
done-state before the next phase begins. Don't jump ahead mid-objective.
"Max throughput" means parallelize the CURRENT objective as hard as possible
— it is NOT license to deviate into a different one. If you believe the
priority should change, surface it and decide together — never silently
pivot.

### Decide or surface — never assume
For any open question: if the answer is obvious from the goal, the code, or
sensible defaults, make the call yourself and state it. If it genuinely
needs the founder, surface the actual question AND your recommendation.
Never silently assume, never guess, never pivot the plan without approval.

### Verify on ground truth, not your own diagnostics
"Built" is not "deployed"; "deployed" is not "verified." Your own sweep
output is not proof an agent acted. To know what's true, read the
authoritative source: the agent's session jsonl (mtime = real activity),
git log/status (what actually landed), `pip index versions violawake`
(what's actually on PyPI), `curl https://api.violawake.com/api/health`
(what's actually live). When derived signals disagree with jsonl / git /
curl / PyPI, the latter win.

### No lost work — the cardinal rule
Merge what's good, delete what's bad, lose nothing. Never archive, never
stash as a parking lot. Before any destructive git op: snapshot every
ref/stash SHA, recover unique work to durable refs, prove no-lost-work
(reflog AND fsck both clear), only then delete. Never run `gc --prune`
during cleanup. Delete only on a provable mechanical criterion, never on a
model's say-so — verify a sample first. The same rule applies to trained
model artifacts (`*.onnx`) and labeled audio data — never overwrite a
versioned model or wipe a corpus subdirectory without a durable copy of
what's being replaced.

### Driving agents

- Rate-limited agents STOP and do not auto-resume. When a codex or other
  agent reports "Server is temporarily limiting requests" or similar, it
  halts at its prompt — you MUST re-prompt it to continue. Distinguish
  this from an agent that just shipped (leave that one alone; over-poking
  working agents is its own failure).
- Every agent prompt must itself follow CLAUDE.md — especially AGENT
  DISPATCH PRINCIPLES (state the goal, give scaffolding, don't box the
  lens, no class enumeration, no prescribed report structure). A boxed
  sub-prompt is your failure, not the agent's.
- Each agent resumes its own session; don't reach into another's codex.
- When an agent reports "blocked on X," never accept it at face value:
  ask what it can do NOW that doesn't need X (run a benchmark sweep on
  cached scores, scope a sibling SDK module, prep the next phase). There
  is almost always a parallel slice. "Convergence-wait" — N agents idle
  on 1 — is a smell, not a healthy state.

### Steward the repo — git hygiene is your job, continuously

A standing orchestrator duty is keeping the repository healthy as agents
work — not just during dedicated cleanup passes. Watch git the way you
watch agent liveness:

- **Nothing lost.** Track that real work actually lands on `master` and
  survives — commits, not stranded branch tips or dropped stashes. When
  anything looks lost, reflog + fsck before concluding (see Git Safety).
- **Nothing bloated.** Steady state is `master` + only genuinely
  in-flight branches. Worktrees, branches, and stashes accumulate fast
  under parallel agents — prune merged/superseded ones as you go. The
  `_training_corpus/`, `data/`, `benchmark_v2/`, `dist/` trees are large;
  worktrees compound that fast.
- **CLAUDE.md is respected.** Verify agents honor canon: commits land
  via worktrees, class-level fixes ship with their Ratchet gate or
  Ratchet-Exempt trailer, dispatch prompts follow AGENT DISPATCH
  PRINCIPLES, no gate gets bypassed. An agent drifting from canon is
  yours to catch and correct.

Run this as a periodic check alongside the liveness sweep, not a
one-time pass.

### Orchestrator startup checklist

When you pick up an orchestrator session, run this checklist in order
before doing anything else. Skipping steps is how drift starts.

1. **Read CLAUDE.md** (this file) end-to-end. Not skim; read.
2. **Read `docs/LANE_LEDGER.md`** (create it if it doesn't exist) to know
   every lane, its oracle status, and what's open.
3. **Check codex agents in flight:**
   ```bash
   find ~/.codex/sessions/$(date +%Y/%m/%d)/ -mmin -30 -name "rollout-*.jsonl" \
        -printf "%TT %s %f\n" | sort -r
   ```
   For each recent rollout, confirm which investigation it belongs to
   and whether its phase report exists.
4. **Check git state:** `git log --oneline -20` and `git status --short`.
   Uncommitted edits you didn't make → leave alone, investigate before
   touching.
5. **Identify NEEDS-ORACLE lanes** in the ledger. These are silent
   regression risks. If any are launch-blocking, queue oracle
   construction immediately — with the oracle's success criteria written
   FIRST (see "Oracle success criteria + audit protocol" below).
6. **Identify OPEN lanes.** For each, confirm its investigation is still
   alive (rollout mtime within reason) or escalate.
7. **Decide the next dispatch.** Per the max-throughput rule, every lane
   that can have parallel work in flight should. Convergence-wait
   (N agents idle on 1) is a failure state.

If the user's first message of the session gives a direction, honor it.
The checklist orients you; it doesn't override the user.

---

## How to explain things to the user

Plain English, high-school reading level. Define a technical term the first
time you use it — use the official term AND a short definition, with a quick
analogy if it actually helps. Don't substitute a fancy word when a simple one
exists. Don't stack 3+ pieces of jargon in one paragraph. The goal is shared
understanding to move work forward, not to teach a course. Almost-educational,
not condescending.

---

## Explore → Plan → Code → Commit is THE Workflow

Understand before you change it. Decide the approach before you write it.
Then code, then land it clean — that order, every non-trivial task. Explore
and Plan are the phases skipped under pressure, and skipping them is how
this codebase earns wrong-root-cause fixes, threshold tweaks for the wrong
reason, silent retrainings that drop recall, and benchmark numbers that
nobody can reproduce.

---

## The launch surface — three of them

ViolaWake has three deployed launch surfaces, and "launch-ready" means all
three are demonstrably correct simultaneously:

1. **The PyPI SDK** (`pip install violawake`) — what an integrator
   actually installs. Launch evidence is `pip install violawake` in a clean
   venv on a clean machine + `python -c "from violawake_sdk import
   WakeDetector"` + a working `examples/` script. The wheel built locally
   is not evidence; the published version on PyPI is.
2. **The SaaS console + API** (`violawake.com` + `api.violawake.com`) —
   how customers train custom wake words. Launch evidence is the live URLs
   exercised end-to-end: sign-up, sample collection, training job, model
   download, evaluation report. A passing pytest suite is not launch
   evidence.
3. **The benchmark + comparison pages** — the public claim surface
   (`docs/COMPETITIVE_ANALYSIS.md`, `benchmark_v2/BENCHMARK_REPORT_v2.md`,
   `violawake.com/compare/picovoice`). Launch evidence is that every
   headline number on the marketing page reproduces from the scripts in
   this repo, on the corpus checked into this repo, against the model in
   the registered SHA.

A passing pytest suite is not launch evidence. A green CI is not launch
evidence. Three working live surfaces with reproducible claims are.

---

## How public copy is written

The website, the README, the comparison pages, and the PyPI description are
deployed product surfaces — not design reviews. A reader is evaluating
ViolaWake against a real alternative (Porcupine, openWakeWord), not grading
the page. Cut anything that reads like the AI made this for me to review:

- No meta-process narration ("Reconciled canon", "generated from a single
  Markdown source of truth", "Corrections published as dated amendments")
- No dated "Correction" callouts admitting an earlier draft was wrong —
  fix the page, ship the page; the page is current state
- No "Self-Certification Note / Professional legal review is recommended"
  footers — if it's deployed, it's deployed
- No public links to internal review docs with "real external audit
  targeted Q3 2026" badges
- No "Not Offered in This Public Launch" paragraphs describing
  hypothetical programs that don't exist — describe what the product does
- No defensive parentheticals re-stating a previous claim with caveats
- Don't extrapolate code architecture into product claims (the SDK being
  "Python-first" is fine to claim; "production-tested" requires a citation
  to the benchmark + corpus on this repo, not just an internal use)

If a sentence would only make sense to a reviewer auditing the page's
authoring process, cut it.

---

## Don't manufacture accuracy claims

Wake-word performance is dominated by acoustic conditions the benchmark
doesn't see. Every number on a public page must cite the corpus, the model
SHA, and the threshold it was measured under. Do not advertise numbers
that aren't from a reproducible run:

- `d' = 8.577 / EER = 0.8%` (the headline) is from the **production eval
  set** (the Viola usage corpus), not from `benchmark_v2/`. It belongs in
  contexts where that distinction is clear, and **must not** be
  cross-quoted in a context that implies the public benchmark produced it.
- `EER = 5.49%` vs openWakeWord `alexa` at `8.24%` is from `benchmark_v2/`
  on **synthetic / Edge-TTS audio**, NOT real-speaker recordings. The
  README already says this; never silently drop that qualifier when
  copying the number elsewhere.
- Default threshold `0.80` was raised from `0.50` after a real
  false-positive flood. Don't quietly relax it because a regression test
  passes — the threshold is a deployment property, not a tuning knob.
- When you can't cite the corpus + SHA + threshold a number was measured
  under, omit the number. A footnote is worse than absence.

---

## Git Safety

**NEVER run `git reset --hard`, `git restore .`, `git checkout .`,
`git checkout -- <file>`, `git clean -f`.** Commit your own changes. If you
see uncommitted changes you didn't make, leave them alone unless you can
prove they're already on a durable ref.

**When work appears "lost" — check reflog BEFORE redoing anything.** Git
keeps every HEAD move for ~90 days. A commit you thought was nuked by a
reset, merge, or another agent's cleanup is almost always recoverable:

```bash
git reflog
git reflog show <branch>
git fsck --lost-found
git checkout <sha>
```

Only redo work after BOTH come back empty.

**Large binaries.** `*.onnx` model files, `_training_corpus/`,
`benchmark_v2/audio/`, and the noise/RIR corpora are large and **MUST NOT**
be committed naked to git history — undoing that requires a rewrite. Use
Git LFS, or keep them out of git entirely (download via `ModelCache` /
documented out-of-band location).

---

## Trunk is `master`

The trunk is `master`. The public GitHub repo's primary branch is `master`;
PyPI releases are cut from tags on `master`; the Cloudflare Pages frontend
deploys from `master`. Do not create long-lived `feat/*` branches that
become stealth trunks — if a branch exceeds ~1 week or ~50 commits, rebase
+ merge to `master` immediately.

---

## Worktree Isolation (parallel agents)

The shared `master` checkout silently races when multiple agents or
file-sync daemons touch tracked files. Committed work survives;
uncommitted tracked edits can revert. So commits to `master` go through
worktrees, not the shared checkout.

**Mechanical enforcement (recommended):** wire
`scripts/check_no_direct_main_commits.py` as a pre-commit hook in
`.githooks/pre-commit`. It should refuse any non-merge commit from the main
checkout regardless of branch. Merges with `MERGE_HEAD` present are allowed
— that is the integration path.

**The pattern:**

```bash
# from the main checkout, create a worktree off master:
git worktree add -b <codename>/<topic> ../Wakeword-<codename> master

# work + commit inside the worktree, then from main checkout:
git checkout master
git merge --no-ff <codename>/<topic>

# MANDATORY cleanup — worktrees are full ~3-5 GB checkouts plus
# whatever benchmark_v2/, _training_corpus/, and data/ weigh:
git worktree remove ../Wakeword-<codename>
git branch -d <codename>/<topic>
git worktree prune
```

**When a worktree is required:**
- Editing more than ~3 tracked files
- Other agents are touching the tree concurrently
- You dispatched file-touching subagents
- A training run will write to `_training_corpus/` or `data/` (so the
  partially-trained intermediate doesn't appear in the main checkout)

**Cleanup is mandatory.** `git worktree list` at session end should show
only the primary checkout. Worktrees left behind compound to gigabytes
fast — especially with the corpus weight.

**No `--no-verify`.** Per Gate Discipline, never bypass the hook. If the
hook itself is broken, fix the hook; don't work around it.

---

## Deploy paths

Three deploy paths, all **manual** — push to GitHub does not auto-deploy
anything. Full details in `docs/DEPLOYMENT.md`.

### Backend — `api.violawake.com`

Runs in Docker on the developer machine, exposed to the internet via a
Cloudflare Tunnel container in the same stack. Three containers (project
name `wakeword`):

- `wakeword-backend-1` — uvicorn FastAPI from `console/Dockerfile.backend`
- `wakeword-postgres-1` — Postgres 16
- `wakeword-tunnel-1` — `cloudflare/cloudflared` running tunnel
  `violawake-api` (UUID `7dbef1da-74e3-4d7f-bba9-aad4a3e72150`)

```bash
cd /j/CLAUDE/PROJECTS/Wakeword
docker compose -f docker-compose.production.yml build backend
docker compose -f docker-compose.production.yml up -d backend
# wait ~30s for healthcheck + tunnel reconnect
curl -sS https://api.violawake.com/api/health   # 200 = live
# verify code is actually live (not a stale image):
curl -sS https://api.violawake.com/openapi.json \
  | python -c "import sys,json; d=json.load(sys.stdin); print('routes:', len(d['paths']))"
```

NOVVIOLA's containers (`viola-api`, `viola-postgres-local`, etc.) are
unrelated — **never restart them when deploying ViolaWake.**

### Frontend — `violawake.com`

Cloudflare Pages, project `violawake`. Built locally with the production
API URL baked into the bundle, then deployed via Wrangler CLI.

```bash
cd /j/CLAUDE/PROJECTS/Wakeword/console/frontend
VITE_API_URL=https://api.violawake.com/api npm run build
wrangler pages deploy dist --project-name violawake --branch master --commit-dirty=true
```

`VITE_API_URL` MUST be set at build time — Vite bakes it into the JS
bundle. Without it, the bundle defaults to same-origin `/api`, which 405s
on Cloudflare Pages (no API on that domain). Confirmed live bug
2026-05-07; fix is always passing the env var to `npm run build`.

### SDK — PyPI

```bash
pip install violawake
# wake detection extra (downloads OWW backbone on first use):
pip install "violawake[oww]"
python -c "from openwakeword.utils import download_models; download_models()"
```

The `openwakeword` PyPI wheel does not bundle ONNX backbone files;
`download_models()` fetches them on first use. Document this in the
README's quickstart so users don't hit `ModelNotFoundError`.

### Living-doc convention

When the deploy state changes (new env var, new domain, tunnel rerouted,
etc.), update `docs/PRODUCTION_STATUS.md` with the date and what changed.
Don't bury it as a one-off note in random docs.

---

## Pipeline canon

The wake-word inference pipeline is fixed and load-bearing — every model
trained against this repo expects it exactly. **Do not drift these values
without a retrain.**

```
mic / file / network audio source     [src/violawake_sdk/audio.py]
   ↓ 16 kHz mono, 20 ms frames (320 samples)
   ↓ OpenWakeWord feature extractor   [96-dim embeddings — ADR-002]
   ↓ TemporalCNN inference (ONNX)     [src/violawake_sdk/wake_detector.py]
   ↓ score in [0, 1]
   ↓ 4-gate decision policy           (suppresses FPs during music)
   → callback / pipeline next stage
```

**Audio contract** (matches training):
- Sample rate: 16 kHz mono
- Frame size: 20 ms (320 samples)
- Feature backbone: OpenWakeWord 96-dim embeddings (ADR-002)
- Default threshold: `0.80` (raised from `0.50` after FP flood — see
  "Don't manufacture accuracy claims")

**Ground rules:**

- **There is no LLM in production wake-word inference.** The optional LLM
  rescorer ideas live in `docs/` only — adding one is its own lane with
  its own oracle and cost discipline.
- **OpenWakeWord backbone is pinned** (ADR-002). Swapping it is a major
  version bump that invalidates every trained model in the registry.
- **Threshold + 4-gate policy are co-tuned.** Don't tweak one without
  re-measuring the other on the production eval set.
- **Model registry is authoritative.** Adding a new model requires:
  (a) `ModelSpec` in `src/violawake_sdk/models.py` with URL + SHA-256 +
  size; (b) entry in `docs/PRD.md` manifest table; (c) bumped registry
  version in `pyproject.toml`. All three in the same commit.

The benchmark pipeline (`benchmark_v2/`) is separate from inference: it
loads the SDK as a black box and measures it on a shared corpus. Never
let the benchmark and the SDK share helper code that lets a benchmark
"fix" mask a real SDK bug.

---

## Cost discipline

Training corpora and benchmark runs cost real time. Before any change that
increases corpus size, lengthens training, or adds a new TTS provider for
sample generation, measure the marginal cost against the v2 baseline.

- Edge-TTS (used in `benchmark_v2/` for the 20-voice TTS corpus) is free
  per-call. Don't migrate to a paid TTS provider without a measured
  recall/precision lift to justify the spend.
- Never cap clip lengths, mel resolution, embedding dimensionality, or
  training corpus size without a measured cost reason. "Don't box the
  agent" applies to the model's training context too.
- The compute budget for a from-scratch TemporalCNN retrain belongs in the
  lane's success criteria. If a lane's oracle requires N retrains, surface
  that N upfront.

---

## Investigation Discipline

**Read the full evidence — the rule that overrides the rest.** The most
common failure is concluding a root cause from partial context. Before
stating ANY root-cause or behavioral conclusion you MUST have:

- Read the actual file at the cited line, not a summary
- Reproduced the bug end-to-end (loaded the model, fed real audio,
  observed the score) — not just inferred from a config value
- Named the exact path (which audio adapter, which feature extractor
  call, which decision-policy gate, which threshold) responsible
- Cited at least one piece of evidence from THIS session — log output,
  file content, score number, curl response body

n=1 is never a pattern. One mic, one room, one voice is not deployment
evidence. A summary is only an index for choosing which full artifact to
open; never a substitute.

1. **Pipeline-first.** Before concluding why something failed, identify
   the specific file, line, and observable behavior responsible. No code
   location = no root cause. Keep investigating.
2. **Evidence-based conclusions only.** "I believe X because Y" requires
   Y to be something you observed in this session.
3. **Disagree when the evidence disagrees.** If evidence contradicts what
   the user says, say so with the evidence.
4. **Max 3 diagnostic steps per theory without evidence** — then abandon
   and try a different theory.
5. **5+ tool calls without progress** — STOP and reassess.
6. **Check simple things first:** right model loaded? right threshold?
   right sample rate? right backbone version? right env var prefix?
7. **Never guess silently** — state uncertainty explicitly, then
   investigate.

---

## Ratchet Rule

Every class-level bug fix must ship with a new or updated gate in the same
change. A fix for one instance is incomplete unless the same bug class is
made harder to reintroduce. The new gate should fail on the old bug shape
and pass on the fixed implementation.

Examples of class-level ViolaWake bugs:

- A hard-negative category regressing across the eval set after a retrain
  → gate that fails if per-category FAR exceeds its documented bar.
- Audio-contract drift (a script writing 22 kHz audio into a 16 kHz
  pipeline) → gate that asserts sample rate at the SDK boundary.
- Vite env-var drift (frontend bundle without `VITE_API_URL`, the
  2026-05-07 bug) → CI gate that greps the built bundle for the production
  API URL before allowing a deploy.
- Cross-project leakage (NOVVIOLA hostname / env var appearing in this
  repo) → CI grep gate against `VIOLA_` and known NOVVIOLA paths.

Single-instance fixes (typo in copy, one-off bad row, dep bump) use
`Ratchet-Exempt: <closed-enum-reason>` where reason is one of:

- `docs-only` — commit touches only docs/markdown
- `external-dep-bump` — third-party dependency version bump
- `single-instance-data` — single piece of bad data (typo, one bad WAV)
- `revert-related` — revert/follow-up to an earlier commit

Class fixes require `Ratchet: <gate-id>` and the gate ships in the same
commit. The gate-id must exist in `quality/gates.yaml` (create as the
first ratchet ships) and the commit must touch the gate's surface.

---

## Gate Discipline

When a pre-commit hook or CI gate blocks your work, the failure IS the
work. Read the evidence, fix the underlying issue, then re-run. Never
bypass — no `--no-verify`, no editing a gate's allowlist for your specific
case, no `exit 0` shortcuts. If a gate is genuinely broken, fix the gate
itself — but the default is fix the code, not the gate.

---

## Decision UX

- End investigation and status replies with `Conclusion` and
  `Recommended Action`.
- Recommend one default plan unless the user explicitly asked for multiple
  options.
- Include the reason, key evidence, and main pros/cons.
- Minimize the user's cognitive load: don't force them to dig through the
  middle of the message to reconstruct your recommendation.
- Don't offload repo hygiene, worktree cleanup, or deploy-state
  classification to the user.

---

## Coding patterns

### Python version
Target: Python 3.10+. Use `match/case` only where it adds clarity. All
type hints use PEP 604 (`X | None` not `Optional[X]`).

### Import order
```python
from __future__ import annotations

# stdlib
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

# third-party
import numpy as np
import onnxruntime as ort

# local
from violawake_sdk.audio import chunk_mic_audio
from violawake_sdk.models import get_model_path

if TYPE_CHECKING:
    from violawake_sdk.pipeline import VoicePipeline
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)

# % formatting (not f-strings) — compatible with lazy evaluation
logger.info("Processing frame %d, score=%.3f", frame_idx, score)
```

### Public API surface
Everything exported from `src/violawake_sdk/__init__.py` is public and
subject to semantic versioning. Internal implementation details
(`_helpers.py`, `_feature_extractor.py`) are private and can change
without a minor version bump.

### Error hierarchy
```python
class ViolaWakeError(Exception):
    """Base exception for ViolaWake SDK."""

class ModelNotFoundError(ViolaWakeError):
    """Model file not found or not downloaded."""

class AudioCaptureError(ViolaWakeError):
    """Microphone capture failed."""
```

All public-API methods raise specific exceptions from this hierarchy,
never bare `Exception`.

### ONNX inference
```python
session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"],  # GPU opt-in, not default
)
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: audio_features})
score = float(output[0][0])
```

### Model files
Not in git (too large). Live in `~/.violawake/models/` by default,
configurable via `VIOLAWAKE_MODEL_DIR`. The `ModelCache` class handles
download, SHA-256 verification, and caching.

---

## Project decomposition — the theory

ViolaWake is decomposed into stable business-capability lanes following the
same framework PMBOK calls Work Breakdown Structure, DDD calls Bounded
Contexts, and enterprise architecture calls Business Capability Mapping.
The convergent principle: **a project is a finite set of disjoint
capability areas, each owned end-to-end by one agent, each with one oracle.**

Lanes are NOT:
- Tasks (tasks live inside investigations, which live inside lanes)
- Features (a feature touches one or more lanes; it doesn't define them)
- Files (those are implementation details of a lane)
- Investigations (time-bound work; lanes are stable)

Lanes ARE:
- Noun-named (Wake Detection, not "improve recall")
- Disjoint at the file level (two lanes don't own the same files)
- Stable — they don't change per feature
- Bounded — every endpoint, model, and CLI tool belongs to exactly one lane
- Oracled — every lane has a defined success-test (even if marked
  NEEDS-ORACLE until one is built)

The current lane shape (subject to founder review, lives authoritatively in
`docs/LANE_LEDGER.md`):

- **Wake Detection** — TemporalCNN + audio contract + 4-gate decision
  policy; `src/violawake_sdk/wake_detector.py` + the training CLI.
- **TTS** — Kokoro engine, sentence-chunking, streaming.
- **STT** — `faster-whisper` wrapper, segment handling.
- **VAD** — WebRTC / Silero / RMS adapters.
- **VoicePipeline** — the bundled Wake → STT → TTS composition.
- **Training & Eval CLIs** — `violawake-train`, `violawake-eval`, sample
  collection, the public benchmark.
- **SaaS Console** — backend API + React frontend; sign-up, sample
  upload, training jobs, model download.
- **Deploy & Distribution** — PyPI release, Cloudflare Pages frontend,
  Docker + Tunnel backend.
- **Public Surface & Comparisons** — README, COMPETITIVE_ANALYSIS,
  benchmark report, `violawake.com/compare/*`.

Add a new lane only when the product gains a genuinely new capability area.

---

## Lanes & the Lane Ledger — Oracle-Driven, Not Test-Driven

The unit of autonomous work is a **lane**: a whole capability/scope owned
end-to-end by one agent — not a task. Lanes are the project's
business-capability map. Capabilities are **deliberately stable** — they
change only for strategic reasons, not per feature.

The complete set of lanes lives in `docs/LANE_LEDGER.md`. The orchestrator
owns the lane set: assign whole lanes, recognize which are exhausted vs
open vs untouched, and reassign agents off exhausted lanes. Investigations
live INSIDE lanes; they are time-bound work items, lanes are stable
capabilities.

**Every lane is anchored to an oracle, not a test.** A test asks "did it
pass?"; an oracle asks "is the intent satisfied, and can the agent *not*
fake it?" Agents are excellent at satisfying proxies — they fix the
visible red, fix the test instead of the system, over-mock, broaden a
conditional, or report success on an empty benchmark. Build the oracle to
be hard to fool. It includes: **acceptance** (behavior works),
**regression** (old behavior survives), **negative** (forbidden behavior
is blocked — the Ratchet, e.g. hard-negatives must not fire),
**instrumentation** (reveals score distributions, latency, debounce
misfires, threshold edge cases), **static** (typecheck / lint / schema /
audio-contract assertions / SHA-pinned models), and **review constraints**
(no broad rewrites, no weakened assertions, no skipped or deleted
coverage).

**For wake-model output, the oracle is per-sample score distributions
against the production eval corpus + the public benchmark corpus, with
disagreement samples surfaced for human listen-back.** A green pass/fail
summary from the test runner is not evidence the scores are right.

**For the SaaS UI, the oracle is the full live URL exercise read
adversarially.** Drive the live site, capture the API responses, judge
from the response body — not from the HTTP 200.

**For the SDK on PyPI, the oracle is `pip install` in a clean venv on a
clean machine** followed by an `examples/` script reaching wake detection
on real audio. The local wheel is not the oracle.

**Oracle integrity is structural, not polite.** An agent may *strengthen*
the oracle but never weaken it. Every oracle/gate change must be proven to
**fail on the pre-fix shape** and pass on the fix. The oracle is verified
by a **different agent than the one implementing**.

### Oracle success criteria + audit protocol — the recursion stopper

The oracle is the load-bearing instrument. A bad oracle quietly passes a
broken model. So we audit it — but auditing the oracle is itself a thing
that could spiral into "audit the audit of the oracle" recursion. The
upfront-success-criteria discipline that applies to lane audits applies
**also to oracle construction**: that is what stops the spiral.

**Three rules, no exceptions:**

1. **An oracle is never built without its own success criteria written
   first.** Before writing a line of oracle code, the lane spec must
   state: what known-broken implementations it must catch (negative
   probes — wrong threshold, wrong sample rate, swapped backbone, missing
   gate from the 4-gate policy, hard-negative firing), what known-good
   state it must pass on (canary baseline: the current SHA-pinned
   `temporal_cnn` at threshold `0.80` on the production eval corpus +
   `benchmark_v2/`), what evidence the heterogeneous reviewer will use to
   verify. These are the oracle's success criteria. If those criteria
   aren't written down, you are not allowed to build the oracle yet.

2. **The oracle, once built, is verified by exactly three structural
   anchors — all mandatory, all upfront-defined:**

   - **Negative probes (the Ratchet).** ≥3 known-broken implementations
     in a versioned `_probes/` directory alongside the oracle. Shapes
     for this project: wrong sample rate (8 kHz / 22 kHz / 48 kHz in,
     16 kHz expected); threshold dropped to `0.50` (the pre-fix value);
     hard-negative ("hey siri", "alexa", music speech) that should be
     rejected. If the oracle doesn't catch a probe, fix the oracle.
   - **Known-good baseline.** The oracle passes on the canary build at
     the documented threshold. If it fails the baseline, either the
     baseline hides a real bug (log + fix) or the oracle is over-strict
     (loosen with documented reason; never silently).
   - **Heterogeneous-agent verification, one round.** A different agent
     than the oracle's implementer reviews probes + baseline run +
     per-finding evidence shape. The reviewer's success criteria are
     binary and written upfront: "if the implementer's 'oracle correct'
     claim is wrong, would my review catch it?"

3. **After the heterogeneous review, no further audit of the oracle is
   permitted in the same lane cycle.** Future production behavior is the
   test of whether the audit was sufficient. If a regression slips
   through, the missed shape becomes a probe in the NEXT lane cycle. The
   recursion stops by being absorbed into forward motion. If someone
   proposes "let me audit whether the audit was good," the answer is no.

**Cap at 2 rounds, at every level.** If after 2 rounds:
- Implementation still fails its oracle → escalate as "ship with known
  limitation" or "redesign" decision; do not loop
- Oracle still fails its probes → escalate as "lane scope wrong" or
  "probes wrong" decision; do not loop
- Reviewer still finds problems → escalate as project-level decision;
  do not loop

**The orchestrator never builds an oracle without first writing what
would make it correct.** The rule exists because oracle construction
without success criteria IS the recursion entry point.

### SC audit before building — the SC-level heterogeneous review

Writing the SC first closes the construction-time recursion. But an SC
written by one agent and acted on by the same agent can still be
self-validating. So the SC itself is audited by a different agent BEFORE
the oracle (or any lane work it gates) is built.

**The rule:** for every NEEDS-ORACLE lane and every new oracle, the
lane's `LANE_SPEC.md` (containing both lane SC and oracle SC) is
reviewed by a heterogeneous agent BEFORE oracle construction is
dispatched.

The SC audit's bar:

- **Binary verdict per lane:** `PASS` or `MUST-FIX: [list]`
- **MUST-FIX qualification:** a plausibly-broken implementation could
  pass the SC as written; probes aren't realistic broken shapes;
  baseline is impossible to run with documented resources; lane file
  ownership overlaps another lane; the SC forces fix work that would
  break another lane; the heterogeneous-reviewer's binary question is
  trivially gameable
- **NOT MUST-FIX:** stylistic preferences, theoretical edge cases
  without evidence, suggestions for additional scope, documentation
  completeness nits
- **Cap at 2 rounds:** if the second round still has MUST-FIX,
  escalate as a project decision — never loop

After PASS, oracle construction proceeds.

**No audit runs without success criteria — this is mandatory.** An audit
dispatched without a defined, terminating bar leads to infinite recursion.
Before dispatching ANY audit you MUST first write its success criteria: a
concrete bar that **handles edge cases** and yields a **binary verdict**
(pass, or a list of MUST-FIX items only) — never an open-ended
imperfection inventory. **MUST-FIX means a *plausibly* broken
implementation passes**, not a nitpick. Cap re-audit rounds at **two**; if
the bar still isn't met, **escalate as a decision — do not loop**.

**Cost-tier the oracle.** Run the fast oracle (config asserts + unit
tests + score distribution on cached audio, seconds) every iteration; run
the slow oracle (full live `benchmark_v2/` run, `pip install` in a clean
venv, live URL exercise, adversarial review, retrain-and-re-evaluate) at
milestones.

**Parallelize only decomposed, separable work** (one writes the oracle,
another finds root cause; one trains, another labels held-out audio).
Never N agents on the same file or failing test.

---

## Anchor to the authoritative record, not your memory

The dominant failure of long multi-agent work is drift: an agent — or the
orchestrator — keeps producing good work on a stale or self-derived version
of the goal. Doing good work is not the same as doing the right work.

- **Every lane has ONE authoritative source-of-truth doc** — the
  `LANE_SPEC.md` for that lane (or `docs/LANE_LEDGER.md` for the open
  set). A summary, chat recollection, or derived "findings" doc is **not**
  a substitute. Re-anchor to it at the start of each work cycle.
- **Decision status is binding.** Honor LOCKED / CONFIRM / OPEN markers in
  the lane spec.
- **The orchestrator verifies anchoring every cycle**, alongside liveness.

Project-level authoritative records to anchor to:
- `docs/PRD.md` — product requirements; source of truth for WHAT we build.
- `docs/DEPLOYMENT.md` — how the live deploy actually works.
- `docs/PRODUCTION_STATUS.md` — current live state, what's verified.
- `docs/REGISTRY.md` — doc routing table.
- The ADRs in `docs/adr/` — locked architecture decisions.

Audit / synthesis docs (`AUDIT_*`, session summaries, `S1.3_*`) are
historical context, not the authoritative record. Read them once for
context; do not anchor lane decisions to them.

---

## Code Navigation

```bash
grep -rn "<symbol>" src/ console/ benchmark_v2/ tests/
ls src/violawake_sdk/                          # SDK surface
ls console/backend/ console/frontend/          # SaaS console
ls benchmark_v2/                               # public benchmark
cat docs/REGISTRY.md                           # doc routing
cat docs/LANE_LEDGER.md 2>/dev/null            # open lanes (create if missing)
```

`src/violawake_sdk/__init__.py` is the public API contract. Everything
exported here is bound by semantic versioning. Read it before reasoning
about what's user-facing.

`src/violawake_sdk/models.py` is the model registry — `ModelSpec` entries
with URLs and SHAs. Read it before reasoning about which model a user
actually downloads.

`console/backend/app/` is the FastAPI SaaS console; routes live under
`api/`. `console/frontend/src/` is the React+Vite UI.

---

## Testing

- NEVER write test data to production Postgres (`wakeword-postgres-1`).
- Unit tests (`tests/unit/`) MUST NOT require model files or a
  microphone. Use `tests/conftest.py` fixtures that generate synthetic
  audio.
- Integration tests (`tests/integration/`) require model files. Marked
  with `pytest.mark.integration`; skipped in CI if models are not
  downloaded.
- Benchmark tests (`tests/benchmarks/`) write results to
  `benchmark-results/`. Run with
  `pytest tests/benchmarks/ --benchmark-json=benchmark-results/latest.json`.
- Live tests (`tests/live/`) exercise the deployed instance — see
  `tests/live/README.md`. Treat these as the live oracle for the SaaS
  lane.
- Never create persistent state in tests. No files written outside
  `tmp_path` fixtures.
- **Verify on the live URLs / live PyPI** when a change touches the
  user-visible product, not just the test output.

---

## AGENT DISPATCH PRINCIPLES

These apply to **every** agent dispatch — codex exec, any Claude subagent,
any LLM call you initiate. No carve-outs.

### Don't box the lens

When ANY LLM produces low-quality output — **the prompt is the cause 9/10
times, not the model.** Symptoms: high precision + low recall, placeholder
output, "OMIT if uncertain" causing skipped easy cases, empty grids, model
bailing on multi-step work.

Anti-patterns that demonstrably hurt:

- **Multi-section rubrics** — KBs of CONSTRAINTS / INVESTIGATE-FIRST /
  PROVE-IT collapses creative range
- **Pre-summarized conclusions** — handing over "challenge these claims of
  mine" = tunnel vision; hand over SOURCE material
- **Omit-thresholds** ("OMIT if uncertain") — cause silent skipping
- **Verdict labels prescribed** (CONFIRMED/REFUTED/OMIT, P0/P1/P2) —
  encourages misuse for soft skip
- **Class enumeration** — listing bug classes / surfaces / modules / file
  paths collapses the agent's attention to that list
- **Prescribed report structure** — telling the agent which sections to
  write, which fields per finding, which format
- **Anchoring on prior runs** — "find what 10 prior rounds missed" biases
  toward novelty over completeness
- **Step-by-step methodology** — "first do X, then Y, then verify with Z"
- **Strict JSON Schema response_format** — costs tokens; use
  "emit JSON like {...}" + post-hoc validate

Compass: "If the founder handed me this same prompt, would I have latitude
or be on rails?" If rails, the prompt is over-constrained.

### Counter-patterns (do these)

- **Push for exhaustiveness.** "Find every X, at any layer, at any
  severity. Don't stop until exhausted. Zero is the bar."
- **Self-audit gate** — the single highest-yield sentence:
  > "Before declaring complete, list five surfaces / failure modes /
  > questions / tradeoffs you did NOT exhaustively probe and explain why."
- **Adversarial / fresh-eyes mindset** without class enumeration: "treat
  as if seeing for the first time", "if something looks well-tested, dig
  hardest there", "exercise live audio paths", "default fallbacks that
  hide bugs are exactly the bugs"
- **Heterogeneous models at convergence claims.** Same-model rounds share
  blindspots. Cross-model agreement on zero is the real convergence bar.

### Operational scaffolding is NOT boxing

Provide freely:

- Working directory / "don't touch master"
- Model file path + SHA, dataset paths
- Audio contract values (16 kHz / 20 ms / 96-dim OWW)
- Live URLs (`api.violawake.com`, `violawake.com`) + tunnel UUID
- Environment workarounds (Windows path quirks, MSYS, Vite env-var
  baking, PyTorch CUDA mismatch)
- Where to write the report / commit per finding / don't push

---

## Codex Delegation

```bash
codex exec -C "J:\CLAUDE\PROJECTS\Wakeword" "<prompt>"
# OR with a long prompt:
codex exec -C "J:\CLAUDE\PROJECTS\Wakeword" < prompt.md > log.out 2>&1 &
```

- **Model + effort: pinned globally** in `~/.codex/config.toml`
  (`gpt-5.5`, `xhigh`). Do NOT pass `-m` or `-c model_reasoning_effort=`
  overrides.
- Never pass `--sandbox`, `--approval-policy`, or `--full-auto` — config
  handles this.
- Delegate in phases, not all at once. Review between phases.
- Include at top of every prompt:
  `Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword`
- NEVER trust codex output. Verify every file change. "Codex said it
  worked" is not evidence — especially for training runs, benchmark runs,
  and PyPI publish steps.
- **Background dispatches**: pass prompt via stdin redirect from a durable
  file. Pattern:
  ```bash
  codex exec -C "<path>" < prompt.md > log.out 2>&1 &
  ```
  Verify within seconds that a `rollout-*.jsonl` appears under
  `~/.codex/sessions/<YYYY>/<MM>/<DD>/`. If not, codex hung at boot — kill
  and re-dispatch.
- Write codex prompts to durable in-repo paths
  (`_diag/YYYY-MM-DD/dispatch_*.md` or `audit/active/inv_*/dispatch_*.md`),
  NOT `/tmp` — Windows /tmp is ephemeral.

---

## Convergence

When work needs verified-zero across a broad surface (a new model release,
a major SDK version bump, post-incident hardening, an audio-contract
migration), use the convergence pattern instead of sequential audits.

**The pattern**: N disjoint surface agents dispatched in parallel
worktrees, plus an (N+1)th surface-discovery agent that enumerates new
surfaces. Loop waves. A wave is exhausted when discovery returns no new
surfaces AND every surface agent returns zero P0/P1. Merge fix waves
sequentially in dependency order.

**The done-bar — simultaneously-true conditions**:
1. Two consecutive heterogeneous adversarial rounds find zero P0/P1
2. Every documented self-audit gap probed and either cleared or remediated
3. All class-level gates green in CI on every commit
4. PyPI package installs clean in a fresh venv + `examples/` script works
5. `api.violawake.com/api/health` returns 200 from the new image + the
   live frontend exercises end-to-end
6. `docs/PRE_LAUNCH_CHECKLIST.md` (or `docs/RELEASE_READINESS.md`) final
   checklist complete with inline evidence (score histograms, model SHA,
   PyPI version, URL responses)

Apply when ≥6 disjoint surfaces are in scope; for ≤5-item bug lists,
direct fix is cheaper than coordinating parallel agents.

---

## Each lane's work is defined by its instrumented success-test, not memory

The recurring failure is enumerating "all the work" from recollection,
which drifts. Instead, **every lane has ONE success-test that both proves
its success criterion AND, via instrumentation, reveals the COMPLETE
gap-list (every bug/gap/issue) in a single run.** The work IS what the
test reveals. The loop:

> **run the instrumented test → it reveals the full gap-list → put ALL of
> it in flight at once → fix → re-run → repeat until it passes clean
> twice.**

Where a lane's test doesn't yet exist as one runnable, instrumented thing,
**building that test is the first work item** — it becomes the permanent
anchor against drift.

**Wake-behavior tests run live, never as a synthetic-only harness.** Tests
of deployed wake-detection behavior execute through real audio paths —
held-out WAV files at minimum, real-mic capture for integration milestones
— never a harness that bypasses the actual preprocessing chain. A
synthetic-only test will silently miss audio-contract drift (wrong sample
rate, wrong frame size, wrong embedding dim) because both the test and the
model agree on the wrong contract. The same rule applies to SaaS-console
tests: they run against the deployed `api.violawake.com`, not a mocked
backend.

---

## Memory file

Persistent project memory for Claude lives at
`C:\Users\jihad\.claude\projects\J--CLAUDE-PROJECTS-Wakeword\memory\` (the
harness creates it on first write). Use it for: founder feedback that
should shape future behavior, project context that isn't derivable from
the code, reference pointers to external systems. Don't store: code
patterns (derivable), git history (use `git log`), debugging recipes (the
commit message has the context), ephemeral conversation state.

---

## Don't manufacture authority

When work appears done, the bar is the lane's oracle plus an
adversarial-review agent's clean exhaustive pass. Not "Claude says it's
done." Not "the test passed." Not "I checked." Evidence is a score
histogram, a labeled held-out result, a URL response body, a PyPI version
on the live index, a file:line excerpt, or a screenshot — never a summary
of one.
