# ViolaWake SDK — Documentation Registry

> Routing table for all project documentation. Check here before creating a new doc to avoid duplication.
> Update this table whenever a doc is added, moved, archived, or superseded.

## Authority Levels

| Badge | Meaning |
|-------|---------|
| **LIVING** | Authoritative, actively maintained. Treat as ground truth. |
| **ARCHIVED** | Historical record only. Superseded by a newer doc. |
| **ADR** | Architecture Decision Record. Immutable once accepted. Captures WHY, not just WHAT. |
| **DRAFT** | Work in progress. Not yet authoritative. |

---

## Living Documents

| Document | Path | Scope | Last Updated | Staleness Signals |
|----------|------|-------|-------------|-------------------|
| **Project Contract** | `CLAUDE.md` | Repo rules, launch evidence, public-copy canon, lane discipline, and operator workflow | 2026-06-03 | Any lane, release, deploy, public-copy, or workflow rule change |
| **Lane Ledger** | `docs/LANE_LEDGER.md` | Authoritative lane ownership, scope status, oracle status, and lane success criteria | 2026-06-03 | Lane ownership, scope, oracle, or capability-map change |
| **Production Status** | `docs/PRODUCTION_STATUS.md` | Current live state, what's verified, operational levers — read FIRST after any deploy | 2026-05-07 | Any deploy, env-var change, or smoke-test result |
| **Deployment** | `docs/DEPLOYMENT.md` | How frontend + backend + tunnel actually deploy. Manual steps, not auto. | 2026-05-07 | Hosting platform change, tunnel reconfig, env-var schema change |
| **Operations Runbook** | `docs/OPERATIONS_RUNBOOK.md` | Procedures for dashboard-mediated config changes (Resend, Stripe live mode, key rotation). Things only the operator can do. | 2026-05-07 | New external service added, account / key change |
| **Launch Runbook** | `docs/RUNBOOK.md` | Pre-launch health monitoring, backups, restore, billing verification, and human handoff steps. | 2026-05-08 | Operational script, monitor, backup, restore, or launch handoff changes |
| **SEO Audit** | `docs/SEO_AUDIT.md` | Crawler audit, competitor facts, keyword landscape, and static marketing architecture decision. | 2026-05-08 | SEO architecture, crawler behavior, competitor positioning, or SERP target change |
| **SEO Runbook** | `docs/SEO_RUNBOOK.md` | Search Console, Bing Webmaster, IndexNow, Cloudflare Crawler Hints, Plausible funnel setup, and post-deploy crawler checks. | 2026-05-08 | Indexing workflow, analytics event, sitemap, or crawler-file change |
| **SEO Outreach** | `docs/SEO_OUTREACH.md` | Rules-aware GitHub/dev.to/Reddit/HN outreach targets for cited discovery. | 2026-05-08 | New outreach target, launch post, or citation strategy change |
| Developer Docs Landing | `docs/index.html` | Static SDK docs landing page that links README, API docs, and contributor docs | 2026-06-03 | README, PyPI package metadata, API docs path, or public install guidance change |
| Generated API Docs | `docs/api/index.html` | pdoc-generated public API reference entry point | 2026-06-03 | Public SDK symbol added, removed, renamed, or doc generation tool changes |
| Product Requirements | `docs/PRD.md` | What we build, feature priorities, metrics | 2026-03-17 | New competitive entrant, market shift, major pivot |
| Test Strategy | `docs/TEST_STRATEGY.md` | Testing philosophy, tiers, coverage requirements | 2026-03-17 | New test tier added, CI pipeline change |
| Business Plan | `docs/BUSINESS_PLAN.md` | Revenue model, pricing, growth | 2026-03-26 | Pricing or market strategy change |
| Competitive Analysis | `docs/COMPETITIVE_ANALYSIS.md` | Feature comparison vs Porcupine, OWW, etc. | 2026-03-26 | New competitor or feature parity shift |
| Pre-Launch Checklist | `docs/PRE_LAUNCH_CHECKLIST.md` | Launch readiness checklist | 2026-03-26 | All items checked off = launch ready |
| Show HN Draft | `docs/SHOW_HN_DRAFT.md` | Hacker News launch post draft | 2026-03-26 | Post published or product pivot |
| Roadmap | `docs/ROADMAP_10_OF_10.md` | Multi-phase product roadmap | 2026-03-26 | Phase completion or priority change |
| Proven Training Recipe | `docs/PROVEN_TRAINING_RECIPE.md` | Canonical training pipeline, parameters, Console parity | 2026-04-05 | Pipeline change, new architecture, hyperparameter change |
| Architecture | `docs/ARCHITECTURE.md` | System architecture overview | 2026-04-05 | Major structural change |
| Changelog | `CHANGELOG.md` | Release history and notable changes | 2026-04-05 | New release shipped |
| Security | `SECURITY.md` | Security policy, vulnerability reporting | 2026-04-05 | New threat model or disclosure process change |
| Console Security Notes | `docs/SECURITY.md` | Console upload hardening, container security notes, and WAF rule status | 2026-06-03 | Upload limits, decoder sidecar, container hardening, WAF rules, or security-control change |
| Contributing | `CONTRIBUTING.md` | Contributor guidelines, dev setup | 2026-04-05 | Process or tooling change |
| Progress | `docs/PROGRESS.md` | Current development progress tracker | 2026-04-05 | Milestone completion or priority shift |
| Release Notes | `RELEASE_NOTES.md` | User-facing release notes | 2026-04-05 | New release shipped |
| This registry | `docs/REGISTRY.md` | Doc routing | 2026-04-05 | New doc added without updating registry |

## Architecture Decision Records (ADRs)

ADRs are immutable once accepted. To change an architecture decision, create a new ADR that supersedes the old one.

| ADR | Title | Status | Path |
|-----|-------|--------|------|
| ADR-001 | ONNX Runtime for all model inference | Superseded by ADR-006 | `docs/adr/ADR-001-onnx-runtime.md` |
| ADR-002 | OpenWakeWord embeddings as feature extractor backbone | Superseded by ADR-007 | `docs/adr/ADR-002-oww-feature-extractor.md` |
| ADR-003 | Python SDK first (not C library) | Accepted | `docs/adr/ADR-003-python-first.md` |
| ADR-004 | Open-core licensing strategy | Accepted | `docs/adr/ADR-004-open-core.md` |
| ADR-005 | PyPI distribution + separate model hosting | Accepted | `docs/adr/ADR-005-packaging.md` |
| ADR-006 | Multi-runtime inference (ONNX + TFLite) | Accepted | `docs/adr/ADR-006-multi-runtime-inference.md` |
| ADR-007 | TemporalCNN wake head on frozen OWW backbone | Accepted | `docs/adr/ADR-007-temporal-cnn-wake-head.md` |

---

## Quick Navigation

**"What are we building?"** → `docs/PRD.md` → Section 2 (Scope) and Section 4 (Feature Catalog)

**"Why did we choose ONNX?"** → `docs/adr/ADR-001-onnx-runtime.md`

**"Why Python and not C?"** → `docs/adr/ADR-003-python-first.md`

**"How do we test?"** → `docs/TEST_STRATEGY.md`

**"What's the benchmark number and where does it come from?"** → `docs/PRD.md` → Section 5 (Metrics)

**"What's the open-source vs paid split?"** → `docs/adr/ADR-004-open-core.md`

**"How are models distributed?"** → `docs/adr/ADR-005-packaging.md`

**"Competitive analysis?"** → `docs/COMPETITIVE_ANALYSIS.md`

**"Are we ready to launch?"** → `docs/PRE_LAUNCH_CHECKLIST.md`

**"What does the HN post say?"** → `docs/SHOW_HN_DRAFT.md`

---

## Archived Documents

Moved to `docs/archive/` on 2026-04-05. These are MLP-era records superseded by the temporal CNN production model.

| Document | Path | Original Location | Why Archived |
|----------|------|--------------------|-------------|
| OWW Benchmark Report | `docs/archive/BENCHMARK_REPORT_oww.md` | `benchmark_oww/` | MLP vs OWW comparison; production model is now temporal_cnn |
| Meta Analysis (MLP era) | `docs/archive/META_ANALYSIS_mlp_era.md` | `experiments/` | 30 patterns from MLP accuracy campaign; FAPH crisis resolved by temporal CNN |
| Streaming vs Clip Analysis | `docs/archive/STREAMING_VS_CLIP_ANALYSIS.md` | `experiments/` | Mean-pooling validity analysis; temporal CNN uses 9-frame windows |
| FAR/FRR Report (MLP) | `docs/archive/far_frr_report_mlp.md` | `eval_clean/` | MLP FAR/FRR metrics; superseded by temporal_cnn EER 0.8% |
| Clean Eval Results (MLP) | `docs/archive/RESULTS_eval_clean_mlp.md` | `eval_clean/` | MLP d'=4.14; superseded by temporal_cnn d'=8.577 |

### Archive Candidates

Documents at the repo root that are audit-era records and candidates for archival.

| Document | Path | Why Archive Candidate |
|----------|------|-----------------------|
| Launch Readiness | `docs/LAUNCH_READINESS.md` | Point-in-time launch audit doc |
| Functional Gap Analysis | `docs/FUNCTIONAL_GAP_ANALYSIS.md` | Point-in-time gap audit doc |
| E2E Readiness | `docs/E2E_READINESS.md` | Point-in-time end-to-end readiness audit doc |
| Adversary Audit | `docs/ADVERSARY_AUDIT.md` | Point-in-time adversary/security audit doc |
| Build vs Buy Audit | `docs/BUILD_VS_BUY_AUDIT.md` | Point-in-time console build-vs-buy audit doc |

---

## Doc Maintenance Rules

1. Every living doc must have a `<!-- doc-meta -->` block at the top with: scope, authority, code paths, staleness signals.
2. When an ADR is superseded, update its status to "Superseded by ADR-XXX" — do not delete.
3. Archive docs that are more than 6 months stale and not referenced by active code. Move to `docs/archive/`.
4. This registry is the canonical index. If a doc isn't listed here, it's either stale or shouldn't exist.
