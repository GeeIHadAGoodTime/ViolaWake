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
| Product Requirements | `docs/PRD.md` | What we build, feature priorities, metrics | 2026-03-17 | New competitive entrant, market shift, major pivot |
| Test Strategy | `docs/TEST_STRATEGY.md` | Testing philosophy, tiers, coverage requirements | 2026-03-17 | New test tier added, CI pipeline change |
| Business Plan | `docs/BUSINESS_PLAN.md` | Revenue model, pricing, growth | 2026-03-26 | Pricing or market strategy change |
| Competitive Analysis | `docs/COMPETITIVE_ANALYSIS.md` | Feature comparison vs Porcupine, OWW, etc. | 2026-03-26 | New competitor or feature parity shift |
| Pre-Launch Checklist | `docs/PRE_LAUNCH_CHECKLIST.md` | Launch readiness checklist | 2026-03-26 | All items checked off = launch ready |
| Show HN Draft | `docs/SHOW_HN_DRAFT.md` | Hacker News launch post draft | 2026-03-26 | Post published or product pivot |
| Roadmap | `docs/ROADMAP_10_OF_10.md` | Multi-phase product roadmap | 2026-03-26 | Phase completion or priority change |
| This registry | `docs/REGISTRY.md` | Doc routing | 2026-03-26 | New doc added without updating registry |

## Architecture Decision Records (ADRs)

ADRs are immutable once accepted. To change an architecture decision, create a new ADR that supersedes the old one.

| ADR | Title | Status | Path |
|-----|-------|--------|------|
| ADR-001 | ONNX Runtime for all model inference | Accepted | `docs/adr/ADR-001-onnx-runtime.md` |
| ADR-002 | OpenWakeWord embeddings as feature extractor backbone | Accepted | `docs/adr/ADR-002-oww-feature-extractor.md` |
| ADR-003 | Python SDK first (not C library) | Accepted | `docs/adr/ADR-003-python-first.md` |
| ADR-004 | Open-core licensing strategy | Accepted | `docs/adr/ADR-004-open-core.md` |
| ADR-005 | PyPI distribution + separate model hosting | Accepted | `docs/adr/ADR-005-packaging.md` |

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

## Doc Maintenance Rules

1. Every living doc must have a `<!-- doc-meta -->` block at the top with: scope, authority, code paths, staleness signals.
2. When an ADR is superseded, update its status to "Superseded by ADR-XXX" — do not delete.
3. Archive docs that are more than 6 months stale and not referenced by active code. Move to `docs/archive/`.
4. This registry is the canonical index. If a doc isn't listed here, it's either stale or shouldn't exist.
