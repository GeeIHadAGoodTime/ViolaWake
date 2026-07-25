"""Ratchet: no gate contract in quality/gates.yaml may be silently truncated.

`contract:` values are written as YAML plain (unquoted) scalars, and these
contracts routinely cite issues -- "#1768", "#1775". In a plain scalar a space
followed by `#` starts a COMMENT, so YAML discards everything after it at parse
time with NO error and NO warning.

That silently destroyed 7 of 34 contracts (25-85% of their text each), including
both gates protecting the #1775 silence subgrade. The gates still RAN correctly
-- own_tests are separate list items -- so nothing was red. What was lost is the
human-readable statement of what each gate protects, which is precisely what an
engineer reads before deciding whether they are allowed to change a guarded
behaviour. A guard whose contract is truncated mid-sentence is a guard nobody can
correctly obey.

Reds if any contract is truncated relative to the text actually on its line.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="pyyaml required to read the gate registry")

GATES = Path(__file__).resolve().parents[2] / "quality" / "gates.yaml"


def _raw_contract_lines() -> dict[str, str]:
    """gate_id -> the literal text after `contract:` on its line."""
    raw = GATES.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    current: str | None = None
    for line in raw.split("\n"):
        m = re.match(r"\s*- gate_id: (\S+)\s*$", line)
        if m:
            current = m.group(1)
            continue
        m = re.match(r"\s*contract: (.*)$", line)
        if m and current:
            out[current] = m.group(1)
            current = None
    return out


def test_the_registry_still_parses() -> None:
    """A malformed entry takes the whole registry offline; fail fast and loudly."""
    data = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data.get("gates"), "gate registry did not parse"
    for gate in data["gates"]:
        assert set(gate.keys()) == {"gate_id", "contract", "detector", "own_tests"}


def test_no_gate_contract_is_silently_truncated() -> None:
    """The load-bearing ratchet."""
    data = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    raw_lines = _raw_contract_lines()

    truncated = []
    for gate in data["gates"]:
        gid = gate["gate_id"]
        raw_text = raw_lines.get(gid)
        if raw_text is None:
            continue
        # An unquoted scalar loses everything from ` #` onward.
        if raw_text.startswith(('"', "'")):
            continue
        parsed = gate["contract"]
        if len(raw_text) - len(parsed) > 5:
            truncated.append((gid, len(parsed), len(raw_text)))

    assert not truncated, (
        "these gate contracts are silently truncated by a YAML comment -- the text "
        "is on disk but never reaches anyone reading the registry. Quote the "
        "contract scalar:\n"
        + "\n".join(f"  {g}: parses to {p} of {r} chars" for g, p, r in truncated)
    )


def test_a_contract_containing_an_issue_reference_survives_parsing() -> None:
    """Direct statement of the bug class, independent of current file contents."""
    unquoted = yaml.safe_load("contract: see #1775 for the reason\n")["contract"]
    assert unquoted == "see", (
        "expected YAML to prove the hazard: an unquoted scalar drops everything "
        f"from ' #' onward, got {unquoted!r}"
    )
    quoted = yaml.safe_load('contract: "see #1775 for the reason"\n')["contract"]
    assert quoted == "see #1775 for the reason", "quoting is the fix"


def test_the_1775_gates_carry_their_full_contract() -> None:
    """The specific gates this ticket depends on must be readable end to end."""
    data = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    by_id = {g["gate_id"]: g for g in data["gates"]}

    for gid, must_contain in (
        ("sdk-silence-subgrade-multi-probe-median", "median"),
        ("quality-gate-tts-outage-is-not-a-model-verdict", "SCORED SAMPLES"),
        ("quality-gate-failure-is-not-a-circuit-breaker-fault", "resume_user"),
    ):
        assert gid in by_id, f"missing gate {gid}"
        contract = by_id[gid]["contract"]
        assert must_contain in contract, (
            f"{gid}'s contract is missing {must_contain!r} -- it parses to "
            f"{len(contract)} chars, which usually means it was truncated"
        )
