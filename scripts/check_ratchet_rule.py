#!/usr/bin/env python3
"""Enforce the CLAUDE.md ratchet trailer rule for fix-like production commits."""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


FIX_LIKE_RE = re.compile(r"\b(fix|hotfix|security|incident|hardening|regression)\b", re.I)
GATE_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
RATCHET_RE = re.compile(r"^Ratchet:\s*(.+?)\s*$", re.M)
EXEMPT_RE = re.compile(r"^Ratchet-Exempt:\s*(.+?)\s*$", re.M)
ALLOWED_EXEMPTIONS = {
    "docs-only",
    "external-dep-bump",
    "single-instance-data",
    "revert-related",
}

PRODUCTION_PREFIXES = (
    "src/",
    "console/backend/app/",
    "console/backend/alembic/",
    "console/decoder/",
    "console/frontend/src/",
    "wasm/src/",
    "workers/",
)
PRODUCTION_FILES = {
    "_write_wake_detector.py",
    "docker-compose.production.yml",
    "docker-compose.viola-bridge.yml",
    "pyproject.toml",
    "railway.json",
    "railway.toml",
}


@dataclass(frozen=True)
class Gate:
    gate_id: str
    contract: str
    detector: str
    own_tests: tuple[str, ...]

    @property
    def surface(self) -> tuple[str, ...]:
        return ("quality/gates.yaml", self.detector, *self.own_tests)


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("./")


def unquote_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index].rstrip()
    return line.rstrip()


def load_gate_registry(path: Path) -> dict[str, Gate]:
    if not path.exists():
        raise ValueError(f"{path} does not exist")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [strip_comment(line) for line in raw_lines]
    lines = [line.rstrip() for line in lines if line.strip()]

    if not lines:
        raise ValueError(f"{path} is empty")

    if lines[0] == "gates: []":
        return {}
    if lines[0] != "gates:":
        raise ValueError("top-level schema must start with 'gates:' or 'gates: []'")

    gates: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    in_tests = False

    for line_number, line in enumerate(lines[1:], start=2):
        if line.startswith("  - gate_id:"):
            if current is not None:
                gates.append(current)
            current = {"gate_id": unquote_scalar(line.split(":", 1)[1])}
            in_tests = False
            continue

        if current is None:
            raise ValueError(f"line {line_number}: expected a gate entry")

        if line.startswith("    contract:"):
            current["contract"] = unquote_scalar(line.split(":", 1)[1])
            in_tests = False
        elif line.startswith("    detector:"):
            current["detector"] = normalize_path(unquote_scalar(line.split(":", 1)[1]))
            in_tests = False
        elif line == "    own_tests:":
            current["own_tests"] = []
            in_tests = True
        elif in_tests and line.startswith("      - "):
            tests = current.setdefault("own_tests", [])
            assert isinstance(tests, list)
            tests.append(normalize_path(unquote_scalar(line[8:])))
        else:
            raise ValueError(f"line {line_number}: unsupported gate schema line: {line!r}")

    if current is not None:
        gates.append(current)

    registry: dict[str, Gate] = {}
    for index, item in enumerate(gates, start=1):
        required = {"gate_id", "contract", "detector", "own_tests"}
        missing = sorted(required - set(item))
        extra = sorted(set(item) - required)
        if missing:
            raise ValueError(f"gate {index}: missing required keys: {', '.join(missing)}")
        if extra:
            raise ValueError(f"gate {index}: unsupported keys: {', '.join(extra)}")

        gate_id = str(item["gate_id"])
        contract = str(item["contract"]).strip()
        detector = normalize_path(str(item["detector"]))
        tests_object = item["own_tests"]
        if not GATE_ID_RE.match(gate_id):
            raise ValueError(f"gate {index}: invalid gate_id {gate_id!r}")
        if gate_id in registry:
            raise ValueError(f"gate {index}: duplicate gate_id {gate_id!r}")
        if not contract:
            raise ValueError(f"gate {gate_id}: contract must be non-empty")
        if not detector:
            raise ValueError(f"gate {gate_id}: detector must be non-empty")
        if not isinstance(tests_object, list) or not tests_object:
            raise ValueError(f"gate {gate_id}: own_tests must be a non-empty list")
        own_tests = tuple(normalize_path(str(test)) for test in tests_object)
        if any(not test for test in own_tests):
            raise ValueError(f"gate {gate_id}: own_tests entries must be non-empty")
        registry[gate_id] = Gate(gate_id, contract, detector, own_tests)

    return registry


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def staged_files(repo: Path) -> list[str]:
    output = run_git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR")
    return [normalize_path(line) for line in output.splitlines() if line.strip()]


def commit_files(repo: Path, commit: str) -> list[str]:
    output = run_git(
        repo,
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return [normalize_path(line) for line in output.splitlines() if line.strip()]


def commit_message(repo: Path, commit: str) -> str:
    return run_git(repo, "log", "-1", "--format=%B", commit)


def commits_in_range(repo: Path, commit_range: str) -> list[str]:
    output = run_git(repo, "log", "--format=%H", "--reverse", commit_range)
    return [line.strip() for line in output.splitlines() if line.strip()]


def is_production_file(path: str) -> bool:
    path = normalize_path(path)
    return path in PRODUCTION_FILES or path.startswith(PRODUCTION_PREFIXES)


def is_fix_like(message: str) -> bool:
    return bool(FIX_LIKE_RE.search(message))


def path_matches_surface(path: str, surface: str) -> bool:
    path = normalize_path(path)
    surface = normalize_path(surface)
    if any(char in surface for char in "*?["):
        return fnmatch.fnmatch(path, surface)
    if surface.endswith("/"):
        return path.startswith(surface)
    return path == surface or path.startswith(surface.rstrip("/") + "/")


def gate_surface_touched(gate: Gate, files: list[str]) -> bool:
    return any(
        path_matches_surface(changed_file, surface_path)
        for changed_file in files
        for surface_path in gate.surface
    )


def check_message_and_files(
    *,
    message: str,
    files: list[str],
    registry: dict[str, Gate],
    label: str,
) -> tuple[bool, list[str]]:
    normalized_files = [normalize_path(path) for path in files]
    production_files = [path for path in normalized_files if is_production_file(path)]
    if not is_fix_like(message):
        return True, [f"PASS: {label}: commit is not fix-like; ratchet not required."]
    if not production_files:
        return True, [f"PASS: {label}: fix-like commit does not touch production code."]

    ratchets = RATCHET_RE.findall(message)
    exemptions = EXEMPT_RE.findall(message)
    if ratchets and exemptions:
        return False, [f"FAIL: {label}: use either Ratchet or Ratchet-Exempt, not both."]
    if not ratchets and not exemptions:
        return False, [
            f"FAIL: {label}: fix-like production commit requires "
            "Ratchet: <gate-id> or Ratchet-Exempt: <closed-enum-reason>."
        ]

    if exemptions:
        invalid = [reason for reason in exemptions if reason not in ALLOWED_EXEMPTIONS]
        if invalid:
            allowed = ", ".join(sorted(ALLOWED_EXEMPTIONS))
            return False, [
                f"FAIL: {label}: invalid Ratchet-Exempt reason(s): "
                f"{', '.join(invalid)}. Allowed: {allowed}."
            ]
        return True, [f"PASS: {label}: accepted Ratchet-Exempt: {exemptions[-1]}."]

    invalid_ids = [gate_id for gate_id in ratchets if not GATE_ID_RE.match(gate_id)]
    if invalid_ids:
        return False, [f"FAIL: {label}: invalid Ratchet gate id(s): {', '.join(invalid_ids)}."]

    missing = [gate_id for gate_id in ratchets if gate_id not in registry]
    if missing:
        return False, [
            f"FAIL: {label}: Ratchet gate id(s) not found in quality/gates.yaml: "
            f"{', '.join(missing)}."
        ]

    untouched = [
        gate_id for gate_id in ratchets if not gate_surface_touched(registry[gate_id], normalized_files)
    ]
    if untouched:
        return False, [
            f"FAIL: {label}: commit does not touch the registered gate surface for: "
            f"{', '.join(untouched)}."
        ]

    return True, [f"PASS: {label}: Ratchet trailer references registered touched gate(s)."]


def read_message(args: argparse.Namespace) -> str | None:
    if args.message is not None:
        return args.message
    if args.message_file is not None:
        return Path(args.message_file).read_text(encoding="utf-8")
    return None


def pre_commit_message(repo: Path) -> str | None:
    candidate = Path(run_git(repo, "rev-parse", "--path-format=absolute", "--git-path", "COMMIT_EDITMSG"))
    if candidate.exists():
        return candidate.read_text(encoding="utf-8", errors="replace")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository path (default: current directory)")
    parser.add_argument(
        "--gates",
        default="quality/gates.yaml",
        help="gate registry path relative to repo or absolute path",
    )
    parser.add_argument("--validate-gates", action="store_true", help="validate only the gate registry")
    parser.add_argument("--pre-commit", action="store_true", help="check staged files for hook use")
    parser.add_argument("--staged", action="store_true", help="use staged files as the changed-file set")
    parser.add_argument("--message", help="commit message text to check")
    parser.add_argument("--message-file", help="path to a commit message file to check")
    parser.add_argument("--file", action="append", default=[], help="changed file path; repeatable")
    parser.add_argument("--commit-range", help="git commit range to check, for example base..head")
    args = parser.parse_args(argv)

    try:
        repo = Path(args.repo).resolve()
        repo = Path(run_git(repo, "rev-parse", "--show-toplevel")).resolve()
        gates_path = Path(args.gates)
        if not gates_path.is_absolute():
            gates_path = repo / gates_path
        registry = load_gate_registry(gates_path)
    except (RuntimeError, ValueError) as exc:
        print(f"FAIL: gate registry validation failed: {exc}")
        return 1

    if args.validate_gates:
        print(f"PASS: {gates_path} validates; registered gates: {len(registry)}.")
        return 0

    all_messages: list[str] = []
    success = True

    try:
        if args.commit_range:
            commits = commits_in_range(repo, args.commit_range)
            if not commits:
                print(f"PASS: no commits found in range {args.commit_range}.")
                return 0
            for commit in commits:
                ok, messages = check_message_and_files(
                    message=commit_message(repo, commit),
                    files=commit_files(repo, commit),
                    registry=registry,
                    label=commit[:12],
                )
                success = success and ok
                all_messages.extend(messages)
        else:
            message = read_message(args)
            files = [normalize_path(path) for path in args.file]
            if args.staged or args.pre_commit:
                files = staged_files(repo)
            if args.pre_commit and message is None:
                message = pre_commit_message(repo)
                if message is None:
                    print(
                        "PASS: ratchet pre-commit validation ran; commit message is not "
                        "available to pre-commit, so trailer enforcement is deferred to "
                        "commit-message-aware invocation or CI."
                    )
                    return 0
            if message is None:
                print("FAIL: provide --message, --message-file, --commit-range, or --pre-commit.")
                return 1
            ok, messages = check_message_and_files(
                message=message,
                files=files,
                registry=registry,
                label="working tree",
            )
            success = ok
            all_messages.extend(messages)
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1

    for message in all_messages:
        print(message)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
