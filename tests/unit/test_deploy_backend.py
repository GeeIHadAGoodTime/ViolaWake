"""Detector for the `backend-deploy-automation` gate.

The regression shape this catches: a merged backend fix that never reaches the
running container, because deploying is a human SSH session that nothing
requires anybody to run. Measured on 2026-07-31 -- four backend fixes merged
2026-07-29/30 with no automated path to the box, and a customer-facing billing
fix (PR #36) still not serving two hours after it merged.

These tests are deliberately behavioural rather than "the file exists": they
drive `BackendDeployer` with an injected command runner so every refusal, the
rollback, and the serving-revision oracle are exercised without a Docker
daemon. Delete the automation and this module cannot import; weaken a guard and
the matching test goes red.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "deploy_backend.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("deploy_backend_under_test", SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


deploy_backend = _load_module()

TARGET = "3697759f2ca60a01dcada4531faf6892dabc2744"
PREVIOUS = "577a5f7aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
IMAGE_ID = "sha256:3e3a28a148d4f0804033f3f9bf8b9930582f557cce7dc56486ac0dc9cabe031d"


class FakeRunner:
    """Command runner that answers from a scripted table and records calls.

    Matching is on the joined argv, so a test overrides exactly the leg it is
    about and inherits a healthy default for everything else.
    """

    def __init__(self, overrides: dict[str, tuple[int, str]] | None = None):
        self.overrides = overrides or {}
        self.calls: list[str] = []

    def default(self, joined: str) -> tuple[int, str]:
        if "git fetch" in joined:
            return 0, ""
        if "git rev-parse origin/master" in joined:
            return 0, TARGET
        if "git rev-parse HEAD" in joined:
            return 0, PREVIOUS
        if "git rev-parse --abbrev-ref HEAD" in joined:
            return 0, "master"
        if "git status --porcelain" in joined:
            return 0, ""
        if "git diff --name-only" in joined:
            return 0, ""
        if "git merge --ff-only" in joined:
            return 0, ""
        if "docker inspect" in joined and "{{.Image}}" in joined:
            return 0, IMAGE_ID
        if "docker image inspect" in joined and "revision" in joined:
            # After a build the running image carries the target revision.
            return 0, TARGET if self._built else PREVIOUS
        if "docker inspect" in joined and "State.Health" in joined:
            return 0, "healthy"
        if "docker tag" in joined:
            return 0, ""
        if "check_in_flight_jobs.py" in joined:
            return 0, "queue idle"
        if "docker compose" in joined and " build " in f" {joined} ":
            self._built = True
            return 0, "built"
        if "docker compose" in joined and "run --rm --no-deps" in joined:
            return 0, ""
        if "docker compose" in joined and "up -d" in joined:
            return 0, ""
        return 0, ""

    _built = False

    def __call__(self, cmd, cwd=None, timeout=300, env=None):
        joined = " ".join(str(c) for c in cmd)
        self.calls.append(joined)
        for needle, (code, out) in self.overrides.items():
            if needle in joined:
                if "docker compose" in joined and " build " in f" {joined} ":
                    self._built = True
                return deploy_backend.CommandResult(code, out, "" if code == 0 else out)
        code, out = self.default(joined)
        return deploy_backend.CommandResult(code, out, "" if code == 0 else out)

    def ran(self, needle: str) -> bool:
        return any(needle in c for c in self.calls)

    def count(self, needle: str) -> int:
        return sum(1 for c in self.calls if needle in c)


def make_deployer(tmp_path: Path, overrides=None, http=(200, "ok"), free_gb=100.0, **cfg_kwargs):
    cfg = deploy_backend.DeployConfig(
        repo=REPO_ROOT,
        state_dir=tmp_path / "state",
        alert_sink=tmp_path / "alerts.jsonl",
        health_timeout_s=10,
        poll_interval_s=0.0,
        **cfg_kwargs,
    )
    runner = FakeRunner(overrides)
    deployer = deploy_backend.BackendDeployer(
        cfg,
        runner=runner,
        http_get=lambda url, timeout=20: http,
        free_gb=lambda path: free_gb,
        sleep=lambda s: None,
    )
    return deployer, runner


def alerts_written(tmp_path: Path) -> list[dict]:
    sink = tmp_path / "alerts.jsonl"
    if not sink.exists():
        return []
    return [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
#  The automation exists at all (red on the pre-fix tree)
# --------------------------------------------------------------------------- #


def test_deploy_entrypoint_is_shipped_and_runnable():
    assert SCRIPT.is_file(), "scripts/deploy_backend.py is the automated deploy entrypoint"
    assert hasattr(deploy_backend, "BackendDeployer")
    assert hasattr(deploy_backend, "main")


def test_timer_and_service_units_are_shipped():
    """A script nobody runs on a schedule is still a manual deploy."""
    service = REPO_ROOT / "infra" / "deploy" / "violawake-deploy.service"
    timer = REPO_ROOT / "infra" / "deploy" / "violawake-deploy.timer"
    installer = REPO_ROOT / "infra" / "deploy" / "install.sh"
    for path in (service, timer, installer):
        assert path.is_file(), f"{path.relative_to(REPO_ROOT)} must ship with the repo"

    service_text = service.read_text(encoding="utf-8")
    assert "scripts/deploy_backend.py" in service_text, "the unit must actually run the deploy script"
    assert "Type=oneshot" in service_text

    timer_text = timer.read_text(encoding="utf-8")
    assert "Unit=violawake-deploy.service" in timer_text
    interval = re.search(r"OnUnitActiveSec=(\d+)(min|h)", timer_text)
    assert interval, "the timer must declare a repeating interval"
    minutes = int(interval.group(1)) * (60 if interval.group(2) == "h" else 1)
    assert 0 < minutes <= 60, "a merge must reach customers within the hour, unattended"


def test_build_is_labelled_with_the_commit_it_was_built_from():
    """Without this label, 'what is serving?' has no ground-truth answer."""
    compose = (REPO_ROOT / "docker-compose.production.yml").read_text(encoding="utf-8")
    assert "org.opencontainers.image.revision: ${VIOLAWAKE_BUILD_SHA" in compose


# --------------------------------------------------------------------------- #
#  Reconciliation
# --------------------------------------------------------------------------- #


def test_no_drift_is_a_noop(tmp_path):
    deployer, runner = make_deployer(
        tmp_path, overrides={"docker image inspect": (0, TARGET)}
    )
    assert deployer.deploy() == 0
    assert deployer.report.outcome == "up-to-date"
    assert not runner.ran("compose"), "an up-to-date host must not rebuild or recreate anything"
    assert alerts_written(tmp_path) == []


def test_happy_path_builds_preflights_recreates_and_verifies(tmp_path):
    deployer, runner = make_deployer(tmp_path)
    assert deployer.deploy() == 0
    assert deployer.report.outcome == "deployed"
    assert runner.ran("git merge --ff-only")
    assert runner.ran("build backend")
    assert runner.ran("run --rm --no-deps"), "the image must be import-preflighted before traffic moves"
    assert runner.ran("up -d backend")
    phases = {p["phase"]: p["ok"] for p in deployer.report.phases}
    for required in ("serving-revision", "live-endpoint", "container-health", "rollback-point"):
        assert phases.get(required) is True, f"{required} must be verified on a real deploy"
    assert alerts_written(tmp_path) == []


def test_preflight_runs_before_the_container_is_recreated(tmp_path):
    deployer, runner = make_deployer(tmp_path)
    deployer.deploy()
    preflight = next(i for i, c in enumerate(runner.calls) if "run --rm --no-deps" in c)
    recreate = next(i for i, c in enumerate(runner.calls) if "up -d backend" in c)
    assert preflight < recreate


def test_dry_run_touches_nothing(tmp_path):
    deployer, runner = make_deployer(tmp_path, dry_run=True)
    assert deployer.deploy() == 0
    assert deployer.report.outcome == "dry-run"
    assert not runner.ran("build")
    assert not runner.ran("up -d")
    assert not runner.ran("git merge")


# --------------------------------------------------------------------------- #
#  Guards -- each one refuses, and refuses without side effects
# --------------------------------------------------------------------------- #


def test_in_flight_training_jobs_defer_instead_of_killing_the_job(tmp_path):
    """Job 51 (2026-05-07): recreating the backend kills a running job."""
    deployer, runner = make_deployer(
        tmp_path, overrides={"check_in_flight_jobs.py": (1, "1 job RUNNING")}
    )
    assert deployer.deploy() == 0, "a deferral is not a failure -- the next tick retries"
    assert deployer.report.outcome == "deferred"
    assert not runner.ran("up -d"), "the container must not be recreated while a job is running"
    assert not runner.ran("build")


def test_unreadable_job_queue_is_not_permission_to_deploy(tmp_path):
    """Exit 2 from the guard means it could not see. Fail closed."""
    deployer, runner = make_deployer(
        tmp_path, overrides={"check_in_flight_jobs.py": (2, "docker not running")}
    )
    assert deployer.deploy() == 1
    assert deployer.report.outcome == "refused"
    assert not runner.ran("up -d")
    assert any("REFUSED" in a["message"] for a in alerts_written(tmp_path))


def test_destructive_migration_blocks_the_unattended_deploy(tmp_path):
    revision = "console/backend/alembic/versions/0099_drop_recordings.py"
    deployer, runner = make_deployer(
        tmp_path,
        overrides={
            "git diff --name-only": (0, revision),
            "git show": (0, "def upgrade():\n    op.drop_column('recordings', 'transcript')\n"),
        },
    )
    assert deployer.deploy() == 1
    assert deployer.report.outcome == "refused"
    assert "destructive" in deployer.report.reason
    assert not runner.ran("build"), "a timer must not drop customer columns on its own"


def test_destructive_migration_deploys_when_explicitly_allowed(tmp_path):
    revision = "console/backend/alembic/versions/0099_drop_recordings.py"
    deployer, runner = make_deployer(
        tmp_path,
        overrides={
            "git diff --name-only": (0, revision),
            "git show": (0, "def upgrade():\n    op.drop_column('recordings', 'transcript')\n"),
        },
        allow_destructive_migrations=True,
    )
    assert deployer.deploy() == 0
    assert deployer.report.outcome == "deployed"
    assert runner.ran("build")


def test_additive_migration_deploys_automatically(tmp_path):
    revision = "console/backend/alembic/versions/0100_add_flag.py"
    deployer, runner = make_deployer(
        tmp_path,
        overrides={
            "git diff --name-only": (0, revision),
            "git show": (0, "def upgrade():\n    op.add_column('users', sa.Column('flag', sa.Boolean()))\n"),
        },
    )
    assert deployer.deploy() == 0
    assert deployer.report.outcome == "deployed"


def test_low_disk_refuses_to_build(tmp_path):
    """The host also serves other production stacks; a full disk is their outage too."""
    deployer, runner = make_deployer(tmp_path, free_gb=2.0)
    assert deployer.deploy() == 1
    assert deployer.report.outcome == "refused"
    assert not runner.ran("build")


def test_dirty_deploy_checkout_refuses(tmp_path):
    deployer, runner = make_deployer(
        tmp_path, overrides={"git status --porcelain": (0, " M console/backend/app/main.py")}
    )
    assert deployer.deploy() == 1
    assert not runner.ran("git merge"), "never fast-forward over somebody's uncommitted work"
    assert not runner.ran("build")


def test_diverged_checkout_is_never_force_moved(tmp_path):
    deployer, runner = make_deployer(
        tmp_path, overrides={"git merge --ff-only": (1, "Not possible to fast-forward")}
    )
    assert deployer.deploy() in (2, 3)
    assert not runner.ran("reset --hard")
    assert not runner.ran("checkout ."), "a diverged checkout is a human's problem, not a reset"


# --------------------------------------------------------------------------- #
#  Failure handling: rollback + paging
# --------------------------------------------------------------------------- #


def test_unhealthy_container_rolls_back_to_the_previous_image(tmp_path):
    deployer, runner = make_deployer(
        tmp_path, overrides={"State.Health": (0, "unhealthy")}
    )
    code = deployer.deploy()
    assert code == 3, "rollback health also fails in this fixture, so this is the loud exit"
    assert runner.ran("docker tag ghcr.io/geeihadagoodtime/wakeword-backend:rollback")
    alerts = alerts_written(tmp_path)
    assert alerts and "FAILED" in alerts[0]["message"]
    assert alerts[0]["business"] == "violawake", "the ops bridge routes on this field"


def test_rollback_restores_service_and_reports_exit_2(tmp_path):
    """New image unhealthy, previous image healthy -> service restored, still paged."""
    state = {"ups": 0}

    class RollbackRunner(FakeRunner):
        def __call__(self, cmd, cwd=None, timeout=300, env=None):
            joined = " ".join(str(c) for c in cmd)
            if "up -d" in joined:
                state["ups"] += 1
            if "State.Health" in joined:
                self.calls.append(joined)
                # Unhealthy until the rollback's `up -d` has run.
                return deploy_backend.CommandResult(
                    0, "healthy" if state["ups"] >= 2 else "unhealthy", ""
                )
            return super().__call__(cmd, cwd=cwd, timeout=timeout, env=env)

    cfg = deploy_backend.DeployConfig(
        repo=REPO_ROOT,
        state_dir=tmp_path / "state",
        alert_sink=tmp_path / "alerts.jsonl",
        health_timeout_s=10,
        poll_interval_s=0.0,
    )
    runner = RollbackRunner()
    deployer = deploy_backend.BackendDeployer(
        cfg, runner=runner, http_get=lambda u, t=20: (200, "ok"),
        free_gb=lambda p: 100.0, sleep=lambda s: None,
    )
    assert deployer.deploy() == 2
    assert deployer.report.outcome == "failed"
    assert any(p["phase"] == "rollback" and p["ok"] for p in deployer.report.phases)
    assert any("Rolled back" in a["message"] for a in alerts_written(tmp_path))


def test_healthy_container_serving_the_old_revision_is_a_failed_deploy(tmp_path):
    """A green healthcheck is not proof the new code is serving."""
    deployer, runner = make_deployer(
        tmp_path, overrides={"docker image inspect": (0, PREVIOUS)}
    )
    assert deployer.deploy() in (2, 3)
    assert "not the deployed target" in deployer.report.reason


def test_live_endpoint_not_answering_rolls_back(tmp_path):
    deployer, runner = make_deployer(tmp_path, http=(502, "bad gateway"))
    assert deployer.deploy() in (2, 3)
    assert deployer.report.outcome == "failed"
    assert "did not return 200" in deployer.report.reason


def test_build_failure_never_recreates_the_container(tmp_path):
    deployer, runner = make_deployer(
        tmp_path, overrides={"build backend": (1, "ERROR: failed to solve")}
    )
    assert deployer.deploy() in (2, 3)
    assert not runner.ran("up -d backend"), "a failed build must leave the live container alone"


def test_import_preflight_failure_never_recreates_the_container(tmp_path):
    """The untested image must neither serve nor stay tagged `:latest`."""
    deployer, runner = make_deployer(
        tmp_path, overrides={"run --rm --no-deps": (1, "ModuleNotFoundError: audio_core")}
    )
    assert deployer.deploy() in (2, 3)
    assert "import preflight failed" in deployer.report.reason
    assert not runner.ran("up -d backend"), (
        "the container was never recreated, so rolling it back would kill a job for nothing"
    )
    assert runner.ran(
        "docker tag ghcr.io/geeihadagoodtime/wakeword-backend:rollback "
        "ghcr.io/geeihadagoodtime/wakeword-backend:latest"
    ), "`:latest` must not be left pointing at an image that failed preflight"


# --------------------------------------------------------------------------- #
#  Drift never rots silently
# --------------------------------------------------------------------------- #


def test_persistent_undeployed_drift_pages(tmp_path):
    state = tmp_path / "state"
    state.mkdir(parents=True)
    long_ago = datetime.now(timezone.utc) - timedelta(hours=9)
    (state / "first_seen_drift.json").write_text(
        json.dumps({"target_sha": TARGET, "first_seen": long_ago.isoformat()}), encoding="utf-8"
    )
    deployer, _ = make_deployer(
        tmp_path, overrides={"check_in_flight_jobs.py": (1, "1 job RUNNING")}, stale_hours=6.0
    )
    assert deployer.deploy() == 0
    alerts = alerts_written(tmp_path)
    assert alerts and "STALE" in alerts[0]["message"]
    assert alerts[0]["source"] == "scripts/deploy_backend.py"


def test_fresh_drift_does_not_page(tmp_path):
    deployer, _ = make_deployer(
        tmp_path, overrides={"check_in_flight_jobs.py": (1, "1 job RUNNING")}, stale_hours=6.0
    )
    assert deployer.deploy() == 0
    assert alerts_written(tmp_path) == []


def test_successful_deploy_clears_the_drift_marker(tmp_path):
    state = tmp_path / "state"
    state.mkdir(parents=True)
    marker = state / "first_seen_drift.json"
    marker.write_text(
        json.dumps({"target_sha": TARGET, "first_seen": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )
    deployer, _ = make_deployer(tmp_path)
    assert deployer.deploy() == 0
    assert not marker.exists()


def test_every_run_is_journalled(tmp_path):
    deployer, _ = make_deployer(tmp_path)
    deployer.deploy()
    journal = tmp_path / "state" / "journal.jsonl"
    assert journal.exists()
    record = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
    assert record["outcome"] == "deployed"
    assert record["target_sha"] == TARGET
    assert record["deployed_sha"] == PREVIOUS


# --------------------------------------------------------------------------- #
#  Standalone-repo boundary
# --------------------------------------------------------------------------- #


def test_no_sibling_project_paths_or_env_are_hard_coded():
    """ViolaWake is standalone: the alert sink is configuration, not a path.

    Naming a sibling project's postmortem in a comment is fine and useful.
    Reaching into its filesystem, hosts or env namespace is not.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    without_own_prefix = text.replace("VIOLAWAKE_", "")
    for forbidden in ("/opt/viola/", "red_alerts.jsonl", "useviola.com", "VIOLA_", "api.useviola"):
        assert forbidden not in without_own_prefix, f"{forbidden} must not be hard-coded here"


@pytest.mark.parametrize(
    "snippet",
    [
        "op.drop_table('jobs')",
        "op.drop_column('users', 'email')",
        "op.alter_column('jobs', 'status', type_=sa.Integer())",
        "op.execute('TRUNCATE recordings')",
        "op.drop_constraint('fk_jobs_user', 'jobs')",
    ],
)
def test_destructive_patterns_are_recognised(snippet):
    assert any(p.search(snippet) for p in deploy_backend.DESTRUCTIVE_MIGRATION_PATTERNS), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        "op.add_column('users', sa.Column('flag', sa.Boolean()))",
        "op.create_index('ix_jobs_user', 'jobs', ['user_id'])",
        "op.create_table('audit', sa.Column('id', sa.Integer()))",
    ],
)
def test_additive_patterns_are_not_flagged(snippet):
    assert not any(p.search(snippet) for p in deploy_backend.DESTRUCTIVE_MIGRATION_PATTERNS), snippet
