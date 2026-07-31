#!/usr/bin/env python3
"""Automated, guarded deploy of the ViolaWake Console backend.

Why this exists
---------------
Until 2026-07-31 the backend behind ``https://api.violawake.com`` had no
automated path from a merged commit to the running container: every deploy
was a human opening an SSH session and running the sequence in
``docs/DEPLOYMENT.md`` by hand. Four backend fixes merged on 2026-07-29/30
(PRs #26, #28, #32, #33) and nobody could say whether any of them was live.
The measured shape of the failure: on 2026-07-31 at 22:00 UTC the deployed
image was built from ``577a5f7`` while ``origin/master`` was already at
``3697759`` -- a merged customer-facing billing fix that was not serving,
two hours after it landed.

This script is the automation. It is **pull-based**: it runs ON the host that
runs the container, on a timer, and reconciles "what is running" toward
"what is on ``origin/master``". Nothing inbound is opened, no CI credential
touches the box, and no self-hosted runner is attached to what is a *public*
repository (a fork PR on a self-hosted runner with a docker socket would own
the whole box).

It is safe to run every few minutes: with no drift it is a no-op, and every
phase is guarded.

Guards (each one exists because of a real incident or a real hazard)
-------------------------------------------------------------------
* **Dirty checkout** -- refuse. Somebody is mid-hand-edit on the box; a deploy
  would either clobber them or bake their scratch work into a customer image.
* **Disk floor** -- refuse to build below the floor. The backend image is
  ~10 GB and this host also serves other production stacks; a build that
  fills the disk is an outage of everything on the box, not just ViolaWake.
* **In-flight training jobs** -- defer, do not recreate. Recreating the
  container kills a RUNNING job, and a slow progress event during the new
  container's warmup flips it to FAILED (Job 51, 2026-05-07). Reuses the
  existing ``scripts/check_in_flight_jobs.py``.
* **Destructive migration** -- refuse. The entrypoint runs
  ``alembic upgrade head`` on every start. Additive revisions deploy
  automatically; a revision that drops a table/column, retypes a column or
  executes destructive SQL is a maintenance-window decision a timer must not
  make for a customer database. Pass ``--allow-destructive-migrations`` to
  deploy one deliberately.
* **Import preflight** -- run the new image's interpreter against the real
  compose environment before any traffic-affecting recreate. A route module
  that imports something absent from the image otherwise produces a
  crash-looping container instead of a refused deploy (NOVVIOLA's 2026-07-05
  outage #2, same failure shape, different stack).
* **Health verification + rollback** -- after the recreate, the container must
  report healthy, the live endpoint must answer 200, and the running image's
  revision label must equal the target commit. Any of those failing rolls the
  previous image back and pages.

Nothing rots silently: a failed deploy pages immediately, and drift that
persists past ``--stale-hours`` pages too, so "the timer quietly stopped
working" cannot look the same as "everything is deployed".

Usage
-----
    python scripts/deploy_backend.py                 # reconcile toward origin/master
    python scripts/deploy_backend.py --dry-run       # print the plan, touch nothing
    python scripts/deploy_backend.py --force         # deploy despite in-flight jobs
    python scripts/deploy_backend.py --target-ref origin/master --stale-hours 6

Exit codes
----------
    0  -- nothing to do, or the deploy succeeded, or it was deliberately
          deferred (a deferral is not a failure; the next tick retries)
    1  -- a guard refused the deploy (disk, destructive migration, dirty tree)
    2  -- the deploy was attempted and failed; a rollback was attempted
    3  -- the deploy failed AND the rollback failed (the loud case)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGE = "ghcr.io/geeihadagoodtime/wakeword-backend"
DEFAULT_CONTAINER = "wakeword-backend-1"
DEFAULT_SERVICE = "backend"
DEFAULT_COMPOSE_FILES = ("docker-compose.production.yml", "docker-compose.viola-bridge.yml")
DEFAULT_TARGET_REF = "origin/master"
DEFAULT_HEALTH_URL = "https://api.violawake.com/api/health"
DEFAULT_DISK_FLOOR_GB = 15.0
DEFAULT_HEALTH_TIMEOUT_S = 420
DEFAULT_STALE_HOURS = 6.0
DEFAULT_BUILD_TIMEOUT_S = 3600

REVISION_LABEL = "org.opencontainers.image.revision"
ROLLBACK_TAG = "rollback"

# See _real_http_get: the public health check must look like a browser or
# Cloudflare's WAF answers it 403 before it reaches us.
HEALTH_CHECK_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

# Statuses that will not change by waiting. A 403/401 from the edge is a
# standing refusal, not a service still warming up, so retrying it for the full
# health timeout only delays the rollback.
NON_RETRYABLE_HTTP = frozenset({401, 403, 404, 405})
NON_RETRYABLE_ATTEMPTS = 3

ALERT_SOURCE = "scripts/deploy_backend.py"
ALERT_BUSINESS = "violawake"
ALERT_STREAM = "violawake-deploy"

# Alembic operations that are not safely automatable against a customer
# database. `op.execute` is included because raw SQL can hide anything.
DESTRUCTIVE_MIGRATION_PATTERNS = (
    re.compile(r"\bop\.drop_table\s*\(", re.M),
    re.compile(r"\bop\.drop_column\s*\(", re.M),
    re.compile(r"\bop\.drop_constraint\s*\(", re.M),
    re.compile(r"\bop\.rename_table\s*\(", re.M),
    re.compile(r"\bop\.alter_column\s*\([^)]*\btype_\s*=", re.M | re.S),
    re.compile(r"\bop\.execute\s*\(", re.M),
)

MIGRATIONS_DIR = "console/backend/alembic/versions/"


class DeployError(RuntimeError):
    """A phase failed in a way that must abort the deploy."""


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def out(self) -> str:
        return self.stdout.strip()


def _real_runner(cmd: Sequence[str], cwd: Path | None = None, timeout: int = 300,
                 env: dict[str, str] | None = None) -> CommandResult:
    """Execute a command for real. Injected away in tests."""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            list(cmd),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(returncode=124, stdout=exc.stdout or "", stderr=f"timeout after {timeout}s")
    return CommandResult(proc.returncode, proc.stdout or "", proc.stderr or "")


def _real_http_get(url: str, timeout: int = 20) -> tuple[int, str]:
    """GET the public health URL the way a customer's browser would.

    The User-Agent is not cosmetic. Measured on the deploy host 2026-07-31:
    `curl https://api.violawake.com/api/health` returned **200** while
    `urllib.request.urlopen(...)` on the same host, same second, returned
    **403** -- Cloudflare's WAF rejects the default `Python-urllib/3.x` agent
    before the request ever reaches our tunnel. The first live run of this
    reconciler deployed correctly (built, preflighted, recreated, healthy,
    correct revision label) and then rolled itself back because our own edge
    refused our own verification. A verification that our own infrastructure
    blocks is not a verification.
    """
    request = urllib.request.Request(  # noqa: S310 - fixed https URL
        url,
        headers={"User-Agent": HEALTH_CHECK_USER_AGENT, "Accept": "application/json, */*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception as exc:  # noqa: BLE001 - any transport failure is "not 200"
        return 0, str(exc)


def _real_free_gb(path: Path) -> float:
    return shutil.disk_usage(str(path)).free / (1024 ** 3)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DeployConfig:
    repo: Path = REPO_ROOT
    target_ref: str = DEFAULT_TARGET_REF
    image: str = DEFAULT_IMAGE
    container: str = DEFAULT_CONTAINER
    service: str = DEFAULT_SERVICE
    compose_files: tuple[str, ...] = DEFAULT_COMPOSE_FILES
    env_file: str | None = None
    health_url: str = DEFAULT_HEALTH_URL
    disk_floor_gb: float = DEFAULT_DISK_FLOOR_GB
    health_timeout_s: int = DEFAULT_HEALTH_TIMEOUT_S
    build_timeout_s: int = DEFAULT_BUILD_TIMEOUT_S
    stale_hours: float = DEFAULT_STALE_HOURS
    allow_destructive_migrations: bool = False
    force: bool = False
    dry_run: bool = False
    state_dir: Path = Path("/var/lib/violawake-deploy")
    alert_sink: Path | None = None
    poll_interval_s: float = 5.0


@dataclass
class DeployReport:
    outcome: str = "unknown"
    reason: str = ""
    deployed_sha: str | None = None
    deployed_sha_source: str = "unknown"
    target_sha: str | None = None
    phases: list[dict] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: _utcnow().isoformat())

    def phase(self, name: str, ok: bool, detail: str = "") -> None:
        self.phases.append({"phase": name, "ok": ok, "detail": detail})

    def as_record(self) -> dict:
        return {
            "timestamp": self.started_at,
            "finished_at": _utcnow().isoformat(),
            "outcome": self.outcome,
            "reason": self.reason,
            "deployed_sha": self.deployed_sha,
            "deployed_sha_source": self.deployed_sha_source,
            "target_sha": self.target_sha,
            "phases": self.phases,
        }


class BackendDeployer:
    """Reconcile the running backend container toward a target commit.

    Every side effect goes through ``self.run`` / ``self.http_get`` /
    ``self.free_gb`` so the whole decision tree -- including the rollback and
    the refusals -- is exercisable without a Docker daemon.
    """

    def __init__(
        self,
        config: DeployConfig,
        runner: Callable[..., CommandResult] = _real_runner,
        http_get: Callable[[str, int], tuple[int, str]] = _real_http_get,
        free_gb: Callable[[Path], float] = _real_free_gb,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.cfg = config
        self.run = runner
        self.http_get = http_get
        self.free_gb = free_gb
        self.sleep = sleep
        self.now = now
        self.report = DeployReport()
        # Whether this run has already recreated the live container. Decides
        # whether a rollback may touch it (see rollback()).
        self._recreated = False
        # The image generation displaced by this run's rollback point; reclaimed
        # only after the deploy is proven good (see prune_superseded_image()).
        self._superseded_image_id: str | None = None

    # ---------------------------------------------------------------- helpers

    def _git(self, *args: str, timeout: int = 180) -> CommandResult:
        return self.run(["git", *args], cwd=self.cfg.repo, timeout=timeout)

    def _compose(self, *args: str, timeout: int = 300, env: dict[str, str] | None = None) -> CommandResult:
        cmd = ["docker", "compose"]
        if self.cfg.env_file:
            cmd += ["--env-file", self.cfg.env_file]
        for f in self.cfg.compose_files:
            cmd += ["-f", f]
        cmd += list(args)
        return self.run(cmd, cwd=self.cfg.repo, timeout=timeout, env=env)

    def _docker(self, *args: str, timeout: int = 120) -> CommandResult:
        return self.run(["docker", *args], cwd=self.cfg.repo, timeout=timeout)

    def _log(self, msg: str) -> None:
        sys.stderr.write(f"[deploy_backend] {msg}\n")
        sys.stderr.flush()

    def alert(self, message: str, alert_id: str, context: str = "") -> None:
        """Emit a red-alert record.

        The record shape is the shared operator red-alert inbox schema, and it
        carries ``business: violawake`` so the ops-ticket bridge routes it to
        this repository's issue queue. The sink PATH is configuration, never a
        hard-coded cross-project path -- ViolaWake stays standalone and the
        host decides where operator alerts land.
        """
        record = {
            "timestamp": self.now().isoformat(),
            "source": ALERT_SOURCE,
            "message": message,
            "stream": ALERT_STREAM,
            "context": context,
            "id": alert_id,
            "business": ALERT_BUSINESS,
        }
        self.report.alerts.append(record)
        self._log(f"ALERT {alert_id}: {message}")
        sink = self.cfg.alert_sink
        if not sink or self.cfg.dry_run:
            return
        try:
            sink.parent.mkdir(parents=True, exist_ok=True)
            with sink.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            self._log(f"could not write alert sink {sink}: {exc}")

    # ------------------------------------------------------------- resolution

    def resolve_target_sha(self) -> str:
        fetch = self._git("fetch", "--quiet", "origin", "master")
        if not fetch.ok:
            raise DeployError(f"git fetch failed: {fetch.stderr.strip() or fetch.stdout.strip()}")
        rev = self._git("rev-parse", self.cfg.target_ref)
        if not rev.ok or not rev.out:
            raise DeployError(f"cannot resolve {self.cfg.target_ref}: {rev.stderr.strip()}")
        return rev.out.split()[0]

    def running_image_id(self) -> str | None:
        res = self._docker("inspect", self.cfg.container, "--format", "{{.Image}}")
        if not res.ok or not res.out:
            return None
        return res.out

    def image_revision(self, image_ref: str) -> str | None:
        """Read the git revision baked into an image by ``build.labels``."""
        res = self._docker(
            "image", "inspect", image_ref,
            "--format", '{{index .Config.Labels "' + REVISION_LABEL + '"}}',
        )
        if not res.ok:
            return None
        value = res.out
        if not value or value in {"<no value>", "unknown", "null"}:
            return None
        return value

    def resolve_deployed_sha(self) -> tuple[str | None, str]:
        """What commit is actually serving right now?

        Authoritative source is the label on the running container's image.
        Before the first labelled build there is no label, so fall back to the
        checkout's HEAD -- which is what the last hand-run
        ``docker compose build`` used -- and say so, rather than guessing
        silently.
        """
        image_id = self.running_image_id()
        if image_id:
            revision = self.image_revision(image_id)
            if revision:
                return revision, "image-label"
        head = self._git("rev-parse", "HEAD")
        if head.ok and head.out:
            return head.out.split()[0], "checkout-head-assumed"
        return None, "unknown"

    # ----------------------------------------------------------------- guards

    def guard_clean_checkout(self) -> None:
        status = self._git("status", "--porcelain")
        if not status.ok:
            raise DeployError(f"git status failed: {status.stderr.strip()}")
        if status.out:
            raise DeployError(
                "the deploy checkout has uncommitted changes; refusing to build or "
                "fast-forward over somebody's work:\n" + status.out
            )

    def guard_disk(self) -> None:
        free = self.free_gb(self.cfg.repo)
        if free < self.cfg.disk_floor_gb:
            raise DeployError(
                f"only {free:.1f} GiB free on the deploy host, floor is "
                f"{self.cfg.disk_floor_gb:.1f} GiB. Refusing to build: this host also runs "
                "other production stacks and a full disk takes all of them down."
            )
        self.report.phase("disk", True, f"{free:.1f} GiB free")

    def guard_in_flight_jobs(self) -> bool:
        """True to proceed, False to defer until the training queue drains."""
        if self.cfg.force:
            self.report.phase("in-flight-jobs", True, "bypassed with --force")
            return True
        script = self.cfg.repo / "scripts" / "check_in_flight_jobs.py"
        res = self.run(
            [sys.executable, str(script), "--container", self.cfg.container],
            cwd=self.cfg.repo,
            timeout=120,
        )
        if res.returncode == 0:
            self.report.phase("in-flight-jobs", True, "queue idle")
            return True
        if res.returncode == 1:
            self.report.phase("in-flight-jobs", False, "jobs in flight -- deferring")
            return False
        # Exit 2 (or anything else) means the guard could not read the queue. A
        # guard that cannot see is not permission to proceed.
        raise DeployError(
            f"in-flight-job guard could not query {self.cfg.container} "
            f"(exit {res.returncode}): {res.stderr.strip() or res.stdout.strip()}"
        )

    def changed_migrations(self, from_sha: str | None, to_sha: str) -> list[str]:
        if not from_sha:
            return []
        res = self._git("diff", "--name-only", "--diff-filter=AM", f"{from_sha}..{to_sha}", "--", MIGRATIONS_DIR)
        if not res.ok:
            raise DeployError(f"could not diff migrations {from_sha}..{to_sha}: {res.stderr.strip()}")
        return [line.strip() for line in res.out.splitlines() if line.strip()]

    def guard_migrations(self, from_sha: str | None, to_sha: str) -> None:
        paths = self.changed_migrations(from_sha, to_sha)
        if not paths:
            self.report.phase("migrations", True, "no new revisions")
            return
        destructive: list[str] = []
        for path in paths:
            show = self._git("show", f"{to_sha}:{path}")
            if not show.ok:
                raise DeployError(f"could not read migration {path} at {to_sha}: {show.stderr.strip()}")
            for pattern in DESTRUCTIVE_MIGRATION_PATTERNS:
                match = pattern.search(show.stdout)
                if match:
                    destructive.append(f"{path}: {match.group(0).strip()}")
                    break
        if destructive and not self.cfg.allow_destructive_migrations:
            raise DeployError(
                "new alembic revision(s) contain destructive operations; an unattended "
                "timer must not run these against the customer database. Deploy by hand "
                "with --allow-destructive-migrations after reviewing:\n  "
                + "\n  ".join(destructive)
            )
        detail = f"{len(paths)} new revision(s)"
        if destructive:
            detail += f", {len(destructive)} destructive (explicitly allowed)"
        self.report.phase("migrations", True, detail)

    # ------------------------------------------------------------- deploy legs

    def fast_forward(self, target_sha: str) -> None:
        branch = self._git("rev-parse", "--abbrev-ref", "HEAD")
        if not branch.ok or branch.out == "HEAD":
            raise DeployError("deploy checkout is not on a branch (detached HEAD); refusing to move it")
        res = self._git("merge", "--ff-only", target_sha)
        if not res.ok:
            raise DeployError(
                "fast-forward to the target commit failed -- the deploy checkout has "
                f"diverged from {self.cfg.target_ref}: {res.stderr.strip() or res.stdout.strip()}"
            )
        self.report.phase("fast-forward", True, f"{branch.out} -> {target_sha[:12]}")

    def tag_rollback_point(self, image_id: str | None, previous_sha: str | None) -> bool:
        """Pin the currently-running image under a stable tag before we build.

        Without this the old image becomes dangling the moment the build
        retags ``:latest``, and any prune between build and rollback would
        delete the only thing we could roll back to.

        Exactly ONE generation is pinned. Tagging every deployed commit under
        its own tag would read nicely and leak ~2 GiB of disk per deploy, which
        on a shared host walks straight into this script's own disk floor and
        stops all future deploys. The image's revision label already says what
        each image is; the journal already says when it ran.
        """
        if not image_id:
            self.report.phase("rollback-point", False, "no running image to pin")
            return False
        # Remember the generation we are about to displace so it can be
        # reclaimed after a SUCCESSFUL deploy (never before -- until then it is
        # still a rollback target).
        self._superseded_image_id = self.image_id_of(f"{self.cfg.image}:{ROLLBACK_TAG}")
        res = self._docker("tag", image_id, f"{self.cfg.image}:{ROLLBACK_TAG}")
        if not res.ok:
            raise DeployError(f"could not tag the rollback point: {res.stderr.strip()}")
        self.report.phase("rollback-point", True, f"{self.cfg.image}:{ROLLBACK_TAG} -> {image_id[:19]}")
        return True

    def image_id_of(self, ref: str) -> str | None:
        res = self._docker("image", "inspect", ref, "--format", "{{.Id}}")
        return res.out if res.ok and res.out else None

    def clear_stale_recreate_backups(self) -> None:
        """Remove a leftover rename-backup container before recreating.

        Compose recreates a service by renaming the old container to
        ``<hash>_<name>``, creating the new one, then deleting the old. If a
        recreate is interrupted -- and this script interrupts recreates, that
        is what a rollback IS -- the renamed container survives and the NEXT
        recreate dies with `Conflict. The container name
        "/<hash>_wakeword-backend-1" is already in use`. Observed live on
        2026-07-31 at 23:19 UTC after three back-to-back recreates.

        Only stopped backups are removed, and only ones whose name is exactly
        our container's rename shape. A RUNNING one means a recreate is in
        flight right now, which is a refusal, not something to force.
        """
        res = self._docker(
            "ps", "-a", "--format", "{{.ID}}\t{{.Names}}\t{{.State}}",
            "--filter", f"name=_{self.cfg.container}",
        )
        if not res.ok:
            raise DeployError(f"could not list containers: {res.stderr.strip()}")
        removed = []
        for line in res.out.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            cid, name, state = (p.strip() for p in parts)
            if name == self.cfg.container or not name.endswith("_" + self.cfg.container):
                continue
            if state == "running":
                raise DeployError(
                    f"container {name} is RUNNING -- another recreate of "
                    f"{self.cfg.container} is in flight; refusing to race it"
                )
            rm = self._docker("rm", "-f", cid)
            if not rm.ok:
                raise DeployError(f"could not remove stale recreate backup {name}: {rm.stderr.strip()}")
            removed.append(name)
        self.report.phase(
            "stale-recreate-backups", True,
            ", ".join(removed) if removed else "none",
        )

    def prune_superseded_image(self) -> None:
        """Reclaim the generation two deploys back, after a proven-good deploy.

        Keeps exactly two images: what is serving and what we would roll back
        to. Removal is by explicit ID -- never a blanket `docker image prune`,
        which on this shared host would reach into other projects' images.
        Failure here is logged, never fatal: a deploy that worked must not be
        reported as failed because a cleanup did not.
        """
        stale = self._superseded_image_id
        if not stale:
            return
        keep = {self.image_id_of(f"{self.cfg.image}:latest"), self.image_id_of(f"{self.cfg.image}:{ROLLBACK_TAG}")}
        if stale in keep:
            return
        res = self._docker("rmi", stale)
        detail = f"{stale[:19]} reclaimed" if res.ok else f"{stale[:19]} still in use"
        self.report.phase("prune-superseded", res.ok, detail)

    def build(self, target_sha: str) -> None:
        res = self._compose(
            "build", self.cfg.service,
            timeout=self.cfg.build_timeout_s,
            env={"VIOLAWAKE_BUILD_SHA": target_sha},
        )
        if not res.ok:
            raise DeployError(f"image build failed: {(res.stderr or res.stdout)[-2000:]}")
        self.report.phase("build", True, target_sha[:12])

    def import_preflight(self) -> None:
        """Prove the new image can import the app before any traffic moves.

        Runs in the faithful compose environment (compose injects far more env
        than a bare ``docker run --env-file``) with ``--no-deps`` so it cannot
        disturb the live stack.
        """
        res = self._compose(
            "run", "--rm", "--no-deps", "--entrypoint", "python",
            self.cfg.service, "-c", "import app.main",
            timeout=300,
        )
        if not res.ok:
            raise DeployError(f"import preflight failed on the new image: {(res.stderr or res.stdout)[-2000:]}")
        self.report.phase("import-preflight", True, "import app.main")

    def recreate(self) -> None:
        res = self._compose("up", "-d", self.cfg.service, timeout=600)
        self._recreated = True
        if not res.ok:
            raise DeployError(f"compose up failed: {(res.stderr or res.stdout)[-2000:]}")
        self.report.phase("recreate", True, self.cfg.service)

    def _poll_attempts(self) -> int:
        """How many times a wait loop may re-check before giving up.

        Derived rather than looped-on-elapsed-time so a zero poll interval (as
        injected by tests) cannot spin forever -- a wait that never advances is
        indistinguishable from a hung deploy.
        """
        interval = self.cfg.poll_interval_s if self.cfg.poll_interval_s > 0 else 1.0
        return max(1, int(self.cfg.health_timeout_s // interval))

    def wait_for_container_health(self) -> None:
        last = ""
        for attempt in range(self._poll_attempts()):
            res = self._docker(
                "inspect", self.cfg.container,
                "--format", "{{if .State.Health}}{{.State.Health.Status}}{{else}}nohealthcheck{{end}}",
            )
            last = res.out
            if res.ok and last in {"healthy", "nohealthcheck"}:
                self.report.phase("container-health", True, last)
                return
            if res.ok and last == "unhealthy" and attempt > 0:
                # One "unhealthy" straight after a recreate is just the health
                # check not having run yet; a persistent one is the real thing.
                raise DeployError("container reports unhealthy after recreate")
            self.sleep(self.cfg.poll_interval_s)
        raise DeployError(
            f"container did not become healthy within {self.cfg.health_timeout_s}s "
            f"(last status: {last or 'unknown'})"
        )

    def verify_serving_revision(self, target_sha: str) -> None:
        """The running container must be serving the commit we deployed.

        A healthy container is not proof of a deploy: compose can decide the
        service is up-to-date and leave the old container in place. Comparing
        the revision label is what makes this a deployment oracle instead of a
        liveness check.
        """
        image_id = self.running_image_id()
        if not image_id:
            raise DeployError("cannot inspect the running container after recreate")
        revision = self.image_revision(image_id)
        if revision != target_sha:
            raise DeployError(
                f"the running container serves revision {revision or 'unlabelled'}, "
                f"not the deployed target {target_sha}"
            )
        self.report.phase("serving-revision", True, target_sha[:12])

    def verify_live_endpoint(self) -> None:
        status, body = 0, ""
        attempts = self._poll_attempts()
        for attempt in range(attempts):
            status, body = self.http_get(self.cfg.health_url, 20)
            if status == 200:
                self.report.phase("live-endpoint", True, f"{self.cfg.health_url} 200")
                return
            if status in NON_RETRYABLE_HTTP and attempt + 1 >= NON_RETRYABLE_ATTEMPTS:
                raise DeployError(
                    f"{self.cfg.health_url} answered {status} on {NON_RETRYABLE_ATTEMPTS} "
                    "consecutive attempts; that is a standing refusal, not a warming service"
                )
            self.sleep(self.cfg.poll_interval_s)
        raise DeployError(
            f"{self.cfg.health_url} did not return 200 within {self.cfg.health_timeout_s}s "
            f"(last: {status} {body[:200]})"
        )

    def rollback(self) -> bool:
        """Undo as much as was actually done, and no more.

        Two distinct situations, and conflating them is its own bug:

        * The container was never recreated (build or preflight failed). The
          live container is untouched and must STAY untouched -- a needless
          recreate here would kill a training job that started in the meantime.
          But ``:latest`` may now point at the untested image, so the tag must
          still be restored or the next ``docker compose up -d`` anyone runs
          silently ships it.
        * The container WAS recreated and did not verify. Restore the tag and
          recreate again on the previous image.
        """
        self._log("restoring the previously running image tag")
        retag = self._docker("tag", f"{self.cfg.image}:{ROLLBACK_TAG}", f"{self.cfg.image}:latest")
        if not retag.ok:
            self.report.phase("rollback", False, f"retag failed: {retag.stderr.strip()}")
            return False
        if not self._recreated:
            self.report.phase("rollback", True, "tag restored; running container never touched")
            return True
        up = self._compose("up", "-d", self.cfg.service, timeout=600)
        if not up.ok:
            self.report.phase("rollback", False, f"compose up failed: {(up.stderr or up.stdout)[-500:]}")
            return False
        try:
            self.wait_for_container_health()
        except DeployError as exc:
            self.report.phase("rollback", False, f"previous image also unhealthy: {exc}")
            return False
        self.report.phase("rollback", True, "previous image restored")
        return True

    # -------------------------------------------------------------- staleness

    def note_drift(self, deployed_sha: str, target_sha: str, deferred_reason: str) -> None:
        """Page when undeployed drift has been sitting too long.

        A single deferral is normal (a training job was running). Drift that
        survives the staleness window means the automation is not converging,
        which is the very failure this script exists to prevent -- so it has to
        be loud rather than another quiet tick.
        """
        marker = self.cfg.state_dir / "first_seen_drift.json"
        now = self.now()
        first_seen = now
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            if data.get("target_sha") == target_sha:
                first_seen = datetime.fromisoformat(data["first_seen"])
        except (OSError, ValueError, KeyError):
            pass
        if not self.cfg.dry_run:
            try:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    json.dumps({"target_sha": target_sha, "first_seen": first_seen.isoformat()}),
                    encoding="utf-8",
                )
            except OSError as exc:
                self._log(f"could not persist drift marker: {exc}")
        age_h = (now - first_seen).total_seconds() / 3600.0
        if age_h >= self.cfg.stale_hours:
            self.alert(
                f"ViolaWake backend deploy is STALE: {target_sha[:12]} has been undeployed for "
                f"{age_h:.1f}h (deployed {deployed_sha[:12]}). Reason each tick: {deferred_reason}",
                alert_id=f"violawake-deploy:stale:{target_sha[:12]}",
                context=(
                    f"deployed={deployed_sha}\ntarget={target_sha}\nfirst_seen={first_seen.isoformat()}\n"
                    f"reason={deferred_reason}\nhost_repo={self.cfg.repo}"
                ),
            )

    def clear_drift_marker(self) -> None:
        if self.cfg.dry_run:
            return
        with contextlib.suppress(OSError):
            (self.cfg.state_dir / "first_seen_drift.json").unlink()

    def journal(self) -> None:
        if self.cfg.dry_run:
            return
        path = self.cfg.state_dir / "journal.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(self.report.as_record()) + "\n")
        except OSError as exc:
            self._log(f"could not write journal {path}: {exc}")

    # ------------------------------------------------------------------ driver

    def deploy(self) -> int:
        try:
            target_sha = self.resolve_target_sha()
        except DeployError as exc:
            self.report.outcome = "error"
            self.report.reason = str(exc)
            self.alert(
                f"ViolaWake backend deploy could not resolve its target: {exc}",
                alert_id="violawake-deploy:resolve-failed",
            )
            self.journal()
            return 1
        self.report.target_sha = target_sha

        deployed_sha, source = self.resolve_deployed_sha()
        self.report.deployed_sha = deployed_sha
        self.report.deployed_sha_source = source

        if deployed_sha == target_sha and source == "image-label":
            self.report.outcome = "up-to-date"
            self.report.reason = f"{target_sha[:12]} already serving"
            self.report.phase("drift", True, "none")
            self.clear_drift_marker()
            self._log(f"up to date at {target_sha[:12]}")
            self.journal()
            return 0

        self._log(
            f"drift: serving {(deployed_sha or 'unknown')[:12]} ({source}), "
            f"target {target_sha[:12]}"
        )
        self.report.phase("drift", True, f"{(deployed_sha or 'unknown')[:12]} -> {target_sha[:12]}")

        try:
            self.guard_clean_checkout()
            self.guard_disk()
            self.guard_migrations(deployed_sha, target_sha)
            proceed = self.guard_in_flight_jobs()
        except DeployError as exc:
            self.report.outcome = "refused"
            self.report.reason = str(exc)
            self.alert(
                f"ViolaWake backend deploy REFUSED: {exc}",
                alert_id="violawake-deploy:refused",
                context=f"deployed={deployed_sha}\ntarget={target_sha}",
            )
            if deployed_sha:
                self.note_drift(deployed_sha, target_sha, f"refused: {exc}")
            self.journal()
            return 1

        if not proceed:
            self.report.outcome = "deferred"
            self.report.reason = "training jobs in flight"
            self._log("deferring: training jobs in flight")
            if deployed_sha:
                self.note_drift(deployed_sha, target_sha, "training jobs in flight")
            self.journal()
            return 0

        if self.cfg.dry_run:
            self.report.outcome = "dry-run"
            self.report.reason = f"would deploy {target_sha[:12]}"
            self._log(f"dry run: would deploy {target_sha[:12]}")
            self.journal()
            return 0

        image_id = self.running_image_id()
        try:
            self.tag_rollback_point(image_id, deployed_sha)
            self.fast_forward(target_sha)
            self.build(target_sha)
            self.import_preflight()
            self.clear_stale_recreate_backups()
            self.recreate()
            self.wait_for_container_health()
            self.verify_serving_revision(target_sha)
            self.verify_live_endpoint()
        except DeployError as exc:
            self.report.outcome = "failed"
            self.report.reason = str(exc)
            recreated = self._recreated
            rolled_back = self.rollback() if image_id else False
            if not rolled_back:
                aftermath = "ROLLBACK ALSO FAILED -- api.violawake.com needs a human now."
            elif recreated:
                aftermath = "Rolled back to the previous image."
            else:
                aftermath = "The live container was never recreated and is still serving the previous image."
            self.alert(
                f"ViolaWake backend deploy of {target_sha[:12]} FAILED: {exc}. " + aftermath,
                alert_id="violawake-deploy:failed",
                context=(
                    f"deployed={deployed_sha}\ntarget={target_sha}\nrolled_back={rolled_back}\n"
                    f"phases={json.dumps(self.report.phases)}"
                ),
            )
            self.journal()
            return 2 if rolled_back else 3

        self.report.outcome = "deployed"
        self.report.reason = f"{(deployed_sha or 'unknown')[:12]} -> {target_sha[:12]}"
        self.prune_superseded_image()
        self.clear_drift_marker()
        self._log(f"deployed {target_sha[:12]}")
        self.journal()
        return 0


def build_config(args: argparse.Namespace) -> DeployConfig:
    repo = Path(args.repo).resolve()
    state_dir = Path(args.state_dir) if args.state_dir else Path(
        os.environ.get("VIOLAWAKE_DEPLOY_STATE_DIR", "/var/lib/violawake-deploy")
    )
    sink_raw = args.alert_sink or os.environ.get("VIOLAWAKE_DEPLOY_ALERT_SINK")
    return DeployConfig(
        repo=repo,
        target_ref=args.target_ref,
        image=args.image,
        container=args.container,
        service=args.service,
        compose_files=tuple(args.compose_file) if args.compose_file else DEFAULT_COMPOSE_FILES,
        env_file=args.env_file,
        health_url=args.health_url,
        disk_floor_gb=args.disk_floor_gb,
        health_timeout_s=args.health_timeout,
        stale_hours=args.stale_hours,
        allow_destructive_migrations=args.allow_destructive_migrations,
        force=args.force or os.environ.get("VIOLAWAKE_DEPLOY_FORCE") == "1",
        dry_run=args.dry_run,
        state_dir=state_dir,
        alert_sink=Path(sink_raw) if sink_raw else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=str(REPO_ROOT), help="deploy checkout (default: this script's repo)")
    p.add_argument("--target-ref", default=DEFAULT_TARGET_REF)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--container", default=DEFAULT_CONTAINER)
    p.add_argument("--service", default=DEFAULT_SERVICE)
    p.add_argument("--compose-file", action="append", help="repeatable; defaults to the production + bridge pair")
    p.add_argument("--env-file", default=None)
    p.add_argument("--health-url", default=DEFAULT_HEALTH_URL)
    p.add_argument("--disk-floor-gb", type=float, default=DEFAULT_DISK_FLOOR_GB)
    p.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT_S)
    p.add_argument("--stale-hours", type=float, default=DEFAULT_STALE_HOURS)
    p.add_argument("--state-dir", default=None)
    p.add_argument("--alert-sink", default=None, help="JSONL red-alert inbox to append operator alerts to")
    p.add_argument("--allow-destructive-migrations", action="store_true")
    p.add_argument("--force", action="store_true", help="deploy even with training jobs in flight")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    deployer = BackendDeployer(build_config(args))
    return deployer.deploy()


if __name__ == "__main__":
    raise SystemExit(main())
