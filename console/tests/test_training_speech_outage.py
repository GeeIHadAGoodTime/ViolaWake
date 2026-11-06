"""Losing speech synthesis must fail fast, cost the customer nothing, and read plainly.

From 2026-10-14 to 2026-11-06 edge-tts 7.2.8 got a 403 on every connection, the Kokoro
fallback was not installed, and nothing noticed. 71 of 71 jobs from 36 customers each
ran about 55 minutes in a one-at-a-time queue, then failed with a sentence about
"confusable probe audio". Each one spent a training attempt and a strike, and user
355 ended up locked. ``/api/health`` said ok the whole time.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.routes import jobs as jobs_route
from app.routes.training import training_readiness
from app.schemas import JobSubmitRequest
from app.services import speech_readiness
from app.services import training_service
from app.services.speech_readiness import (
    TRAINING_UNAVAILABLE_MESSAGE,
    SpeechReadiness,
    get_speech_readiness,
    reset_speech_readiness_cache,
)
from app.services.training_service import (
    SharedInfrastructureUnavailableError,
    SpeechSynthesisUnavailableError,
    run_training_job_sync,
)


class _Probe:
    def __init__(self, edge: bool, kokoro: bool) -> None:
        self.edge_tts, self.kokoro = edge, kokoro
        self.edge_tts_error = None if edge else "WSServerHandshakeError: 403"
        self.calls = 0

    @property
    def available(self) -> bool:
        return self.edge_tts or self.kokoro


def _install_probe(monkeypatch: pytest.MonkeyPatch, probe: _Probe) -> None:
    train = types.ModuleType("violawake_sdk.tools.train")

    def probe_speech_synthesis(tmp_dir=None):
        probe.calls += 1
        return probe

    train.probe_speech_synthesis = probe_speech_synthesis
    train.reset_edge_tts_reporting = lambda: None
    monkeypatch.setitem(sys.modules, "violawake_sdk.tools.train", train)


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_speech_readiness_cache()
    yield
    reset_speech_readiness_cache()


def test_no_speech_backend_stops_the_job_before_it_spends_anything(monkeypatch, tmp_path: Path) -> None:
    probe = _Probe(edge=False, kokoro=False)
    _install_probe(monkeypatch, probe)
    downloads: list[str] = []

    class _Storage:
        def exists(self, identifier: str) -> bool:
            downloads.append(identifier)
            return True

    monkeypatch.setattr(training_service, "get_storage", lambda: _Storage())

    with pytest.raises(SpeechSynthesisUnavailableError) as caught:
        run_training_job_sync(
            job_id=1,
            wake_word="zephyra",
            recording_identifiers=["a", "b", "c", "d", "e"],
            output_path=tmp_path / "m.onnx",
            epochs=1,
            timeout_seconds=60,
            progress_callback=lambda event: None,
            is_cancelled=lambda: False,
        )

    assert probe.calls == 1
    assert downloads == [], "not one recording may be touched once speech is known to be down"
    assert isinstance(caught.value, SharedInfrastructureUnavailableError)
    assert "Your recordings are fine" in str(caught.value)
    assert "synthesize" not in str(caught.value), "the customer must not get operator wording"


def test_a_late_gate_outage_also_reads_plainly_and_counts_as_ours(monkeypatch, tmp_path: Path) -> None:
    """The preflight can pass and the service die mid-job. The customer's text and the
    no-strike treatment must be the same either way."""
    _install_probe(monkeypatch, _Probe(edge=True, kokoro=False))

    class QualityGateUnavailableError(RuntimeError):
        pass

    class _Storage:
        def exists(self, identifier: str) -> bool:
            raise QualityGateUnavailableError("Quality gate could not synthesize any speech probe audio")

    monkeypatch.setattr(training_service, "get_storage", lambda: _Storage())

    with pytest.raises(SpeechSynthesisUnavailableError) as caught:
        run_training_job_sync(
            job_id=2,
            wake_word="zephyra",
            recording_identifiers=["a"],
            output_path=tmp_path / "m.onnx",
            epochs=1,
            timeout_seconds=60,
            progress_callback=lambda event: None,
            is_cancelled=lambda: False,
        )
    assert str(caught.value) == TRAINING_UNAVAILABLE_MESSAGE


async def test_readiness_reports_the_outage_without_provider_internals(monkeypatch) -> None:
    _install_probe(monkeypatch, _Probe(edge=False, kokoro=False))

    response = await training_readiness()

    assert response.ready is False
    assert response.status == "down", "the uptime worker pages on this"
    assert response.message == TRAINING_UNAVAILABLE_MESSAGE
    assert "403" not in response.model_dump_json()


async def test_readiness_is_ready_on_the_fallback_alone(monkeypatch) -> None:
    _install_probe(monkeypatch, _Probe(edge=False, kokoro=True))

    response = await training_readiness()

    assert response.ready is True
    assert response.backend == "kokoro"


async def test_readiness_is_cached_so_public_polling_cannot_hammer_the_tts_service(monkeypatch) -> None:
    probe = _Probe(edge=True, kokoro=False)
    _install_probe(monkeypatch, probe)

    await asyncio.gather(*(get_speech_readiness() for _ in range(20)))

    assert probe.calls == 1


async def test_a_failed_answer_expires_sooner_so_recovery_is_noticed(monkeypatch) -> None:
    probe = _Probe(edge=False, kokoro=False)
    _install_probe(monkeypatch, probe)
    await get_speech_readiness()

    clock = speech_readiness.time.monotonic()
    monkeypatch.setattr(
        speech_readiness.time,
        "monotonic",
        lambda: clock + speech_readiness.NOT_READY_CACHE_SECONDS + 1,
    )
    probe.edge_tts = True

    assert (await get_speech_readiness()).ready is True
    assert probe.calls == 2


async def test_a_hung_check_counts_as_down_instead_of_hanging_the_caller(monkeypatch) -> None:
    def _hang() -> SpeechReadiness:
        raise TimeoutError("stalled")

    monkeypatch.setattr(speech_readiness, "_probe_blocking", _hang)

    assert (await get_speech_readiness()).ready is False


async def test_submit_is_refused_and_not_charged_while_speech_is_down(monkeypatch) -> None:
    breaker = types.SimpleNamespace(paused=False)
    queue = types.SimpleNamespace(
        get_circuit_breaker=AsyncMock(return_value=breaker),
        submit_job=AsyncMock(return_value=1),
        mark_usage_charged=AsyncMock(),
    )
    monkeypatch.setattr(jobs_route, "init_job_queue", AsyncMock(return_value=queue))
    monkeypatch.setattr(
        jobs_route,
        "validate_training_request",
        AsyncMock(return_value=("zephyra", [1, 2, 3, 4, 5], 80)),
    )
    record_usage = AsyncMock()
    monkeypatch.setattr(jobs_route, "record_usage", record_usage)
    monkeypatch.setattr(
        jobs_route,
        "get_speech_readiness",
        AsyncMock(return_value=SpeechReadiness(False, None, 0.0, "403")),
    )

    with pytest.raises(HTTPException) as caught:
        await jobs_route.submit_training_job(
            JobSubmitRequest(wake_word="zephyra", recording_ids=[1, 2, 3, 4, 5]),
            types.SimpleNamespace(id=7),
            None,
            types.SimpleNamespace(),
        )

    assert caught.value.status_code == 503
    assert caught.value.detail == TRAINING_UNAVAILABLE_MESSAGE
    queue.submit_job.assert_not_awaited()
    record_usage.assert_not_awaited()
