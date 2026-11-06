"""A word that cannot work is refused up front, with a reason a person can act on."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.routes.recordings import _normalize_wake_word
from app.routes.training import check_word
from app.schemas import WakeWordCheckRequest
from app.services.wake_word_policy import check_wake_word, count_syllables


@pytest.mark.parametrize(
    "word",
    ["jarvis", "hey jarvis", "ares", "hello maya", "hey lisa wake up", "kuh_pree", "quinta feira", "maya", "ultron"],
)
def test_words_people_really_trained_are_still_allowed(word: str) -> None:
    assert check_wake_word(word).ok, word


@pytest.mark.parametrize("word,code", [("no", "too_short"), ("hi", "too_short"), ("void", "too_short")])
def test_too_short_is_refused_with_a_reason(word: str, code: str) -> None:
    verdict = check_wake_word(word)
    assert not verdict.ok
    assert verdict.code == code
    assert "too short" in (verdict.reason or "")


@pytest.mark.parametrize("word", ["hello", "okay", "good morning", "thank you", "hey there", "what time"])
def test_everyday_words_are_refused_because_they_would_false_wake(word: str) -> None:
    verdict = check_wake_word(word)
    assert not verdict.ok
    assert verdict.code == "too_common"
    assert "too common" in (verdict.reason or "")


def test_one_rare_word_makes_a_phrase_distinctive() -> None:
    assert check_wake_word("hello zephyra").ok


def test_a_phrase_that_will_not_fit_the_listening_window_is_refused() -> None:
    verdict = check_wake_word("supercalifragilistic expialidocious")
    assert not verdict.ok
    assert verdict.code == "too_long"


@pytest.mark.parametrize("word", ["hey 123", "r2d2", "<script>", "wake!!"])
def test_non_letters_are_refused(word: str) -> None:
    verdict = check_wake_word(word)
    assert not verdict.ok
    assert verdict.code == "invalid_characters"


def test_blank_is_refused() -> None:
    assert check_wake_word("   ").code == "empty"


@pytest.mark.parametrize("word,syllables", [("void", 1), ("ares", 2), ("jarvis", 2), ("zephyra", 3), ("mike", 1)])
def test_syllable_estimate(word: str, syllables: int) -> None:
    assert count_syllables(word) == syllables


def test_upload_refuses_a_bad_word_before_storing_any_audio() -> None:
    with pytest.raises(HTTPException) as caught:
        _normalize_wake_word("no")
    assert caught.value.status_code == 400
    assert "too short" in str(caught.value.detail)


async def test_check_word_route_returns_the_reason() -> None:
    response = await check_word(WakeWordCheckRequest(wake_word="hello"))
    assert response.ok is False
    assert response.code == "too_common"


def test_the_service_account_is_not_held_to_one_free_users_storage(monkeypatch) -> None:
    """It uploads for a whole upstream user base. On the free tier, 500 live recordings
    and 50 MB a day would be one shared budget for everybody."""
    import asyncio
    from types import SimpleNamespace

    from app.routes import recordings

    monkeypatch.setattr(recordings, "is_service_user", lambda user: user.id == 47)

    async def _free(db, user_id):
        return "free"

    monkeypatch.setattr(recordings, "_subscription_tier", _free)

    assert asyncio.run(recordings._storage_tier(None, SimpleNamespace(id=47))) == "enterprise"
    assert asyncio.run(recordings._storage_tier(None, SimpleNamespace(id=48))) == "free"
