"""Refuse a wake word that cannot work, before anyone records anything.

A wake word has to be long enough to carry a pattern and rare enough not to fire
on normal talk. Nothing checked either: "no" would have cost the user a recording
session, a training attempt and about ten minutes, and then false-fired all day
or failed the quality gate with a message that blamed run-to-run variance.

The reason returned here is shown to the user as written, so it says what is wrong
and what to do. ``None`` means the word is fine.

Rules are about the word, not the audio:

* Letters only (spaces, hyphens, apostrophes between letters).
* At least 2 syllables and at most 6. The model listens to a 1.5 second window,
  and about 6 syllables is what a person says in that time.
* Not made only of everyday words. "hey jarvis" is fine, "good morning" is not.
  One rare word is enough to make a phrase distinctive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_LETTERS = 4
MIN_SYLLABLES = 2
MAX_SYLLABLES = 6

_ALLOWED = re.compile(r"\A[^\W\d_]+(?:[ '\-_][^\W\d_]+)*\Z")
_SEPARATORS = re.compile(r"[ \-_]+")
_VOWEL_GROUP = re.compile(r"[aeiou]+|y(?![aeiou])")

# Words people say all day. Only a phrase made ENTIRELY of these is refused. The
# list is short on purpose: it should catch the obvious, not argue with names.
_EVERYDAY_WORDS = frozenset(
    """
    a about after again all also always am an and another any are around as ask at
    away back bad be because been before being best better big both but buddy by
    call came can come could day did do does done down each even ever every
    everyone everything excuse fine first for friend from get give go goes going
    gone good got great had happy has have he hello help her here hey hi him his
    home how i if in into is it its just keep kind know last later least left let
    like listen little long look lot love made make man many maybe me mean might
    more morning most much music must my name need never new next nice night no
    nope not nothing now of off oh ok okay old on once one only open or other our
    out over people play please put quite really right said same say see she
    should show so some someone something sorry start still stop sure take talk
    tell than thank thanks that the their them then there these they thing things
    think this those through time to today together tomorrow tonight too turn two
    under until up us very wait wake want was watch water way we well went were
    what when where which while who why will with without wonderful work would
    wow yeah year yes yesterday yet you your
    """.split()
)


@dataclass(frozen=True, slots=True)
class WakeWordVerdict:
    ok: bool
    reason: str | None = None
    code: str | None = None


def normalize_wake_word(wake_word: str) -> str:
    return " ".join(wake_word.strip().lower().split())


def _words(normalized: str) -> list[str]:
    return [word for word in _SEPARATORS.split(normalized) if word]


def count_syllables(word: str) -> int:
    """Estimate syllables in one written word (vowel groups, minus a silent e)."""
    word = word.replace("'", "")
    groups = _VOWEL_GROUP.findall(word)
    count = len(groups)
    if not count:
        # Consonant-only spellings, and scripts this estimator has no rule for.
        return max(1, len(word) // 2)
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1
    return count


def check_wake_word(wake_word: str) -> WakeWordVerdict:
    """Return whether ``wake_word`` can make a working wake word, and why not."""
    normalized = normalize_wake_word(wake_word)
    if not normalized:
        return WakeWordVerdict(False, "Type the word or phrase you want to use.", "empty")

    if not _ALLOWED.match(normalized):
        return WakeWordVerdict(
            False,
            "Use letters only, with no numbers or symbols. Spell a number out as a word.",
            "invalid_characters",
        )

    words = _words(normalized)
    letters = sum(len(word.replace("'", "")) for word in words)
    syllables = sum(count_syllables(word) for word in words)

    if letters < MIN_LETTERS or syllables < MIN_SYLLABLES:
        return WakeWordVerdict(
            False,
            f'"{normalized}" is too short to recognize reliably. Use 2 to 4 syllables, '
            f'for example a name like "hey nova".',
            "too_short",
        )

    if syllables > MAX_SYLLABLES:
        return WakeWordVerdict(
            False,
            f'"{normalized}" is too long. You need to be able to say it in about a '
            "second and a half, so keep it to 2 to 4 syllables.",
            "too_long",
        )

    if all(word.replace("'", "") in _EVERYDAY_WORDS for word in words):
        return WakeWordVerdict(
            False,
            f'"{normalized}" is too common. It would wake up during normal conversation. '
            "Pick something you do not say every day, like a name.",
            "too_common",
        )

    return WakeWordVerdict(True)


__all__ = [
    "MAX_SYLLABLES",
    "MIN_LETTERS",
    "MIN_SYLLABLES",
    "WakeWordVerdict",
    "check_wake_word",
    "count_syllables",
    "normalize_wake_word",
]
