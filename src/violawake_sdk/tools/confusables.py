"""Phonetic confusable generation utilities for wake word evaluation."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z]+")
_VOWEL_RE = re.compile(r"[aeiouy]+")
_DUPLICATE_RE = re.compile(r"(.)\1+")

_PHONETIC_NORMALIZATIONS: tuple[tuple[str, str], ...] = (
    ("tion", "shun"),
    ("sion", "zhun"),
    ("ough", "off"),
    ("eaux", "oh"),
    ("ph", "f"),
    ("gh", "g"),
    ("ck", "k"),
    ("qu", "kw"),
    ("x", "ks"),
)

_PHONETIC_GROUPS: tuple[tuple[str, ...], ...] = (
    ("b", "p"),
    ("d", "t"),
    ("f", "v", "ph"),
    ("g", "j", "ch"),
    ("k", "c", "q", "qu", "ck"),
    ("m", "n"),
    ("s", "z", "sh"),
    ("l", "r"),
    ("w", "v"),
    ("th", "d", "f"),
    ("x", "ks", "z"),
    ("a", "ah", "o", "uh"),
    ("e", "eh", "i", "ee"),
    ("i", "ee", "y"),
    ("o", "oh", "u", "aw"),
    ("u", "oo", "oh"),
    ("er", "ur", "ar"),
    ("an", "en", "in"),
)

COMMON_ONSETS: tuple[str, ...] = (
    "b",
    "bl",
    "br",
    "c",
    "ch",
    "cl",
    "cr",
    "d",
    "dr",
    "f",
    "g",
    "gr",
    "h",
    "j",
    "k",
    "kl",
    "m",
    "n",
    "p",
    "pl",
    "r",
    "s",
    "sh",
    "st",
    "t",
    "tr",
    "v",
    "w",
    "z",
)

COMMON_SUFFIXES: tuple[str, ...] = (
    "a",
    "ah",
    "al",
    "an",
    "and",
    "ant",
    "ent",
    "er",
    "ia",
    "ian",
    "in",
    "ing",
    "is",
    "ish",
    "la",
    "lin",
    "lyn",
    "on",
    "or",
    "ot",
    "ous",
    "us",
)

COMMON_SYLLABLES: tuple[str, ...] = (
    "a",
    "ah",
    "ay",
    "ee",
    "ia",
    "io",
    "la",
    "li",
    "lo",
    "na",
    "oh",
    "ola",
    "ra",
    "va",
    "ya",
)

HARDCODED_CONFUSABLES: dict[str, tuple[str, ...]] = {
    "viola": (
        "voila",
        "crayola",
        "ebola",
        "biola",
        "viala",
        "violet",
        "violent",
        "violin",
        "vee ola",
        "via la",
        "buy a lot",
        "the viola",
        "viola instrument",
        "viola the instrument",
    ),
}

_CONSONANT_CLASS_MAP = str.maketrans(
    {
        "b": "v",
        "c": "k",
        "d": "t",
        "f": "v",
        "g": "k",
        "h": "",
        "j": "k",
        "k": "k",
        "p": "v",
        "q": "k",
        "r": "l",
        "s": "s",
        "t": "t",
        "v": "v",
        "w": "v",
        "x": "k",
        "y": "i",
        "z": "s",
    }
)


def _build_similarity_map() -> dict[str, tuple[str, ...]]:
    mapping: defaultdict[str, set[str]] = defaultdict(set)
    for group in _PHONETIC_GROUPS:
        for source in group:
            for target in group:
                if source != target:
                    mapping[source].add(target)

    return {key: tuple(sorted(values)) for key, values in sorted(mapping.items())}


PHONETIC_SIMILARITY_MAP: dict[str, tuple[str, ...]] = _build_similarity_map()


def _normalize_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _normalize_phrase(text: str) -> str:
    return " ".join(_normalize_tokens(text))


def _collapse_repeated(text: str) -> str:
    return _DUPLICATE_RE.sub(r"\1", text)


def _count_syllables(token: str) -> int:
    matches = _VOWEL_RE.findall(token)
    return max(1, len(matches))


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _suffix_overlap(left: str, right: str) -> float:
    overlap = 0
    max_length = min(len(left), len(right))
    while overlap < max_length and left[-(overlap + 1)] == right[-(overlap + 1)]:
        overlap += 1

    if max_length == 0:
        return 0.0
    return overlap / max_length


def _normalized_similarity(left: str, right: str) -> float:
    max_length = max(len(left), len(right))
    if max_length == 0:
        return 0.0
    return 1.0 - (_levenshtein_distance(left, right) / max_length)


def _replace_substring_once(token: str, source: str, target: str) -> set[str]:
    variants: set[str] = set()
    start = token.find(source)
    while start != -1:
        end = start + len(source)
        variants.add(token[:start] + target + token[end:])
        start = token.find(source, start + 1)
    return variants


def _generate_rhyme_variants(token: str) -> set[str]:
    match = _VOWEL_RE.search(token)
    if match is None:
        return set()

    onset = token[: match.start()]
    rime = token[match.start() :]
    variants = set()
    for onset_candidate in COMMON_ONSETS:
        if onset_candidate != onset:
            variants.add(onset_candidate + rime)
    return variants


def _generate_vowel_variants(token: str) -> set[str]:
    variants: set[str] = set()
    for match in _VOWEL_RE.finditer(token):
        chunk = match.group(0)
        for replacement in PHONETIC_SIMILARITY_MAP.get(chunk, ()):
            variants.add(token[: match.start()] + replacement + token[match.end() :])

        if len(chunk) == 1:
            for replacement in PHONETIC_SIMILARITY_MAP.get(chunk[0], ()):
                variants.add(token[: match.start()] + replacement + token[match.end() :])

    return variants


def _generate_consonant_variants(token: str) -> set[str]:
    variants: set[str] = set()
    for source in sorted(PHONETIC_SIMILARITY_MAP, key=len, reverse=True):
        if source not in token:
            continue
        for target in PHONETIC_SIMILARITY_MAP[source]:
            variants.update(_replace_substring_once(token, source, target))
    return variants


def _generate_prefix_suffix_variants(token: str) -> set[str]:
    variants: set[str] = set()
    stem_lengths = {
        max(2, len(token) - 1),
        max(2, len(token) - 2),
    }
    if len(token) >= 5:
        stem_lengths.add(max(3, len(token) - 3))

    for stem_length in stem_lengths:
        stem = token[:stem_length]
        for suffix in COMMON_SUFFIXES:
            variants.add(stem + suffix)

    match = _VOWEL_RE.search(token)
    if match is not None:
        rime = token[match.start() :]
        for onset in COMMON_ONSETS:
            variants.add(onset + rime)

    return variants


def _generate_syllable_variants(token: str) -> set[str]:
    variants: set[str] = set()
    vowel_matches = list(_VOWEL_RE.finditer(token))
    insertion_points = {1, len(token) // 2, max(1, len(token) - 1)}
    insertion_points.update(match.end() for match in vowel_matches)

    for point in insertion_points:
        for syllable in COMMON_SYLLABLES:
            variants.add(token[:point] + syllable + token[point:])

    if len(vowel_matches) > 1:
        for match in vowel_matches[1:]:
            variants.add(token[: match.start()] + token[match.end() :])

    for match in vowel_matches:
        if match.end() < len(token):
            variants.add(token[: match.start()] + token[match.end() + 1 :])

    return variants


def _is_viable_candidate(original: str, candidate: str) -> bool:
    if candidate == original:
        return False
    if not candidate.isalpha():
        return False
    if len(candidate) < 2:
        return False
    if abs(len(candidate) - len(original)) > 5:
        return False
    return _VOWEL_RE.search(candidate) is not None


def _generate_token_confusables(token: str) -> set[str]:
    raw_candidates = (
        _generate_rhyme_variants(token)
        | _generate_vowel_variants(token)
        | _generate_consonant_variants(token)
        | _generate_prefix_suffix_variants(token)
        | _generate_syllable_variants(token)
    )
    return {
        candidate
        for candidate in raw_candidates
        if _is_viable_candidate(token, candidate)
    }


def _is_viable_phrase(original: str, candidate: str) -> bool:
    normalized_candidate = _normalize_phrase(candidate)
    if not normalized_candidate:
        return False
    if normalized_candidate == original:
        return False
    return len(normalized_candidate) <= max(len(original) + 16, len(original) * 2 + 4)


def simple_phonetic_key(text: str) -> str:
    """Return a lightweight phonetic key for a word or phrase."""
    normalized = "".join(_normalize_tokens(text))
    if not normalized:
        return ""

    for source, target in _PHONETIC_NORMALIZATIONS:
        normalized = normalized.replace(source, target)

    normalized = re.sub(r"c(?=[eiy])", "s", normalized)
    normalized = normalized.translate(_CONSONANT_CLASS_MAP)
    normalized = re.sub(r"[aeiou]+", "a", normalized)
    normalized = _collapse_repeated(normalized)
    return normalized.strip()


def phonetic_similarity(left: str, right: str) -> float:
    """Estimate phonetic similarity between two words or phrases."""
    left_normalized = _normalize_phrase(left)
    right_normalized = _normalize_phrase(right)
    if not left_normalized or not right_normalized:
        return 0.0

    left_key = simple_phonetic_key(left_normalized)
    right_key = simple_phonetic_key(right_normalized)

    key_similarity = _normalized_similarity(left_key, right_key)
    token_similarity = _normalized_similarity(
        left_normalized.replace(" ", ""),
        right_normalized.replace(" ", ""),
    )
    suffix_similarity = _suffix_overlap(
        simple_phonetic_key(left_normalized[-4:]),
        simple_phonetic_key(right_normalized[-4:]),
    )

    left_syllables = sum(_count_syllables(token) for token in left_normalized.split())
    right_syllables = sum(_count_syllables(token) for token in right_normalized.split())
    syllable_similarity = 1.0 / (1.0 + abs(left_syllables - right_syllables))

    score = (
        0.5 * key_similarity
        + 0.25 * token_similarity
        + 0.15 * suffix_similarity
        + 0.10 * syllable_similarity
    )
    return max(0.0, min(1.0, score))


def generate_confusables(wake_word: str, count: int = 50) -> list[str]:
    """Generate ranked phonetic confusables for a wake word."""
    normalized_phrase = _normalize_phrase(wake_word)
    if not normalized_phrase or count <= 0:
        return []

    tokens = normalized_phrase.split()
    seed_candidates = set(HARDCODED_CONFUSABLES.get(normalized_phrase, ()))
    candidates: set[str] = set(seed_candidates)

    if len(tokens) == 1:
        candidates.update(_generate_token_confusables(tokens[0]))
    else:
        for index, token in enumerate(tokens):
            replacements = _generate_token_confusables(token)
            for replacement in replacements:
                phrase_tokens = list(tokens)
                phrase_tokens[index] = replacement
                candidates.add(" ".join(phrase_tokens))

    viable_candidates = {
        _normalize_phrase(candidate)
        for candidate in candidates
        if _is_viable_phrase(normalized_phrase, candidate)
    }

    ranked = sorted(
        viable_candidates,
        key=lambda candidate: (
            1 if candidate in seed_candidates else 0,
            phonetic_similarity(normalized_phrase, candidate),
            -abs(len(candidate.replace(" ", "")) - len(normalized_phrase.replace(" ", ""))),
            candidate,
        ),
        reverse=True,
    )

    logger.debug(
        "Generated %d confusables for wake word %s",
        len(ranked),
        normalized_phrase,
    )
    return ranked[:count]


def is_kokoro_tts_available() -> bool:
    """Return True when the Kokoro TTS package is importable."""
    try:
        import kokoro_onnx  # noqa: F401
    except ImportError:
        return False
    return True


def generate_confusable_tts_audio(
    confusables: Sequence[str],
    voice: str = "af_heart",
    sample_rate: int = 16_000,
) -> dict[str, np.ndarray]:
    """Generate Kokoro TTS audio for confusable phrases."""
    if not is_kokoro_tts_available():
        raise ImportError(
            "kokoro-onnx is not installed. Install with: pip install 'violawake[tts]'"
        )

    from violawake_sdk.tts import TTSEngine

    engine = TTSEngine(voice=voice, sample_rate=sample_rate)
    generated_audio: dict[str, np.ndarray] = {}
    for confusable in confusables:
        audio = np.asarray(engine.synthesize(confusable), dtype=np.float32)
        if audio.size == 0:
            logger.warning("Skipping empty Kokoro output for %s", confusable)
            continue
        generated_audio[confusable] = audio

    logger.info("Generated Kokoro audio for %d confusables", len(generated_audio))
    return generated_audio


__all__ = [
    "HARDCODED_CONFUSABLES",
    "PHONETIC_SIMILARITY_MAP",
    "generate_confusable_tts_audio",
    "generate_confusables",
    "is_kokoro_tts_available",
    "phonetic_similarity",
    "simple_phonetic_key",
]
