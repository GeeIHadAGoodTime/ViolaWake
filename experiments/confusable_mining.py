"""Phoneme-based confusable word mining for wake word "viola".

Checklist item 1.6 from ACCURACY_MISSION.md.
Answers Q3: "Does phoneme-distance-based mining find confusables we missed manually?"

Uses:
- g2p-en: grapheme-to-phoneme for arbitrary words
- pronouncing (CMU dict): 135K word pronunciations
- Phoneme edit distance to find words that sound like "viola"

Then generates TTS samples for NEW confusables using Edge TTS.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFUSABLE_DIR = BASE_DIR / "training_data" / "confusable_negatives"
CONFUSABLE_V2_DIR = BASE_DIR / "training_data" / "confusable_negatives_v2"
SAMPLE_RATE = 16000

# Current confusable words (from fast_generate.py)
EXISTING_WORDS = {
    "vanilla", "villa", "violet", "vinyl", "villain", "village",
    "viper", "vista", "vivid", "valor", "valley", "valid",
    "venture", "vintage", "visual", "vital", "volume", "via",
}
EXISTING_PHRASES = {
    "she plays the viola in the orchestra",
    "the viola section sounds beautiful",
    "hand me that viola please",
    "buy a villa", "fly over", "I'll be over",
}

# Same 15 NEG_VOICES from fast_generate.py
NEG_VOICES = [
    "en-US-DavisNeural", "en-US-JaneNeural", "en-US-JasonNeural",
    "en-US-SaraNeural", "en-US-TonyNeural",
    "en-US-NancyNeural", "en-US-AmberNeural", "en-US-AshleyNeural",
    "en-US-BrandonNeural", "en-US-RogerNeural",
    "en-GB-AbbiNeural", "en-GB-AlfieNeural", "en-GB-BellaNeural",
    "en-GB-ElliotNeural", "en-GB-NoahNeural",
]

# Phoneme distance thresholds
MAX_EDIT_DISTANCE = 3  # up to 3 phoneme edits
MIN_EDIT_DISTANCE = 1  # at least 1 edit (exclude exact matches)

# Manual confusable phrases to also mine (cross-word boundary confusables)
# These are phrases where "viola" could be misheard in running speech
MANUAL_PHRASE_CANDIDATES = [
    # V-initial phrases that could blur into "viola"
    "buy a villa",
    "fly over",
    "I'll be over",
    "pile of",
    "dial up",
    "file a",
    "I owe a lot",
    "by a lot",
    "higher law",
    "hire a lot",
    "via la",
    "viva la",
    "try a lot",
    "viola da gamba",  # the instrument — should NOT trigger wake word
    "my viola teacher",
    "play viola for me",
    # Near-miss vowel swaps
    "viol",
    "voila",
    "vial",
    "viral",
    "final",
    "rival",
    "bridal",
    "title",
    "idle",
    "bible",
    "liable",
    "viable",
    "dial",
    "trial",
    "denial",
    "spiral",
    "arrival",
    # Rhythm-similar (same stress/syllable pattern as vee-OH-la)
    "payola",
    "crayola",
    "granola",
    "angola",
    "ebola",
    "cupola",
    "gondola",
    "pianola",
    "victrola",
    "ayatollah",
    "coca-cola",
]


# ---------------------------------------------------------------------------
# Phoneme edit distance
# ---------------------------------------------------------------------------

def strip_stress(phoneme: str) -> str:
    """Remove stress markers (0, 1, 2) from a phoneme. AH0 -> AH."""
    return phoneme.rstrip("012")


def phoneme_edit_distance(p1: list[str], p2: list[str]) -> int:
    """Levenshtein edit distance between two phoneme sequences.

    Stress markers are stripped before comparison so AH0 == AH1.
    """
    s1 = [strip_stress(p) for p in p1]
    s2 = [strip_stress(p) for p in p2]
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def weighted_phoneme_distance(p1: list[str], p2: list[str]) -> float:
    """Weighted phoneme distance that penalizes consonant changes more.

    Vowel substitutions cost 0.5 (common in accents), consonant subs cost 1.0.
    This better models perceptual similarity.
    """
    VOWELS = {
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
        "IH", "IY", "OW", "OY", "UH", "UW",
    }
    s1 = [strip_stress(p) for p in p1]
    s2 = [strip_stress(p) for p in p2]
    m, n = len(s1), len(s2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0.0
            elif s1[i - 1] in VOWELS and s2[j - 1] in VOWELS:
                cost = 0.5  # vowel-to-vowel swap is cheaper
            else:
                cost = 1.0
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


# ---------------------------------------------------------------------------
# CMU dictionary mining
# ---------------------------------------------------------------------------

def mine_cmu_confusables(
    target_phonemes_list: list[list[str]],
    max_distance: int = MAX_EDIT_DISTANCE,
) -> dict[str, dict]:
    """Search entire CMU dictionary for words within phoneme edit distance.

    Args:
        target_phonemes_list: List of alternative pronunciations for target.
        max_distance: Maximum edit distance to consider.

    Returns:
        Dict of word -> {phonemes, distance, weighted_distance, pronunciation_variant}.
    """
    import pronouncing

    results = {}
    entries = pronouncing.cmudict.entries()

    for word, phonemes_list in entries:
        # Skip very short words (1 phoneme) and very long words (> 10 phonemes)
        if len(phonemes_list) < 2 or len(phonemes_list) > 10:
            continue
        # Skip words that are just numbers or have punctuation
        if not word.isalpha():
            continue

        best_dist = float("inf")
        best_weighted = float("inf")
        best_variant = 0

        for vi, target_ph in enumerate(target_phonemes_list):
            d = phoneme_edit_distance(target_ph, phonemes_list)
            if d <= max_distance:
                w = weighted_phoneme_distance(target_ph, phonemes_list)
                if w < best_weighted:
                    best_dist = d
                    best_weighted = w
                    best_variant = vi

        if best_dist <= max_distance and best_dist >= MIN_EDIT_DISTANCE:
            # Use lowercase canonical form
            canonical = word.lower()
            if canonical not in results or results[canonical]["weighted_distance"] > best_weighted:
                results[canonical] = {
                    "phonemes": " ".join(phonemes_list),
                    "distance": best_dist,
                    "weighted_distance": best_weighted,
                    "pronunciation_variant": best_variant,
                }

    return results


def mine_g2p_confusables(
    target_phonemes_list: list[list[str]],
    candidate_words: list[str],
) -> dict[str, dict]:
    """Use g2p-en to check words not in CMU dict.

    Useful for proper nouns, slang, brand names.
    """
    from g2p_en import G2p
    g2p = G2p()

    results = {}
    for word in candidate_words:
        phonemes = g2p(word)
        # g2p returns a mix of phonemes and spaces; filter to just phonemes
        phonemes = [p for p in phonemes if p.strip() and p != " "]

        best_dist = float("inf")
        best_weighted = float("inf")

        for target_ph in target_phonemes_list:
            d = phoneme_edit_distance(target_ph, phonemes)
            w = weighted_phoneme_distance(target_ph, phonemes)
            if w < best_weighted:
                best_dist = d
                best_weighted = w

        if best_dist <= MAX_EDIT_DISTANCE:
            results[word.lower()] = {
                "phonemes": " ".join(phonemes),
                "distance": best_dist,
                "weighted_distance": best_weighted,
            }

    return results


# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------

def categorize_confusables(
    all_confusables: dict[str, dict],
) -> dict[str, list[tuple[str, dict]]]:
    """Categorize confusables by risk level based on weighted distance."""
    categories = defaultdict(list)
    for word, info in sorted(all_confusables.items(), key=lambda x: x[1]["weighted_distance"]):
        wd = info["weighted_distance"]
        if wd <= 1.5:
            categories["HIGH_RISK"].append((word, info))
        elif wd <= 2.5:
            categories["MEDIUM_RISK"].append((word, info))
        else:
            categories["LOW_RISK"].append((word, info))
    return dict(categories)


def find_new_confusables(
    mined: dict[str, dict],
    existing: set[str],
) -> dict[str, dict]:
    """Return only confusables not in the existing set."""
    existing_lower = {w.lower() for w in existing}
    return {w: info for w, info in mined.items() if w.lower() not in existing_lower}


# ---------------------------------------------------------------------------
# TTS generation
# ---------------------------------------------------------------------------

async def generate_tts(text: str, voice: str, output_path: Path, retries: int = 2) -> bool:
    """Generate a single TTS clip, save as 16kHz mono WAV.

    Handles Edge TTS rate limiting with retries and delays.
    For very short text, wraps in a carrier sentence to avoid empty audio.
    """
    if output_path.exists():
        return True

    # Edge TTS needs enough text to generate audio. For single short words,
    # use "Say X" as a carrier to ensure audio generation.
    tts_text = text
    if len(text.split()) == 1 and len(text) < 8:
        tts_text = f"Say {text} please"
    else:
        tts_text = text + "."

    for attempt in range(retries + 1):
        try:
            import edge_tts
            import tempfile
            from pydub import AudioSegment

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            communicate = edge_tts.Communicate(tts_text, voice)
            await communicate.save(tmp_path)

            audio = AudioSegment.from_mp3(tmp_path)
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            audio.export(str(output_path), format="wav")

            import os
            os.unlink(tmp_path)
            return True
        except Exception as e:
            if attempt < retries:
                # Back off before retry
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            print(f"  FAIL: {voice} '{text[:30]}': {e}", file=sys.stderr)
            return False

    return False


async def generate_new_confusable_clips(
    new_words: list[str],
    new_phrases: list[str],
    voices: list[str],
    output_dir: Path,
    existing_dir: Path | None = None,
) -> int:
    """Generate TTS clips for new confusables. Skip if already in existing_dir.

    Adds a small delay between requests to avoid Edge TTS rate limiting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    failed = 0
    skipped = 0

    all_texts = new_words + new_phrases

    for vi, voice in enumerate(voices):
        print(f"  Voice {vi + 1}/{len(voices)}: {voice}")
        voice_generated = 0
        for text in all_texts:
            safe_text = text.replace(" ", "_").replace("'", "").replace("-", "_")[:40]

            # Check if already exists in v1 confusable dir
            if existing_dir:
                v1_path = existing_dir / f"{voice}_{safe_text}.wav"
                if v1_path.exists():
                    skipped += 1
                    continue

            out_path = output_dir / f"{voice}_{safe_text}.wav"
            if out_path.exists():
                total += 1
                continue

            ok = await generate_tts(text, voice, out_path)
            if ok:
                total += 1
                voice_generated += 1
            else:
                failed += 1

            # Small delay to avoid rate limiting (0.3s between requests)
            await asyncio.sleep(0.3)

        print(f"    -> {voice_generated} new, {failed} failed so far")

    print(f"\n  Generated: {total}, Failed: {failed}, Skipped (v1 exists): {skipped}")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mining() -> dict:
    """Run full confusable mining pipeline. Returns results dict."""
    import pronouncing
    from g2p_en import G2p

    g2p = G2p()

    print("=" * 70)
    print("PHONEME-BASED CONFUSABLE MINING FOR 'VIOLA'")
    print("=" * 70)

    # Step 1: Get phoneme representation of "viola"
    print("\n--- Step 1: Phoneme Representation ---")
    viola_g2p = g2p("viola")
    viola_g2p_clean = [p for p in viola_g2p if p.strip() and p != " "]
    print(f"  g2p-en:  {'  '.join(viola_g2p_clean)}")

    viola_cmu = pronouncing.phones_for_word("viola")
    viola_cmu_parsed = []
    for pron in viola_cmu:
        parsed = pron.split()
        viola_cmu_parsed.append(parsed)
        print(f"  CMU:     {pron}")

    # Use CMU pronunciations as primary (they include alternate pronunciations)
    target_phonemes = viola_cmu_parsed if viola_cmu_parsed else [viola_g2p_clean]
    print(f"\n  Using {len(target_phonemes)} pronunciation variant(s) for matching")

    # Step 2: Mine CMU dictionary
    print("\n--- Step 2: CMU Dictionary Mining ---")
    print(f"  Searching {len(list(pronouncing.cmudict.entries()))} entries...")
    print(f"  Max edit distance: {MAX_EDIT_DISTANCE}")
    t0 = time.time()
    cmu_results = mine_cmu_confusables(target_phonemes, MAX_EDIT_DISTANCE)
    elapsed = time.time() - t0
    print(f"  Found {len(cmu_results)} words within distance {MAX_EDIT_DISTANCE} ({elapsed:.1f}s)")

    # Step 3: Check manual phrase candidates with g2p
    print("\n--- Step 3: Manual Candidate Checking ---")
    # Filter to single words from manual list for g2p check
    manual_single_words = [w for w in MANUAL_PHRASE_CANDIDATES if " " not in w and "-" not in w]
    g2p_results = mine_g2p_confusables(target_phonemes, manual_single_words)
    print(f"  Checked {len(manual_single_words)} manual single-word candidates")
    print(f"  {len(g2p_results)} are within distance threshold")

    # Merge results (CMU + g2p, prefer CMU if both have it)
    all_confusables = {**g2p_results, **cmu_results}

    # Step 4: Categorize by risk
    print("\n--- Step 4: Risk Categorization ---")
    categories = categorize_confusables(all_confusables)
    for cat, items in categories.items():
        print(f"\n  {cat} ({len(items)} words):")
        for word, info in items[:20]:  # show top 20 per category
            marker = " [EXISTING]" if word in {w.lower() for w in EXISTING_WORDS} else " **NEW**"
            print(f"    {word:20s}  d={info['distance']}  wd={info['weighted_distance']:.1f}  "
                  f"ph: {info['phonemes']}{marker}")
        if len(items) > 20:
            print(f"    ... and {len(items) - 20} more")

    # Step 5: Find NEW confusables
    print("\n--- Step 5: New Confusables ---")
    new_confusables = find_new_confusables(all_confusables, EXISTING_WORDS)
    new_high = [(w, i) for w, i in new_confusables.items() if i["weighted_distance"] <= 1.5]
    new_medium = [(w, i) for w, i in new_confusables.items() if 1.5 < i["weighted_distance"] <= 2.5]
    new_low = [(w, i) for w, i in new_confusables.items() if i["weighted_distance"] > 2.5]

    print(f"  Total new: {len(new_confusables)}")
    print(f"  HIGH risk: {len(new_high)}")
    print(f"  MEDIUM risk: {len(new_medium)}")
    print(f"  LOW risk: {len(new_low)}")

    if new_high:
        print("\n  HIGH RISK new confusables (MUST ADD):")
        for word, info in sorted(new_high, key=lambda x: x[1]["weighted_distance"]):
            print(f"    {word:20s}  d={info['distance']}  wd={info['weighted_distance']:.1f}  "
                  f"ph: {info['phonemes']}")

    if new_medium:
        print("\n  MEDIUM RISK new confusables (SHOULD ADD):")
        for word, info in sorted(new_medium, key=lambda x: x[1]["weighted_distance"]):
            print(f"    {word:20s}  d={info['distance']}  wd={info['weighted_distance']:.1f}  "
                  f"ph: {info['phonemes']}")

    # Step 6: Also check phrase candidates
    print("\n--- Step 6: Confusable Phrases ---")
    phrase_candidates = [p for p in MANUAL_PHRASE_CANDIDATES if " " in p or "-" in p]
    existing_phrases_lower = {p.lower() for p in EXISTING_PHRASES}
    new_phrases = [p for p in phrase_candidates if p.lower() not in existing_phrases_lower]
    print(f"  New phrase candidates: {len(new_phrases)}")
    for p in new_phrases:
        print(f"    '{p}'")

    # Step 7: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Viola phonemes (CMU): {viola_cmu}")
    print(f"  Total CMU words scanned: {len(list(pronouncing.cmudict.entries()))}")
    print(f"  Total confusables found: {len(all_confusables)}")
    print(f"  Existing in our list: {len(all_confusables) - len(new_confusables)}")
    print(f"  NEW confusables: {len(new_confusables)}")
    print(f"    HIGH risk: {len(new_high)}")
    print(f"    MEDIUM risk: {len(new_medium)}")
    print(f"    LOW risk: {len(new_low)}")
    print(f"  New phrases: {len(new_phrases)}")

    # Words recommended for TTS generation (high + medium risk)
    recommended_words = sorted(
        [(w, i) for w, i in new_confusables.items() if i["weighted_distance"] <= 2.5],
        key=lambda x: x[1]["weighted_distance"],
    )
    print(f"\n  Recommended for TTS generation: {len(recommended_words)} words + {len(new_phrases)} phrases")
    print(f"  Estimated clips: {(len(recommended_words) + len(new_phrases)) * len(NEG_VOICES)}")

    return {
        "viola_phonemes_cmu": viola_cmu,
        "viola_phonemes_g2p": " ".join(viola_g2p_clean),
        "total_scanned": len(list(pronouncing.cmudict.entries())),
        "total_confusables": len(all_confusables),
        "new_confusables": {w: i for w, i in sorted(new_confusables.items(), key=lambda x: x[1]["weighted_distance"])},
        "new_high_risk": [w for w, _ in new_high],
        "new_medium_risk": [w for w, _ in new_medium],
        "new_low_risk": [w for w, _ in new_low],
        "new_phrases": new_phrases,
        "recommended_words": [w for w, _ in recommended_words],
        "existing_confirmed": [w for w in EXISTING_WORDS if w.lower() in all_confusables],
        "categories": {cat: [(w, i["weighted_distance"]) for w, i in items] for cat, items in categories.items()},
    }


def select_tts_targets(results: dict) -> tuple[list[str], list[str]]:
    """Select the most valuable words for TTS generation.

    Strategy:
    - ALL high-risk words (distance 1, weighted <= 1.5)
    - MEDIUM risk words that are common English words (not obscure names)
    - New confusable phrases
    - Cap at ~80 total items to keep generation time reasonable (~80 * 15 = 1200 clips)
    """
    # All high risk
    high_risk = results.get("new_high_risk", [])

    # For medium risk, filter to common/dangerous words
    # These are words that would actually appear in everyday speech
    MEDIUM_PRIORITY_WORDS = {
        # Common English words that sound like viola
        "viol", "voila", "vial", "vials", "viral", "vinyl",
        "ebola", "cola", "corolla", "gorilla", "guerrilla",
        "koala", "payola", "crayola",
        # -ola ending words (same rhythm)
        "granola", "gondola", "cupola", "pianola",
        "enola", "loyola", "iota", "iowa",
        # V-initial common words
        "voila", "violet", "violets", "violent", "violence",
        "violin", "violins", "violinist",
        "venom", "villa", "valley",
        # Near-rhymes people say often
        "viable", "dial", "trial", "denial",
        "spiral", "arrival", "rival", "bridal",
        "title", "idle", "bible", "liable", "final",
        # Instrument context (most dangerous — same domain)
        "violas", "violists", "viole", "violett", "violette",
        # Names that TTS might say
        "fiona", "verona", "vienna",
    }

    medium_risk_selected = [
        w for w in results.get("new_medium_risk", [])
        if w.lower() in MEDIUM_PRIORITY_WORDS
    ]

    # Also add medium-priority words from high risk that aren't already there
    all_words = list(dict.fromkeys(high_risk + medium_risk_selected))  # deduplicate, preserve order

    # Add any priority words from the full new_confusables that we might have missed
    all_new = results.get("new_confusables", {})
    for w in MEDIUM_PRIORITY_WORDS:
        if w in all_new and w not in all_words and w.lower() not in {x.lower() for x in EXISTING_WORDS}:
            all_words.append(w)

    new_phrases = results.get("new_phrases", [])
    return all_words, new_phrases


async def run_tts_generation(results: dict) -> int:
    """Generate TTS clips for new confusables."""
    recommended_words, new_phrases = select_tts_targets(results)

    if not recommended_words and not new_phrases:
        print("\nNo new confusables to generate TTS for.")
        return 0

    print(f"\n--- TTS Generation ---")
    print(f"  Words: {len(recommended_words)}")
    for w in recommended_words:
        wd = results.get("new_confusables", {}).get(w, {}).get("weighted_distance", "?")
        print(f"    {w} (wd={wd})")
    print(f"  Phrases: {len(new_phrases)}")
    for p in new_phrases:
        print(f"    '{p}'")
    print(f"  Voices: {len(NEG_VOICES)}")
    print(f"  Estimated total clips: {(len(recommended_words) + len(new_phrases)) * len(NEG_VOICES)}")

    total = await generate_new_confusable_clips(
        new_words=recommended_words,
        new_phrases=new_phrases,
        voices=NEG_VOICES,
        output_dir=CONFUSABLE_V2_DIR,
        existing_dir=CONFUSABLE_DIR,
    )
    return total


async def main():
    t0 = time.time()

    # Run mining
    results = run_mining()

    # Save results to JSON
    results_path = BASE_DIR / "confusable_mining_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Generate TTS if there are new confusables
    should_generate = "--generate" in sys.argv or "--gen" in sys.argv
    if should_generate:
        n_clips = await run_tts_generation(results)
        print(f"\n  Total new TTS clips generated: {n_clips}")
    else:
        n_words = len(results["recommended_words"])
        n_phrases = len(results["new_phrases"])
        est_clips = (n_words + n_phrases) * len(NEG_VOICES)
        print(f"\n  To generate TTS clips ({est_clips} estimated), run:")
        print(f"    python experiments/confusable_mining.py --generate")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
