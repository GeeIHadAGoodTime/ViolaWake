"""Phoneme-based confusable word mining and TTS generation for ViolaWake.

Mines the CMU pronouncing dictionary and g2p-en for words phonemically similar
to "viola", ranks them by phoneme edit distance and phonetic feature similarity,
then generates diverse TTS audio for wake-word hard-negative training.

Usage:
    python experiments/mine_confusables.py                # Mine only (print ranked list)
    python experiments/mine_confusables.py --generate     # Mine + generate TTS audio
    python experiments/mine_confusables.py --top 75       # Adjust number of words (default 100)
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
CONFUSABLE_V1_DIR = BASE_DIR / "training_data" / "confusable_negatives"
CONFUSABLE_V2_DIR = BASE_DIR / "training_data" / "confusable_negatives_v2"
RESULTS_PATH = BASE_DIR / "mine_confusables_results.json"
SAMPLE_RATE = 16000

# Words already present in v1 confusable negatives
EXISTING_WORDS = {
    "vanilla", "villa", "violet", "vinyl", "villain", "village",
    "viper", "vista", "vivid", "valor", "valley", "valid",
    "venture", "vintage", "visual", "vital", "volume", "via",
}

EXISTING_PHRASES = {
    "she plays the viola in the orchestra",
    "the viola section sounds beautiful",
    "hand me that viola please",
    "buy a villa", "fly over", "I'll be over", "I'll be over there",
}

# Phoneme distance thresholds
MAX_EDIT_DISTANCE = 3
MIN_EDIT_DISTANCE = 1

# TTS voices -- diverse accents for training robustness.
# Excludes all eval-only voices per project policy.
TTS_VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-SoniaNeural",
    "en-AU-NatashaNeural",
    "en-IN-NeerjaNeural",
    "en-US-DavisNeural",
    "en-US-JaneNeural",
    "en-US-SaraNeural",
]

# Delay between TTS requests to avoid Edge TTS rate limiting (seconds)
TTS_DELAY = 0.5
TTS_RETRY_DELAY = 3.0
TTS_MAX_RETRIES = 2

# Eval-only voices -- NEVER use these for training data generation.
EVAL_ONLY_VOICES = {
    "en-AU-WilliamNeural", "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-GB-LibbyNeural", "en-GB-RyanNeural", "en-GB-ThomasNeural",
    "en-IE-ConnorNeural", "en-IE-EmilyNeural", "en-IN-PrabhatNeural",
    "en-US-AnaNeural", "en-US-AndrewNeural", "en-US-AriaNeural",
    "en-US-BrianNeural", "en-US-ChristopherNeural", "en-US-EmmaNeural",
    "en-ZA-LeahNeural", "en-ZA-LukeNeural",
}

# ARPABET vowel phonemes (stress markers stripped)
VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}

# Phonetic feature classes for fine-grained similarity
MANNER_CLASSES = {
    "stops": {"P", "B", "T", "D", "K", "G"},
    "fricatives": {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"},
    "affricates": {"CH", "JH"},
    "nasals": {"M", "N", "NG"},
    "liquids": {"L", "R"},
    "glides": {"W", "Y"},
}

# Voicing pairs -- substituting one for the other is perceptually close
VOICING_PAIRS = {
    "P": "B", "B": "P", "T": "D", "D": "T", "K": "G", "G": "K",
    "F": "V", "V": "F", "TH": "DH", "DH": "TH",
    "S": "Z", "Z": "S", "SH": "ZH", "ZH": "SH",
    "CH": "JH", "JH": "CH",
}


# ---------------------------------------------------------------------------
# Phoneme utilities
# ---------------------------------------------------------------------------

def strip_stress(phoneme: str) -> str:
    """Remove stress markers (0, 1, 2) from ARPABET phoneme."""
    return phoneme.rstrip("012")


def get_stress_pattern(phonemes: list[str]) -> list[int]:
    """Extract stress pattern from phonemes. Returns list of stress levels for vowels."""
    pattern = []
    for p in phonemes:
        if p[-1:] in ("0", "1", "2"):
            pattern.append(int(p[-1]))
    return pattern


def extract_vowels(phonemes: list[str]) -> list[str]:
    """Extract just the vowel phonemes (stress stripped)."""
    return [strip_stress(p) for p in phonemes if strip_stress(p) in VOWELS]


def extract_consonants(phonemes: list[str]) -> list[str]:
    """Extract just the consonant phonemes."""
    return [strip_stress(p) for p in phonemes if strip_stress(p) not in VOWELS]


def same_manner_class(p1: str, p2: str) -> bool:
    """Check if two consonants belong to the same manner of articulation class."""
    for cls in MANNER_CLASSES.values():
        if p1 in cls and p2 in cls:
            return True
    return False


def is_voicing_pair(p1: str, p2: str) -> bool:
    """Check if two phonemes differ only in voicing (e.g., P/B, F/V)."""
    return VOICING_PAIRS.get(p1) == p2


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def phoneme_edit_distance(p1: list[str], p2: list[str]) -> int:
    """Standard Levenshtein distance on phoneme sequences (stress stripped)."""
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
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def weighted_phoneme_distance(p1: list[str], p2: list[str]) -> float:
    """Perceptually-weighted edit distance.

    Costs:
      - Identical phoneme:           0.0
      - Vowel-to-vowel swap:         0.4  (accent variation)
      - Voicing pair swap (P/B etc): 0.5  (very close perceptually)
      - Same manner class swap:      0.7  (e.g., L/R)
      - Any other substitution:      1.0
      - Insertion / deletion:        1.0
    """
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
            a, b = s1[i - 1], s2[j - 1]
            if a == b:
                cost = 0.0
            elif a in VOWELS and b in VOWELS:
                cost = 0.4
            elif is_voicing_pair(a, b):
                cost = 0.5
            elif same_manner_class(a, b):
                cost = 0.7
            else:
                cost = 1.0
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def vowel_pattern_similarity(target_vowels: list[str], candidate_vowels: list[str]) -> float:
    """Score 0-1 for how similar the vowel backbones are.

    Same length + same vowels = 1.0.  Length mismatch or different vowels reduce score.
    """
    if not target_vowels or not candidate_vowels:
        return 0.0
    # Length penalty
    max_len = max(len(target_vowels), len(candidate_vowels))
    min_len = min(len(target_vowels), len(candidate_vowels))
    length_ratio = min_len / max_len
    # Match the shorter against the longer
    matches = 0
    for i in range(min_len):
        if target_vowels[i] == candidate_vowels[i]:
            matches += 1
    match_ratio = matches / max_len
    return (length_ratio * 0.4) + (match_ratio * 0.6)


def stress_pattern_similarity(target_stress: list[int], candidate_stress: list[int]) -> float:
    """Score 0-1 for how similar stress patterns are.

    Same primary stress position is most important for wake-word confusion.
    """
    if not target_stress or not candidate_stress:
        return 0.0
    # Find primary stress positions
    t_primary = [i for i, s in enumerate(target_stress) if s == 1]
    c_primary = [i for i, s in enumerate(candidate_stress) if s == 1]
    if not t_primary or not c_primary:
        return 0.3  # no primary stress info, neutral score
    # Same primary stress position?
    if t_primary[0] == c_primary[0] and len(target_stress) == len(candidate_stress):
        return 1.0
    elif t_primary[0] == c_primary[0]:
        return 0.7
    else:
        return 0.2


def composite_similarity_score(
    target_phonemes: list[str],
    candidate_phonemes: list[str],
) -> dict:
    """Compute a composite similarity score combining multiple metrics.

    Returns dict with individual scores and a combined score (lower = more similar).
    """
    edit_dist = phoneme_edit_distance(target_phonemes, candidate_phonemes)
    weighted_dist = weighted_phoneme_distance(target_phonemes, candidate_phonemes)

    t_vowels = extract_vowels(target_phonemes)
    c_vowels = extract_vowels(candidate_phonemes)
    vowel_sim = vowel_pattern_similarity(t_vowels, c_vowels)

    t_stress = get_stress_pattern(target_phonemes)
    c_stress = get_stress_pattern(candidate_phonemes)
    stress_sim = stress_pattern_similarity(t_stress, c_stress)

    # Starts with same consonant onset?
    t_onset = strip_stress(target_phonemes[0]) if target_phonemes else ""
    c_onset = strip_stress(candidate_phonemes[0]) if candidate_phonemes else ""
    onset_match = 1.0 if t_onset == c_onset else (0.5 if is_voicing_pair(t_onset, c_onset) else 0.0)

    # Combined score: lower is more confusable
    # Weighted distance is the primary signal; vowel/stress/onset are bonuses
    combined = (
        weighted_dist
        - (vowel_sim * 0.3)       # reward similar vowel patterns
        - (stress_sim * 0.2)      # reward similar stress
        - (onset_match * 0.2)     # reward same onset consonant
    )

    return {
        "edit_distance": edit_dist,
        "weighted_distance": round(weighted_dist, 2),
        "vowel_similarity": round(vowel_sim, 2),
        "stress_similarity": round(stress_sim, 2),
        "onset_match": round(onset_match, 2),
        "combined_score": round(combined, 2),
    }


# ---------------------------------------------------------------------------
# CMU dictionary mining
# ---------------------------------------------------------------------------

def mine_cmu_confusables(
    target_phonemes_list: list[list[str]],
    max_distance: int = MAX_EDIT_DISTANCE,
) -> dict[str, dict]:
    """Search the entire CMU dictionary for phonemically similar words."""
    import pronouncing

    results: dict[str, dict] = {}
    entries = list(pronouncing.cmudict.entries())

    for word, phoneme_str in entries:
        phonemes = phoneme_str if isinstance(phoneme_str, list) else phoneme_str.split()
        # Skip very short or very long words
        if len(phonemes) < 2 or len(phonemes) > 10:
            continue
        if not word.isalpha():
            continue
        # Skip the wake word itself
        if word.lower() == "viola":
            continue

        best_combined = float("inf")
        best_scores = None

        for target_ph in target_phonemes_list:
            ed = phoneme_edit_distance(target_ph, phonemes)
            if ed > max_distance or ed < MIN_EDIT_DISTANCE:
                continue
            scores = composite_similarity_score(target_ph, phonemes)
            if scores["combined_score"] < best_combined:
                best_combined = scores["combined_score"]
                best_scores = scores

        if best_scores is not None:
            canonical = word.lower()
            if canonical not in results or results[canonical]["combined_score"] > best_combined:
                results[canonical] = {
                    "phonemes": " ".join(phonemes) if isinstance(phonemes, list) else phoneme_str,
                    **best_scores,
                }

    return results


def mine_g2p_extras(
    target_phonemes_list: list[list[str]],
    extra_words: list[str],
) -> dict[str, dict]:
    """Use g2p-en to score words not in CMU dict (proper nouns, brands, etc.)."""
    from g2p_en import G2p
    g2p = G2p()

    results: dict[str, dict] = {}
    for word in extra_words:
        phonemes_raw = g2p(word)
        phonemes = [p for p in phonemes_raw if p.strip() and p != " "]
        if len(phonemes) < 2:
            continue
        if word.lower() == "viola":
            continue

        best_combined = float("inf")
        best_scores = None

        for target_ph in target_phonemes_list:
            ed = phoneme_edit_distance(target_ph, phonemes)
            if ed > MAX_EDIT_DISTANCE:
                continue
            scores = composite_similarity_score(target_ph, phonemes)
            if scores["combined_score"] < best_combined:
                best_combined = scores["combined_score"]
                best_scores = scores

        if best_scores is not None:
            results[word.lower()] = {
                "phonemes": " ".join(phonemes),
                **best_scores,
            }

    return results


# ---------------------------------------------------------------------------
# Confusable phrase candidates (cross-word-boundary confusion)
# ---------------------------------------------------------------------------

PHRASE_CANDIDATES = [
    # Phrases where "viola" might be mis-heard at word boundaries
    "pile of", "dial up", "file a", "I owe a lot", "by a lot",
    "higher law", "hire a lot", "via la", "viva la", "try a lot",
    # Instrument references (should NOT trigger wake word)
    "viola da gamba", "my viola teacher", "play viola for me",
    "she plays the viola", "is that a viola or violin",
    # Rhyme/rhythm matches in running speech
    "payola scheme", "drive a corolla", "eating granola",
    "riding a gondola", "coca cola please",
]

# Extra single words to check via g2p (proper nouns, brand names, etc.)
G2P_EXTRA_WORDS = [
    "voila", "viola", "viol", "violas", "violist", "violists",
    "fiona", "verona", "vienna", "angola", "ebola", "cupola",
    "gondola", "pianola", "victrola", "crayola", "payola",
    "granola", "corolla", "guerrilla", "gorilla", "koala",
    "loyola", "enola", "ayatollah", "coca-cola",
    "viable", "dial", "trial", "denial", "spiral", "arrival",
    "rival", "bridal", "title", "idle", "bible", "liable", "final",
    "viral", "vial", "violin", "violent", "violence",
]


# ---------------------------------------------------------------------------
# TTS generation
# ---------------------------------------------------------------------------

async def generate_tts_clip(text: str, voice: str, output_path: Path, retries: int = TTS_MAX_RETRIES) -> bool:
    """Generate a single TTS clip as 16 kHz mono WAV with retry on rate limit."""
    if output_path.exists():
        return True  # skip existing
    import edge_tts
    import tempfile
    import os
    from pydub import AudioSegment

    for attempt in range(retries + 1):
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            communicate = edge_tts.Communicate(text + ".", voice)
            await communicate.save(tmp_path)

            # Verify file has content
            if os.path.getsize(tmp_path) < 100:
                os.unlink(tmp_path)
                raise RuntimeError("Empty audio file received")

            audio = AudioSegment.from_mp3(tmp_path)
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            audio.export(str(output_path), format="wav")

            os.unlink(tmp_path)
            return True
        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(TTS_RETRY_DELAY * (attempt + 1))
                continue
            print(f"  FAIL: {voice} '{text[:40]}': {e}", file=sys.stderr)
            return False


async def generate_all_tts(
    words: list[str],
    phrases: list[str],
    voices: list[str],
    output_dir: Path,
) -> tuple[int, int]:
    """Generate TTS for all words and phrases across all voices.

    Returns (generated_count, failed_count).
    """
    # Safety check: no eval-only voices
    for v in voices:
        if v in EVAL_ONLY_VOICES:
            raise ValueError(f"Voice {v} is eval-only and must not be used for training data")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_texts = words + phrases
    generated = 0
    failed = 0
    skipped = 0

    for vi, voice in enumerate(voices):
        print(f"  Voice {vi + 1}/{len(voices)}: {voice}")
        voice_gen = 0
        for ti, text in enumerate(all_texts):
            safe_name = (
                text.lower()
                .replace(" ", "_")
                .replace("'", "")
                .replace("-", "_")
                .replace(",", "")
            )[:50]
            out_path = output_dir / f"{voice}_{safe_name}.wav"
            if out_path.exists():
                skipped += 1
                continue
            ok = await generate_tts_clip(text, voice, out_path)
            if ok:
                generated += 1
                voice_gen += 1
            else:
                failed += 1
            # Throttle to avoid rate limiting
            await asyncio.sleep(TTS_DELAY)
        print(f"    -> {voice_gen} new clips generated")

    print(f"\n  Skipped (already exist): {skipped}")
    return generated, failed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_mining(top_n: int = 100) -> dict:
    """Run the full confusable mining pipeline.

    Returns a results dict with ranked word list, phoneme info, and categories.
    """
    import pronouncing
    from g2p_en import G2p

    g2p = G2p()

    print("=" * 70)
    print("PHONEME-BASED CONFUSABLE MINING FOR 'VIOLA'")
    print("=" * 70)

    # --- Step 1: Get phoneme representations of "viola" ---
    print("\n[Step 1] Phoneme representations of 'viola'")
    viola_cmu = pronouncing.phones_for_word("viola")
    viola_cmu_parsed = [pron.split() for pron in viola_cmu]
    for pron in viola_cmu:
        print(f"  CMU dict: {pron}")

    viola_g2p_raw = g2p("viola")
    viola_g2p_clean = [p for p in viola_g2p_raw if p.strip() and p != " "]
    print(f"  g2p-en:   {' '.join(viola_g2p_clean)}")

    target_phonemes = viola_cmu_parsed if viola_cmu_parsed else [viola_g2p_clean]
    print(f"  Using {len(target_phonemes)} pronunciation variant(s)")

    # --- Step 2: Mine CMU dictionary ---
    print(f"\n[Step 2] Mining CMU dictionary (edit distance {MIN_EDIT_DISTANCE}-{MAX_EDIT_DISTANCE})")
    n_entries = len(list(pronouncing.cmudict.entries()))
    print(f"  Scanning {n_entries} entries...")
    t0 = time.time()
    cmu_results = mine_cmu_confusables(target_phonemes, MAX_EDIT_DISTANCE)
    elapsed = time.time() - t0
    print(f"  Found {len(cmu_results)} candidate words ({elapsed:.1f}s)")

    # --- Step 3: Check extra words via g2p ---
    print(f"\n[Step 3] Checking {len(G2P_EXTRA_WORDS)} extra words via g2p-en")
    g2p_results = mine_g2p_extras(target_phonemes, G2P_EXTRA_WORDS)
    print(f"  {len(g2p_results)} within distance threshold")

    # Merge (CMU takes priority)
    all_results = {**g2p_results, **cmu_results}

    # --- Step 4: Remove existing and "viola" itself ---
    existing_lower = {w.lower() for w in EXISTING_WORDS} | {"viola"}
    new_results = {w: info for w, info in all_results.items() if w not in existing_lower}
    print(f"\n[Step 4] Filtering")
    print(f"  Total unique candidates: {len(all_results)}")
    print(f"  Already in v1 set: {len(all_results) - len(new_results)}")
    print(f"  New candidates: {len(new_results)}")

    # --- Step 5: Rank by combined score and pick top N ---
    ranked = sorted(new_results.items(), key=lambda x: x[1]["combined_score"])
    top_words = ranked[:top_n]

    # Categorize
    high_risk = [(w, i) for w, i in top_words if i["combined_score"] <= 0.5]
    medium_risk = [(w, i) for w, i in top_words if 0.5 < i["combined_score"] <= 1.5]
    low_risk = [(w, i) for w, i in top_words if i["combined_score"] > 1.5]

    print(f"\n[Step 5] Top {len(top_words)} confusable words (ranked by combined score)")
    print(f"  HIGH risk (score <= 0.5):  {len(high_risk)}")
    print(f"  MEDIUM risk (0.5-1.5):     {len(medium_risk)}")
    print(f"  LOW risk (> 1.5):          {len(low_risk)}")

    print(f"\n{'Word':<20s} {'Edit':>4s} {'Wt':>5s} {'Vowel':>5s} {'Strs':>5s} {'Onset':>5s} {'Comb':>6s}  Phonemes")
    print("-" * 90)
    for word, info in top_words:
        print(
            f"  {word:<18s} {info['edit_distance']:>4d} {info['weighted_distance']:>5.1f} "
            f"{info['vowel_similarity']:>5.2f} {info['stress_similarity']:>5.2f} "
            f"{info['onset_match']:>5.1f} {info['combined_score']:>6.2f}  {info['phonemes']}"
        )

    # --- Step 6: New confusable phrases ---
    existing_phrases_lower = {p.lower() for p in EXISTING_PHRASES}
    new_phrases = [p for p in PHRASE_CANDIDATES if p.lower() not in existing_phrases_lower]
    print(f"\n[Step 6] Confusable phrases")
    print(f"  New phrase candidates: {len(new_phrases)}")
    for p in new_phrases:
        print(f"    '{p}'")

    # --- Summary ---
    word_list = [w for w, _ in top_words]
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Viola phonemes (CMU): {viola_cmu}")
    print(f"  CMU entries scanned:  {n_entries}")
    print(f"  Total candidates:     {len(all_results)}")
    print(f"  New top-{top_n} words:   {len(word_list)}")
    print(f"  New phrases:          {len(new_phrases)}")
    print(f"  TTS voices:           {len(TTS_VOICES)}")
    est_clips = (len(word_list) + len(new_phrases)) * len(TTS_VOICES)
    print(f"  Estimated TTS clips:  {est_clips}")

    results = {
        "viola_phonemes_cmu": viola_cmu,
        "viola_phonemes_g2p": " ".join(viola_g2p_clean),
        "total_cmu_entries": n_entries,
        "total_candidates": len(all_results),
        "top_n": top_n,
        "word_list": word_list,
        "word_details": {w: info for w, info in top_words},
        "phrase_list": new_phrases,
        "high_risk": [w for w, _ in high_risk],
        "medium_risk": [w for w, _ in medium_risk],
        "low_risk": [w for w, _ in low_risk],
        "voices": TTS_VOICES,
        "estimated_clips": est_clips,
    }
    return results


async def main():
    t_start = time.time()

    # Parse args
    top_n = 100
    if "--top" in sys.argv:
        idx = sys.argv.index("--top")
        if idx + 1 < len(sys.argv):
            top_n = int(sys.argv[idx + 1])
    do_generate = "--generate" in sys.argv or "--gen" in sys.argv

    # Run mining
    results = run_mining(top_n=top_n)

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_PATH}")

    # TTS generation
    if do_generate:
        print(f"\n{'=' * 70}")
        print("TTS GENERATION")
        print(f"{'=' * 70}")
        word_list = results["word_list"]
        phrase_list = results["phrase_list"]
        print(f"  Words: {len(word_list)}, Phrases: {len(phrase_list)}, Voices: {len(TTS_VOICES)}")

        generated, failed = await generate_all_tts(
            words=word_list,
            phrases=phrase_list,
            voices=TTS_VOICES,
            output_dir=CONFUSABLE_V2_DIR,
        )
        print(f"\n  Generated: {generated}")
        print(f"  Failed:    {failed}")
        print(f"  Output:    {CONFUSABLE_V2_DIR}")

        # Count total files in output dir
        total_files = len(list(CONFUSABLE_V2_DIR.glob("*.wav")))
        print(f"  Total WAV files in v2 dir: {total_files}")
    else:
        est = results["estimated_clips"]
        print(f"\n  To generate TTS clips (~{est} files), run:")
        print(f"    python experiments/mine_confusables.py --generate")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
