"""
Generate a fresh held-out TTS evaluation set for "Viola" wake word.

Uses edge-tts voices NOT in any English training set -- specifically non-English
locale voices (Italian, German, Spanish, Portuguese, French, Dutch, etc.) that
naturally pronounce "Viola" with different accents. Also uses local SAPI5 voices
via pyttsx3 for additional diversity.

The newer English edge-tts voices (Cora, Elizabeth, Jacob, etc.) fail on
single-word text, so we use multilingual voices that can handle short text
and produce diverse, natural "Viola" pronunciations.

Usage:
    python experiments/generate_fresh_eval.py
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import edge_tts
import numpy as np
import onnxruntime as ort
import soundfile as sf
from openwakeword.utils import AudioFeatures

# ── Project root ──────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT / "experiments" / "eval_fresh"
POS_DIR = OUTPUT_DIR / "positives"
NEG_DIR = OUTPUT_DIR / "negatives"
MODELS_DIR = PROJECT / "experiments" / "models"

# ── Voice exclusion lists ─────────────────────────────────────────────
# ALL English voices ever used in training or existing eval — NEVER reuse these.
EXCLUDED_VOICES = {
    # Existing eval set (22 voices)
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    "en-GB-LibbyNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-GB-ThomasNeural",
    "en-IE-ConnorNeural", "en-IE-EmilyNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-US-AnaNeural", "en-US-AndrewNeural", "en-US-AriaNeural", "en-US-BrianNeural",
    "en-US-ChristopherNeural", "en-US-EmmaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-ZA-LeahNeural", "en-ZA-LukeNeural",
    # Training confusable voices
    "en-US-AvaNeural", "en-US-EricNeural", "en-US-MichelleNeural",
    "en-US-RogerNeural", "en-US-SteffanNeural",
    "en-US-AvaMultilingualNeural", "en-US-BrianMultilingualNeural",
    "en-GB-MaisieNeural", "en-GB-MiaNeural",
    "en-KE-AsiliaNeural", "en-KE-ChilembaNeural",
    "en-NG-AbeoNeural", "en-NG-EzinneNeural",
    # Training positive voices
    "en-AU-WilliamMultilingualNeural", "en-US-AndrewMultilingualNeural",
    "en-US-EmmaMultilingualNeural",
    "en-HK-SamNeural", "en-HK-YanNeural",
    "en-NZ-MitchellNeural", "en-NZ-MollyNeural",
    "en-PH-JamesNeural", "en-PH-RosaNeural",
    "en-SG-LunaNeural", "en-SG-WayneNeural",
    "en-TZ-ElimuNeural", "en-TZ-ImaniNeural",
    # fast_generate.py training voices
    "en-US-DavisNeural", "en-US-JaneNeural", "en-US-JasonNeural",
    "en-US-SaraNeural", "en-US-TonyNeural",
    "en-US-NancyNeural", "en-US-AmberNeural", "en-US-AshleyNeural",
    "en-US-BrandonNeural",
    "en-GB-AbbiNeural", "en-GB-AlfieNeural", "en-GB-BellaNeural",
    "en-GB-ElliotNeural", "en-GB-NoahNeural",
    "en-AU-AnnetteNeural", "en-AU-CarlyNeural", "en-AU-DarrenNeural",
    "en-AU-FreyaNeural", "en-AU-KenNeural",
    "en-GB-HollieNeural", "en-GB-OliverNeural", "en-GB-OliviaNeural",
}

# ── Fresh eval voices: non-English locales (NONE used in training) ────
# "Viola" is a real word/name in Italian, German, Spanish, Portuguese, French,
# Dutch, Swedish, etc. These voices produce natural but accent-diverse
# pronunciations — exactly what we need for robust eval.
FRESH_VOICES = [
    # Italian (Viola = name + musical instrument)
    "it-IT-IsabellaNeural",        # Female, Italian
    "it-IT-DiegoNeural",           # Male, Italian
    # German (Viola = name + instrument)
    "de-DE-KatjaNeural",           # Female, German
    "de-DE-ConradNeural",          # Male, German
    "de-AT-IngridNeural",          # Female, Austrian German
    "de-CH-LeniNeural",            # Female, Swiss German
    # Spanish (Viola = related word, diff pronunciation)
    "es-ES-ElviraNeural",          # Female, Castilian Spanish
    "es-MX-JorgeNeural",           # Male, Mexican Spanish
    # Portuguese (Viola = guitar-like instrument)
    "pt-BR-FranciscaNeural",       # Female, Brazilian Portuguese
    "pt-PT-DuarteNeural",          # Male, European Portuguese
    # French (different vowel treatment)
    "fr-FR-DeniseNeural",          # Female, French
    "fr-CA-JeanNeural",            # Male, Quebec French
    # Dutch
    "nl-NL-FennaNeural",           # Female, Dutch
    "nl-NL-MaartenNeural",         # Male, Dutch
    # Scandinavian
    "sv-SE-SofieNeural",           # Female, Swedish
    "da-DK-JeppeNeural",           # Male, Danish
    # Polish
    "pl-PL-ZofiaNeural",           # Female, Polish
    # Finnish
    "fi-FI-HarriNeural",           # Male, Finnish
]

# Verify no overlap with excluded English voices
assert not set(FRESH_VOICES) & EXCLUDED_VOICES, \
    f"OVERLAP: {set(FRESH_VOICES) & EXCLUDED_VOICES}"

# ── Prosodic styles (SSML rate adjustments) ───────────────────────────
STYLES = [
    ("normal", "+0%"),
    ("slow", "-10%"),
    ("fast", "+10%"),
]

# ── Negative (confusable) words ───────────────────────────────────────
CONFUSABLE_WORDS = [
    "voila",
    "villa",
    "vanilla",
    "viva",
    "via",
    "hola",
    "viola the instrument",
    "playing the viola",
    "I love my viola",
    "violent",
]


async def generate_tts(text: str, voice: str, rate: str, output_path: Path) -> bool:
    """Generate a single TTS utterance as 16kHz mono WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path = output_path.with_suffix(".mp3")

    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(str(mp3_path))

        # Convert MP3 to 16kHz mono WAV using ffmpeg
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1",
             str(output_path)],
            capture_output=True, timeout=30
        )
        mp3_path.unlink(missing_ok=True)

        if result.returncode != 0:
            print(f"  ffmpeg failed for {output_path.name}: {result.stderr.decode()[:100]}")
            return False

        # Verify the file
        audio, sr = sf.read(str(output_path), dtype="float32")
        if sr != 16000 or len(audio) < 1600:  # at least 0.1s
            print(f"  Bad audio: {output_path.name} sr={sr} len={len(audio)}")
            output_path.unlink(missing_ok=True)
            return False

        return True
    except Exception as e:
        print(f"  FAIL: {voice} '{text[:20]}' rate={rate}: {e}")
        mp3_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return False


def generate_pyttsx3(text: str, voice_id: str, voice_label: str,
                     rate: int, output_path: Path) -> bool:
    """Generate a TTS utterance using local SAPI5 voices via pyttsx3."""
    try:
        import pyttsx3

        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = output_path.with_suffix(".raw.wav")

        engine = pyttsx3.init()
        engine.setProperty("voice", voice_id)
        engine.setProperty("rate", rate)
        engine.save_to_file(text, str(raw_path))
        engine.runAndWait()

        if not raw_path.exists():
            return False

        # Convert to 16kHz mono WAV
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_path), "-ar", "16000", "-ac", "1",
             str(output_path)],
            capture_output=True, timeout=30
        )
        raw_path.unlink(missing_ok=True)

        if result.returncode != 0:
            return False

        audio, sr = sf.read(str(output_path), dtype="float32")
        if sr != 16000 or len(audio) < 1600:
            output_path.unlink(missing_ok=True)
            return False

        return True
    except Exception as e:
        print(f"  FAIL pyttsx3: {voice_label} '{text[:20]}' rate={rate}: {e}")
        return False


async def generate_positives() -> list[Path]:
    """Generate 50+ positive 'Viola' utterances.

    18 edge-tts voices x 3 styles = 54 files, plus 2 pyttsx3 voices x 3 rates = 6.
    Total target: ~60 files.
    """
    print("\n=== Generating Positives ===")
    created = []

    # Edge-TTS voices (non-English locales)
    for voice in FRESH_VOICES:
        for style_name, rate in STYLES:
            safe_name = f"{voice}_{style_name}"
            out_path = POS_DIR / f"{safe_name}.wav"
            ok = await generate_tts("Viola", voice, rate, out_path)
            if ok:
                created.append(out_path)
                print(f"  OK: {out_path.name}")
            else:
                print(f"  SKIP: {out_path.name}")

    # pyttsx3 local voices (SAPI5 - David and Zira)
    sapi_voices = [
        ("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0",
         "sapi5_david"),
        ("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
         "sapi5_zira"),
    ]
    sapi_rates = [
        ("normal", 150),
        ("slow", 120),
        ("fast", 200),
    ]

    for voice_id, voice_label in sapi_voices:
        for style_name, rate in sapi_rates:
            out_path = POS_DIR / f"{voice_label}_{style_name}.wav"
            ok = generate_pyttsx3("Viola", voice_id, voice_label, rate, out_path)
            if ok:
                created.append(out_path)
                print(f"  OK: {out_path.name}")
            else:
                print(f"  SKIP: {out_path.name}")

    print(f"\nPositives generated: {len(created)}")
    return created


async def generate_negatives() -> list[Path]:
    """Generate 50+ negative confusable utterances.

    Use 5 diverse edge-tts voices x 10 confusable words = 50 files.
    """
    print("\n=== Generating Negatives ===")
    created = []

    # Use 5 voices from different language families for variety
    neg_voices = [
        "it-IT-IsabellaNeural",    # Italian female
        "de-DE-ConradNeural",      # German male
        "es-MX-JorgeNeural",       # Mexican Spanish male
        "fr-FR-DeniseNeural",      # French female
        "nl-NL-MaartenNeural",     # Dutch male
    ]

    for voice in neg_voices:
        for word in CONFUSABLE_WORDS:
            safe_word = word.replace(" ", "_").replace("'", "")
            safe_name = f"{voice}_{safe_word}"
            out_path = NEG_DIR / f"{safe_name}.wav"
            ok = await generate_tts(word, voice, "+0%", out_path)
            if ok:
                created.append(out_path)
                print(f"  OK: {out_path.name}")
            else:
                print(f"  SKIP: {out_path.name}")

    # Add pyttsx3 negatives for extra coverage
    sapi_voice = (
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0",
        "sapi5_david"
    )
    for word in CONFUSABLE_WORDS:
        safe_word = word.replace(" ", "_").replace("'", "")
        out_path = NEG_DIR / f"{sapi_voice[1]}_{safe_word}.wav"
        ok = generate_pyttsx3(word, sapi_voice[0], sapi_voice[1], 150, out_path)
        if ok:
            created.append(out_path)
            print(f"  OK: {out_path.name}")
        else:
            print(f"  SKIP: {out_path.name}")

    print(f"\nNegatives generated: {len(created)}")
    return created


def extract_embedding(filepath: Path, preprocessor: AudioFeatures) -> np.ndarray | None:
    """Extract OWW embedding from a WAV file."""
    try:
        audio, sr = sf.read(str(filepath), dtype="float32")
        if sr != 16000:
            print(f"  WARNING: {filepath.name} sr={sr}, expected 16000")
            return None

        # Convert to int16
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        # Pad/trim to 24000 samples (1.5s at 16kHz)
        if len(audio_int16) < 24000:
            audio_int16 = np.pad(audio_int16, (0, 24000 - len(audio_int16)))
        else:
            audio_int16 = audio_int16[:24000]

        embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
        embedding = embeddings.mean(axis=1)[0].astype(np.float32)
        return embedding
    except Exception as e:
        print(f"  Embedding error for {filepath.name}: {e}")
        return None


def score_embeddings(
    embeddings: dict[str, np.ndarray],
    model_path: Path,
) -> dict[str, float]:
    """Score embeddings through an ONNX model."""
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    scores = {}

    for name, emb in embeddings.items():
        inp = emb.reshape(1, -1)
        result = session.run(None, {input_name: inp})
        # Model outputs a score (sigmoid or raw)
        score = float(result[0][0][0]) if result[0].ndim > 1 else float(result[0][0])
        scores[name] = score

    return scores


def compute_metrics(
    pos_scores: dict[str, float],
    neg_scores: dict[str, float],
    thresholds: list[float],
) -> dict:
    """Compute detection rate and false accept rate at each threshold."""
    results = {}
    for thresh in thresholds:
        tp = sum(1 for s in pos_scores.values() if s >= thresh)
        fn = sum(1 for s in pos_scores.values() if s < thresh)
        fp = sum(1 for s in neg_scores.values() if s >= thresh)
        tn = sum(1 for s in neg_scores.values() if s < thresh)

        det_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results[str(thresh)] = {
            "threshold": thresh,
            "detection_rate": round(det_rate, 4),
            "false_accept_rate": round(fa_rate, 4),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
        }
    return results


async def main():
    print("=" * 60)
    print("Fresh Held-Out Eval Set Generator")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Edge-TTS voices: {len(FRESH_VOICES)} (non-English, none in training)")
    print(f"SAPI5 voices: 2 (David, Zira -- never used in any training)")
    print(f"Excluded English voices: {len(EXCLUDED_VOICES)}")

    # Create output dirs
    POS_DIR.mkdir(parents=True, exist_ok=True)
    NEG_DIR.mkdir(parents=True, exist_ok=True)

    # Clear any old files
    for d in [POS_DIR, NEG_DIR]:
        for f in d.glob("*.wav"):
            f.unlink()

    # Step 1: Generate audio
    pos_files = await generate_positives()
    neg_files = await generate_negatives()

    if not pos_files:
        print("\nERROR: No positive files generated. Check edge-tts and ffmpeg.")
        sys.exit(1)
    if not neg_files:
        print("\nERROR: No negative files generated.")
        sys.exit(1)

    # Step 2: Extract embeddings
    print("\n=== Extracting Embeddings ===")
    preprocessor = AudioFeatures()

    pos_embeddings = {}
    for f in pos_files:
        emb = extract_embedding(f, preprocessor)
        if emb is not None:
            pos_embeddings[f.name] = emb
    print(f"Positive embeddings: {len(pos_embeddings)}")

    neg_embeddings = {}
    for f in neg_files:
        emb = extract_embedding(f, preprocessor)
        if emb is not None:
            neg_embeddings[f.name] = emb
    print(f"Negative embeddings: {len(neg_embeddings)}")

    # Step 3: Score through models
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    models = {
        "D_combined_bce_s42 (baseline)": MODELS_DIR / "D_combined_bce_s42.onnx",
        "faph_hardened_s43 (current best)": MODELS_DIR / "faph_hardened_s43.onnx",
    }

    report = {
        "voices_used": FRESH_VOICES + ["sapi5_david", "sapi5_zira"],
        "voice_count": len(FRESH_VOICES) + 2,
        "excluded_voices_count": len(EXCLUDED_VOICES),
        "positives_generated": len(pos_files),
        "negatives_generated": len(neg_files),
        "positives_embedded": len(pos_embeddings),
        "negatives_embedded": len(neg_embeddings),
        "models": {},
    }

    for model_name, model_path in models.items():
        if not model_path.exists():
            print(f"\nWARNING: Model not found: {model_path}")
            continue

        print(f"\n=== Scoring: {model_name} ===")
        pos_scores = score_embeddings(pos_embeddings, model_path)
        neg_scores = score_embeddings(neg_embeddings, model_path)

        metrics = compute_metrics(pos_scores, neg_scores, thresholds)
        report["models"][model_name] = {
            "metrics": metrics,
            "positive_score_stats": {
                "min": round(min(pos_scores.values()), 4),
                "max": round(max(pos_scores.values()), 4),
                "mean": round(np.mean(list(pos_scores.values())), 4),
                "median": round(float(np.median(list(pos_scores.values()))), 4),
            },
            "negative_score_stats": {
                "min": round(min(neg_scores.values()), 4),
                "max": round(max(neg_scores.values()), 4),
                "mean": round(np.mean(list(neg_scores.values())), 4),
                "median": round(float(np.median(list(neg_scores.values()))), 4),
            },
            "per_file_scores": {
                "positives": {k: round(v, 4) for k, v in sorted(pos_scores.items())},
                "negatives": {k: round(v, 4) for k, v in sorted(neg_scores.items())},
            },
        }

        # Print results
        print(f"\n  Positive score range: "
              f"{min(pos_scores.values()):.4f} - {max(pos_scores.values()):.4f} "
              f"(mean={np.mean(list(pos_scores.values())):.4f})")
        print(f"  Negative score range: "
              f"{min(neg_scores.values()):.4f} - {max(neg_scores.values()):.4f} "
              f"(mean={np.mean(list(neg_scores.values())):.4f})")

        print(f"\n  {'Threshold':>10} {'Det Rate':>10} {'FA Rate':>10} "
              f"{'TP':>5} {'FN':>5} {'FP':>5} {'TN':>5}")
        print(f"  {'-' * 55}")
        for thresh in thresholds:
            m = metrics[str(thresh)]
            print(f"  {thresh:>10.2f} {m['detection_rate']:>10.4f} "
                  f"{m['false_accept_rate']:>10.4f} "
                  f"{m['true_positives']:>5} {m['false_negatives']:>5} "
                  f"{m['false_positives']:>5} {m['true_negatives']:>5}")

    # Save report
    report_path = OUTPUT_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Voices used ({len(FRESH_VOICES) + 2} total):")
    print("  Edge-TTS (non-English locales):")
    for v in FRESH_VOICES:
        print(f"    - {v}")
    print("  SAPI5 (local Windows voices):")
    print("    - sapi5_david (Microsoft David)")
    print("    - sapi5_zira (Microsoft Zira)")
    print(f"\nFiles generated:")
    print(f"  Positives: {len(pos_files)} ({len(pos_embeddings)} embedded)")
    print(f"  Negatives: {len(neg_files)} ({len(neg_embeddings)} embedded)")
    print(f"\nAll files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
