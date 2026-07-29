"""Ratchet: the synthetic RIR must behave like a room, not dissolve a clip into noise.

#1775. `generate_synthetic_rir` planted the direct impulse at 1.0 and *then*
peak-normalised the whole IR. The peak is the noise tail's, not the direct
path's, so the direct impulse was divided down to 0.31-0.51 and the
direct-to-reverberant ratio came out at -22 to -30 dB -- the tail carrying
roughly 500x the energy of the direct sound. A real room at 1-3 m is 0 to +15 dB.

Convolving a wake-word clip with such an IR does not add reverberation, it
destroys the clip: correlation with the dry signal falls to ~0.22 and the energy
smears out of the speech burst into what should be silence. One in three
auto-generated TTS positives is built this way (`tools/train.py`
`_generate_tts_positives`), so the trainer learned near-silence-like audio as the
wake word -- the same region the post-training quality gate probes to decide
grade F.

These tests fail on the pre-fix shape (measured DRR ~-27 dB) and pass on the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from violawake_sdk.training.augment import (
    _SYNTHETIC_RIR_DRR_DB_MAX,
    _SYNTHETIC_RIR_DRR_DB_MIN,
    apply_rir,
    generate_synthetic_rir,
)

SR = 16000


def _drr_db(rir: np.ndarray) -> float:
    """Direct-to-reverberant ratio: sample 0 against the rest of the tail."""
    direct = float(rir[0]) ** 2
    tail = float(np.sum(np.asarray(rir[1:], dtype=np.float64) ** 2))
    return 10.0 * np.log10(direct / max(tail, 1e-20))


def _speech_like(duration_s: float = 1.5, burst=(0.5, 0.8)) -> np.ndarray:
    """A silent clip with one short voiced burst, like a wake word in a clip."""
    t = np.arange(int(SR * duration_s)) / SR
    sig = np.zeros(len(t), dtype=np.float32)
    m = (t >= burst[0]) & (t < burst[1])
    for f in (300.0, 900.0, 2400.0):
        sig[m] += np.sin(2 * np.pi * f * t[m]).astype(np.float32)
    return (sig / np.abs(sig).max()).astype(np.float32)


class TestSyntheticRIRIsARoom:
    def test_direct_path_is_the_loudest_tap(self) -> None:
        """The direct sound must dominate; it is what makes an IR a room."""
        for seed in range(16):
            rir = generate_synthetic_rir(rng=np.random.default_rng(seed))
            assert rir[0] == pytest.approx(np.abs(rir).max(), rel=1e-6), (
                f"seed {seed}: direct path {rir[0]:.4f} is not the peak "
                f"({np.abs(rir).max():.4f}) -- the tail was normalised over it"
            )

    def test_drr_is_physically_plausible(self) -> None:
        """Pre-fix this measured -22 to -30 dB across every seed."""
        drrs = [
            _drr_db(generate_synthetic_rir(rng=np.random.default_rng(seed)))
            for seed in range(24)
        ]
        assert min(drrs) >= _SYNTHETIC_RIR_DRR_DB_MIN - 1.0, (
            f"min DRR {min(drrs):.2f} dB is below a real room's range; "
            "the reverberant tail is drowning the direct path"
        )
        assert max(drrs) <= _SYNTHETIC_RIR_DRR_DB_MAX + 1.0, (
            f"max DRR {max(drrs):.2f} dB is above the configured range"
        )

    def test_requested_drr_is_honoured(self) -> None:
        for target in (0.0, 3.0, 7.5, 15.0):
            rir = generate_synthetic_rir(rng=np.random.default_rng(7), drr_db=target)
            assert _drr_db(rir) == pytest.approx(target, abs=0.5)

    def test_reverb_preserves_the_clip(self) -> None:
        """A reverberated wake word must still be recognisably the wake word."""
        dry = _speech_like()
        for seed in range(8):
            wet = apply_rir(dry, generate_synthetic_rir(rng=np.random.default_rng(seed)))
            corr = float(np.corrcoef(dry, wet)[0, 1])
            assert corr >= 0.5, (
                f"seed {seed}: reverberated clip correlates {corr:.3f} with the dry "
                "signal -- this is a smear, not a room"
            )

    def test_energy_stays_inside_the_speech_burst(self) -> None:
        """Reverb adds a tail; it must not move the clip's energy into the silence."""
        dry = _speech_like()
        t = np.arange(len(dry)) / SR
        inside = (t >= 0.5) & (t < 0.8)
        for seed in range(8):
            wet = apply_rir(dry, generate_synthetic_rir(rng=np.random.default_rng(seed)))
            frac = float((wet[inside] ** 2).sum() / (wet**2).sum())
            assert frac >= 0.85, (
                f"seed {seed}: only {frac:.1%} of the reverberated clip's energy is "
                "still inside the spoken burst; the rest leaked into the silence, "
                "which is what teaches the trainer that near-silence is the wake word"
            )


class TestRIRRegressionsStillHold:
    """The properties the pre-fix code did get right must survive the fix."""

    def test_peak_is_unit(self) -> None:
        rir = generate_synthetic_rir(sample_rate=SR, rt60=0.5, rng=np.random.default_rng(42))
        assert np.abs(rir).max() == pytest.approx(1.0, abs=1e-6)

    def test_energy_still_decays(self) -> None:
        rir = generate_synthetic_rir(sample_rate=SR, rt60=0.5, rng=np.random.default_rng(42))
        n = len(rir)
        assert np.mean(rir[1 : n // 4] ** 2) > np.mean(rir[3 * n // 4 :] ** 2)

    def test_reproducible(self) -> None:
        a = generate_synthetic_rir(rng=np.random.default_rng(42))
        b = generate_synthetic_rir(rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

    def test_very_short_rt60_is_safe(self) -> None:
        rir = generate_synthetic_rir(sample_rate=SR, rt60=0.0001)
        assert len(rir) >= 2
        assert np.isfinite(rir).all()
        assert rir[0] == pytest.approx(1.0)
