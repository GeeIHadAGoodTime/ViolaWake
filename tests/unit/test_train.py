"""Targeted unit tests for violawake_sdk.tools.train CLI behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from violawake_sdk.tools import train


def _touch_audio_files(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (directory / f"{idx:03d}.wav").write_bytes(b"wav")


def _write_probe_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    duration_s: float = 0.1,
) -> None:
    import wave

    import numpy as np

    n_samples = max(1, int(sample_rate * duration_s))
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pcm = (audio * 32767).astype(np.int16)
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1).reshape(-1)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _path_exists_without_corpus(original_exists):
    def _exists(path: Path) -> bool:
        if "corpus" in {part.lower() for part in path.parts}:
            return False
        return original_exists(path)

    return _exists


class TestTrainHelpers:
    def test_held_out_count_keeps_at_least_one_training_file(self) -> None:
        assert train._held_out_count(0) == 0
        assert train._held_out_count(1) == 0
        assert train._held_out_count(2) == 1
        assert train._held_out_count(10) == 5

    def test_auto_eval_verdict_thresholds(self) -> None:
        assert train._auto_eval_verdict(9.9) == "GOOD (EER < 10%)"
        assert train._auto_eval_verdict(10.0) == "ACCEPTABLE (EER <= 15%)"
        assert train._auto_eval_verdict(20.0) == "WARNING (EER > 15%)"
        assert train._auto_eval_verdict(30.0) == "CRITICAL (EER > 25%)"

    def test_update_auto_eval_config_merges_existing_json(self, tmp_path: Path) -> None:
        config_path = tmp_path / "model.config.json"
        config_path.write_text(json.dumps({"wake_word": "viola"}), encoding="utf-8")

        train._update_auto_eval_config(config_path, {"status": "ok", "eer_percent": 12.5})

        saved = json.loads(config_path.read_text(encoding="utf-8"))
        assert saved["wake_word"] == "viola"
        assert saved["auto_eval"]["status"] == "ok"

    def test_edge_tts_synthesize_decodes_with_soundfile(self, tmp_path: Path) -> None:
        import numpy as np

        class FakeCommunicate:
            def __init__(self, text: str, voice: str) -> None:
                self.text = text
                self.voice = voice

            async def stream(self):
                yield {"type": "audio", "data": b"fake mp3 bytes" * 20}

        edge_tts_module = ModuleType("edge_tts")
        edge_tts_module.Communicate = FakeCommunicate
        soundfile_module = ModuleType("soundfile")
        soundfile_module.read = MagicMock(
            return_value=(np.zeros(16000, dtype=np.float32), 16000)
        )

        out_path = tmp_path / "tts.wav"
        with patch.dict(
            sys.modules,
            {"edge_tts": edge_tts_module, "soundfile": soundfile_module},
        ):
            assert train._edge_tts_synthesize("hello", "en-US-JennyNeural", out_path)

        assert out_path.stat().st_size > 44
        soundfile_module.read.assert_called_once()

    def test_edge_tts_synthesize_retries_transient_stream_failure(
        self, tmp_path: Path
    ) -> None:
        import numpy as np

        attempts = {"count": 0}

        class FakeCommunicate:
            def __init__(self, text: str, voice: str) -> None:
                self.text = text
                self.voice = voice

            async def stream(self):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("503 Service Unavailable")
                yield {"type": "audio", "data": b"fake mp3 bytes" * 20}

        edge_tts_module = ModuleType("edge_tts")
        edge_tts_module.Communicate = FakeCommunicate
        soundfile_module = ModuleType("soundfile")
        soundfile_module.read = MagicMock(
            return_value=(np.zeros(16000, dtype=np.float32), 16000)
        )

        out_path = tmp_path / "tts.wav"
        with (
            patch.dict(
                sys.modules,
                {"edge_tts": edge_tts_module, "soundfile": soundfile_module},
            ),
            patch("violawake_sdk.tools.train._sleep_with_cancel") as sleep_mock,
        ):
            assert train._edge_tts_synthesize("hello", "en-US-JennyNeural", out_path)

        assert attempts["count"] == 2
        sleep_mock.assert_called_once()
        assert out_path.stat().st_size > 44

    def test_load_training_audio_accepts_16khz_mono(self, tmp_path: Path) -> None:
        import numpy as np

        wav_path = tmp_path / "ok_16khz_mono.wav"
        _write_probe_wav(wav_path, sample_rate=16_000, channels=1)

        audio = train._load_training_audio(wav_path)

        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert audio.size > 0

    def test_load_training_audio_rejects_wrong_sample_rate(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "bad_22khz.wav"
        _write_probe_wav(wav_path, sample_rate=22_050, channels=1)

        with pytest.raises(train.TrainingError, match="16000 Hz"):
            train._load_training_audio(wav_path)

    def test_load_training_audio_rejects_stereo(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "bad_stereo.wav"
        _write_probe_wav(wav_path, sample_rate=16_000, channels=2)

        with pytest.raises(train.TrainingError, match="mono"):
            train._load_training_audio(wav_path)

    def test_prepare_audio_for_oww_outputs_20ms_aligned_clip(self) -> None:
        import numpy as np

        from violawake_sdk._constants import CLIP_SAMPLES

        audio = np.full(CLIP_SAMPLES - 123, 0.05, dtype=np.float32)
        prepared = train._prepare_audio_for_oww(audio, clip_name="short.wav", verbose=False)

        assert train._TRAINING_FRAME_SAMPLES == 320
        assert prepared is not None
        assert len(prepared) == CLIP_SAMPLES
        assert len(prepared) % train._TRAINING_FRAME_SAMPLES == 0

    def test_confusable_generation_logs_zero_edge_tts_outputs(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        train._LAST_EDGE_TTS_ERROR = "pydub decode failed: missing ffprobe"

        with (
            caplog.at_level("ERROR", logger=train.logger.name),
            patch(
                "violawake_sdk.tools.confusables.generate_confusables",
                return_value=["violas"],
            ),
            patch("violawake_sdk.tools.train._edge_tts_synthesize", return_value=False),
        ):
            generated = train._generate_confusable_negatives(
                "viola",
                tmp_path,
                n_confusables=1,
                voices_per_word=1,
                verbose=False,
            )

        assert generated == []
        assert "edge-tts confusable negative generation produced 0 files" in caplog.text
        assert "missing ffprobe" in caplog.text

    def test_edge_tts_retry_backoff_honors_cancellation(self, tmp_path: Path) -> None:
        class FakeCommunicate:
            def __init__(self, text: str, voice: str) -> None:
                self.text = text
                self.voice = voice

            async def stream(self):
                raise RuntimeError("503 Service Unavailable")
                yield  # pragma: no cover

        edge_tts_module = ModuleType("edge_tts")
        edge_tts_module.Communicate = FakeCommunicate

        checks = {"count": 0}

        def _cancel_after_first_check() -> None:
            checks["count"] += 1
            if checks["count"] >= 2:
                raise RuntimeError("cancelled")

        out_path = tmp_path / "tts.wav"
        with (
            patch.dict(sys.modules, {"edge_tts": edge_tts_module}),
            patch("violawake_sdk.tools.train.time.sleep") as sleep_mock,
            pytest.raises(RuntimeError, match="cancelled"),
        ):
            train._edge_tts_synthesize(
                "hello",
                "en-US-JennyNeural",
                out_path,
                check_cancelled=_cancel_after_first_check,
            )

        sleep_mock.assert_not_called()

    def test_confusable_generation_honors_cancellation_between_words(
        self, tmp_path: Path
    ) -> None:
        def _cancel_immediately() -> None:
            raise RuntimeError("cancelled")

        with (
            patch(
                "violawake_sdk.tools.confusables.generate_confusables",
                return_value=["violas", "violah"],
            ),
            patch("violawake_sdk.tools.train._edge_tts_synthesize") as synth_mock,
            pytest.raises(RuntimeError, match="cancelled"),
        ):
            train._generate_confusable_negatives(
                "viola",
                tmp_path,
                n_confusables=2,
                voices_per_word=1,
                verbose=False,
                check_cancelled=_cancel_immediately,
            )

        synth_mock.assert_not_called()

    def test_confusable_generation_reports_incremental_progress(
        self, tmp_path: Path
    ) -> None:
        progress_events: list[dict[str, int | str]] = []

        def _fake_synthesize(
            text: str,
            voice: str,
            out_path: Path,
            *,
            check_cancelled=None,
        ) -> bool:
            out_path.write_bytes(b"wav")
            return True

        with (
            patch(
                "violawake_sdk.tools.confusables.generate_confusables",
                return_value=["violas", "violah"],
            ),
            patch("violawake_sdk.tools.train._edge_tts_synthesize", side_effect=_fake_synthesize),
        ):
            generated = train._generate_confusable_negatives(
                "viola",
                tmp_path,
                n_confusables=2,
                voices_per_word=2,
                verbose=False,
                progress_callback=progress_events.append,
            )

        assert len(generated) == 4
        assert [event["completed_samples"] for event in progress_events] == [1, 2, 3, 4]
        assert all(event["total_samples"] == 4 for event in progress_events)
        assert progress_events[-1]["generated_files"] == 4
        assert progress_events[-1]["word_index"] == 2
        assert progress_events[-1]["total_words"] == 2

    def test_extract_temporal_embeddings_streams_files_incrementally(
        self, tmp_path: Path
    ) -> None:
        import numpy as np

        from violawake_sdk._constants import CLIP_SAMPLES

        audio_files = [tmp_path / f"neg_{idx:03d}.wav" for idx in range(3)]
        for path in audio_files:
            path.write_bytes(b"wav")

        class _FakePreprocessor:
            def __init__(self) -> None:
                self.calls: list[tuple[int, ...]] = []

            def embed_clips(self, audio_batch, ncpu: int = 1):
                self.calls.append(tuple(audio_batch.shape))
                return np.ones((1, 12, 96), dtype=np.float32)

        fake_preprocessor = _FakePreprocessor()

        class _FakeModel:
            def __init__(self, inference_framework: str) -> None:
                assert inference_framework == "onnx"
                self.preprocessor = fake_preprocessor

        fake_openwakeword = ModuleType("openwakeword")
        fake_openwakeword_model = ModuleType("openwakeword.model")
        fake_openwakeword_model.Model = _FakeModel
        fake_openwakeword.model = fake_openwakeword_model

        with (
            patch.dict(
                sys.modules,
                {
                    "openwakeword": fake_openwakeword,
                    "openwakeword.model": fake_openwakeword_model,
                },
            ),
            patch(
                "violawake_sdk.tools.train._load_training_audio",
                side_effect=lambda _path: np.full(CLIP_SAMPLES * 40, 0.05, dtype=np.float32),
            ),
            patch(
                "violawake_sdk.tools.train._extract_temporal_windows_from_audio",
                side_effect=AssertionError(
                    "negative temporal extraction must process files incrementally"
                ),
            ),
        ):
            embeddings, source_indices, tags = train._extract_temporal_embeddings(
                audio_files,
                "neg",
                verbose=False,
                seq_len=9,
            )

        assert fake_preprocessor.calls == [(1, CLIP_SAMPLES)] * 3
        assert len(embeddings) == 12
        assert source_indices == ([0] * 4) + ([1] * 4) + ([2] * 4)
        assert tags == ["neg"] * 12

    def test_extract_temporal_embeddings_rejects_wrong_sample_rate_before_embedding(
        self, tmp_path: Path
    ) -> None:
        wav_path = tmp_path / "bad_22khz.wav"
        _write_probe_wav(wav_path, sample_rate=22_050, channels=1)

        class _FakePreprocessor:
            def embed_clips(self, audio_batch, ncpu: int = 1):
                raise AssertionError("bad-rate audio must fail before embedding extraction")

        class _FakeModel:
            def __init__(self, inference_framework: str) -> None:
                assert inference_framework == "onnx"
                self.preprocessor = _FakePreprocessor()

        fake_openwakeword = ModuleType("openwakeword")
        fake_openwakeword_model = ModuleType("openwakeword.model")
        fake_openwakeword_model.Model = _FakeModel
        fake_openwakeword.model = fake_openwakeword_model

        with (
            patch.dict(
                sys.modules,
                {
                    "openwakeword": fake_openwakeword,
                    "openwakeword.model": fake_openwakeword_model,
                },
            ),
            pytest.raises(train.TrainingError, match="16000 Hz"),
        ):
            train._extract_temporal_embeddings([wav_path], "neg", verbose=False, seq_len=9)


class TestTrainMainValidation:
    def test_main_exits_when_positives_dir_is_missing(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ]

        with patch.object(sys, "argv", argv), pytest.raises(SystemExit, match="1"):
            train.main()

        assert "Positives directory not found" in capsys.readouterr().err

    def test_main_exits_when_negatives_dir_is_missing(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        positives = tmp_path / "positives"
        positives.mkdir()
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(positives),
            "--negatives",
            str(tmp_path / "missing-neg"),
            "--output",
            str(output),
        ]

        with patch.object(sys, "argv", argv), pytest.raises(SystemExit, match="1"):
            train.main()

        assert "Negatives directory not found" in capsys.readouterr().err

    def test_main_rejects_legacy_architecture_flag(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--output",
            str(output),
            "--architecture",
            "mlp",
        ]

        with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc_info:
            train.main()

        assert exc_info.value.code == 2
        assert "unrecognized arguments: --architecture mlp" in capsys.readouterr().err

    def test_main_temporal_exits_with_too_few_positive_files(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        positives = tmp_path / "positives"
        positives.mkdir()
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(positives),
            "--output",
            str(output),
            "--no-auto-corpus",
            "--quiet",
        ]

        with patch.object(sys, "argv", argv), pytest.raises(SystemExit, match="1"):
            train.main()

    def test_main_exits_cleanly_when_internal_temporal_training_fails(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        positives = tmp_path / "positives"
        negatives = tmp_path / "negatives"
        _touch_audio_files(positives, 5)
        _touch_audio_files(negatives, 5)
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(positives),
            "--negatives",
            str(negatives),
            "--no-auto-corpus",
            "--output",
            str(output),
        ]

        with (
            patch.object(sys, "argv", argv),
            patch(
                "violawake_sdk.tools.train._train_temporal_cnn",
                side_effect=train.TrainingError("temporal pipeline failed"),
            ),
            pytest.raises(SystemExit, match="1"),
        ):
            train.main()

        assert "temporal pipeline failed" in capsys.readouterr().err

    def test_main_temporal_exits_with_too_few_negative_files(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        positives = tmp_path / "positives"
        negatives = tmp_path / "negatives"
        _touch_audio_files(positives, 5)
        negatives.mkdir()
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(positives),
            "--negatives",
            str(negatives),
            "--output",
            str(output),
            "--no-auto-corpus",
            "--quiet",
        ]

        original_exists = Path.exists
        with (
            patch.object(sys, "argv", argv),
            patch("pathlib.Path.exists", new=_path_exists_without_corpus(original_exists)),
            pytest.raises(SystemExit, match="1"),
        ):
            train.main()

        assert "Enable --auto-corpus or provide negatives via --negatives" in capsys.readouterr().err

    def test_main_parses_args_and_invokes_temporal_training(self, tmp_path: Path) -> None:
        positives = tmp_path / "positives"
        negatives = tmp_path / "negatives"
        eval_dir = tmp_path / "eval"
        (eval_dir / "positives").mkdir(parents=True)
        (eval_dir / "negatives").mkdir(parents=True)
        _touch_audio_files(positives, 6)
        _touch_audio_files(negatives, 6)
        output = tmp_path / "model.onnx"
        evaluate_module = ModuleType("violawake_sdk.tools.evaluate")
        evaluate_module.evaluate_onnx_model = MagicMock(
            return_value={
                "architecture": "temporal_cnn",
                "n_positives": 6,
                "n_negatives": 6,
                "eer_approx": 0.08,
                "roc_auc": 0.95,
                "optimal_far": 0.02,
                "optimal_frr": 0.03,
                "optimal_threshold": 0.82,
            }
        )
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(positives),
            "--negatives",
            str(negatives),
            "--output",
            str(output),
            "--eval-dir",
            str(eval_dir),
            "--epochs",
            "12",
            "--batch-size",
            "32",
            "--lr",
            "0.002",
            "--patience",
            "4",
            "--no-auto-corpus",
            "--no-augment",
            "--quiet",
        ]

        original_exists = Path.exists
        with (
            patch.object(sys, "argv", argv),
            patch.dict(sys.modules, {"violawake_sdk.tools.evaluate": evaluate_module}),
            patch("pathlib.Path.exists", new=_path_exists_without_corpus(original_exists)),
            patch("violawake_sdk.tools.train._train_temporal_cnn") as train_temporal,
        ):
            train.main()

        train_temporal.assert_called_once()
        kwargs = train_temporal.call_args.kwargs
        assert kwargs["wake_word"] == "viola"
        assert kwargs["epochs"] == 12
        assert kwargs["batch_size"] == 32
        assert kwargs["lr"] == 0.002
        assert kwargs["patience"] == 4
        assert kwargs["augment"] is False
        assert kwargs["verbose"] is False
        assert len(kwargs["pos_files"]) == 6
        assert len(kwargs["neg_files"]) == 6
