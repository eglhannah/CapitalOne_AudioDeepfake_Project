from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aasist_inference.audio import (  # noqa: E402
    decode_audio_bytes,
    detect_audio_format,
    resolve_ffmpeg_binary,
)
from aasist_inference.errors import (  # noqa: E402
    AudioDecodeError,
    AudioTooLongError,
    UnsupportedAudioFormatError,
    UploadTooLargeError,
)
from inference_contract import MAX_UPLOAD_BYTES, SAMPLE_RATE  # noqa: E402


class AudioSignatureTest(unittest.TestCase):
    def test_recognizes_supported_container_signatures(self) -> None:
        fixtures = {
            "flac": b"fLaC" + b"\x00" * 64,
            "wav": b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32,
            "ogg": b"OggS" + b"\x00" * 64,
            "webm": b"\x1aE\xdf\xa3" + b"\x00" * 64,
            "m4a": b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 48,
            "mp3": b"ID3" + b"\x00" * 64,
            "mp3-or-aac": b"\xff\xf1" + b"\x00" * 64,
        }
        for expected, payload in fixtures.items():
            with self.subTest(expected=expected):
                self.assertEqual(detect_audio_format(payload), expected)

    def test_rejects_unknown_signature(self) -> None:
        with self.assertRaises(UnsupportedAudioFormatError):
            detect_audio_format(b"not an audio container")


class AudioDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ffmpeg = resolve_ffmpeg_binary()

    def encode(self, waveform: np.ndarray, output_arguments: list[str]) -> bytes:
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            *output_arguments,
            "pipe:1",
        ]
        completed = subprocess.run(
            command,
            input=waveform.astype("<f4").tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode:
            self.fail(f"Test audio encoding failed for {output_arguments!r}")
        return completed.stdout

    def test_decodes_all_supported_presentation_formats(self) -> None:
        time_axis = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
        waveform = (0.1 * np.sin(2 * np.pi * 440 * time_axis)).astype(np.float32)
        formats = {
            "wav": ["-c:a", "pcm_s16le", "-f", "wav"],
            "flac": ["-c:a", "flac", "-f", "flac"],
            "mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
            "m4a": [
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "frag_keyframe+empty_moov",
                "-f",
                "mp4",
            ],
            "ogg": ["-c:a", "libvorbis", "-q:a", "5", "-f", "ogg"],
            "webm": ["-c:a", "libopus", "-b:a", "96k", "-f", "webm"],
        }
        for expected_format, arguments in formats.items():
            with self.subTest(expected_format=expected_format):
                encoded = self.encode(waveform, arguments)
                decoded = decode_audio_bytes(encoded, ffmpeg_binary=self.ffmpeg)
                self.assertEqual(decoded.detected_format, expected_format)
                self.assertEqual(decoded.sample_rate, SAMPLE_RATE)
                self.assertGreater(decoded.waveform.size, int(0.95 * SAMPLE_RATE))
                self.assertLess(decoded.waveform.size, int(1.1 * SAMPLE_RATE))
                self.assertTrue(np.isfinite(decoded.waveform).all())

    def test_lossless_wav_and_flac_decoding_are_equivalent(self) -> None:
        waveform = np.random.default_rng(7).normal(0.0, 0.05, SAMPLE_RATE).astype(np.float32)
        wav = decode_audio_bytes(
            self.encode(waveform, ["-c:a", "pcm_f32le", "-f", "wav"]),
            ffmpeg_binary=self.ffmpeg,
        )
        flac = decode_audio_bytes(
            self.encode(waveform, ["-c:a", "flac", "-sample_fmt", "s32", "-f", "flac"]),
            ffmpeg_binary=self.ffmpeg,
        )
        self.assertEqual(wav.waveform.size, flac.waveform.size)
        np.testing.assert_allclose(wav.waveform, flac.waveform, atol=1e-6, rtol=0.0)

    def test_converts_stereo_44100_hz_audio_to_mono_16000_hz(self) -> None:
        source_rate = 44_100
        time_axis = np.arange(source_rate, dtype=np.float32) / source_rate
        stereo = np.column_stack(
            (
                0.1 * np.sin(2 * np.pi * 440 * time_axis),
                0.1 * np.sin(2 * np.pi * 880 * time_axis),
            )
        ).astype(np.float32)
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(source_rate),
            "-ac",
            "2",
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
            "pipe:1",
        ]
        completed = subprocess.run(
            command,
            input=stereo.astype("<f4").tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        decoded = decode_audio_bytes(completed.stdout, ffmpeg_binary=self.ffmpeg)
        self.assertEqual(decoded.sample_rate, SAMPLE_RATE)
        self.assertEqual(decoded.waveform.ndim, 1)
        self.assertEqual(decoded.waveform.size, SAMPLE_RATE)

    def test_rejects_compressed_audio_over_decoded_duration_limit(self) -> None:
        long_silence = np.zeros(int(30.1 * SAMPLE_RATE), dtype=np.float32)
        encoded = self.encode(long_silence, ["-c:a", "flac", "-f", "flac"])
        self.assertLess(len(encoded), MAX_UPLOAD_BYTES)
        with self.assertRaises(AudioTooLongError):
            decode_audio_bytes(encoded, ffmpeg_binary=self.ffmpeg)

    def test_rejects_empty_oversized_and_corrupt_uploads(self) -> None:
        with self.assertRaises(AudioDecodeError):
            decode_audio_bytes(b"", ffmpeg_binary=self.ffmpeg)
        with self.assertRaises(UploadTooLargeError):
            decode_audio_bytes(b"fLaC" + b"\x00" * (MAX_UPLOAD_BYTES - 3), ffmpeg_binary=self.ffmpeg)
        with self.assertRaises(AudioDecodeError):
            decode_audio_bytes(b"fLaC-invalid", ffmpeg_binary=self.ffmpeg)


if __name__ == "__main__":
    unittest.main()
