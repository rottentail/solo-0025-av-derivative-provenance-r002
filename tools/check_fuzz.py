#!/usr/bin/env python3
"""Randomized differential fuzzing between frozen and NumPy media kernels.

The curated golden fixtures exercise deterministic edge cases; this script
adds breadth by generating random WAV/Y4M inputs and random operation plans and
rendering each pair through both the frozen standard-library implementation
under ``tools/reference`` and the migrated package.  Every output must match
byte-for-byte.  Random seeds are fixed so the run is reproducible.

Run with the project virtual environment, e.g.
``.venv/bin/python -I tools/check_fuzz.py [iterations]``.
"""

from __future__ import annotations

import io
import random
import struct
import sys
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = PROJECT_ROOT / "tools" / "reference"
for path in (str(PROJECT_ROOT), str(REFERENCE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from avprovenance.media import apply_media_operations as numpy_render  # noqa: E402
from avprovenance_ref.media.audio_ops import render_audio as ref_render_audio  # noqa: E402
from avprovenance_ref.media.video_ops import render_video as ref_render_video  # noqa: E402


SAMPLE_RATE = 8000
INTERESTING_SAMPLES = (-32768, -32767, -16384, -3, -2, -1, 0, 1, 2, 3, 16383, 32766, 32767)
INTERESTING_BYTES = (0, 1, 2, 127, 128, 253, 254, 255)


def make_wav(channels: list[list[int]]) -> bytes:
    frame_count = len(channels[0])
    interleaved: list[int] = []
    for frame_index in range(frame_count):
        for channel in channels:
            interleaved.append(channel[frame_index])
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(len(channels))
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(struct.pack(f"<{len(interleaved)}h", *interleaved))
    return payload.getvalue()


def make_y4m(width: int, height: int, frame_count: int, rng: random.Random) -> bytes:
    header = (
        "YUV4MPEG2 "
        f"W{width} H{height} F12:1 Ip C420jpeg\n"
    ).encode("ascii")
    parts = [header]
    chroma_width = width // 2
    chroma_height = height // 2
    for _frame in range(frame_count):
        parts.append(b"FRAME\n")
        parts.append(bytes(rng.randrange(256) for _ in range(width * height)))
        parts.append(bytes(rng.randrange(256) for _ in range(chroma_width * chroma_height)))
        parts.append(bytes(rng.randrange(256) for _ in range(chroma_width * chroma_height)))
    return b"".join(parts)


def random_channel(rng: random.Random, frames: int) -> list[int]:
    values: list[int] = []
    for _ in range(frames):
        choice = rng.random()
        if choice < 0.25:
            values.append(rng.choice(INTERESTING_SAMPLES))
        elif choice < 0.5:
            values.append(rng.randrange(-50, 50))
        else:
            values.append(rng.randrange(-32768, 32768))
    return values


def random_audio_plan(rng: random.Random, frame_count: int, channels: int) -> tuple[list[dict], dict | None]:
    operations: list[dict] = []
    secondary_fixture = None
    current_frames = frame_count
    steps = rng.randrange(1, 5)
    for _ in range(steps):
        kind = rng.choice(
            ["gain", "normalize", "trim", "mix"]
            if secondary_fixture is None
            else ["gain", "normalize", "trim"]
        )
        if kind == "gain":
            operations.append({"type": "audio_gain", "db": rng.uniform(-48, 24)})
        elif kind == "normalize":
            operations.append(
                {"type": "audio_normalize", "target_peak": rng.uniform(0.05, 1.0)}
            )
        elif kind == "trim":
            start = rng.randrange(0, current_frames)
            duration = rng.randrange(1, current_frames - start + 1)
            operations.append(
                {
                    "type": "media_trim",
                    "start_seconds": start / SAMPLE_RATE,
                    "duration_seconds": duration / SAMPLE_RATE,
                }
            )
            current_frames = min(current_frames, start + duration) - start
        else:
            secondary_frames = rng.randrange(1, frame_count + 5)
            secondary_channels = [
                random_channel(rng, secondary_frames) for _ in range(channels)
            ]
            secondary_fixture = {"data": make_wav(secondary_channels)}
            operations.append(
                {
                    "type": "audio_mix",
                    "secondary_gain": rng.uniform(-48, 24),
                    "primary_weight": rng.uniform(0, 1.5),
                    "secondary_weight": rng.uniform(0, 1.5),
                }
            )
    return operations, secondary_fixture


def random_video_plan(rng: random.Random, width: int, height: int, frame_count: int) -> list[dict]:
    operations: list[dict] = []
    current_width, current_height = width, height
    current_frames = frame_count
    for _ in range(rng.randrange(1, 5)):
        kind = rng.choice(["trim", "crop", "luma", "mirror"])
        if kind == "trim":
            start = rng.randrange(0, current_frames)
            count = rng.randrange(1, current_frames - start + 1)
            operations.append(
                {"type": "media_trim", "start_frame": start, "frame_count": count}
            )
            current_frames = min(current_frames, start + count)
        elif kind == "crop" and current_width >= 4 and current_height >= 4:
            crop_width = rng.randrange(2, current_width + 1) & ~1
            crop_height = rng.randrange(2, current_height + 1) & ~1
            x = rng.randrange(0, current_width - crop_width + 1) & ~1
            y = rng.randrange(0, current_height - crop_height + 1) & ~1
            operations.append(
                {
                    "type": "video_crop",
                    "x": x,
                    "y": y,
                    "width": crop_width,
                    "height": crop_height,
                }
            )
            current_width, current_height = crop_width, crop_height
        elif kind == "luma":
            operations.append(
                {"type": "video_luma_offset", "amount": rng.randrange(-64, 65)}
            )
        else:
            operations.append({"type": "video_mirror"})
    return operations


def compare_audio(rng: random.Random, index: int) -> None:
    channels = rng.choice((1, 2))
    frame_count = rng.randrange(8, 128)
    primary = make_wav([random_channel(rng, frame_count) for _ in range(channels)])
    operations, secondary_fixture = random_audio_plan(rng, frame_count, channels)
    secondary = secondary_fixture["data"] if secondary_fixture else None

    try:
        expected = ref_render_audio(primary, operations, secondary)
    except Exception as ref_exc:  # noqa: BLE001
        try:
            numpy_render(primary, operations, secondary)
        except Exception as numpy_exc:  # noqa: BLE001
            # The frozen reference is a separate package, so compare the
            # stable domain contract (error code) rather than class identity.
            assert getattr(ref_exc, "code", None) == getattr(numpy_exc, "code", None), (
                f"audio[{index}] error code differs: {ref_exc!r} vs {numpy_exc!r}"
            )
            assert str(ref_exc) == str(numpy_exc)
            return
        raise AssertionError(f"audio[{index}] reference failed but numpy succeeded")

    outcome = numpy_render(primary, operations, secondary)
    assert outcome.data == expected, (
        f"audio[{index}] bytes differ for operations={operations}"
    )


def compare_video(rng: random.Random, index: int) -> None:
    width = rng.choice((2, 4, 8, 16, 32, 64))
    height = rng.choice((2, 4, 8, 12, 36))
    frame_count = rng.randrange(1, 8)
    primary = make_y4m(width, height, frame_count, rng)
    operations = random_video_plan(rng, width, height, frame_count)

    try:
        expected = ref_render_video(primary, operations, None)
    except Exception as ref_exc:  # noqa: BLE001
        try:
            numpy_render(primary, operations, None)
        except Exception as numpy_exc:  # noqa: BLE001
            assert getattr(ref_exc, "code", None) == getattr(numpy_exc, "code", None), (
                f"video[{index}] error code differs: {ref_exc!r} vs {numpy_exc!r}"
            )
            assert str(ref_exc) == str(numpy_exc)
            return
        raise AssertionError(f"video[{index}] reference failed but numpy succeeded")

    outcome = numpy_render(primary, operations, None)
    assert outcome.data == expected, (
        f"video[{index}] bytes differ for operations={operations}"
    )


def main() -> int:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = random.Random(20261112)
    np.random.seed(20261112)
    for index in range(iterations):
        compare_audio(rng, index)
        compare_video(rng, index)
    print(f"fuzz: {iterations} audio and {iterations} video random plans matched byte-for-byte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
