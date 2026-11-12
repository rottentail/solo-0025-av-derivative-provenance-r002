#!/usr/bin/env python3
"""Assert the two semantics highlighted for the NumPy migration.

1. Stereo PCM: when one channel holds extreme negative samples and the other
   is quiet, gain and normalization must not alter the relationship between
   the two channels (a single shared scalar applies to both, and the extreme
   negative value saturates at -32768 instead of wrapping positive).
2. Y4M luma: values at the 0/255 boundaries must clamp when offset, never
   wrap into another color because of an integer representation change.

These assertions deliberately inspect decoded sample/byte values rather than
media summary fields.  Run with the project virtual environment.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import numpy as np  # noqa: E402

from avprovenance.media.audio_ops import (  # noqa: E402
    gain_audio,
    normalize_audio,
)
from avprovenance.media.video_ops import offset_luma  # noqa: E402
from avprovenance.media.wav import AudioClip, decode_wav, encode_wav  # noqa: E402
from avprovenance.media.y4m import Y4mClip, Y4mFrame, decode_y4m  # noqa: E402
import generate_golden as generator  # noqa: E402


def check_stereo_relationships() -> None:
    data = generator.build_audio_fixtures()["stereo_extreme"]["data"]
    clip = decode_wav(data)
    left_before, right_before = clip.channels
    assert int(left_before[0]) == -32768
    assert int(right_before[0]) == 0

    gained = gain_audio(clip, {"db": 24})
    left_gain, right_gain = (channel.astype(np.float64) for channel in gained.channels)
    # A gain > 1 must leave the extreme negative sample saturated, not wrapped
    # into a positive value.
    assert int(gained.channels[0][0]) == -32768
    assert int(gained.channels[0][8]) == 32767

    normalized = normalize_audio(clip, {"target_peak": 1.0})
    left_norm, right_norm = normalized.channels
    # Both channels share one factor, so per-sample ratios between channels
    # are preserved wherever neither sample saturates.  Compare ratios against
    # the pre-normalization float values.
    factor = 32767.0 / 32768.0  # peak is |-32768|; target_peak 1.0 -> desired 32767
    for index in range(clip.frame_count):
        expected_left = np.clip(np.rint(int(left_before[index]) * factor), -32768, 32767)
        expected_right = np.clip(np.rint(int(right_before[index]) * factor), -32768, 32767)
        assert int(left_norm[index]) == int(expected_left), (index, left_norm[index])
        assert int(right_norm[index]) == int(expected_right), (index, right_norm[index])

    # The quiet channel must remain quiet relative to the extreme channel:
    # every quiet-channel sample stays within its original tiny magnitude
    # scaled by the same factor (and never inherits the extreme channel's
    # scale).
    assert np.max(np.abs(right_norm.astype(np.int32))) <= max(
        2, round(np.max(np.abs(right_before.astype(np.float64))) * factor)
    )

    # Round trip through the encoder keeps signed values intact.
    encoded = encode_wav(normalized)
    round_trip = decode_wav(encoded)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(normalized.channels, round_trip.channels)
    )

    # Silent clip normalizes to itself (peak == 0 early return).
    silent = AudioClip(
        sample_rate=8000,
        channels=(np.zeros(8, dtype=np.int16), np.zeros(8, dtype=np.int16)),
    )
    assert normalize_audio(silent, {"target_peak": 0.5}) is silent
    print("semantics: stereo gain/normalize preserve channel relationships")


def check_luma_boundaries() -> None:
    data = generator.build_video_fixtures()["luma_boundary"]["data"]
    clip = decode_y4m(data)
    y_before = np.frombuffer(clip.frames[0].y_plane, dtype=np.uint8).reshape(
        clip.height, clip.width
    )
    assert int(y_before[0, 0]) == 0 and int(y_before[1, 0]) == 255

    up = offset_luma(clip, {"amount": 64})
    y_up = np.frombuffer(up.frames[0].y_plane, dtype=np.uint8)
    # No wraparound: 255 + 64 stays 255; small values rise by exactly 64.
    assert int(y_up[clip.width]) == 255  # row 1 was all 255
    assert int(y_up[2 * clip.width]) == 65  # row 2 was all 1
    assert int(y_up[0]) == 64  # row 0 was all 0 -> 64, never a wrapped value
    # Every output is the saturated sum, never a uint8 wrap-around.
    expected_up = np.clip(y_before.astype(np.int16) + 64, 0, 255).astype(np.uint8)
    assert np.array_equal(y_up.reshape(clip.height, clip.width), expected_up)

    down = offset_luma(clip, {"amount": -64})
    y_down = np.frombuffer(down.frames[0].y_plane, dtype=np.uint8)
    assert int(y_down[0]) == 0  # row 0 was all 0, stays 0, never 192
    assert int(y_down[3 * clip.width]) == 190  # row 3 was all 254

    # Chroma planes pass through untouched by luma offset.
    for frame_before, frame_after in zip(clip.frames, up.frames):
        assert frame_before.u_plane == frame_after.u_plane
        assert frame_before.v_plane == frame_after.v_plane

    # A full-range sweep: every byte value 0..255 appears, twice, and each
    # clamps correctly.
    sweep_bytes = bytes(range(256)) * 2
    sweep_frame = Y4mFrame(
        y_plane=sweep_bytes,
        u_plane=b"\x00" * 128,
        v_plane=b"\xff" * 128,
    )
    sweep = Y4mClip(
        width=16,
        height=32,
        rate_numerator=12,
        rate_denominator=1,
        interlace="p",
        pixel_format="C420jpeg",
        frames=(sweep_frame,),
    )
    shifted = offset_luma(sweep, {"amount": 64})
    values = np.frombuffer(shifted.frames[0].y_plane, dtype=np.uint8)
    expected = np.clip(np.arange(256, dtype=np.int16) + 64, 0, 255).astype(np.uint8)
    assert values.shape == (512,)
    assert np.array_equal(values[:256], expected)
    assert np.array_equal(values[256:], expected)  # second copy of the same sweep
    assert int(values.min()) == 64 and int(values.max()) == 255
    print("semantics: luma offset saturates at 0/255 without color wraparound")


def main() -> int:
    check_stereo_relationships()
    check_luma_boundaries()
    print("semantics: all highlighted migration semantics hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
