#!/usr/bin/env python3
"""Generate golden media fixtures and pre-migration render outputs.

This script runs against the frozen stdlib implementation copied under
tools/reference/avprovenance_ref.  It must execute with a plain interpreter that
has no third-party packages so the golden bytes come exclusively from the
original standard-library kernels.

Outputs land in tools/golden/<fixture>/.  The parity check re-renders the same
fixtures with the migrated NumPy kernels and demands byte-for-byte equality;
matching only lengths or descriptor fields is treated as failure.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import sys
import wave
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
REFERENCE_ROOT = TOOLS_DIR / "reference"
GOLDEN_ROOT = TOOLS_DIR / "golden"
sys.path.insert(0, str(REFERENCE_ROOT))

from avprovenance_ref.media.audio_ops import render_audio  # noqa: E402
from avprovenance_ref.media.inspector import inspect_media  # noqa: E402
from avprovenance_ref.media.video_ops import render_video  # noqa: E402


SAMPLE_RATE = 8000


def make_wav(channels: list[list[int]], sample_rate: int = SAMPLE_RATE) -> bytes:
    frame_count = len(channels[0])
    interleaved: list[int] = []
    for frame_index in range(frame_count):
        for channel in channels:
            interleaved.append(channel[frame_index])
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(len(channels))
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(interleaved)}h", *interleaved))
    return payload.getvalue()


def make_y4m(
    width: int,
    height: int,
    y_rows: list[list[int]] | None = None,
    u_rows: list[list[int]] | None = None,
    v_rows: list[list[int]] | None = None,
    *,
    frame_count: int = 3,
    rate_numerator: int = 12,
    rate_denominator: int = 1,
    pixel_format: str = "C420jpeg",
) -> bytes:
    header = (
        "YUV4MPEG2 "
        f"W{width} H{height} "
        f"F{rate_numerator}:{rate_denominator} "
        f"Ip {pixel_format}\n"
    ).encode("ascii")
    chroma_width = width // 2
    chroma_height = height // 2
    parts = [header]
    for frame_index in range(frame_count):
        parts.append(b"FRAME\n")
        if y_rows is not None:
            y_plane = bytes(value for row in y_rows for value in row)
        else:
            y_plane = bytes(
                (x * 7 + y * 13 + frame_index * 17) % 256
                for y in range(height)
                for x in range(width)
            )
        if u_rows is not None:
            u_plane = bytes(value for row in u_rows for value in row)
        else:
            u_plane = bytes(
                (96 + x * 3 + frame_index * 5) % 256
                for _y in range(chroma_height)
                for x in range(chroma_width)
            )
        if v_rows is not None:
            v_plane = bytes(value for row in v_rows for value in row)
        else:
            v_plane = bytes(
                (160 + y * 3 + frame_index * 7) % 256
                for y in range(chroma_height)
                for _x in range(chroma_width)
            )
        parts.extend((y_plane, u_plane, v_plane))
    return b"".join(parts)


def sine(frames: int, frequency: float, amplitude: int, phase: float = 0.0) -> list[int]:
    return [
        int(amplitude * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE + phase))
        for index in range(frames)
    ]


def build_audio_fixtures() -> dict[str, dict[str, object]]:
    frames = 96  # 12 ms at 8 kHz, keeps samples easy to reason about

    # Stereo: one channel carries extreme negative / boundary samples while the
    # other stays quiet.  Gain and normalization must preserve the relationship
    # between the two channels exactly.
    extreme_left = [
        -32768,
        -32767,
        -16384,
        -1,
        0,
        1,
        16383,
        32766,
        32767,
    ] + sine(frames - 9, 440, 30000, phase=math.pi)
    quiet_right = [0, 0, 1, -1, 0, 2, -2, 1, 0] + sine(
        frames - 9, 440, 4, phase=math.pi / 2
    )
    stereo_extreme = make_wav([extreme_left, quiet_right])

    # Half-even rounding boundary values: .5 ties must round like Python round().
    boundary_left = [
        -32767,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        -5,
        -4,
    ] + sine(frames - 12, 330, 12000)
    boundary_right = sine(frames, 220, 9000)
    stereo_rounding = make_wav([boundary_left, boundary_right])

    mono_tone = make_wav([sine(frames, 440, 10000)])
    mono_secondary = make_wav([sine(frames, 220, 7000)])
    stereo_tone = make_wav(
        [sine(frames, 440, 12000), sine(frames, 330, 6000, phase=0.7)]
    )
    stereo_secondary = make_wav(
        [sine(frames, 550, 5000), sine(frames, 110, 9000, phase=1.3)]
    )

    # Entirely silent clip exercises the normalize peak == 0 early return.
    silent = make_wav([[0] * frames, [0] * frames])

    # Different lengths exercise zero-padding inside audio_mix.
    short_primary = make_wav([sine(40, 440, 11000), sine(40, 330, 8000)])
    long_secondary = make_wav([sine(72, 220, 9000), sine(72, 110, 6000)])

    return {
        "stereo_extreme": {"kind": "audio", "data": stereo_extreme},
        "stereo_rounding": {"kind": "audio", "data": stereo_rounding},
        "mono_tone": {"kind": "audio", "data": mono_tone},
        "mono_secondary": {"kind": "audio", "data": mono_secondary},
        "stereo_tone": {"kind": "audio", "data": stereo_tone},
        "stereo_secondary": {"kind": "audio", "data": stereo_secondary},
        "silent": {"kind": "audio", "data": silent},
        "short_primary": {"kind": "audio", "data": short_primary},
        "long_secondary": {"kind": "audio", "data": long_secondary},
    }


def build_video_fixtures() -> dict[str, dict[str, object]]:
    width, height = 16, 12
    # Luma rows that sit on both clamp boundaries.
    boundary_y = [
        [0] * width,
        [255] * width,
        [1] * width,
        [254] * width,
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 250, 254, 255],
    ]
    while len(boundary_y) < height:
        boundary_y.append([(index * 17) % 256 for index in range(width)])
    chroma_width = width // 2
    boundary_u = [
        [0] * chroma_width,
        [255] * chroma_width,
        [250, 251, 252, 253, 254, 255, 0, 1],
    ]
    boundary_v = [
        [255] * chroma_width,
        [0] * chroma_width,
        [5, 4, 3, 2, 1, 0, 255, 254],
    ]
    while len(boundary_u) < height // 2:
        boundary_u.append([(index * 23 + 7) % 256 for index in range(chroma_width)])
        boundary_v.append([(index * 29 + 11) % 256 for index in range(chroma_width)])

    luma_boundary = make_y4m(width, height, boundary_y, boundary_u, boundary_v)
    pattern = make_y4m(64, 36)
    tiny = make_y4m(2, 2, frame_count=2)
    odd_like = make_y4m(10, 6, frame_count=4)
    pixel_formats = make_y4m(8, 8, frame_count=2, pixel_format="C420p")

    return {
        "luma_boundary": {"kind": "video", "data": luma_boundary},
        "pattern": {"kind": "video", "data": pattern},
        "tiny": {"kind": "video", "data": tiny},
        "odd_like": {"kind": "video", "data": odd_like},
        "pixel_formats": {"kind": "video", "data": pixel_formats},
    }


AUDIO_PLANS: dict[str, list[dict[str, object]]] = {
    "gain_up": [{"type": "audio_gain", "db": 6.5}],
    "gain_down": [{"type": "audio_gain", "db": -12.25}],
    "gain_max": [{"type": "audio_gain", "db": 24}],
    "normalize_full": [{"type": "audio_normalize", "target_peak": 1.0}],
    "normalize_quiet": [{"type": "audio_normalize", "target_peak": 0.05}],
    "normalize_mid": [{"type": "audio_normalize", "target_peak": 0.8}],
    "trim_head": [
        {"type": "media_trim", "start_seconds": 0.002, "duration_seconds": 0.005}
    ],
    "trim_tail": [
        {"type": "media_trim", "start_seconds": 0.008, "duration_seconds": 0.05}
    ],
    "gain_then_normalize": [
        {"type": "audio_gain", "db": -6},
        {"type": "audio_normalize", "target_peak": 0.9},
    ],
    "normalize_silent": [{"type": "audio_normalize", "target_peak": 0.5}],
    "mix_basic": [
        {
            "type": "audio_mix",
            "secondary_gain": -6,
            "primary_weight": 0.75,
            "secondary_weight": 0.4,
        }
    ],
    "mix_full_chain": [
        {"type": "media_trim", "start_seconds": 0.001, "duration_seconds": 0.01},
        {"type": "audio_gain", "db": -3},
        {
            "type": "audio_mix",
            "secondary_gain": 3.5,
            "primary_weight": 1.0,
            "secondary_weight": 0.6,
        },
        {"type": "audio_normalize", "target_peak": 0.8},
    ],
    "mix_extremes": [
        {
            "type": "audio_mix",
            "secondary_gain": 12,
            "primary_weight": 1.5,
            "secondary_weight": 1.5,
        },
        {"type": "audio_normalize", "target_peak": 1.0},
    ],
    "mix_padding": [
        {
            "type": "audio_mix",
            "secondary_gain": -1.5,
            "primary_weight": 0.9,
            "secondary_weight": 1.1,
        }
    ],
    "trim_gain_mix_normalize": [
        {"type": "media_trim", "start_seconds": 0.0, "duration_seconds": 0.006},
        {"type": "audio_gain", "db": 2.25},
        {
            "type": "audio_mix",
            "secondary_gain": -10,
            "primary_weight": 0.5,
            "secondary_weight": 0.25,
        },
        {"type": "audio_normalize", "target_peak": 0.65},
    ],
}

VIDEO_PLANS: dict[str, list[dict[str, object]]] = {
    "trim": [{"type": "media_trim", "start_frame": 1, "frame_count": 2}],
    "crop": [{"type": "video_crop", "x": 4, "y": 4, "width": 8, "height": 6}],
    "crop_full": [{"type": "video_crop", "x": 0, "y": 0, "width": 16, "height": 12}],
    "luma_up_clamp": [{"type": "video_luma_offset", "amount": 64}],
    "luma_down_clamp": [{"type": "video_luma_offset", "amount": -64}],
    "luma_small": [{"type": "video_luma_offset", "amount": 7}],
    "mirror": [{"type": "video_mirror"}],
    "full_chain": [
        {"type": "media_trim", "start_frame": 0, "frame_count": 2},
        {"type": "video_crop", "x": 2, "y": 2, "width": 10, "height": 8},
        {"type": "video_luma_offset", "amount": -17},
        {"type": "video_mirror"},
    ],
    "mirror_then_clamp": [
        {"type": "video_mirror"},
        {"type": "video_luma_offset", "amount": 64},
    ],
    "crop_then_crop": [
        {"type": "video_crop", "x": 2, "y": 2, "width": 12, "height": 8},
        {"type": "video_crop", "x": 2, "y": 0, "width": 8, "height": 6},
    ],
}


# Which secondary fixture (if any) each plan needs.
AUDIO_SECONDARY = {
    "mix_basic": "stereo_secondary",
    "mix_full_chain": "stereo_secondary",
    "mix_extremes": "stereo_secondary",
    "mix_padding": "long_secondary",
    "trim_gain_mix_normalize": "mono_secondary",
}

# Plans restricted to fixtures whose geometry fits them.
VIDEO_PLAN_FIXTURES = {
    "crop": ("luma_boundary", "pattern", "odd_like", "pixel_formats"),
    "crop_full": ("luma_boundary",),
    "full_chain": ("luma_boundary",),
    "trim": ("luma_boundary", "pattern", "tiny", "odd_like", "pixel_formats"),
    "luma_up_clamp": ("luma_boundary", "pattern", "tiny", "odd_like", "pixel_formats"),
    "luma_down_clamp": ("luma_boundary", "pattern", "tiny", "odd_like", "pixel_formats"),
    "luma_small": ("luma_boundary", "pattern", "tiny", "odd_like", "pixel_formats"),
    "mirror": ("luma_boundary", "pattern", "tiny", "odd_like", "pixel_formats"),
    "mirror_then_clamp": ("luma_boundary", "pattern", "odd_like", "pixel_formats"),
    "crop_then_crop": ("luma_boundary",),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    fixtures: dict[str, dict[str, object]] = {}
    fixtures.update(build_audio_fixtures())
    fixtures.update(build_video_fixtures())

    manifest: dict[str, object] = {
        "fixtures": {},
        "audio_renders": [],
        "video_renders": [],
    }

    GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
    for name, fixture in fixtures.items():
        fixture_dir = GOLDEN_ROOT / name
        fixture_dir.mkdir(parents=True, exist_ok=True)
        suffix = "wav" if fixture["kind"] == "audio" else "y4m"
        source_path = fixture_dir / f"source.{suffix}"
        source_bytes = fixture["data"]
        source_path.write_bytes(source_bytes)
        descriptor = inspect_media(source_bytes).to_dict()
        manifest["fixtures"][name] = {
            "kind": fixture["kind"],
            "source_file": source_path.name,
            "sha256": _sha256(source_bytes),
            "descriptor": descriptor,
        }

    # Audio renders.  Every audio plan is exercised against every audio fixture
    # where a secondary source is available and channel counts line up.
    for plan_name, operations in AUDIO_PLANS.items():
        secondary_name = AUDIO_SECONDARY.get(plan_name)
        for fixture_name, fixture in fixtures.items():
            if fixture["kind"] != "audio":
                continue
            secondary_bytes = None
            if secondary_name is not None:
                secondary = fixtures[secondary_name]
                primary_channels = manifest["fixtures"][fixture_name]["descriptor"][
                    "properties"
                ]["channels"]
                secondary_channels = manifest["fixtures"][secondary_name][
                    "descriptor"
                ]["properties"]["channels"]
                if primary_channels != secondary_channels:
                    continue
                secondary_bytes = secondary["data"]
            try:
                rendered = render_audio(fixture["data"], operations, secondary_bytes)
            except Exception as exc:  # noqa: BLE001 - record error expectations too
                manifest["audio_renders"].append(
                    {
                        "fixture": fixture_name,
                        "plan": plan_name,
                        "secondary": secondary_name,
                        "error": type(exc).__name__,
                        "code": getattr(exc, "code", None),
                    }
                )
                continue
            output_name = f"{plan_name}.wav"
            output_path = GOLDEN_ROOT / fixture_name / output_name
            output_path.write_bytes(rendered)
            manifest["audio_renders"].append(
                {
                    "fixture": fixture_name,
                    "plan": plan_name,
                    "secondary": secondary_name,
                    "file": output_name,
                    "sha256": _sha256(rendered),
                    "length": len(rendered),
                    "descriptor": inspect_media(rendered).to_dict(),
                }
            )

    # Video renders.
    for plan_name, operations in VIDEO_PLANS.items():
        for fixture_name in VIDEO_PLAN_FIXTURES[plan_name]:
            fixture = fixtures[fixture_name]
            try:
                rendered = render_video(fixture["data"], operations, None)
            except Exception as exc:  # noqa: BLE001
                manifest["video_renders"].append(
                    {
                        "fixture": fixture_name,
                        "plan": plan_name,
                        "error": type(exc).__name__,
                        "code": getattr(exc, "code", None),
                    }
                )
                continue
            output_name = f"{plan_name}.y4m"
            output_path = GOLDEN_ROOT / fixture_name / output_name
            output_path.write_bytes(rendered)
            manifest["video_renders"].append(
                {
                    "fixture": fixture_name,
                    "plan": plan_name,
                    "file": output_name,
                    "sha256": _sha256(rendered),
                    "length": len(rendered),
                    "descriptor": inspect_media(rendered).to_dict(),
                }
            )

    manifest_path = GOLDEN_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    audio_count = sum(
        1 for item in manifest["audio_renders"] if "error" not in item
    )
    video_count = sum(
        1 for item in manifest["video_renders"] if "error" not in item
    )
    print(
        f"golden fixtures: {len(fixtures)} "
        f"audio renders: {audio_count} video renders: {video_count} "
        f"manifest: {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
