"""Planar Y4M operations that preserve deterministic frame layout.

Plane bytes remain the public frame representation used by
:mod:`avprovenance.media.y4m`; NumPy only replaces the per-element Python
loops.  Luma arithmetic works on unsigned bytes with an explicit saturated
add/subtract (``np.clip`` on the ``uint8`` range), so values at the 0/255
boundaries clamp instead of wrapping into another color.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import MediaOperationFailure
from .y4m import Y4mClip, Y4mFrame, decode_y4m, encode_y4m


def trim_video(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    start = int(operation["start_frame"])
    count = int(operation["frame_count"])
    if start < 0 or start >= clip.frame_count:
        raise MediaOperationFailure(
            "VIDEO_TRIM_OUT_OF_RANGE",
            "The requested video trim starts outside the source.",
        )
    end = min(clip.frame_count, start + max(1, count))
    frames = clip.frames[start:end]
    if not frames:
        raise MediaOperationFailure(
            "VIDEO_TRIM_EMPTY",
            "The requested video trim produces no frames.",
        )
    return Y4mClip(
        width=clip.width,
        height=clip.height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=frames,
    )


def _crop_plane(
    plane: bytes,
    plane_width: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> bytes:
    matrix = np.frombuffer(plane, dtype=np.uint8).reshape(-1, plane_width)
    cropped = matrix[y : y + height, x : x + width]
    return np.ascontiguousarray(cropped).tobytes()


def crop_video(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    x = int(operation["x"])
    y = int(operation["y"])
    width = int(operation["width"])
    height = int(operation["height"])
    if any(value % 2 for value in (x, y, width, height)):
        raise MediaOperationFailure(
            "VIDEO_CROP_ALIGNMENT",
            "Video crop geometry must align to even pixels.",
        )
    if width < 2 or height < 2:
        raise MediaOperationFailure(
            "VIDEO_CROP_TOO_SMALL",
            "Crop width and height must be at least two pixels.",
        )
    if x + width > clip.width or y + height > clip.height:
        raise MediaOperationFailure(
            "VIDEO_CROP_OUT_OF_RANGE",
            "Crop geometry exceeds the source frame.",
        )

    chroma_x = x // 2
    chroma_y = y // 2
    chroma_width = width // 2
    chroma_height = height // 2
    frames = []
    for frame in clip.frames:
        frames.append(
            Y4mFrame(
                y_plane=_crop_plane(
                    frame.y_plane, clip.width, x, y, width, height
                ),
                u_plane=_crop_plane(
                    frame.u_plane,
                    clip.chroma_width,
                    chroma_x,
                    chroma_y,
                    chroma_width,
                    chroma_height,
                ),
                v_plane=_crop_plane(
                    frame.v_plane,
                    clip.chroma_width,
                    chroma_x,
                    chroma_y,
                    chroma_width,
                    chroma_height,
                ),
            )
        )
    return Y4mClip(
        width=width,
        height=height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=tuple(frames),
    )


def _offset_plane(plane: bytes, amount: int) -> bytes:
    matrix = np.frombuffer(plane, dtype=np.uint8)
    # Saturating arithmetic: cast to a wider type, clamp back into [0, 255].
    adjusted = np.clip(matrix.astype(np.int16) + amount, 0, 255).astype(np.uint8)
    return adjusted.tobytes()


def offset_luma(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    amount = int(operation["amount"])
    frames = []
    for frame in clip.frames:
        frames.append(
            Y4mFrame(
                y_plane=_offset_plane(frame.y_plane, amount),
                u_plane=frame.u_plane,
                v_plane=frame.v_plane,
            )
        )
    return Y4mClip(
        width=clip.width,
        height=clip.height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=tuple(frames),
    )


def _mirror_plane(plane: bytes, width: int, height: int) -> bytes:
    matrix = np.frombuffer(plane, dtype=np.uint8).reshape(height, width)
    mirrored = np.ascontiguousarray(matrix[:, ::-1])
    return mirrored.tobytes()


def mirror_video(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    del operation
    frames = tuple(
        Y4mFrame(
            y_plane=_mirror_plane(frame.y_plane, clip.width, clip.height),
            u_plane=_mirror_plane(
                frame.u_plane, clip.chroma_width, clip.chroma_height
            ),
            v_plane=_mirror_plane(
                frame.v_plane, clip.chroma_width, clip.chroma_height
            ),
        )
        for frame in clip.frames
    )
    return Y4mClip(
        width=clip.width,
        height=clip.height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=frames,
    )


def render_video(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> bytes:
    if secondary_bytes is not None:
        raise MediaOperationFailure(
            "VIDEO_SECONDARY_UNSUPPORTED",
            "The current video operation set accepts one source.",
        )
    clip = decode_y4m(primary_bytes)
    try:
        for operation in operations:
            operation_type = operation["type"]
            if operation_type == "media_trim":
                clip = trim_video(clip, operation)
            elif operation_type == "video_crop":
                clip = crop_video(clip, operation)
            elif operation_type == "video_luma_offset":
                clip = offset_luma(clip, operation)
            elif operation_type == "video_mirror":
                clip = mirror_video(clip, operation)
            else:
                raise MediaOperationFailure(
                    "UNSUPPORTED_VIDEO_OPERATION",
                    f"Unsupported video operation: {operation_type}",
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaOperationFailure(
            "VIDEO_OPERATION_INVALID",
            "The video operation payload is malformed.",
        ) from exc
    return encode_y4m(clip)
