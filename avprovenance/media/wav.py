"""PCM WAV inspection, decoding, and deterministic encoding.

The sample plane is held in NumPy ``int16`` arrays while the RIFF framing stays
in the standard-library :mod:`wave` module.  Decoding and encoding are
explicitly little-endian so the produced bytes match the previous
array-based implementation on every host byte order.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..errors import UnsupportedMedia
from ..models import MediaDescriptor


@dataclass(frozen=True)
class AudioClip:
    sample_rate: int
    channels: tuple[np.ndarray, ...]

    @property
    def frame_count(self) -> int:
        return int(self.channels[0].shape[0]) if self.channels else 0

    @property
    def channel_count(self) -> int:
        return len(self.channels)


def _open_reader(data: bytes) -> wave.Wave_read:
    try:
        reader = wave.open(io.BytesIO(data), "rb")
    except (EOFError, wave.Error) as exc:
        raise UnsupportedMedia(
            "The audio payload is not a readable RIFF/WAVE file.",
            details={"reason": str(exc)},
        ) from exc
    return reader


def _validate_params(reader: wave.Wave_read) -> None:
    if reader.getcomptype() != "NONE":
        raise UnsupportedMedia(
            "Compressed WAV payloads are not supported.",
            details={"compression": reader.getcomptype()},
        )
    sample_width = reader.getsampwidth()
    if sample_width != 2:
        raise UnsupportedMedia(
            "Only signed 16-bit PCM WAV is supported.",
            details={"sample_width_bytes": sample_width},
        )
    channels = reader.getnchannels()
    if channels not in (1, 2):
        raise UnsupportedMedia(
            "WAV input must contain one or two channels.",
            details={"channels": channels},
        )
    sample_rate = reader.getframerate()
    if sample_rate < 8000 or sample_rate > 192000:
        raise UnsupportedMedia(
            "WAV sample rate is outside the supported range.",
            details={"sample_rate": sample_rate},
        )
    if reader.getnframes() <= 0:
        raise UnsupportedMedia("WAV input has no sample frames.")


def inspect_wav(data: bytes) -> MediaDescriptor:
    reader = _open_reader(data)
    try:
        _validate_params(reader)
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        channels = reader.getnchannels()
        return MediaDescriptor(
            kind="audio",
            mime_type="audio/wav",
            container="wav",
            properties={
                "codec": "pcm_s16le",
                "sample_rate": sample_rate,
                "channels": channels,
                "frame_count": frame_count,
                "duration_seconds": round(frame_count / sample_rate, 9),
            },
        )
    finally:
        reader.close()


def decode_wav(data: bytes) -> AudioClip:
    reader = _open_reader(data)
    try:
        _validate_params(reader)
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        raw = reader.readframes(reader.getnframes())
    finally:
        reader.close()

    interleaved = np.frombuffer(raw, dtype="<i2")
    if interleaved.shape[0] % channels != 0:
        raise UnsupportedMedia("WAV sample data is not frame-aligned.")
    frame_count = interleaved.shape[0] // channels
    planar = interleaved.reshape(frame_count, channels)
    # Native int16 arrays feed the NumPy kernels; byte order only matters at
    # encode time where little-endian output is requested explicitly.
    channel_values = tuple(
        np.ascontiguousarray(planar[:, index], dtype=np.int16)
        for index in range(channels)
    )
    return AudioClip(sample_rate=sample_rate, channels=channel_values)


def _interleave(channels: tuple[np.ndarray, ...], frame_count: int) -> np.ndarray:
    channel_count = len(channels)
    stacked = np.stack(channels, axis=1)
    interleaved = stacked.reshape(frame_count * channel_count)
    return np.ascontiguousarray(interleaved, dtype="<i2")


def encode_wav(clip: AudioClip) -> bytes:
    if not clip.channels:
        raise UnsupportedMedia("Cannot encode an audio clip without channels.")
    if any(channel.shape[0] != clip.frame_count for channel in clip.channels):
        raise UnsupportedMedia("Audio channels do not share one frame count.")
    values = _interleave(clip.channels, clip.frame_count)
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(len(clip.channels))
        writer.setsampwidth(2)
        writer.setframerate(clip.sample_rate)
        writer.writeframes(values.tobytes())
    return payload.getvalue()


def replace_channels(
    clip: AudioClip,
    channels: Iterable[np.ndarray],
) -> AudioClip:
    values = tuple(np.asarray(channel, dtype=np.int16) for channel in channels)
    if not values:
        raise UnsupportedMedia("Audio operations produced no channels.")
    expected = values[0].shape[0]
    if any(channel.shape[0] != expected for channel in values):
        raise UnsupportedMedia("Audio operation produced misaligned channels.")
    return AudioClip(sample_rate=clip.sample_rate, channels=values)
