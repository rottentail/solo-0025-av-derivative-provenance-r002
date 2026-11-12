# AV Derivative Provenance

AV Derivative Provenance is a Python backend for deterministic audio and video
derivation. It accepts PCM WAV and planar Y4M media, applies bounded
operations, records content digests and lineage, and reproduces a completed
result on demand to verify that the stored bytes still match the plan.

The application runs on the Python standard runtime plus one pinned third-party
dependency, NumPy, which powers the PCM and planar-YUV compute kernels. It does
not need FFmpeg, network access at runtime, credentials, or an online database.

## Install (reproducible)

Runtime dependencies are pinned with SHA-256 hashes in `requirements.txt`, so
installs verify the exact wheels they download. Create a virtual environment
and install from the hash-pinned file:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
```

On minimal interpreters that ship without `ensurepip` (for example a Debian
system Python), bootstrap pip into the virtual environment once with
`python -I get-pip.py` (downloaded from <https://bootstrap.pypa.io/get-pip/>)
and then run the same `pip install --require-hashes` command. The hash-to-file
mapping is recorded in `tools/numpy-hashes.txt`.

## What it does

- Ingests one- or two-channel signed 16-bit PCM WAV.
- Ingests planar 4:2:0 YUV4MPEG2 in the `C420` family.
- Applies audio trim, gain, normalization, and two-source mixing as vectorized
  NumPy kernels.
- Applies video trim, even-aligned crop, luma offset, and horizontal mirror as
  vectorized NumPy kernels.
- Persists immutable assets, rendition state, provenance edges, and integrity
  checks.
- Replays the stored operation plan and compares both bytes and media
  descriptors.

## Quick start

```bash
.venv/bin/python -m compileall -q avprovenance scripts
.venv/bin/python -m avprovenance --host 127.0.0.1 --port 8765 --data-dir .avprovenance-data
```

The startup line includes the active host, port, and data directory. Stop the
process with `Ctrl+C`.

## HTTP endpoints

### Health

```http
GET /health
```

### Upload an asset

```http
POST /v1/assets
Content-Type: application/json

{
  "filename": "source.wav",
  "data_base64": "<strict base64>"
}
```

### Read an asset

```http
GET /v1/assets/{asset_id}
```

### Create a rendition

```http
POST /v1/renditions
Content-Type: application/json

{
  "primary_asset_id": "asset_...",
  "secondary_asset_id": null,
  "output_name": "derived.wav",
  "operations": [
    {
      "type": "media_trim",
      "start_seconds": 0.1,
      "duration_seconds": 0.3
    },
    {"type": "audio_gain", "db": -3},
    {"type": "audio_normalize", "target_peak": 0.8}
  ]
}
```

Rendition creation is synchronous at the HTTP boundary. The durable transition
history still records `queued`, `running`, and the terminal state.

Audio-only operations:

- `media_trim` with `start_seconds` and `duration_seconds`
- `audio_gain` with `db`
- `audio_normalize` with `target_peak`
- `audio_mix` with `secondary_gain`, `primary_weight`, and `secondary_weight`

Video-only operations:

- `media_trim` with `start_frame` and `frame_count`
- `video_crop` with even `x`, `y`, `width`, and `height`
- `video_luma_offset` with `amount`
- `video_mirror`

### Read rendition state

```http
GET /v1/renditions/{rendition_id}
```

### Read lineage

```http
GET /v1/renditions/{rendition_id}/lineage
```

### Verify reproduction

```http
POST /v1/renditions/{rendition_id}/integrity-checks
Content-Type: application/json

{}
```

The successful response has `status: "verified"`. A byte or descriptor
difference returns HTTP `409` with code `INTEGRITY_MISMATCH`; the mismatch
record remains durable.

## Media operation examples

Audio mixing plan:

```json
[
  {"type": "media_trim", "start_seconds": 0.05, "duration_seconds": 0.3},
  {"type": "audio_gain", "db": -3},
  {
    "type": "audio_mix",
    "secondary_gain": -6,
    "primary_weight": 0.75,
    "secondary_weight": 0.4
  },
  {"type": "audio_normalize", "target_peak": 0.8}
]
```

Video plan:

```json
[
  {"type": "media_trim", "start_frame": 1, "frame_count": 4},
  {
    "type": "video_crop",
    "x": 8,
    "y": 6,
    "width": 32,
    "height": 18
  },
  {"type": "video_luma_offset", "amount": 12},
  {"type": "video_mirror"}
]
```

## Data layout

```text
.avprovenance-data/
├── state.json
└── content/
    └── <digest-prefix>/
        └── <digest-prefix>/
            └── <sha256>
```

`state.json` is replaced atomically. Binary objects are content-addressed by
SHA-256 and written through temporary siblings before replacement.

## Source layout

```text
avprovenance/
├── application.py
├── http_api.py
├── models.py
├── validation.py
├── media/
├── renditions/
├── storage/
└── transport/
```

The HTTP boundary calls the application layer. The application coordinates
plan handling, media operations, and persistence. The media layer does not
depend on HTTP or the repository.

## Workflow checks

The baseline intentionally has no automated test suite. Tests are deferred to
the later task stage, which will add red/green verification coverage. The
current production workflow checks (run inside the virtual environment) are:

```bash
.venv/bin/python scripts/workflow_checks.py ingest-media-asset
.venv/bin/python scripts/workflow_checks.py execute-rendition
.venv/bin/python scripts/workflow_checks.py verify-reproduction
```

Each check starts a temporary HTTP server, generates media bytes in memory,
exercises the public endpoints, and shuts the server down.

### NumPy migration parity checks

The `tools/` directory holds the frozen pre-NumPy reference implementation and
checks that prove the NumPy kernels reproduce its exact bytes rather than just
matching durations or descriptor fields:

```bash
python3 -I tools/generate_golden.py            # stdlib-only golden generation
.venv/bin/python -I tools/check_parity.py             # byte/descriptor/error parity
.venv/bin/python -I tools/check_fuzz.py 400           # randomized differential runs
.venv/bin/python -I tools/check_semantics.py          # stereo and luma edge semantics
.venv/bin/python -I tools/check_cross_implementation.py
```

The cross-implementation check renders durable renditions with the frozen
implementation and confirms the migrated application reproduces them as
`verified` through the integrity-check entry point. See `tools/README.md`.

## Environment

No environment variables are required. Runtime choices use `--host`, `--port`,
and `--data-dir`. Incoming request bodies are limited to 36 MiB, decoded media
payloads to 24 MiB, WAV geometry to one or two channels, and Y4M geometry to
1920x1080 with at most 240 frames.

The only third-party runtime package is NumPy (`requirements.txt`, hash-pinned
to `numpy==2.2.6`); everything else uses the Python 3.11+ standard library.
