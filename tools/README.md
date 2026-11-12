# NumPy migration verification tooling

The production package under `avprovenance/` has a single media processing
implementation: the NumPy kernels.  Nothing in this directory is imported by
the application or exposed through an alternative processing chain.  These
scripts exist to prove that the NumPy migration preserved the exact byte-level
computation of the previous standard-library kernels.

## Layout

- `reference/avprovenance_ref/` — frozen copy of the pre-migration
  standard-library media package (PCM lists, per-element plane loops).  It is
  used only as an oracle by the checks below and is never imported from
  `avprovenance/`.
- `generate_golden.py` — renders edge-case-heavy WAV/Y4M fixtures through the
  frozen reference and stores the outputs plus a SHA-256 manifest under
  `golden/`.  Runs on a plain standard-library interpreter.
- `check_parity.py` — re-renders every golden plan through the migrated
  `avprovenance.media.apply_media_operations` entry point (the same call used
  by the rendition executor and the integrity verifier) and requires
  byte-for-byte equality, equal media descriptors, and equal stable error
  codes.  Length or descriptor equality alone is a failure.
- `check_fuzz.py` — fixed-seed randomized differential testing of random
  stereo/mono PCM and C420 inputs across random multi-operation plans.
- `check_cross_implementation.py` — builds durable renditions (state file plus
  content-addressed blobs) with the frozen implementation, reopens the data
  directory with the migrated application, and runs the real integrity-check
  use case to confirm the recorded outputs replay as `verified`.
- `check_semantics.py` — direct assertions for the two highlighted semantics:
  stereo extreme-negative vs. quiet channel relationships under gain and
  normalization, and saturated (non-wrapping) luma offsets at 0/255.
- `numpy-hashes.txt` — maps every SHA-256 hash pinned in the root
  `requirements.txt` to its NumPy distribution file.
- `golden/` — generated fixtures, golden render bytes, and `manifest.json`.

## Running the checks

```bash
# Golden bytes only need the standard library:
python3 -I tools/generate_golden.py

# The remaining checks need the project virtual environment (NumPy installed):
.venv/bin/python -I tools/check_parity.py
.venv/bin/python -I tools/check_fuzz.py 400
.venv/bin/python -I tools/check_semantics.py
.venv/bin/python -I tools/check_cross_implementation.py
```

The golden directory is regenerated from the frozen reference on demand and is
not required at application runtime.
