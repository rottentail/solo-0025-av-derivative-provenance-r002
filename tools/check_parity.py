#!/usr/bin/env python3
"""Compare NumPy kernels against the frozen standard-library golden outputs.

For every fixture and plan recorded in ``tools/golden/manifest.json`` this
check:

* re-renders through the migrated :mod:`avprovenance.media` entry points
  (``apply_media_operations`` — the same call used by the rendition executor
  and integrity verifier),
* requires byte-for-byte equality with the pre-migration output, including the
  WAV/Y4M container framing, and not merely equal lengths or descriptor
  fields,
* requires equal media descriptors, and
* requires the same stable error code for plans that must fail.

Run with the project virtual environment (NumPy available), e.g.
``.venv/bin/python -I tools/check_parity.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_ROOT = PROJECT_ROOT / "tools" / "golden"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avprovenance.media import apply_media_operations  # noqa: E402
from avprovenance.media.errors import MediaOperationFailure  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(manifest: dict, name: str, filename: str) -> bytes:
    return (GOLDEN_ROOT / name / filename).read_bytes()


def main() -> int:
    manifest = json.loads((GOLDEN_ROOT / "manifest.json").read_text())
    failures: list[str] = []
    checked_bytes = 0
    render_count = 0
    error_count = 0

    def run_case(
        *,
        fixture_name: str,
        plan_name: str,
        operations: list[dict],
        secondary_name: str | None,
        record: dict,
    ) -> None:
        nonlocal checked_bytes, render_count, error_count
        primary = _load(
            manifest,
            fixture_name,
            manifest["fixtures"][fixture_name]["source_file"],
        )
        secondary = (
            _load(
                manifest,
                secondary_name,
                manifest["fixtures"][secondary_name]["source_file"],
            )
            if secondary_name
            else None
        )
        if "error" in record:
            try:
                apply_media_operations(primary, operations, secondary)
            except MediaOperationFailure as exc:
                error_count += 1
                if exc.code != record["code"]:
                    failures.append(
                        f"{fixture_name}/{plan_name}: expected error code "
                        f"{record['code']}, got {exc.code}"
                    )
                return
            failures.append(
                f"{fixture_name}/{plan_name}: expected failure "
                f"{record['code']} but rendering succeeded"
            )
            return

        outcome = apply_media_operations(primary, operations, secondary)
        render_count += 1
        golden = _load(manifest, fixture_name, record["file"])
        checked_bytes += len(outcome.data)
        if len(outcome.data) != len(golden):
            failures.append(
                f"{fixture_name}/{plan_name}: length differs "
                f"{len(golden)} -> {len(outcome.data)}"
            )
            return
        if outcome.data != golden:
            # Locate the first differing byte to make regressions diagnosable.
            first = next(
                index
                for index, (old, new) in enumerate(zip(golden, outcome.data))
                if old != new
            )
            failures.append(
                f"{fixture_name}/{plan_name}: byte mismatch at offset {first} "
                f"(golden 0x{golden[first]:02x}, got 0x{outcome.data[first]:02x}); "
                f"digest golden={record['sha256']} "
                f"actual={_sha256(outcome.data)}"
            )
            return
        actual_digest = _sha256(outcome.data)
        if actual_digest != record["sha256"]:
            failures.append(
                f"{fixture_name}/{plan_name}: digest differs despite equal bytes"
            )
        actual_descriptor = outcome.descriptor.to_dict()
        if actual_descriptor != record["descriptor"]:
            failures.append(
                f"{fixture_name}/{plan_name}: descriptor differs: "
                f"golden={record['descriptor']} actual={actual_descriptor}"
            )

    # Reconstruct operation lists from the golden generator module so plan
    # definitions stay single-sourced.
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import generate_golden as generator  # noqa: E402

    for record in manifest["audio_renders"]:
        run_case(
            fixture_name=record["fixture"],
            plan_name=record["plan"],
            operations=generator.AUDIO_PLANS[record["plan"]],
            secondary_name=record.get("secondary"),
            record=record,
        )

    for record in manifest["video_renders"]:
        run_case(
            fixture_name=record["fixture"],
            plan_name=record["plan"],
            operations=generator.VIDEO_PLANS[record["plan"]],
            secondary_name=None,
            record=record,
        )

    print(
        f"parity: {render_count} byte-identical renders "
        f"({checked_bytes} output bytes compared), {error_count} error cases matched"
    )
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("parity: all migrated renders reproduce the golden bytes and descriptors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
