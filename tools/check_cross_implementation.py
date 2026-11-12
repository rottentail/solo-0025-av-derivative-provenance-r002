#!/usr/bin/env python3
"""Prove durable pre-migration renditions reproduce under the NumPy kernels.

This is not a same-process replay: the frozen standard-library implementation
under ``tools/reference`` first renders outputs through the real storage and
application layers (assets, renditions, provenance edges, content-addressed
blobs).  The migrated package then reopens that state directory and runs the
real integrity-check use case.  A ``verified`` result means the persisted,
pre-migration WAV/Y4M bytes are reproduced byte-for-byte by the new kernels
through the original derived entry point, rather than merely sharing a
duration or descriptor.

Run with the project virtual environment.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = PROJECT_ROOT / "tools" / "reference"
for path in (str(PROJECT_ROOT), str(REFERENCE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import generate_golden as generator  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    from avprovenance_ref.media.pipeline import (
        apply_media_operations as reference_render,
    )
    from avprovenance.application import MediaProvenanceApplication
    from avprovenance.serialization import digest_json
    from avprovenance.storage import ContentFileRepository

    audio_fixtures = generator.build_audio_fixtures()
    video_fixtures = generator.build_video_fixtures()

    with tempfile.TemporaryDirectory(prefix="avprovenance-cross-") as temporary:
        data_dir = Path(temporary) / "data"

        # Phase 1: build durable state entirely with the frozen stdlib kernels.
        reference_app = MediaProvenanceApplication(data_dir)
        # Swap only the media callable behind the application boundary; the
        # storage, models, and state machine are the shared production code.
        import avprovenance.renditions.executor as executor_module

        original_execute = executor_module.execute_plan

        def reference_execute(primary_bytes, operations, secondary_bytes):
            from avprovenance.media.errors import MediaOperationFailure
            from avprovenance.errors import RenderFailed, UnsupportedMedia
            from avprovenance.media import RenderOutcome

            try:
                outcome = reference_render(
                    primary_bytes, operations, secondary_bytes
                )
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, MediaOperationFailure):
                    raise RenderFailed(
                        "The rendition could not be produced.",
                        details={"reason": exc.code, "detail": exc.message},
                    ) from exc
                if isinstance(exc, UnsupportedMedia):
                    raise RenderFailed(
                        "The rendition produced unsupported media bytes.",
                        details={"reason": exc.code},
                    ) from exc
                raise
            return RenderOutcome(
                data=outcome.data,
                descriptor=outcome.descriptor,
                operation_types=outcome.operation_types,
            )

        executor_module.execute_plan = reference_execute
        try:
            rendered_records: list[dict] = []

            def ingest(app: MediaProvenanceApplication, name: str, data: bytes) -> str:
                return app.ingest_asset(name, data)["id"]

            # A stereo plan exercising extreme negative vs. quiet channels,
            # gain, mix, and normalization through the full HTTP-shaped flow.
            primary_id = ingest(
                reference_app,
                "stereo-extreme.wav",
                audio_fixtures["stereo_extreme"]["data"],
            )
            secondary_id = ingest(
                reference_app,
                "stereo-secondary.wav",
                audio_fixtures["stereo_secondary"]["data"],
            )
            audio_ops = generator.AUDIO_PLANS["trim_gain_mix_normalize"]
            audio_rendition = reference_app.create_rendition(
                {
                    "primary_asset_id": primary_id,
                    "secondary_asset_id": secondary_id,
                    "output_name": "cross-audio.wav",
                    "operations": audio_ops,
                }
            )
            rendered_records.append(
                {
                    "label": "stereo extreme chain",
                    "rendition_id": audio_rendition["id"],
                    "plan_digest": audio_rendition["plan_digest"],
                }
            )

            # A video plan exercising boundary luma, crop, mirror and trim.
            video_id = ingest(
                reference_app,
                "luma-boundary.y4m",
                video_fixtures["luma_boundary"]["data"],
            )
            video_ops = generator.VIDEO_PLANS["full_chain"]
            video_rendition = reference_app.create_rendition(
                {
                    "primary_asset_id": video_id,
                    "output_name": "cross-video.y4m",
                    "operations": video_ops,
                }
            )
            rendered_records.append(
                {
                    "label": "luma boundary chain",
                    "rendition_id": video_rendition["id"],
                    "plan_digest": video_rendition["plan_digest"],
                }
            )

            # Independently recompute what the frozen implementation produced
            # and confirm the stored blob digest matches it.
            content = ContentFileRepository(data_dir / "content")
            state_path = data_dir / "state.json"
            state = json.loads(state_path.read_text())
            for rendition in state["renditions"].values():
                output_id = rendition["output_asset_id"]
                output_record = state["assets"][output_id]
                stored = content.read(output_record["content_key"])
                assert _sha256(stored) == output_record["sha256"]
        finally:
            executor_module.execute_plan = original_execute

        # Phase 2: reopen the same durable directory with the migrated code.
        # Nothing is re-rendered or migrated on disk; integrity checks replay
        # the recorded normalized plans through the NumPy kernels.
        migrated_app = MediaProvenanceApplication(data_dir)
        results = []
        for record in rendered_records:
            audit = migrated_app.verify_rendition(record["rendition_id"])
            assert audit["status"] == "verified", audit
            assert audit["expected_digest"] == audit["reproduced_digest"]
            assert audit["expected_descriptor"] == audit["reproduced_descriptor"]
            results.append(
                {
                    "label": record["label"],
                    "rendition_id": record["rendition_id"],
                    "status": audit["status"],
                    "digest": audit["reproduced_digest"],
                }
            )

    print(json.dumps({"cross_implementation_checks": results}, indent=2))
    print(
        "cross-impl: pre-migration durable renditions verified byte-for-byte "
        "by the NumPy kernels"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
