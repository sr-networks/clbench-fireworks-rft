"""Task-dispatching rollout processor for the mixed (spectrum + dbx) dataset.

Routes each row by its row_id prefix to the task's own battle-tested processor:
  canon-*  -> SpectrumCanonRolloutProcessor (dormc scaffold)
  dbx-*    -> DbxCanonRolloutProcessor
Both sub-processors share the same RolloutProcessorConfig (semaphore, completion_params),
so concurrency limits apply across the whole mixed batch. Task list is returned in the
original row order (eval-protocol zips results back to rows by position).
"""

from __future__ import annotations

import asyncio
import os
from typing import List

# The spectrum processor reads its regime from the env AT IMPORT TIME; the mixed arm
# uses the dormc scaffold = NOTEPAD mode (ICL-off). Must be set before the import.
os.environ["SPECTRUM_CANON_NOTEPAD"] = "1"

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig

from spectrum_canon_processor import SpectrumCanonRolloutProcessor
from dbx_canon_processor import DbxCanonRolloutProcessor

print("[mix] dispatching processor v1 (canon-* -> spectrum, dbx-* -> dbx)", flush=True)


def route(row_id: str) -> str:
    rid = row_id or ""
    if rid.startswith("dbx-"):
        return "dbx"
    if rid.startswith("canon-"):
        return "spectrum"
    raise ValueError(f"mix processor: unroutable row_id {row_id!r}")


class MixRolloutProcessor(RolloutProcessor):
    def __init__(self):
        self.spectrum = SpectrumCanonRolloutProcessor()
        self.dbx = DbxCanonRolloutProcessor()

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        buckets: dict[str, list[EvaluationRow]] = {"spectrum": [], "dbx": []}
        order: list[tuple[str, int]] = []   # (bucket, index within bucket) per row
        for row in rows:
            b = route(row.input_metadata.row_id)
            order.append((b, len(buckets[b])))
            buckets[b].append(row)
        tasks = {
            "spectrum": self.spectrum(buckets["spectrum"], config) if buckets["spectrum"] else [],
            "dbx": self.dbx(buckets["dbx"], config) if buckets["dbx"] else [],
        }
        return [tasks[b][i] for b, i in order]
