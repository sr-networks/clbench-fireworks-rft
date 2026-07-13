"""Task-dispatching reward for the mixed (spectrum + dbx) dataset.

Same routing key as mix_canon_processor.route: canon-* rows get the validated dormc
spectrum reward, dbx-* rows get the dbx reward. No cross-task scale adjustment — GRPO
computes advantages within one row's candidate group, so the two score scales never
meet. An unroutable row_id raises (fail loud, not silently zero)."""

from eval_protocol.models import EvaluationRow

from mix_canon_processor import route
from spectrum_reward import compute_spectrum_dormant_completion_reward
from dbx_reward import compute_dbx_reward


def compute_mix_reward(row: EvaluationRow):
    task = route(row.input_metadata.row_id)
    if task == "dbx":
        return compute_dbx_reward(row)
    return compute_spectrum_dormant_completion_reward(row)
