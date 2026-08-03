"""Small adapter helpers for optional CheMLFlow training telemetry.

Model implementations call these helpers instead of depending on the concrete
artifact reporter.  That keeps sklearn, PyTorch, Lightning, and time-series
trainers independently usable and guarantees telemetry cannot break training.
"""

from __future__ import annotations

import logging
from typing import Any


def emit(
    progress_reporter: Any | None, method: str, /, *args: Any, **kwargs: Any
) -> None:
    if progress_reporter is None:
        return
    callback = getattr(progress_reporter, method, None)
    if not callable(callback):
        return
    try:
        callback(*args, **kwargs)
    except Exception as exc:  # telemetry is strictly best-effort
        logging.debug("Progress telemetry callback %s failed: %s", method, exc)


def opaque_fit_started(progress_reporter: Any | None, model_type: str) -> None:
    emit(
        progress_reporter,
        "training_indeterminate",
        "fit",
        unit="fit",
        phase="training",
        message=(
            f"{model_type} fit is active; this estimator does not expose a "
            "trustworthy step total."
        ),
    )


def opaque_fit_finished(progress_reporter: Any | None, model_type: str) -> None:
    emit(
        progress_reporter,
        "training_scope_finished",
        "fit",
        message=f"{model_type} fit completed.",
    )
