"""Compatibility patch for missing-date Zarr stores and NumPy datetime arithmetic."""

import datetime
import logging
import os

import numpy as np

from anemoi.datasets.usage.gridded.store import ZarrWithMissingDates
from anemoi.datasets.usage.gridded.store import ZarrWithMissingDatesFix

logger = logging.getLogger(__name__)


def _mutate_missing_dates(self):
    start, end, frequency = self.start_date, self.end_date, self.frequency
    frequency_seconds = int(frequency.total_seconds())
    if frequency_seconds <= 0 or datetime.timedelta(seconds=frequency_seconds) != frequency:
        raise ValueError(f"Invalid dataset frequency: {frequency!r}")

    expected_size = ((end - start) // np.timedelta64(frequency_seconds, "s")) + 1
    if self.data.shape[0] == expected_size:
        return self

    logger.warning(
        "Dataset %s has missing dates encoded incorrectly; applying the experimental in-memory fix.",
        self.path,
    )
    if int(os.environ.get("ANEMOI_DATASETS_MISSING_DATES_FIX_EXPERIMENTAL", 0)):
        return ZarrWithMissingDatesFix(self.store, self.path)

    raise ValueError(
        f"Dataset {self} is not encoded correctly. Set "
        "ANEMOI_DATASETS_MISSING_DATES_FIX_EXPERIMENTAL=1 to enable the in-memory fix."
    )


ZarrWithMissingDates.mutate = _mutate_missing_dates
