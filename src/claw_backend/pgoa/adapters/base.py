"""Abstract base class for PGOA profiling adapters."""

from __future__ import annotations

import abc
from datetime import datetime, UTC
from uuid import uuid4

from claw_backend.pgoa.schema import HardwareInfo, KPIMetrics, ProfileBundle


class BaseAdapter(abc.ABC):
    """All profiling adapters must implement this interface."""

    @abc.abstractmethod
    def name(self) -> str:
        """Return the adapter name."""

    @abc.abstractmethod
    def collect(self, **kwargs) -> ProfileBundle:
        """Collect profiling data and return a ProfileBundle."""

    def _new_bundle(self, kpi: KPIMetrics) -> ProfileBundle:
        """Create a new ProfileBundle with a fresh run_id and current timestamp."""
        return ProfileBundle(
            run_id=str(uuid4()),
            timestamp=datetime.now(tz=UTC),
            hardware=HardwareInfo(),
            kpi=kpi,
        )
