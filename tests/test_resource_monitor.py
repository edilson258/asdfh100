"""Unit tests for memory guard scheduler resource monitoring."""

import pytest
from pipeline.resource_monitor import MemoryGuardScheduler


def test_memory_guard_init_validation() -> None:
    """Test validation of memory ceiling ratio input bounds."""
    with pytest.raises(ValueError, match="max_ram_pct must be between"):
        MemoryGuardScheduler(max_ram_pct=0.05)


def test_get_current_ram_usage_ratio() -> None:
    """Test querying system RAM usage ratio."""
    guard = MemoryGuardScheduler(max_ram_pct=0.90)
    ratio = guard.get_current_ram_usage_ratio()
    assert 0.0 <= ratio <= 1.0


def test_compute_safe_batch_size() -> None:
    """Test computing safe batch size under normal vs high RAM pressure."""
    guard = MemoryGuardScheduler(max_ram_pct=0.90)
    batch_size = guard.compute_safe_batch_size(default_batch_size=64)
    assert batch_size >= 1
