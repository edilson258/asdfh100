"""Resource monitoring and dynamic RAM scheduler module."""

import psutil
from loguru import logger


class MemoryGuardScheduler:
    """Monitors system memory utilization and dynamically throttles processing queues.

    Example:
        >>> guard = MemoryGuardScheduler(max_ram_pct=0.90)
        >>> guard.is_memory_safe()
        True
    """

    def __init__(self, max_ram_pct: float = 0.90) -> None:
        """Initialize scheduler with target maximum memory utilization ceiling.

        Example:
            >>> guard = MemoryGuardScheduler(max_ram_pct=0.85)
        """
        if not (0.1 <= max_ram_pct <= 0.99):
            raise ValueError(
                f"max_ram_pct must be between 0.1 and 0.99, got {max_ram_pct}"
            )
        self._max_ram_pct = max_ram_pct

    def get_current_ram_usage_ratio(self) -> float:
        """Query system RAM utilization ratio using psutil.

        Example:
            >>> ratio = guard.get_current_ram_usage_ratio()
            >>> 0.0 <= ratio <= 1.0
            True
        """
        vm = psutil.virtual_memory()
        return float(vm.percent / 100.0)

    def is_memory_safe(self) -> bool:
        """Check if current system RAM utilization is below configured ceiling ratio.

        Example:
            >>> safe = guard.is_memory_safe()
        """
        current_ratio = self.get_current_ram_usage_ratio()
        is_safe = current_ratio <= self._max_ram_pct
        if not is_safe:
            logger.warning(
                f"RAM usage high: {current_ratio * 100:.1f}% exceeds target ceiling of "
                f"{self._max_ram_pct * 100:.1f}%"
            )
        return is_safe

    def compute_safe_batch_size(self, default_batch_size: int) -> int:
        """Dynamically scale down default batch size if system memory pressure is high.

        Example:
            >>> safe_bs = guard.compute_safe_batch_size(default_batch_size=64)
        """
        current_ratio = self.get_current_ram_usage_ratio()
        if current_ratio <= self._max_ram_pct:
            return default_batch_size

        headroom_ratio = max(0.1, 1.0 - current_ratio)
        scaled_batch = int(default_batch_size * headroom_ratio)
        return max(1, scaled_batch)

    def check_and_log(self) -> None:
        if not self.is_memory_safe():
            logger.warning("RAM usage is high, but continuing processing.")

    def check_and_throttle(self, check_pause_sec: float = 0.5) -> None:
        """Throttle execution loop with micro-pauses if memory exceeds safe ceiling.

        Example:
            >>> guard.check_and_throttle()
        """
        import time

        while not self.is_memory_safe():
            logger.warning(
                f"RAM usage high ({self.get_current_ram_usage_ratio() * 100:.1f}%). "
                f"Pausing {check_pause_sec}s to avoid 100% saturation..."
            )
            time.sleep(check_pause_sec)
