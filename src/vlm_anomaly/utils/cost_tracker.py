"""Per-call cost tracking and budget enforcement.

Usage::

    tracker = CostTracker(budget_usd=5.0)
    tracker.check_budget(estimated_cost=0.01)   # raises BudgetExceeded if over
    tracker.record(cost_usd=0.0042)
    print(tracker.total_usd)                    # 0.0042

Thread-safe via a threading.Lock so concurrent future backends won't race.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised before a call that would push the total over the budget."""

    def __init__(self, total: float, budget: float, next_cost: float) -> None:
        super().__init__(
            f"Budget ${budget:.4f} would be exceeded: "
            f"spent ${total:.4f}, next call ~${next_cost:.4f}"
        )
        self.total = total
        self.budget = budget
        self.next_cost = next_cost


@dataclass
class CostTracker:
    """Running USD total with an optional hard budget cap.

    Args:
        budget_usd: Maximum total spend allowed.  ``None`` means unlimited.
        name: Label shown in log messages and the ``BudgetExceeded`` error.
    """

    budget_usd: float | None = None
    name: str = "experiment"
    _total: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)

    @property
    def total_usd(self) -> float:
        """Total spend so far in USD."""
        return self._total

    @property
    def calls(self) -> int:
        """Number of API calls recorded."""
        return self._calls

    @property
    def remaining_usd(self) -> float | None:
        """Remaining budget, or ``None`` if unlimited."""
        if self.budget_usd is None:
            return None
        return max(0.0, self.budget_usd - self._total)

    def check_budget(self, estimated_cost: float = 0.0) -> None:
        """Raise :exc:`BudgetExceeded` if the next call would exceed the budget.

        Call this *before* making an API request.

        Args:
            estimated_cost: Rough upper-bound cost of the upcoming call.

        Raises:
            BudgetExceeded: If ``total + estimated_cost > budget_usd``.
        """
        if self.budget_usd is None:
            return
        with self._lock:
            if self._total + estimated_cost > self.budget_usd:
                raise BudgetExceeded(self._total, self.budget_usd, estimated_cost)

    def record(self, cost_usd: float) -> None:
        """Add ``cost_usd`` to the running total.

        Args:
            cost_usd: Actual cost of a completed API call (from the response).
        """
        with self._lock:
            self._total += cost_usd
            self._calls += 1

    def reset(self) -> None:
        """Reset total and call count to zero."""
        with self._lock:
            self._total = 0.0
            self._calls = 0

    def summary(self) -> dict[str, float | int | str | None]:
        """Return a loggable summary dict."""
        return {
            "name": self.name,
            "calls": self._calls,
            "total_usd": round(self._total, 6),
            "budget_usd": self.budget_usd,
            "remaining_usd": self.remaining_usd,
        }
