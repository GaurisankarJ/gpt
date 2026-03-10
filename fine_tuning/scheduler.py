import math
from dataclasses import dataclass
from typing import Callable, List, Optional

@dataclass(frozen=True)
class LearningRateSchedulerState:
    total_steps: int
    warmup_steps: int
    cosine_decay: bool
    min_learning_rate_ratio: float
    peak_learning_rates: List[float]
    initial_learning_rates: List[float]


class LearningRateScheduler:
    def __init__(
        self,
        peak_learning_rates: List[float],
        total_steps: int,
        learning_rate_warmup: Optional[float] = None,
        initial_learning_rate: Optional[float] = None,
        initial_learning_rates: Optional[List[float]] = None,
        cosine_decay: bool = False,
        min_learning_rate_ratio: float = 0.0,
    ) -> None:
        if not peak_learning_rates:
            raise ValueError("peak_learning_rates must not be empty.")
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0.")
        if learning_rate_warmup is None:
            learning_rate_warmup = 0.0
        if learning_rate_warmup < 0 or learning_rate_warmup > 100:
            raise ValueError("learning_rate_warmup must be between 0 and 100.")
        if min_learning_rate_ratio < 0 or min_learning_rate_ratio > 1:
            raise ValueError("min_learning_rate_ratio must be between 0 and 1.")
        if initial_learning_rate is not None and initial_learning_rate < 0:
            raise ValueError("initial_learning_rate must be non-negative.")
        if initial_learning_rates is not None and initial_learning_rate is not None:
            raise ValueError(
                "Provide either initial_learning_rate or initial_learning_rates, not both."
            )

        self.total_steps = total_steps
        self.cosine_decay = cosine_decay
        self.min_learning_rate_ratio = min_learning_rate_ratio
        self.peak_learning_rates = [float(lr) for lr in peak_learning_rates]

        warmup_steps = int(total_steps * (learning_rate_warmup / 100.0))
        if learning_rate_warmup > 0:
            warmup_steps = max(1, warmup_steps)
        else:
            warmup_steps = 0
        self.warmup_steps = warmup_steps

        self.initial_learning_rates = self._initialize_initial_learning_rates(
            initial_learning_rate=initial_learning_rate,
            initial_learning_rates=initial_learning_rates,
        )

        self.state = LearningRateSchedulerState(
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            cosine_decay=self.cosine_decay,
            min_learning_rate_ratio=self.min_learning_rate_ratio,
            peak_learning_rates=self.peak_learning_rates.copy(),
            initial_learning_rates=self.initial_learning_rates.copy(),
        )

    def _initialize_initial_learning_rates(
        self,
        initial_learning_rate: Optional[float],
        initial_learning_rates: Optional[List[float]],
    ) -> List[float]:
        if self.warmup_steps == 0:
            return self.peak_learning_rates.copy()

        if initial_learning_rates is not None:
            if len(initial_learning_rates) != len(self.peak_learning_rates):
                raise ValueError(
                    "initial_learning_rates must match peak_learning_rates length."
                )
            return [float(lr) for lr in initial_learning_rates]

        if initial_learning_rate is not None:
            return [
                float(initial_learning_rate) for _ in range(len(self.peak_learning_rates))
            ]

        return [lr / 1e4 for lr in self.peak_learning_rates]

    def _calculate_warmup_learning_rates(self, step_idx: int) -> List[float]:
        if self.warmup_steps <= 0:
            return self.peak_learning_rates.copy()

        if self.warmup_steps == 1:
            progress = 1.0
        else:
            progress = step_idx / (self.warmup_steps - 1)

        return [
            init_lr + progress * (peak_lr - init_lr)
            for init_lr, peak_lr in zip(
                self.initial_learning_rates, self.peak_learning_rates
            )
        ]

    def _calculate_cosine_decay_learning_rates(self, step_idx: int) -> List[float]:
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 1:
            progress = 1.0
        else:
            decay_step_idx = step_idx - self.warmup_steps
            progress = decay_step_idx / (decay_steps - 1)
            progress = min(max(progress, 0.0), 1.0)

        learning_rates = []
        for peak_lr in self.peak_learning_rates:
            min_lr = peak_lr * self.min_learning_rate_ratio
            cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
            learning_rates.append(min_lr + (peak_lr - min_lr) * cosine_multiplier)
        return learning_rates

    def get_learning_rates(self, step_idx: int) -> List[float]:
        if step_idx < 0:
            raise ValueError("step_idx must be >= 0.")

        clamped_step_idx = min(step_idx, self.total_steps - 1)

        if self.warmup_steps > 0 and clamped_step_idx < self.warmup_steps:
            return self._calculate_warmup_learning_rates(clamped_step_idx)

        if self.cosine_decay:
            return self._calculate_cosine_decay_learning_rates(clamped_step_idx)

        return self.peak_learning_rates.copy()


def initialize_learning_rate_scheduler(
    peak_learning_rates: List[float],
    total_steps: int,
    learning_rate_warmup: Optional[float] = None,
    initial_learning_rate: Optional[float] = None,
    initial_learning_rates: Optional[List[float]] = None,
    cosine_decay: bool = False,
    min_learning_rate_ratio: float = 0.0,
) -> tuple[Callable[[int], List[float]], LearningRateSchedulerState]:
    scheduler = LearningRateScheduler(
        peak_learning_rates=peak_learning_rates,
        total_steps=total_steps,
        learning_rate_warmup=learning_rate_warmup,
        initial_learning_rate=initial_learning_rate,
        initial_learning_rates=initial_learning_rates,
        cosine_decay=cosine_decay,
        min_learning_rate_ratio=min_learning_rate_ratio,
    )
    return scheduler.get_learning_rates, scheduler.state
