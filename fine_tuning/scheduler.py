import math
from typing import List, Optional, Tuple

from .utils import save_csv_logs


class LearningRateScheduler:
    def __init__(
        self,
        num_epochs: int,
        len_train_dataloader: int,
    ):
        # Parameters
        if num_epochs <= 0 or len_train_dataloader <= 0:
            raise ValueError(
                "num_epochs and len_train_dataloader must be greater than 0."
            )

        self.num_epochs = num_epochs
        self.len_train_dataloader = len_train_dataloader
        self.learning_rates_warmup = False
        self.learning_rates_cosine_decay = False

        # Learning rates
        self.initial_learning_rates = None
        self.peak_learning_rates = None
        self.learning_rate_increment = None
        self.minimum_learning_rates = None

        # Warmup steps
        self.warmup_steps = None

        # Track learning rate
        self.track_learning_rate = []

    def _validate_learning_rate_lists(
        self,
        initial_learning_rates: List[float],
        peak_learning_rates: List[float],
    ) -> None:
        if not initial_learning_rates or not peak_learning_rates:
            raise ValueError("Learning rate lists must not be empty.")

        if len(initial_learning_rates) != len(peak_learning_rates):
            raise ValueError(
                "Initial and peak learning rate lists must have the same length."
            )

        if any(lr < 0 for lr in initial_learning_rates):
            raise ValueError("Initial learning rates must be non-negative.")

        if any(lr < 0 for lr in peak_learning_rates):
            raise ValueError("Peak learning rates must be non-negative.")

    def reset_scheduler(self) -> None:
        self.learning_rates_warmup = False
        self.learning_rates_cosine_decay = False
        self.initial_learning_rates = None
        self.peak_learning_rates = None
        self.learning_rate_increment = None
        self.minimum_learning_rates = None
        self.warmup_steps = None
        self.track_learning_rate = []

    def set_warmup_steps(
        self,
        warmup_percentage: float,
    ) -> int | None:
        if warmup_percentage < 0 or warmup_percentage > 100:
            raise ValueError("Warmup percentage must be between 0 and 100.")

        total_steps = self.len_train_dataloader * self.num_epochs
        self.warmup_steps = int(warmup_percentage * total_steps / 100)

        if self.warmup_steps == 0:
            self.warmup_steps = None

            return

        self.warmup_steps = max(1, self.warmup_steps)

        return self.warmup_steps

    def set_learning_rates_warmup(
        self,
        initial_learning_rates: List[float],
        peak_learning_rates: List[float],
    ) -> Tuple[List[float], List[float]]:
        self._validate_learning_rate_lists(
            initial_learning_rates=initial_learning_rates,
            peak_learning_rates=peak_learning_rates,
        )

        self.initial_learning_rates = initial_learning_rates
        self.peak_learning_rates = peak_learning_rates

        return self.initial_learning_rates, self.peak_learning_rates

    def initialize_learning_rates_warmup(
        self,
        warmup_percentage: float,
        initial_learning_rates: List[float],
        peak_learning_rates: List[float],
    ) -> None:
        initial_learning_rates, peak_learning_rates = self.set_learning_rates_warmup(
            initial_learning_rates=initial_learning_rates,
            peak_learning_rates=peak_learning_rates,
        )

        warmup_steps = self.set_warmup_steps(warmup_percentage)
        if warmup_steps is None:
            self.learning_rates_warmup = False
            self.learning_rate_increment = None

            return
        self.learning_rates_warmup = True

        self.learning_rate_increment = [
            (peak - init) / warmup_steps
            for init, peak in zip(initial_learning_rates, peak_learning_rates)
        ]

    def get_learning_rates_warmup(self, global_steps: int) -> List[float]:
        if self.learning_rates_warmup is False:
            raise ValueError("Learning rates warmup is not initialized.")
        if (
            self.warmup_steps is None
            or self.initial_learning_rates is None
            or self.learning_rate_increment is None
            or self.peak_learning_rates is None
        ):
            raise ValueError("Warmup state is incomplete.")

        if global_steps < self.warmup_steps:
            return [
                self.initial_learning_rates[i]
                + self.learning_rate_increment[i] * global_steps
                for i in range(len(self.initial_learning_rates))
            ]
        else:
            return self.peak_learning_rates

    def set_learning_rates_cosine_decay(
        self,
        minimum_learning_rates_percentage: float,
        initial_learning_rates: Optional[List[float]] = None,
        peak_learning_rates: Optional[List[float]] = None,
    ) -> List[float]:
        if (
            minimum_learning_rates_percentage < 0
            or minimum_learning_rates_percentage > 100
        ):
            raise ValueError(
                "Minimum learning rates percentage must be between 0 and 100."
            )

        resolved_initial = (
            initial_learning_rates
            if initial_learning_rates is not None
            else self.initial_learning_rates
        )
        resolved_peak = (
            peak_learning_rates
            if peak_learning_rates is not None
            else self.peak_learning_rates
        )

        if resolved_initial is None or resolved_peak is None:
            raise ValueError(
                "Initial and peak learning rates must be provided at least once."
            )

        self._validate_learning_rate_lists(
            initial_learning_rates=resolved_initial,
            peak_learning_rates=resolved_peak,
        )

        self.initial_learning_rates = resolved_initial
        self.peak_learning_rates = resolved_peak

        self.minimum_learning_rates = [
            (minimum_learning_rates_percentage / 100) * lr
            for lr in self.peak_learning_rates
        ]

        return self.minimum_learning_rates

    def initialize_learning_rates_cosine_decay(
        self,
        minimum_learning_rates_percentage: float,
        initial_learning_rates: List[float],
        peak_learning_rates: List[float],
    ) -> None:
        self.learning_rates_cosine_decay = True

        self.set_learning_rates_cosine_decay(
            minimum_learning_rates_percentage=minimum_learning_rates_percentage,
            initial_learning_rates=initial_learning_rates,
            peak_learning_rates=peak_learning_rates,
        )

    def get_learning_rates_cosine_decay(self, global_steps: int) -> List[float]:
        if self.learning_rates_cosine_decay is False:
            raise ValueError("Learning rates cosine decay is not initialized.")
        if self.minimum_learning_rates is None or self.peak_learning_rates is None:
            raise ValueError("Cosine decay state is incomplete.")

        total_steps = self.num_epochs * self.len_train_dataloader

        if self.learning_rates_warmup is True:
            if self.warmup_steps is None:
                raise ValueError("Warmup is enabled but warmup_steps is not set.")

            decay_steps = total_steps - self.warmup_steps
            if decay_steps <= 0:
                return self.minimum_learning_rates

            progress = (global_steps - self.warmup_steps) / (decay_steps)
        else:
            progress = global_steps / total_steps

        progress = min(max(progress, 0.0), 1.0)

        return [
            self.minimum_learning_rates[i]
            + 0.5
            * (self.peak_learning_rates[i] - self.minimum_learning_rates[i])
            * (1 + math.cos(math.pi * progress))
            for i in range(len(self.minimum_learning_rates))
        ]

    def get_scheduled_learning_rates(
        self,
        warmup: bool,
        cosine_decay: bool,
        global_steps: int,
    ) -> List[float]:
        if warmup and not cosine_decay:
            learning_rates = self.get_learning_rates_warmup(global_steps)
            self.track_learning_rate.append(learning_rates)

            return learning_rates
        elif not warmup and cosine_decay:
            learning_rates = self.get_learning_rates_cosine_decay(global_steps)
            self.track_learning_rate.append(learning_rates)

            return learning_rates
        elif warmup and cosine_decay:
            if (
                self.learning_rates_warmup
                and self.warmup_steps is not None
                and global_steps < self.warmup_steps
            ):
                learning_rates = self.get_learning_rates_warmup(global_steps)
            else:
                learning_rates = self.get_learning_rates_cosine_decay(global_steps)
            self.track_learning_rate.append(learning_rates)

            return learning_rates
        elif not warmup and not cosine_decay:
            if self.peak_learning_rates is None:
                raise ValueError("Peak learning rates are not initialized.")
            learning_rates = self.peak_learning_rates
            self.track_learning_rate.append(learning_rates)

            return learning_rates
        else:
            raise ValueError("Invalid learning rate schedule.")

    def save_csv_logs_learning_rate(
        self,
        model_name: str,
    ) -> None:
        data = {
            "Global Steps": range(len(self.track_learning_rate)),
            "Learning Rate": self.track_learning_rate,
        }

        full_path = save_csv_logs(
            data=data,
            name=f"{model_name}_instruct_learning_rate",
        )

        print(f"Learning rates for instruction fine-tuning saved to: {full_path}.")
