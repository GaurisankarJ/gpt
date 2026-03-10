from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.scheduler import LearningRateScheduler  # noqa: E402


def _make_scheduler() -> LearningRateScheduler:
    return LearningRateScheduler(num_epochs=2, len_train_dataloader=5)


def _collect_first_group_lrs(
    scheduler: LearningRateScheduler, warmup: bool, cosine: bool, steps: int
) -> list[float]:
    return [
        scheduler.get_scheduled_learning_rates(
            warmup=warmup,
            cosine_decay=cosine,
            global_steps=step,
        )[0]
        for step in range(steps)
    ]


@pytest.mark.parametrize(
    ("num_epochs", "len_train_dataloader"),
    [(0, 1), (1, 0), (0, 0), (-1, 1), (1, -1)],
)
def test_constructor_rejects_non_positive_step_counts(num_epochs, len_train_dataloader):
    with pytest.raises(
        ValueError, match="num_epochs and len_train_dataloader must be greater than 0"
    ):
        LearningRateScheduler(
            num_epochs=num_epochs,
            len_train_dataloader=len_train_dataloader,
        )


@pytest.mark.parametrize("warmup_percentage", [-1, 101])
def test_set_warmup_steps_rejects_invalid_percentages(warmup_percentage):
    scheduler = _make_scheduler()
    with pytest.raises(ValueError, match="Warmup percentage must be between 0 and 100"):
        scheduler.set_warmup_steps(warmup_percentage)


def test_set_warmup_steps_zero_returns_none():
    scheduler = _make_scheduler()
    assert scheduler.set_warmup_steps(0) is None
    assert scheduler.warmup_steps is None


def test_set_warmup_steps_rounds_down_with_minimum_of_one():
    scheduler = _make_scheduler()
    # total_steps = 10, so 1% -> int(0.1) -> 0 -> None path
    assert scheduler.set_warmup_steps(1) is None
    # 10% -> 1 step
    assert scheduler.set_warmup_steps(10) == 1
    assert scheduler.warmup_steps == 1


@pytest.mark.parametrize(
    ("initial_learning_rates", "peak_learning_rates", "message"),
    [
        ([], [0.1], "must not be empty"),
        ([0.01], [], "must not be empty"),
        ([0.01], [0.1, 0.2], "same length"),
        ([-0.01], [0.1], "non-negative"),
        ([0.01], [-0.1], "non-negative"),
    ],
)
def test_set_learning_rates_warmup_validates_lists(
    initial_learning_rates, peak_learning_rates, message
):
    scheduler = _make_scheduler()
    with pytest.raises(ValueError, match=message):
        scheduler.set_learning_rates_warmup(initial_learning_rates, peak_learning_rates)


def test_initialize_learning_rates_warmup_zero_percent_disables_warmup():
    scheduler = _make_scheduler()
    scheduler.initialize_learning_rates_warmup(
        warmup_percentage=0,
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.1],
    )
    assert scheduler.learning_rates_warmup is False
    assert scheduler.learning_rate_increment is None
    with pytest.raises(ValueError, match="warmup is not initialized"):
        scheduler.get_learning_rates_warmup(global_steps=0)


def test_warmup_schedule_increases_then_plateaus():
    scheduler = _make_scheduler()
    scheduler.initialize_learning_rates_warmup(
        warmup_percentage=20,  # total_steps=10 -> warmup_steps=2
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.1],
    )

    assert scheduler.warmup_steps == 2
    lrs = _collect_first_group_lrs(scheduler, warmup=True, cosine=False, steps=6)
    assert lrs == pytest.approx([0.0, 0.05, 0.1, 0.1, 0.1, 0.1])
    assert len(scheduler.track_learning_rate) == 6


@pytest.mark.parametrize("minimum_learning_rates_percentage", [-1, 101])
def test_set_learning_rates_cosine_decay_rejects_invalid_percentages(
    minimum_learning_rates_percentage,
):
    scheduler = _make_scheduler()
    with pytest.raises(
        ValueError, match="Minimum learning rates percentage must be between 0 and 100"
    ):
        scheduler.set_learning_rates_cosine_decay(
            minimum_learning_rates_percentage=minimum_learning_rates_percentage,
            initial_learning_rates=[0.01],
            peak_learning_rates=[0.1],
        )


def test_set_learning_rates_cosine_decay_requires_lists_once():
    scheduler = _make_scheduler()
    with pytest.raises(ValueError, match="must be provided at least once"):
        scheduler.set_learning_rates_cosine_decay(minimum_learning_rates_percentage=10)


def test_set_learning_rates_cosine_decay_uses_peak_percentage():
    scheduler = _make_scheduler()
    mins = scheduler.set_learning_rates_cosine_decay(
        minimum_learning_rates_percentage=10,
        initial_learning_rates=[0.01, 0.02],
        peak_learning_rates=[0.1, 0.2],
    )
    assert mins == pytest.approx([0.01, 0.02])


def test_set_learning_rates_cosine_decay_can_reuse_stored_lists():
    scheduler = _make_scheduler()
    scheduler.initialize_learning_rates_warmup(
        warmup_percentage=20,
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.1],
    )

    mins = scheduler.set_learning_rates_cosine_decay(
        minimum_learning_rates_percentage=20,
    )
    assert mins == pytest.approx([0.02])


def test_get_learning_rates_cosine_decay_requires_initialization():
    scheduler = _make_scheduler()
    with pytest.raises(ValueError, match="cosine decay is not initialized"):
        scheduler.get_learning_rates_cosine_decay(global_steps=0)


def test_get_learning_rates_cosine_decay_errors_on_incomplete_warmup_state():
    scheduler = _make_scheduler()
    scheduler.learning_rates_cosine_decay = True
    scheduler.minimum_learning_rates = [0.01]
    scheduler.peak_learning_rates = [0.1]
    scheduler.learning_rates_warmup = True
    scheduler.warmup_steps = None
    with pytest.raises(ValueError, match="warmup_steps is not set"):
        scheduler.get_learning_rates_cosine_decay(global_steps=0)


def test_cosine_decay_with_full_warmup_returns_minimum_lrs():
    scheduler = LearningRateScheduler(num_epochs=1, len_train_dataloader=4)
    scheduler.initialize_learning_rates_warmup(
        warmup_percentage=100,
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.1],
    )
    scheduler.initialize_learning_rates_cosine_decay(
        minimum_learning_rates_percentage=10,
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.1],
    )

    assert scheduler.get_learning_rates_cosine_decay(global_steps=4) == pytest.approx(
        [0.01]
    )


def test_cosine_decay_clamps_progress_out_of_range():
    scheduler = _make_scheduler()
    scheduler.initialize_learning_rates_cosine_decay(
        minimum_learning_rates_percentage=10,
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.1],
    )

    # progress < 0 => clamp to 0 => peak lr
    assert scheduler.get_learning_rates_cosine_decay(global_steps=-5) == pytest.approx(
        [0.1]
    )
    # progress > 1 => clamp to 1 => minimum lr
    assert scheduler.get_learning_rates_cosine_decay(global_steps=50) == pytest.approx(
        [0.01]
    )


def test_get_scheduled_learning_rates_requires_peak_when_no_schedule():
    scheduler = _make_scheduler()
    with pytest.raises(ValueError, match="Peak learning rates are not initialized"):
        scheduler.get_scheduled_learning_rates(
            warmup=False, cosine_decay=False, global_steps=0
        )


def test_get_scheduled_learning_rates_all_modes_work():
    # Warmup only
    warmup_only = _make_scheduler()
    warmup_only.initialize_learning_rates_warmup(
        warmup_percentage=20,
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.1],
    )
    lrs_warmup = _collect_first_group_lrs(warmup_only, warmup=True, cosine=False, steps=6)
    assert lrs_warmup[0] < lrs_warmup[1] <= lrs_warmup[2]
    assert lrs_warmup[-1] == pytest.approx(0.1)

    # Cosine only
    cosine_only = _make_scheduler()
    cosine_only.initialize_learning_rates_cosine_decay(
        minimum_learning_rates_percentage=10,
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.1],
    )
    lrs_cosine = _collect_first_group_lrs(cosine_only, warmup=False, cosine=True, steps=11)
    assert lrs_cosine[0] == pytest.approx(0.1)
    assert lrs_cosine[-1] == pytest.approx(0.01)
    assert lrs_cosine[3] > lrs_cosine[7]

    # Warmup + cosine
    both = _make_scheduler()
    both.initialize_learning_rates_warmup(
        warmup_percentage=20,
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.1],
    )
    both.initialize_learning_rates_cosine_decay(
        minimum_learning_rates_percentage=10,
        initial_learning_rates=[0.0],
        peak_learning_rates=[0.1],
    )
    lrs_both = _collect_first_group_lrs(both, warmup=True, cosine=True, steps=11)
    assert lrs_both[0] < lrs_both[1] <= lrs_both[2]
    assert lrs_both[3] < lrs_both[2]
    assert lrs_both[-1] == pytest.approx(0.01)


def test_save_csv_logs_learning_rate_calls_utility(monkeypatch):
    scheduler = _make_scheduler()
    scheduler.track_learning_rate = [[0.01], [0.02], [0.03]]
    called = {}

    def fake_save_csv_logs(data, name):
        called["data"] = data
        called["name"] = name
        return "logs/fake.csv"

    monkeypatch.setattr("fine_tuning.scheduler.save_csv_logs", fake_save_csv_logs)
    scheduler.save_csv_logs_learning_rate(model_name="qwen3_0.6b_base")

    assert called["name"] == "qwen3_0.6b_base_instruct_learning_rate"
    assert list(called["data"]["Global Steps"]) == [0, 1, 2]
    assert called["data"]["Learning Rate"] == [[0.01], [0.02], [0.03]]


def test_plot_saved_for_three_scheduler_combinations():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = 11
    combinations = {
        "warmup_only": {
            "warmup": True,
            "cosine": False,
            "warmup_pct": 20,
            "min_pct": None,
        },
        "cosine_only": {
            "warmup": False,
            "cosine": True,
            "warmup_pct": None,
            "min_pct": 10,
        },
        "warmup_plus_cosine": {
            "warmup": True,
            "cosine": True,
            "warmup_pct": 20,
            "min_pct": 10,
        },
    }
    output_dir = PROJECT_ROOT / "logs" / "scheduler_test_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, config in combinations.items():
        scheduler = _make_scheduler()
        if config["warmup"]:
            scheduler.initialize_learning_rates_warmup(
                warmup_percentage=config["warmup_pct"],
                initial_learning_rates=[0.0],
                peak_learning_rates=[0.1],
            )
        else:
            scheduler.set_learning_rates_warmup(
                initial_learning_rates=[0.0],
                peak_learning_rates=[0.1],
            )

        if config["cosine"]:
            scheduler.initialize_learning_rates_cosine_decay(
                minimum_learning_rates_percentage=config["min_pct"],
                initial_learning_rates=[0.0],
                peak_learning_rates=[0.1],
            )

        lrs = _collect_first_group_lrs(
            scheduler=scheduler,
            warmup=config["warmup"],
            cosine=config["cosine"],
            steps=steps,
        )

        fig_path = output_dir / f"{name}.png"
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(steps), lrs, marker="o")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)

        assert fig_path.exists()
        assert fig_path.stat().st_size > 0
