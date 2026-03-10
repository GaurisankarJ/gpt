import pytest


def test_main_raises_without_mode(run_instruction_script):
    with pytest.raises(ValueError, match="Select at least one mode"):
        run_instruction_script({"train": False, "test": False})


def test_main_train_path_wires_expected_calls(run_instruction_script):
    calls = run_instruction_script(
        {
            "train": True,
            "test": False,
            "grad_clip": 0.0,
            "repo_id": None,
        }
    )

    assert calls["parse_args_calls"] == 1
    assert calls["trainer_init_calls"] == 1
    assert calls["trainer_train_calls"] == 1
    assert calls["evaluator_init_calls"] == 0
    assert calls["print_eval_losses_calls"] == 0
    assert calls["save_json_calls"] == 0
    assert calls["tokenizer_kwargs"]["repo_id"] == "Qwen/Qwen3-0.6B-Base"
    assert calls["trainer_train_kwargs"]["grad_clip"] is None
    assert calls["trainer_train_kwargs"]["show_progress_bar"] is True
    assert calls["trainer_train_kwargs"]["progress_update_freq"] == 20
    assert calls["trainer_train_kwargs"]["freq_checkpoint"] is None


def test_main_test_path_runs_eval_generation_and_save(run_instruction_script):
    calls = run_instruction_script(
        {
            "train": False,
            "test": True,
            "model_name": "my_model",
            "iter_evaluation": 7,
        }
    )

    assert calls["trainer_init_calls"] == 0
    assert calls["trainer_train_calls"] == 0
    assert calls["evaluator_init_calls"] == 1
    assert calls["print_eval_losses_calls"] == 1
    assert calls["print_eval_losses_kwargs"]["iter_evaluation"] == 7
    assert calls["save_json_calls"] == 1
    assert calls["save_json_args"]["file_name"] == "instruction_tuning_data_with_response_my_model.json"
    assert all("model_response" in row for row in calls["save_json_args"]["data"])


def test_main_train_and_test_modes_both_execute(run_instruction_script):
    calls = run_instruction_script({"train": True, "test": True})

    assert calls["trainer_train_calls"] == 1
    assert calls["print_eval_losses_calls"] == 1
    assert calls["save_json_calls"] == 1
