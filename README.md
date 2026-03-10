## Instruction Fine-Tuning

This project includes a script-based pipeline for instruction fine-tuning and evaluation:
- Script entrypoint: `scripts/fine_tune_instruction.py`
- CLI + helpers: `scripts/fine_tune_instruction_utils.py`

## Setup

1. Create/activate your virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Script

At least one mode is required:
- `--train` for fine-tuning
- `--test` for evaluation + generating responses on test split

### Train

```bash
python -m scripts.fine_tune_instruction --train
```

### Test

```bash
python -m scripts.fine_tune_instruction --test
```

### Train + Test in one run

```bash
python -m scripts.fine_tune_instruction --train --test
```

### Example with custom options

```bash
python -m scripts.fine_tune_instruction \
  --train \
  --model_name qwen3_0.6b_base_run1 \
  --checkpoint_path qwen3_0.6b_base \
  --dataset_file_path instruction_tuning_data.json \
  --batch_size 2 \
  --max_length 1024 \
  --learning_rate 5e-5 \
  --num_epochs 1 \
  --freq_evaluation 100 \
  --iter_evaluation 50 \
  --show_progress_bar \
  --save_logs
```

## Testing

Run the targeted script test suite:

```bash
python -m pytest -q tests/scripts/test_fine_tune_instruction_utils.py tests/scripts/test_fine_tune_instruction_main.py
```

Run all tests:

```bash
python -m pytest -q
```

## CLI Arguments (`scripts/fine_tune_instruction.py`)

Note: boolean flags use `BooleanOptionalAction`, so you can pass either `--flag` or `--no-flag`.

### Mode flags

- `--train` (default: `False`): enable training path.
- `--test` (default: `False`): enable evaluation + response generation path.

### Model and tokenizer

- `--model_name` (default: `qwen3_0.6b_base`): run/model label used in output names.
- `--model_size` (default: `0.6B`): model config key for `get_qwen3_config`.
- `--model_type` (default: `base`): model type passed into generator.
- `--tokenizer_file_path` (default: `./tokenizer/qwen_3_instruct_tokenizer.json`): tokenizer JSON path.
- `--repo_id` (default: `Qwen/Qwen3-0.6B-Base`): tokenizer/model repo id.
- `--checkpoint_path` (default: `qwen3_0.6b_base`): checkpoint basename loaded from `./checkpoints/<name>.pth`.

### Prompt formatting

- `--apply_chat_template` / `--no-apply_chat_template` (default: `True`): apply tokenizer chat template.
- `--add_generation_prompt` / `--no-add_generation_prompt` (default: `True`): append generation prompt tokens.
- `--add_thinking` / `--no-add_thinking` (default: `False`): include thinking tags where supported.

### Dataset and split

- `--dataset_file_path` (default: `instruction_tuning_data.json`): input dataset JSON file.
- `--shuffle_before_split` / `--no-shuffle_before_split` (default: `True`): shuffle dataset before 85/10/5 split.
- `--seed` (default: `42`): random seed used for pre-split shuffle.

### Dataloader and sequence settings

- `--batch_size` (default: `4`): batch size for train/val/test dataloaders.
- `--shuffle` / `--no-shuffle` (default: `True`): shuffle training dataloader.
- `--drop_last` / `--no-drop_last` (default: `True`): drop incomplete final batch in training dataloader.
- `--num_workers` (default: `0`): dataloader workers.
- `--ignore_index` (default: `-100`): label ignore index for loss.
- `--mask_inputs` / `--no-mask_inputs` (default: `True`): mask user prompt tokens in labels.
- `--max_length` (default: `256`): max sequence length (also used as generation max length in test mode).

### Optimization and scheduling

- `--learning_rate` (default: `5e-5`): AdamW learning rate.
- `--weight_decay` (default: `0.3`): AdamW weight decay.
- `--num_epochs` (default: `1`): number of training epochs.
- `--grad_clip` (default: `1.0`): gradient clipping max norm; values `<= 0` disable clipping.

### Logging, evaluation, checkpoints

- `--freq_evaluation` (default: `100`): evaluate every N training steps.
- `--iter_evaluation` (default: `50`): number of batches used during each evaluation call.
- `--save_logs` / `--no-save_logs` (default: `True`): save CSV logs from trainer.
- `--show_progress_bar` / `--no-show_progress_bar` (default: `True`): toggle tqdm progress display.
- `--progress_update_freq` (default: `20`): update tqdm postfix every N steps.
- `--freq_checkpoint` (default: `None`): save checkpoint every N epochs (`None` disables periodic checkpointing).