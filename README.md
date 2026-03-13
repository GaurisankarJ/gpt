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

### W&B Setup (optional)

```bash
python -m pip install wandb
wandb login
python -m scripts.fine_tune_instruction --train --wandb --wandb_project sft
```

### Run W&B Locally

```bash
# 1) Start Docker Desktop/daemon first.

# 2) If container already exists, start it:
docker start wandb-local

# 3) If you need a fresh container, recreate it:
docker rm -f wandb-local
docker run --platform linux/amd64 -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local

# 4) Verify it is running:
docker ps -a | grep wandb-local
docker logs -f wandb-local

# 5) Point wandb CLI/SDK to local server and login:
export WANDB_BASE_URL=http://localhost:8080
wandb login --host http://localhost:8080

# 6) Run training against local server:
python -m scripts.fine_tune_instruction --train --wandb --wandb_entity omega --wandb_project sft
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

Run the full validated fine-tuning + script test suites (single command):

```bash
python -m pytest -q tests/fine_tuning/test_scheduler.py tests/fine_tuning/test_instruction_trainer.py tests/scripts/test_fine_tune_instruction_utils.py tests/scripts/test_fine_tune_instruction_main.py tests/scripts/test_wandb.py

python -m pytest -vv -rP tests/fine_tuning/test_scheduler.py tests/fine_tuning/test_instruction_trainer.py tests/scripts/test_fine_tune_instruction_utils.py tests/scripts/test_fine_tune_instruction_main.py tests/scripts/test_wandb.py
```

Run every test in the repository:

```bash
python -m pytest -q

python -m pytest -vv -rP
```

## CLI Arguments (`scripts/fine_tune_instruction.py`)

Note: boolean flags use `BooleanOptionalAction`, so you can pass either `--flag` or `--no-flag`.

### Mode flags

- `--train` (default: `False`): enable training path.
- `--test` (default: `False`): enable evaluation + response generation path.
- `--eval` (default: `False`): run model-judge evaluation on response JSON.
- `--test_data_path` (default: from hyperparameters): optional JSON path used by `--eval`.

### Model and tokenizer

- `--model_name` (default: `qwen3_0.6b_base`): run/model label used in output names.
- `--model_size` (default: `0.6B`): model config key for `get_qwen3_config`.
- `--model_type` (default: `instruct`): model type passed into generator.
- `--tokenizer_file_path` (default: `./tokenizer/qwen_3_instruct_tokenizer.json`): tokenizer JSON path.
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
- `--evaluation_model` (default: `llama3.2:3b`): model used in `--eval` scoring.
- `--save_logs` / `--no-save_logs` (default: `True`): save CSV logs from trainer.
- `--wandb` / `--no-wandb` (default: `False`): enable/disable Weights & Biases experiment tracking.
- `--wandb_project` (default: `omega-instruction-tuning`): wandb project name.
- `--wandb_run_name` (default: `None`): optional wandb run display name.
- `--wandb_entity` (default: `None`): optional wandb entity/user/team.
- `--wandb_tags` (default: `[]`): optional wandb tags (space-separated list).
- `--wandb_artifacts` / `--no-wandb_artifacts` (default: `False`): enable/disable wandb dataset/checkpoint artifact uploads.
- `--show_progress_bar` / `--no-show_progress_bar` (default: `True`): toggle tqdm progress display.
- `--progress_update_freq` (default: `20`): update tqdm postfix every N steps.
- `--freq_checkpoint` (default: `None`): save checkpoint every N epochs (`None` disables periodic checkpointing).

### Scheduler and LoRA

- `--warmup` / `--no-warmup` (default: `True`): enable/disable warmup schedule.
- `--cosine_decay` / `--no-cosine_decay` (default: `True`): enable/disable cosine decay.
- `--initial_learning_rates` (default: `[5e-8]`): initial LR(s), one per optimizer param group.
- `--peak_learning_rates` (default: `[5e-5]`): peak LR(s), one per optimizer param group.
- `--learning_rate_warmup_percentage` (default: `10`): warmup duration as percentage of total steps.
- `--minimum_learning_rates_percentage` (default: `10`): min LR as percentage of peak LR.
- `--lora` / `--no-lora` (default: `True`): enable/disable LoRA replacement.
- `--lora_alpha` (default: `16`): LoRA alpha.
- `--lora_rank` (default: `16`): LoRA rank.

## Newly Added Args

- `--eval`
- `--test_data_path`
- `--evaluation_model`
- `--warmup`, `--cosine_decay`
- `--initial_learning_rates`, `--peak_learning_rates`
- `--learning_rate_warmup_percentage`, `--minimum_learning_rates_percentage`
- `--lora`, `--lora_alpha`, `--lora_rank`
- `--wandb`, `--wandb_project`, `--wandb_run_name`, `--wandb_entity`, `--wandb_tags`, `--wandb_artifacts`