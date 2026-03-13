# python -m scripts.fine_tune_instruction --train
# python -m scripts.fine_tune_instruction --test
# python -m scripts.fine_tune_instruction --eval

from __future__ import annotations

import torch

from evaluation import (
    EvaluatorInstructionFineTuning,
)
from fine_tuning import LearningRateScheduler, TrainerInstructionFineTuning
from generate import Generator_Qwen_3
from parameter_efficient_fine_tuning import replace_linear_with_lora
from tokenizer import Qwen_3_Tokenizer
from utils import get_device

from .fine_tune_instruction_utils import (
    create_and_save_response_data,
    create_dataloaders,
    evaluate_model,
    load_and_split_dataset,
    load_model,
    parse_args,
    print_eval_losses,
)
from .wandb import WandbLogger, build_wandb_config, count_params

HYPERPARAMETER_INSTRUCTION_TUNING = {
    "model_name": "qwen3_0.6b_base",
    "model_size": "0.6B",
    "model_type": "instruct",
    "tokenizer_file_path": "./tokenizer/qwen_3_instruct_tokenizer.json",
    "dataset_file_path": "instruction_tuning_data.json",
    "checkpoint_path": "qwen3_0.6b_base",
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 0.1,
    "num_epochs": 1,
    "freq_evaluation": 25,
    "iter_evaluation": 50,
    "max_length": 1024,
    "ignore_index": -100,
    "mask_inputs": True,
    "shuffle": False,
    "drop_last": True,
    "num_workers": 0,
    "progress_update_freq": 20,
    "grad_clip": 1.0,
    "show_progress_bar": True,
    "freq_checkpoint": None,
    "apply_chat_template": True,
    "add_generation_prompt": True,
    "add_thinking": False,
    "shuffle_before_split": False,
    "seed": 42,
    "evaluation_model": "llama3.2:3b",
    "test_data_path": "instruction_tuning_data_with_response_qwen3_0.6b_base_20260309_152002.json",
    "warmup": True,
    "cosine_decay": True,
    "initial_learning_rates": [5e-8],
    "peak_learning_rates": [1e-4],
    "learning_rate_warmup_percentage": 5,
    "minimum_learning_rates_percentage": 30,
    "lora": True,
    "lora_rank": 16,
    "lora_alpha": 16,
    "wandb": True,
    "wandb_project": "sft",
    "wandb_run_name": "test",
    # "wandb_entity": "gaurisankarj1996-leiden-university",
    "wandb_entity": "omega",
    "wandb_tags": ["sft", "lora"],
    "wandb_artifacts": False,
}
if __name__ == "__main__":
    args = parse_args(HYPERPARAMETER_INSTRUCTION_TUNING)
    if not args.train and not args.test and not args.eval:
        raise ValueError(
            "Select at least one mode: --train and/or --test and/or --eval."
        )

    model_name = args.model_name
    model_size = args.model_size
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    tokenizer_file_path = args.tokenizer_file_path
    apply_chat_template = args.apply_chat_template
    add_generation_prompt = args.add_generation_prompt
    add_thinking = args.add_thinking
    dataset_file_path = args.dataset_file_path
    batch_size = args.batch_size
    shuffle = args.shuffle
    drop_last = args.drop_last
    num_workers = args.num_workers
    ignore_index = args.ignore_index
    mask_inputs = args.mask_inputs
    max_length = args.max_length
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    freq_evaluation = args.freq_evaluation
    iter_evaluation = args.iter_evaluation
    save_logs = args.save_logs
    show_progress_bar = args.show_progress_bar
    progress_update_freq = args.progress_update_freq
    grad_clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
    freq_checkpoint = args.freq_checkpoint
    shuffle_before_split = args.shuffle_before_split
    seed = args.seed
    evaluation_model = args.evaluation_model
    test_data_path = args.test_data_path
    warmup = args.warmup
    cosine_decay = args.cosine_decay
    initial_learning_rates = args.initial_learning_rates
    peak_learning_rates = args.peak_learning_rates
    learning_rate_warmup_percentage = args.learning_rate_warmup_percentage
    minimum_learning_rates_percentage = args.minimum_learning_rates_percentage
    lora = args.lora
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    use_wandb = args.wandb
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name
    wandb_entity = args.wandb_entity
    wandb_tags = args.wandb_tags
    wandb_artifacts = args.wandb_artifacts

    wandb_logger = WandbLogger(
        enabled=use_wandb,
        project=wandb_project,
        run_name=wandb_run_name,
        entity=wandb_entity,
        tags=wandb_tags,
        config=build_wandb_config(args),
    )

    try:
        # Get device
        device = get_device()
        print(f"Device: {device}")
        wandb_logger.set_summary("device", str(device))

        # Load model
        model, model_config = load_model(
            model_size=model_size,
            checkpoint_path=checkpoint_path,
            device=device,
            mode=args.test or args.eval,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        total_params, trainable_params = count_params(model=model)
        wandb_logger.set_summary("model_total_params", int(total_params))
        wandb_logger.set_summary("model_trainable_params", int(trainable_params))
        wandb_logger.watch_model(model=model, log_freq=max(freq_evaluation, 1))

        if lora:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters before: {total_params:,}")

            for param in model.parameters():
                param.requires_grad = False

            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters after: {total_params:,}")
            replace_linear_with_lora(model, rank=lora_rank, alpha=lora_alpha)

            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable LoRA parameters: {total_params:,}")
            wandb_logger.set_summary("lora_trainable_params", int(total_params))

            model.to(device)

        # Get tokenizer
        tokenizer = Qwen_3_Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            model_type=model_type,
            apply_chat_template=apply_chat_template,
            add_generation_prompt=add_generation_prompt,
            add_thinking=add_thinking,
        )

        # Load dataset and split into training, validation and testing
        train_data, val_data, test_data = load_and_split_dataset(
            dataset_file_path=dataset_file_path,
            shuffle_before_split=shuffle_before_split,
            seed=seed,
        )

        wandb_logger.set_summary("train_samples", len(train_data))
        wandb_logger.set_summary("val_samples", len(val_data))
        wandb_logger.set_summary("test_samples", len(test_data))
        wandb_logger.set_summary("optimizer", "AdamW")
        wandb_logger.set_summary("scheduler_warmup_enabled", bool(warmup))
        wandb_logger.set_summary("scheduler_cosine_enabled", bool(cosine_decay))
        if wandb_artifacts:
            wandb_logger.log_dataset_artifact(
                dataset_file_path=dataset_file_path,
                model_name=model_name,
                seed=seed,
                shuffle_before_split=shuffle_before_split,
            )

        # Create dataloaders
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            ignore_index=ignore_index,
            mask_inputs=mask_inputs,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            max_length=max_length,
            device=device,
        )
        wandb_logger.set_summary("train_dataloader_len", len(train_dataloader))
        wandb_logger.set_summary("val_dataloader_len", len(val_dataloader))
        wandb_logger.set_summary("test_dataloader_len", len(test_dataloader))

        # Create generator
        generator = Generator_Qwen_3(
            model=model,
            num_layers=model_config["num_layers"],
            context_length=model_config["context_length"],
            model_size=model_size,
            model_type=model_type,
            tokenizer_file_path=tokenizer_file_path,
        )

        if args.train:
            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            # Create learning rate scheduler
            learning_rate_scheduler = LearningRateScheduler(
                num_epochs=num_epochs,
                len_train_dataloader=len(train_dataloader),
            )

            if warmup:
                learning_rate_scheduler.initialize_learning_rates_warmup(
                    warmup_percentage=learning_rate_warmup_percentage,
                    initial_learning_rates=initial_learning_rates,
                    peak_learning_rates=peak_learning_rates,
                )

            if cosine_decay:
                learning_rate_scheduler.initialize_learning_rates_cosine_decay(
                    minimum_learning_rates_percentage=minimum_learning_rates_percentage,
                    initial_learning_rates=initial_learning_rates,
                    peak_learning_rates=peak_learning_rates,
                )

            # Create trainer
            trainer = TrainerInstructionFineTuning(
                model=model,
                optimizer=optimizer,
                device=device,
            )

            # Wandb callbacks
            def _wandb_step_callback(metrics: dict) -> None:
                wandb_logger.log(metrics)

            def _wandb_eval_callback(metrics: dict) -> None:
                wandb_logger.log(metrics)

            def _wandb_epoch_callback(metrics: dict) -> None:
                wandb_logger.log(metrics)

            def _wandb_checkpoint_callback(
                checkpoint_path: str, checkpoint_metrics: dict
            ) -> None:
                if wandb_artifacts:
                    wandb_logger.log_checkpoint(
                        model_name=model_name,
                        checkpoint_path=checkpoint_path,
                        checkpoint_metrics=checkpoint_metrics,
                    )
                else:
                    wandb_logger.log(checkpoint_metrics)

            # Train
            train_losses, val_losses, total_tokens_seen = trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=num_epochs,
                freq_evaluation=freq_evaluation,
                iter_evaluation=iter_evaluation,
                save_logs=save_logs,
                model_name=model_name,
                generator=generator,
                show_progress_bar=show_progress_bar,
                progress_update_freq=progress_update_freq,
                grad_clip=grad_clip,
                freq_checkpoint=freq_checkpoint,
                learning_rate_scheduler=learning_rate_scheduler,
                warmup=warmup,
                cosine_decay=cosine_decay,
                step_callback=_wandb_step_callback if use_wandb else None,
                eval_callback=_wandb_eval_callback if use_wandb else None,
                epoch_callback=_wandb_epoch_callback if use_wandb else None,
                checkpoint_callback=_wandb_checkpoint_callback if use_wandb else None,
            )
            if train_losses:
                wandb_logger.set_summary("final_train_loss", float(train_losses[-1]))
            if val_losses:
                wandb_logger.set_summary("final_val_loss", float(val_losses[-1]))
            if total_tokens_seen:
                wandb_logger.set_summary(
                    "total_tokens_seen", int(total_tokens_seen[-1])
                )
            step_metrics_history = getattr(trainer, "step_metrics_history", [])
            eval_metrics_history = getattr(trainer, "eval_metrics_history", [])
            epoch_metrics_history = getattr(trainer, "epoch_metrics_history", [])
            checkpoint_paths = getattr(trainer, "checkpoint_paths", [])
            wandb_logger.set_summary("num_train_steps", len(step_metrics_history))
            wandb_logger.set_summary("num_eval_points", len(eval_metrics_history))
            wandb_logger.set_summary(
                "num_epochs_completed",
                len(epoch_metrics_history),
            )
            wandb_logger.set_summary(
                "num_checkpoints_written",
                len(checkpoint_paths),
            )

        if args.test:
            evaluator = EvaluatorInstructionFineTuning(
                model=model,
                device=device,
            )
            print_eval_losses(
                evaluator=evaluator,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                iter_evaluation=iter_evaluation,
            )
            if use_wandb:
                test_loss = evaluator.calculate_loss_dataloader(
                    dataloader=test_dataloader,
                    num_batches=iter_evaluation,
                )
                wandb_logger.log({"test/loss": float(test_loss)})
                wandb_logger.set_summary("test_loss", float(test_loss))

            test_data = create_and_save_response_data(
                model_name=model_name,
                generator=generator,
                test_data=test_data,
                max_length=max_length,
                device=device,
            )
            if use_wandb and test_data:
                wandb_logger.set_summary("num_generated_responses", len(test_data))
                wandb_logger.log_response_table(test_data=test_data, max_rows=20)

        if args.eval:
            scores = evaluate_model(
                test_data=test_data,
                evaluation_model=evaluation_model,
                test_data_path=test_data_path,
            )
            if use_wandb and scores:
                wandb_logger.log(
                    {
                        "eval/model_score_avg": float(sum(scores) / len(scores)),
                        "eval/model_score_count": len(scores),
                    }
                )
                wandb_logger.set_summary(
                    "eval_model_score_avg", sum(scores) / len(scores)
                )
                wandb_logger.set_summary("eval_model_score_count", len(scores))
    finally:
        wandb_logger.finish()
