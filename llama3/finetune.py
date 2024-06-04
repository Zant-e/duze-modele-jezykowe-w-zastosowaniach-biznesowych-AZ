import os

import dataclasses
import fire
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils.configs import train_config as TRAIN_CONFIG
from utils.dataset_utils import get_preprocessed_curious, ConcatDataset, collate_fn

from utils.configs import update_config
from utils.train_utils import train
from llama import Llama
from datasets import load_dataset


def setup_wandb(train_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from utils.configs import wandb_config as WANDB_CONFIG

    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    return run


def main(**kwargs):
    # Update the configuration for training
    train_config = TRAIN_CONFIG()
    update_config((train_config), **kwargs)
    # Set the seeds for reproducibility
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    wandb_run = None

    if train_config.use_wandb:
        wandb_run = setup_wandb(train_config, **kwargs)

    # Load the pre-trained model and setup its configuration
    generator = Llama.build(
        instruct_model=train_config.model_type == "instruct",
        max_seq_len=train_config.context_length,
        seed=train_config.seed,
        lora_target=train_config.lora_target,
        lora_r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        lora_dropout=train_config.lora_dropout,
        quant_type=train_config.quant_type,
        lora_ckpt_path=train_config.lora_ckpt_path,
        use_cache=False,
    )

    generator.prep_for_training(
        layers_to_upcast=train_config.layers_to_upcast,
        output_requires_grad=train_config.output_requires_grad,
        embed_requires_grad=train_config.embed_requires_grad,
    )

    model = generator.model
    tokenizer = generator.tokenizer

    # Load and preprocess the dataset for training and validation
    if train_config.dataset == "curious_dataset":
        dataset = load_dataset("xiyuez/im-feeling-curious", split="train")
        split = dataset.train_test_split(test_size=0.2)
        train_split = split["train"]
        val_split = split["test"]
        dataset_train = get_preprocessed_curious(tokenizer, train_split)

        dataset_val = get_preprocessed_curious(tokenizer, val_split)

    print(f"--> Training Set Length = {len(dataset_train)}")
    print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, seq_length=768)

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, pad_id=tokenizer.eos_id),
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, seq_length=768)

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn=lambda batch: collate_fn(batch, pad_id=tokenizer.eos_id),
        )

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        wandb_run,
    )
    [print(f"Key: {k}, Value: {v}") for k, v in results.items()]
    if train_config.use_wandb:
        for k, v in results.items():
            wandb_run.summary[k] = v


if __name__ == "__main__":
    fire.Fire(main)
