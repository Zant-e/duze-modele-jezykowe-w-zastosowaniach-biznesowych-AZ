from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class train_config:
    # Model settings
    model_name: str = "model"
    model_type: str = "instruct"  # pretrained or instruct
    quant_type: str = "nf4"  # nf4, fp4, 8bit, ""
    lora_ckpt_path: str = ""  # path to lora checkpoint
    output_dir: str = "parameter_tuning/llama3_8b_moe_instruct_customer"

    # LoRA settings
    lora_target: List[str] = field(
        default_factory=lambda: ["all_linear", "output"]
    )  # q, k, v, o, ffn, all_linear (without output), output, embed
    lora_r: int = 16
    lora_alpha: int = 128
    lora_dropout: float = 0.1

    # Training settings
    batch_size_training: int = 1
    val_batch_size: int = 1
    context_length: int = 8192
    gradient_accumulation_steps: int = 64
    num_epochs: int = 1
    max_train_step: int = 99999999
    max_eval_step: int = 9999999
    lr: float = 1e-4
    weight_decay: float = 0
    gamma: float = 0.1
    batching_strategy: str = "padding"  # packing or padding

    # Gradient settings
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0

    # Data loading settings
    num_workers_dataloader: int = 1

    # Validation settings
    run_validation: bool = True

    # Miscellaneous settings
    seed: int = 1
    dataset: str = "cust_support"  # curious_dataset, pure_dove or cust_support

    # Save settings
    save_model: bool = True
    save_metrics: bool = False

    # WandB settings
    use_wandb: bool = True

    # MoE settings
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 3
    penalty_alpha: float = 1e-2


@dataclass
class wandb_config:
    project: str = "llama8b_comparison"  # wandb project name
    entity: Optional[str] = None  # wandb entity name
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")
