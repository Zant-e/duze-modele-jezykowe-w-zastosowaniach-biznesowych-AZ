from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class train_config:
    # Model settings
    model_name: str = "testmodel"
    model_type: str = "pretrained"  # pretrained or instruct
    quant_type: str = "nf4"  # nf4, fp4, 8bit, ""
    lora_ckpt_path: str = ""  # path to lora checkpoint
    output_dir: str = "testrun"
    layers_to_upcast: List[str] = field(default_factory=list)
    output_requires_grad: bool = False
    embed_requires_grad: bool = False

    # LoRA settings
    lora_target: List[str] = field(default_factory=lambda: ["q", "k", "v", "o", "ffn"])
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.1

    # Training settings
    batch_size_training: int = 4
    val_batch_size: int = 4
    context_length: int = 8192
    gradient_accumulation_steps: int = 1
    num_epochs: int = 10
    max_train_step: int = 10
    max_eval_step: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.85
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
    dataset: str = "curious_dataset"

    # Save settings
    save_model: bool = True
    save_metrics: bool = True

    # Profiling settings
    use_profiler: bool = False  # As of now, profiler's CUPTI crashes - fix later.
    profiler_dir: str = "testrun/profiler/results"

    # WandB settings
    use_wandb: bool = True


@dataclass
class wandb_config:
    project: str = "llama_recipes"  # wandb project name
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
                        # In case of specialized config we can warn user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")
