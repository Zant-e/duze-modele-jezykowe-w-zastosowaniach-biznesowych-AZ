from llama.generation import Llama

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import re
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
)
import json
import torch
from pathlib import Path

# options to add here:
# - whether to use lora, qlora or full finetune
# - whether to use add lora layer for every linear layer in model or just keys, queries and values
# - lora parameters
# - quantization precision for qlora
# - what monitoring/logging to use and where to save it
# - whether to use gradient accumulation
# - whether to use gradient checkpointing
# - learning rate, whether to use scheduling/warmup
# - batch size

ckpt_dir = "Meta-Llama-3-8B-Instruct"
tokenizer_path = "Meta-Llama-3-8B-Instruct/tokenizer.model"


def prep(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    seed: int = 1,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        seed=seed,
    )
    return generator


generator = prep(ckpt_dir, tokenizer_path, 512, 6)
model = generator.get_model()
tokenizer = generator.get_tokenizer()

generator.prep_for_training()

# list names of trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

generator.prep_for_training(output_requires_grad=False)

