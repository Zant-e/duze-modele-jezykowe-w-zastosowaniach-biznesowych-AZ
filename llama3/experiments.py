from llama.generation import Llama

from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments

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

    model = generator.get_model()
    tokenizer = generator.get_tokenizer()

    return model, tokenizer


model_path = "D:/LLMs/Meta-Llama-3-8B-Instruct"
tokenizer_path = "D:/LLMs/Meta-Llama-3-8B-Instruct/tokenizer.model"

model, tokenizer = prep(model_path, tokenizer_path, 512, 4)
