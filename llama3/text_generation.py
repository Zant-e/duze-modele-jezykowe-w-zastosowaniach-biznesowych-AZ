from typing import List, Optional, TypedDict
import json
from llama import Dialog, Llama
from llama.tokenizer import Message
import gc
import torch


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]
    logprobs: List[float]


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]
    logprobs: List[float]


def text_completion(
    generator,
    prompts: List[str],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
    logprobs: bool = False,
    echo: bool = False,
    use_cache: bool = False,
) -> List[CompletionPrediction]:
    """
    Perform text completion for a list of prompts using the language generation model.

    Args:
        prompts (List[str]): List of text prompts for completion.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    Returns:
        List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

    Note:
        This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
        If logprobs is True, token log probabilities are computed for each generated token.

    """
    if max_gen_len is None:
        max_gen_len = generator.model.params.max_seq_len - 1
    prompt_tokens = [
        generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts
    ]
    generation_tokens, generation_logprobs = generator.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
        echo=echo,
        use_cache=use_cache,
    )
    if logprobs:
        return [
            {
                "generation": generator.tokenizer.decode(t),
                "tokens": [generator.tokenizer.decode([x]) for x in t],
                "logprobs": logprobs_i,
            }
            for t, logprobs_i in zip(generation_tokens, generation_logprobs)
        ]
    return [{"generation": generator.tokenizer.decode(t)} for t in generation_tokens]


def chat_completion(
    generator,
    dialogs: List[Dialog],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
    logprobs: bool = False,
    use_cache: bool = False,
) -> List[ChatPrediction]:
    """
    Generate assistant responses for a list of conversational dialogs using the language generation model.

    Args:
        dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

    Returns:
        List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

    Note:
        This method generates assistant responses for the provided conversational dialogs.
        It employs nucleus sampling to introduce controlled randomness in text generation.
        If logprobs is True, token log probabilities are computed for each generated token.
    """
    if max_gen_len is None:
        max_gen_len = generator.model.params.max_seq_len - 1

    prompt_tokens = [
        generator.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
    ]
    generation_tokens, generation_logprobs = generator.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
        use_cache=use_cache,
    )
    if logprobs:
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": generator.tokenizer.decode(t),
                },
                "tokens": [generator.tokenizer.decode([x]) for x in t],
                "logprobs": logprobs_i,
            }
            for t, logprobs_i in zip(generation_tokens, generation_logprobs)
        ]
    return [
        {
            "generation": {
                "role": "assistant",
                "content": generator.tokenizer.decode(t),
            },
        }
        for t in generation_tokens
    ]


def main(
    task: str = "chat",  # or "text"
    model_type: str = "instruct",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    seed: int = 1,
    lora_target: List[str] = [],
    lora_ckpt_path: Optional[str] = "",  # "parameter_tuning/llama3_8b.pt"
    lora_r: int = 16,
    lora_alpha: int = 48,
    lora_dropout: float = 0.2,
    quant_type: str = "nf4",
    use_moe: bool = True,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    use_cache: bool = True,
):
    """
    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    """

    if model_type == "instruct":
        instruct_model = True
    else:
        instruct_model = False

    generator = Llama.build(
        instruct_model=instruct_model,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        seed=seed,
        lora_target=lora_target,
        lora_ckpt_path=lora_ckpt_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quant_type=quant_type,
        use_moe=use_moe,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        use_cache=use_cache,
    )

    if task == "chat":
        dialogs: List[Dialog] = [
            [
                {
                    "role": "user",
                    "content": "Good evening, I'd like to file a complaint. My order (number 5528) has arrived damaged.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "I can't log into my account, can you help me reset my password?",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "I'm trying to throw a birthday party for my dad, can you give me some ideas? He loves cars and hiking.",
                },
            ],
        ]
        results = chat_completion(
            generator=generator,
            dialogs=dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")

        del dialogs, results, generator
        gc.collect()
        torch.cuda.empty_cache()

    elif task == "text":
        prompts: List[str] = [
            "I believe the meaning of life is",
            "Simply put, the theory of relativity states that ",
            """A brief message congratulating the team on the launch:
            Hi everyone,
            I just """,
            """Translate English to French:
            sea otter => loutre de mer
            peppermint => menthe poivrée
            plush girafe => girafe peluche
            cheese =>""",
        ]

        results = text_completion(
            generator=generator,
            prompts=prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
        )
        for prompt, result in zip(prompts, results):
            print(prompt)
            print(f"> {result['generation']}")
            print("\n==================================\n")


if __name__ == "__main__":

    customer_path = "tuned_checkpoints/customer_dataset"
    mixed_path = "tuned_checkpoints/mixed_dataset"

    customer_models = [
        "llama3_8b_instruct_customer",
        "llama3_8b_moe_instruct_customer",
        "llama3_8b_pretrained_customer",
        "llama3_8b_moe_pretrained_customer",
    ]
    mixed_models = [
        "llama3_8b_instruct_mixed",
        "llama3_8b_moe_instruct_mixed",
        "llama3_8b_pretrained_mixed",
        "llama3_8b_moe_pretrained_mixed",
    ]

    for i in ["customer", "mixed"]:
        if i == "customer":
            path = customer_path
            models = customer_models
        else:
            path = mixed_path
            models = mixed_models

        for model_name in models:
            modelpath = f"{path}/{model_name}"

            json_config = json.load(open(f"{modelpath}/model_config.json", "r"))

            model_config = {
                "task": "chat",
                "model_type": json_config["model_type"],
                "quant_type": json_config["quant_type"],
                "lora_ckpt_path": f"{modelpath}/model.pt",
                "lora_target": json_config["lora_target"],
                "lora_r": json_config["lora_r"],
                "lora_alpha": json_config["lora_alpha"],
                "lora_dropout": json_config["lora_dropout"],
                "max_seq_len": json_config["context_length"],
                "seed": 1,
                "use_moe": json_config["use_moe"],
                "num_experts": json_config["num_experts"],
                "num_experts_per_tok": json_config["num_experts_per_tok"],
                "max_batch_size": 3,
                "use_cache": True,
            }

            print(
                f"""
                    ######
                    Running {model_name}
                    ######"""
            )
            main(**model_config)

    for i in ["instruct", "pretrained"]:
        model_config = {
            "task": "chat",
            "model_type": i,
            "quant_type": "nf4",
            "lora_r": 0,
            "lora_alpha": 0,
            "lora_dropout": 0,
            "max_seq_len": 8192,
            "seed": 1,
            "use_moe": False,
            "num_experts": 0,
            "num_experts_per_tok": 0,
            "max_batch_size": 3,
            "use_cache": True,
        }

        if i == "pretrained":
            model_config["max_gen_len"] = 1024

        print(
            f"""
                ######
                Running base {i} model in nf4
                ######"""
        )

        main(**model_config)
