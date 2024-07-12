from typing import List, Optional, TypedDict
import fire
from llama import Dialog, Llama
from llama.tokenizer import Message


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


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

    `max_gen_len` is optional in chat, as finetuned models are able to stop generations naturally.
    """

    if task == "text":
        instruct_model = False
    elif task == "chat":
        instruct_model = True

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
            [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
            # [
            #     {"role": "user", "content": "I am going to Paris, what should I see?"},
            #     {
            #         "role": "assistant",
            #         "content": """\
            # Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:
            # 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
            # 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
            # 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.
            # These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            #     },
            #     {"role": "user", "content": "What is so great about #1?"},
            # ],
            [
                {
                    "role": "user",
                    "content": "what do I need to do to cancel purchase {{Order Number}}?",
                }
            ],
            [
                {"role": "system", "content": "Always answer with Haiku"},
                {"role": "user", "content": "I am going to Paris, what should I see?"},
            ],
            [
                {
                    "role": "system",
                    "content": "Always answer with emojis",
                },
                {"role": "user", "content": "How to go from Beijing to NY?"},
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

    elif task == "text":
        prompts: List[str] = [
            # For these prompts, the expected answer is the natural continuation of the prompt
            "I believe the meaning of life is",
            "Simply put, the theory of relativity states that ",
            """A brief message congratulating the team on the launch:
            Hi everyone,
            I just """,
            # Few shot prompt (providing a few examples before asking model to complete more);
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
    fire.Fire(main)
