from typing import List, Optional, TypedDict

import fire

from llama import Llama


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
    use_cache: bool = True,
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


def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 1,
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """

    generator = Llama.build(
        instruct_model=False,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        seed=1,
        lora_target=["q"],
        lora_r=4,
        lora_alpha=1,
        lora_dropout=0.1,
        quant_type="nf4",
        lora_ckpt_path="",
        use_cache=False,
    )

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
        use_cache=False,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
