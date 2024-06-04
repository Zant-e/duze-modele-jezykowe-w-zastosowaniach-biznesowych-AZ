import json
import time
import re
import gc
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from llama.model import ModelArgs, Transformer
from llama.tokenizer import ChatFormat, Tokenizer
import loralib as lora

import torch.nn.functional as F


class Llama:
    @staticmethod
    def build(
        instruct_model: bool,
        max_seq_len: int,
        max_batch_size: int = 1,
        seed: int = 1,
        lora_target: list = [],
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        quant_type: str = "",
        lora_ckpt_path: str = "",
        use_cache: bool = True,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            instruct_model (bool): Flag indicating whether to use the instruction-tuned model.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            seed (int, optional): Random seed. Defaults to 1.
            lora_target (list, optional): List of targets for LoRA adaptation. Accpeted values are: "q", "k", "v", "o", "ffn", "all_linear". Defaults to an empty list.
            lora_r (int, optional): Rank of the LoRA matrix. Defaults to 0.
            lora_alpha (int, optional): Scaling factor for the LoRA matrix. Defaults to 1.
            lora_dropout (float, optional): Dropout rate for the LoRA layers. Defaults to 0.0.
            quant_type (str, optional): String specifying quantization parameters. Accepted values are: "nf4", "fp4", "8bit". Defaults to an empty string.
            lora_ckpt_path (str, optional): Path to a LoRA checkpoint file. Defaults to an empty string.
            use_cache (bool, optional): Flag indicating whether to use cache for inference. Defaults to True.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Note:
            This method initializes the device to CUDA and loads the pre-trained model and tokenizer.
        """
        # seed must be the same in all processes
        torch.manual_seed(seed)
        start_time = time.time()

        # Set the checkpoint directory and tokenizer path based on the model type
        if instruct_model:
            ckpt_dir = "Meta-Llama-3-8B-Instruct"
            tokenizer_path = "Meta-Llama-3-8B-Instruct/tokenizer.model"
        else:
            ckpt_dir = "Meta-Llama-3-8B"
            tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"

        # Set the quantization type to None if it is an empty string
        if quant_type == "":
            quant = None
        else:
            quant = quant_type

        # Define a parameter dictionary based on the provided arguments
        params = {
            "dim": 4096,
            "n_layers": 32,
            "n_heads": 32,
            "n_kv_heads": 8,
            "vocab_size": 128256,
            "multiple_of": 1024,
            "ffn_dim_multiplier": 1.3,
            "norm_eps": 1e-05,
            "rope_theta": 500000.0,
            "lora_target": lora_target,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "quant_type": quant,
            "use_cache": use_cache,
            "max_seq_len": max_seq_len,
            "max_batch_size": max_batch_size,
        }

        # Pass the parameters to the ModelArgs class
        model_args: ModelArgs = ModelArgs(
            **params,
        )

        # Load the model checkpoint to CUDA (loading directly to cuda avoids excessive ram usage)
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        ckpt_path = checkpoints[0]
        checkpoint = torch.load(ckpt_path, map_location="cuda")

        # Modify the state dictionary if LoRA is applied to any layer
        # Regex used, as LoRA modifies model architecture
        new_state_dict = {}
        if params["lora_target"]:
            target_map = {
                "q": (
                    r"layers\.(\d+)\.attention\.wq\.weight",
                    "layers.{}.attention.wq.linear.weight",
                ),
                "k": (
                    r"layers\.(\d+)\.attention\.wk\.weight",
                    "layers.{}.attention.wk.linear.weight",
                ),
                "v": (
                    r"layers\.(\d+)\.attention\.wv\.weight",
                    "layers.{}.attention.wv.linear.weight",
                ),
                "o": (
                    r"layers\.(\d+)\.attention\.wo\.weight",
                    "layers.{}.attention.wo.linear.weight",
                ),
                "ffn": (
                    r"layers\.(\d+)\.feed_forward\.w(1|2|3)\.weight",
                    "layers.{}.feed_forward.w{}.linear.weight",
                ),
            }
            for lora_target_re in params["lora_target"]:
                for key, value in checkpoint.items():
                    for target, (pattern, key_format) in target_map.items():
                        if lora_target_re in {"all_linear", target}:
                            match = re.match(pattern, key)
                            if match:
                                groups = match.groups()
                                new_key = key_format.format(*groups)
                                new_state_dict[new_key] = value
                            else:
                                new_state_dict[key] = value
                        else:
                            new_state_dict[key] = value
        else:
            new_state_dict = checkpoint

        # Load the tokenizer, assert vocabulary size
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words

        # Create a Transformer model instance (in half precision, for lower RAM usage)
        model = Transformer(model_args).half()

        # Load the model state dictionary and LoRA checkpoint (if provided)
        model.load_state_dict(new_state_dict, strict=False)
        if lora_ckpt_path != "":
            model.load_state_dict(torch.load(lora_ckpt_path), strict=False)

        # Delete the state dictionary and checkpoint to free up memory
        del new_state_dict, checkpoint
        gc.collect()
        torch.cuda.empty_cache()

        # Move the model to CUDA and set the data type to bfloat16
        # Quantization happens here
        model.to("cuda", dtype=torch.bfloat16)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    def prep_for_training(
        self,
        layers_to_upcast: list = [],
        output_requires_grad: bool = False,
        embed_requires_grad: bool = False,
    ):
        """
        Prepare the model for training by freezing non-LoRA layers.
        Optional settings for upcasting layers to float32 and enabling gradient for output and embedding layers.

        Args:
            layers_to_upcast (list, optional): List of layers to upcast to float32. Accepted layers are: "tok_embeddings.weight", "output", "norm". Defaults to an empty list.
            output_requires_grad (bool, optional): Flag indicating whether to enable gradient for the output layer. Defaults to False.
            embed_requires_grad (bool, optional): Flag indicating whether to enable gradient for the embedding layer. Defaults to False.
        """
        # Making sure provided layers are valid
        accepted_upcast_params = ["tok_embeddings.weight", "output", "norm", None]
        if not all(layer in accepted_upcast_params for layer in layers_to_upcast):
            raise ValueError(
                "Invalid layer in layers_to_upcast. Accepted layers are: "
                + ", ".join(accepted_upcast_params)
            )

        # Making sure only LoRA layers are trainable
        lora.mark_only_lora_as_trainable(self.model)

        # Setting requires_grad for output and embedding layers
        if output_requires_grad:
            self.model.output.weight.requires_grad = True

        if embed_requires_grad and "embed" not in self.model.params.lora_target:
            self.model.tok_embeddings.weight.requires_grad = True

        # Upcasting layers to float32 based on the provided list
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layers_to_upcast):
                param.data = param.data.to(torch.float32)

        # print percentage of trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Trainable parameters: {trainable_params / total_params:.2%}")

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        additional_stop_tokens: Optional[
            List[str]
        ] = None,  # variable used by some lm_eval harness tasks
        use_cache: bool = True,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
            additional_stop_tokens (Optional[List[str]], optional): List of additional stop tokens. Defaults to None.
            use_cache (bool, optional): Flag indicating whether to use cache for inference. Defaults to True.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        params = self.model.params
        bsz = len(prompt_tokens)
        if use_cache:
            assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = list(self.tokenizer.stop_tokens)

        # encode additional stop tokens
        if additional_stop_tokens:
            for token in additional_stop_tokens:
                stop_tokens.extend(self.tokenizer.encode(token, bos=False, eos=False))

        stop_tokens = torch.tensor(stop_tokens).to("cuda")

        for cur_pos in range(min_prompt_len, total_len):
            if use_cache:
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            else:
                logits = self.model.forward(tokens[:, :cur_pos])
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any

            for stop_token in stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
