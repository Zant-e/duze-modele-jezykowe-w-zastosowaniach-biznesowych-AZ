import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch
from torch import nn
from bitsandbytes.nn import LinearFP4, LinearNF4, Linear8bitLt
import loralib as lora
import torch.nn.functional as F
from typing import List, Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    lora_target: List[str] = field(default_factory=lambda: ["all_linear"])
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    quant_type: str = "nf4"

    max_batch_size: int = 32
    max_seq_len: int = 2048
    use_cache: bool = True

    moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def quant_layer(quant_type, in_features: int, out_features: int, bias: bool):
    if quant_type == "fp4":
        return LinearFP4(in_features, out_features, bias)
    elif quant_type == "nf4":
        return LinearNF4(in_features, out_features, bias)
    elif quant_type == "8bit":
        return Linear8bitLt(in_features, out_features, bias, has_fp16_weights=False)
    elif quant_type == "":
        return nn.Linear(in_features, out_features, bias)
    else:
        raise ValueError("Invalid quant type")


class LoraQuantLinear(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        quant_type: str = "",
        bias: bool = False,
        expert_core: Optional[nn.Module] = None,
    ):
        nn.Module.__init__(self)
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        if expert_core is None:
            self.linear = quant_layer(quant_type, in_features, out_features, bias)
        else:
            self.linear = expert_core

        # Actual trainable parameters
        if r > 0 and expert_core is None:
            self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.linear.weight.requires_grad = False
        else:
            self.lora_A = nn.Parameter(
                self.linear.w1.weight.new_zeros((r, in_features))
            )
            self.lora_B = nn.Parameter(
                self.linear.w1.weight.new_zeros((out_features, r))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.linear.w1.weight.requires_grad = False
            self.linear.w2.weight.requires_grad = False
            self.linear.w3.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        try:
            self.linear.reset_parameters()
        except AttributeError:
            self.linear.w1.reset_parameters()
            self.linear.w2.reset_parameters()
            self.linear.w3.reset_parameters()
        if hasattr(self, "lora_A"):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.linear.train(mode)

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            result = self.linear(x)
            result += (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
            return result
        else:
            return self.linear(x)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.use_cache = args.use_cache

        self.wq = (
            LoraQuantLinear(
                args.dim,
                args.n_heads * self.head_dim,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                quant_type=args.quant_type,
                bias=False,
            )
            if "all_linear" in args.lora_target or "q" in args.lora_target
            else quant_layer(
                args.quant_type, args.dim, args.n_heads * self.head_dim, bias=False
            )
        )
        self.wk = (
            LoraQuantLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                quant_type=args.quant_type,
                bias=False,
            )
            if "all_linear" in args.lora_target or "k" in args.lora_target
            else quant_layer(
                args.quant_type, args.dim, self.n_kv_heads * self.head_dim, bias=False
            )
        )
        self.wv = (
            LoraQuantLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                quant_type=args.quant_type,
                bias=False,
            )
            if "all_linear" in args.lora_target or "v" in args.lora_target
            else quant_layer(
                args.quant_type, args.dim, self.n_kv_heads * self.head_dim, bias=False
            )
        )
        self.wo = (
            LoraQuantLinear(
                args.n_heads * self.head_dim,
                args.dim,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                quant_type=args.quant_type,
                bias=False,
            )
            if "all_linear" in args.lora_target or "o" in args.lora_target
            else quant_layer(
                args.quant_type, args.n_heads * self.head_dim, args.dim, bias=False
            )
        )

        if self.use_cache:
            self.cache_k = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()
        else:
            self.cache_k = None
            self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.use_cache:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        args: ModelArgs,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        l = ["all_linear", "ffn"]
        if any([x in args.lora_target for x in l]) and args.moe is False:
            self.w1 = LoraQuantLinear(
                dim,
                hidden_dim,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                quant_type=args.quant_type,
                bias=False,
            )
            self.w2 = LoraQuantLinear(
                hidden_dim,
                dim,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                quant_type=args.quant_type,
                bias=False,
            )
            self.w3 = LoraQuantLinear(
                dim,
                hidden_dim,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                quant_type=args.quant_type,
                bias=False,
            )
        else:
            self.w1 = quant_layer(args.quant_type, dim, hidden_dim, bias=False)
            self.w2 = quant_layer(args.quant_type, hidden_dim, dim, bias=False)
            self.w3 = quant_layer(args.quant_type, dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, args: ModelArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = args
        self.experts_logits = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs dim: (bs, seqlen, dim)
        gate_logits = self.gate(inputs)  # (bs, seqlen, num_experts)
        self.experts_logits = gate_logits
        weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok, dim=-1
        )  # (bs, seqlen, num_experts_per_tok)

        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)  # (bs, seqlen, dim)

        for i, expert in enumerate(self.experts):
            batch_idx, seqlen, nth_expert = torch.where(selected_experts == i)
            # which batch was selected, which token, which expert (selected_tokens)

            expert_output = expert(inputs[batch_idx, seqlen])  # (selected_tokens, dim)
            weight = weights[batch_idx, seqlen, nth_expert].unsqueeze(
                -1
            )  # (selected_tokens, 1)
            results[batch_idx, seqlen] += weight * expert_output
        return results


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        if args.moe is False:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                args=args,
            )
        else:
            self.expert_core = FeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                args=args,
            )
            self.feed_forward = MoeLayer(
                experts=[
                    LoraQuantLinear(
                        args.dim,
                        args.dim,
                        args.lora_r,
                        args.lora_alpha,
                        args.lora_dropout,
                        args.quant_type,
                        bias=False,
                        expert_core=self.expert_core,
                    )
                    for _ in range(args.num_experts)
                ],
                gate=nn.Linear(args.dim, args.num_experts, bias=False),
                args=args,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_cache = args.use_cache
        self.experts_logits = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        if isinstance(self.feed_forward, MoeLayer):
            self.experts_logits = self.feed_forward.experts_logits
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        if "embed" in params.lora_target:
            self.tok_embeddings = lora.Embedding(
                params.vocab_size,
                params.dim,
                r=params.lora_r,
                lora_alpha=params.lora_alpha,
            )
        else:
            self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = (
            LoraQuantLinear(
                params.dim,
                params.vocab_size,
                r=params.lora_r,
                lora_alpha=params.lora_alpha,
                lora_dropout=params.lora_dropout,
                quant_type="",
                bias=False,
            )
            if "output" in params.lora_target
            else nn.Linear(params.dim, params.vocab_size, bias=False)
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

        self.use_cache = params.use_cache
        self.all_expert_logits = []

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass of the model

        Args:
            tokens (torch.Tensor): The input tensor of shape (bsz, seqlen)
            start_pos (int): The starting position in the cache. If `use_cache` is False, `start_pos` is ignored.

        Returns:
            torch.Tensor: The output tensor
            If `moe` is True, the function also returns a tuple of expert logits.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        # freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        freqs_cis = (
            self.freqs_cis[start_pos : start_pos + seqlen]
            if self.use_cache
            else self.freqs_cis[:seqlen]
        )

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            if self.use_cache:
                # When performing key-value caching, we compute the attention scores
                # only for the new sequence. Thus, the matrix of scores is of size
                # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
                # j > cache_len + i, since row i corresponds to token cache_len + i.
                mask = torch.hstack(
                    [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
                ).type_as(h)

        self.all_expert_logits = []
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
            if layer.experts_logits is not None:
                # self.all_expert_logits.append(layer.experts_logits)
                batch_size, sequence_length, num_experts = layer.experts_logits.shape
                reshaped_logits = layer.experts_logits.reshape(
                    batch_size * sequence_length, num_experts
                )
                self.all_expert_logits.append(reshaped_logits)
        h = self.norm(h)
        output = self.output(h).float()

        if self.params.moe:
            # return output, self.all_expert_logits
            return output, tuple(self.all_expert_logits)
        else:
            return output
