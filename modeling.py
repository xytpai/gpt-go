import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor
from transformers import AutoTokenizer

import distributed_utils as dutils
from map_state_names import map_qwen3_moe_name
from tensor_parallel import ColumnParallelLinear, RowParallelLinear


@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 48
    n_heads: int = 32
    n_kv_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 151936
    moe_intermediate_size: int = 768
    norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    max_seq_len: int = 262144
    dropout_prob: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    num_experts: int = 128
    num_activated_experts: int = 8
    norm_experts_prob: bool = True
    eos_token_ids: tuple[int, ...] = (151645, 151643)
    ignore_index: int = -100

    @classmethod
    def from_hf_config(
        cls,
        config: dict,
        max_seq_len: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "ModelArgs":
        if config.get("model_type") != "qwen3_moe":
            raise ValueError(
                f"Expected model_type='qwen3_moe', got {config.get('model_type')!r}"
            )

        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        config_dtype = dtype_by_name.get(config.get("torch_dtype"), torch.bfloat16)
        eos_token_ids = config.get("eos_token_id", (151645, 151643))
        if isinstance(eos_token_ids, int):
            eos_token_ids = (eos_token_ids,)
        else:
            eos_token_ids = tuple(eos_token_ids)

        return cls(
            dim=config["hidden_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config["num_key_value_heads"],
            head_dim=config.get(
                "head_dim",
                config["hidden_size"] // config["num_attention_heads"],
            ),
            vocab_size=config["vocab_size"],
            moe_intermediate_size=config["moe_intermediate_size"],
            norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
            max_seq_len=max_seq_len or config["max_position_embeddings"],
            dropout_prob=config.get("attention_dropout", 0.0),
            dtype=dtype or config_dtype,
            num_experts=config["num_experts"],
            num_activated_experts=config["num_experts_per_tok"],
            norm_experts_prob=config.get("norm_topk_prob", True),
            eos_token_ids=eos_token_ids,
        )


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_size)
        self.register_buffer(
            "k_cache", torch.empty(cache_shape, dtype=dtype, device=device)
        )
        self.register_buffer(
            "v_cache", torch.empty(cache_shape, dtype=dtype, device=device)
        )
        self.cache_length = 0

    def update(
        self, input_pos: Tensor, k_val: Tensor, v_val: Tensor
    ) -> tuple[Tensor, Tensor]:
        end_pos = int(input_pos.max().item()) + 1
        if end_pos > self.k_cache.size(2):
            raise ValueError(
                f"KV cache length {end_pos} exceeds maximum {self.k_cache.size(2)}"
            )
        self.k_cache[: k_val.size(0), :, input_pos] = k_val
        self.v_cache[: v_val.size(0), :, input_pos] = v_val
        self.cache_length = max(self.cache_length, end_pos)
        return (
            self.k_cache[: k_val.size(0), :, : self.cache_length],
            self.v_cache[: v_val.size(0), :, : self.cache_length],
        )


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_eps: float,
        dtype: torch.dtype,
        device: str | torch.device,
    ):
        super().__init__()
        self.eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(
            x_float.pow(2).mean(-1, keepdim=True) + self.eps
        )
        return self.weight * x_norm.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        theta: float,
        device: str | torch.device,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, input_pos: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        freqs = torch.outer(input_pos.float(), self.inv_freq.float())
        embeddings = torch.cat((freqs, freqs), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: Tensor, xk: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (
        xq * cos + rotate_half(xq) * sin,
        xk * cos + rotate_half(xk) * sin,
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, device: str | torch.device):
        super().__init__()
        parallel_size = dutils.get_tensor_parallel_world_size()
        dutils.ensure_divisibility(args.n_heads, parallel_size)
        dutils.ensure_divisibility(args.n_kv_heads, parallel_size)

        self.dropout_prob = args.dropout_prob
        self.n_local_heads = args.n_heads // parallel_size
        self.n_local_kv_heads = args.n_kv_heads // parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * args.head_dim,
            bias=False,
            dtype=args.dtype,
            device=device,
            gather_output=False,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False,
            dtype=args.dtype,
            device=device,
            gather_output=False,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False,
            dtype=args.dtype,
            device=device,
            gather_output=False,
        )
        self.wo = RowParallelLinear(
            args.n_heads * args.head_dim,
            args.dim,
            bias=False,
            dtype=args.dtype,
            device=device,
            input_is_parallel=True,
        )
        self.q_norm = RMSNorm(args.head_dim, args.norm_eps, args.dtype, device)
        self.k_norm = RMSNorm(args.head_dim, args.norm_eps, args.dtype, device)
        self.rotary_emb = RotaryEmbedding(args.head_dim, args.rope_theta, device)
        self.kv_cache: Optional[KVCache] = None

    def forward(self, x: Tensor, input_pos: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.shape

        xq = self.wq(x).view(
            batch_size, seq_length, self.n_local_heads, self.head_dim
        )
        xk = self.wk(x).view(
            batch_size, seq_length, self.n_local_kv_heads, self.head_dim
        )
        xv = self.wv(x).view(
            batch_size, seq_length, self.n_local_kv_heads, self.head_dim
        )
        xq = self.q_norm(xq).transpose(1, 2)
        xk = self.k_norm(xk).transpose(1, 2)
        xv = xv.transpose(1, 2)

        cos, sin = self.rotary_emb(input_pos, xq.dtype)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        if self.kv_cache is not None:
            xk, xv = self.kv_cache.update(input_pos, xk, xv)

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)
        key_length = xk.size(2)

        starts_at_zero = int(input_pos[0].item()) == 0
        is_prefill = starts_at_zero and seq_length == key_length
        attention_mask = None
        if not is_prefill and seq_length > 1:
            key_positions = torch.arange(key_length, device=x.device)
            attention_mask = (
                key_positions[None, None, None, :]
                <= input_pos[None, None, :, None]
            )

        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_prefill,
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, -1)
        )
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device: str | torch.device,
        reduce_output: bool,
    ):
        super().__init__()
        hidden_dim = args.moe_intermediate_size
        self.w1 = ColumnParallelLinear(
            args.dim,
            hidden_dim,
            bias=False,
            dtype=args.dtype,
            device=device,
            gather_output=False,
        )
        self.w2 = RowParallelLinear(
            hidden_dim,
            args.dim,
            bias=False,
            dtype=args.dtype,
            device=device,
            input_is_parallel=True,
            reduce_output=reduce_output,
        )
        self.w3 = ColumnParallelLinear(
            args.dim,
            hidden_dim,
            bias=False,
            dtype=args.dtype,
            device=device,
            gather_output=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Router(nn.Module):
    def __init__(self, args: ModelArgs, device: str | torch.device):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                args.num_experts,
                args.dim,
                dtype=args.dtype,
                device=device,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)


class MOEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs, device: str | torch.device):
        super().__init__()
        self.num_experts = args.num_experts
        self.num_activated_experts = args.num_activated_experts
        self.norm_experts_prob = args.norm_experts_prob
        self.gate = Router(args, device)
        # Expert outputs are reduced once after routing instead of once per expert.
        self.experts = nn.ModuleList(
            [
                FeedForward(args, device, reduce_output=False)
                for _ in range(args.num_experts)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length, dim = x.shape
        flat_x = x.view(-1, dim)
        router_logits = self.gate(flat_x)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_activated_experts, dim=-1
        )
        if self.norm_experts_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights.to(flat_x.dtype)

        output = torch.zeros_like(flat_x)
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero().flatten()

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor.item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = flat_x[token_idx]
            current_state = self.experts[expert_idx](current_state)
            current_state = (
                current_state * routing_weights[token_idx, top_k_pos, None]
            )
            output.index_add_(0, token_idx, current_state.to(output.dtype))

        output = dutils.reduce_from_tensor_parallel_region(output)
        return output.view(batch_size, seq_length, dim)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        device: str | torch.device,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args, device)
        self.feed_forward = MOEFeedForward(args, device)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)

    def forward(self, x: Tensor, input_pos: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x), input_pos)
        return x + self.feed_forward(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device: str | torch.device = "cuda:0",
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.max_seq_len = args.max_seq_len

        embedding_weight = torch.empty(
            args.vocab_size, args.dim, dtype=args.dtype, device=device
        )
        self.tok_embeddings = nn.Embedding(
            args.vocab_size,
            args.dim,
            _weight=embedding_weight,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(layer_id, args, device)
                for layer_id in range(args.n_layers)
            ]
        )
        self.norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
        self.output = ColumnParallelLinear(
            args.dim,
            args.vocab_size,
            bias=False,
            dtype=args.dtype,
            device=device,
            gather_output=True,
        )

    @property
    def device(self) -> torch.device:
        return self.output.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.output.weight.dtype

    def size(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def assign_kv_cache(
        self, max_batch_size: int, cache_seq_len: Optional[int] = None
    ) -> None:
        cache_seq_len = cache_seq_len or self.max_seq_len
        if cache_seq_len > self.max_seq_len:
            raise ValueError(
                f"KV cache length {cache_seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )
        parallel_size = dutils.get_tensor_parallel_world_size()
        local_kv_heads = self.args.n_kv_heads // parallel_size
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size,
                cache_seq_len,
                local_kv_heads,
                self.args.head_dim,
                self.dtype,
                self.device,
            )

    def forward(
        self,
        tokens: Tensor,
        input_pos: Optional[Tensor] = None,
        logits_to_keep: int = 0,
    ) -> Tensor:
        _, seq_length = tokens.shape
        if input_pos is None:
            input_pos = torch.arange(seq_length, device=tokens.device)
        if int(input_pos.max().item()) >= self.max_seq_len:
            raise ValueError(
                f"Position exceeds configured max_seq_len={self.max_seq_len}"
            )

        hidden_states = self.tok_embeddings(tokens)
        for layer in self.layers:
            hidden_states = layer(hidden_states, input_pos)
        hidden_states = self.norm(hidden_states)
        if logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        return self.output(hidden_states)

    @staticmethod
    def post_loss(logits: Tensor, target_ids: Tensor, ignore_index: int = -100):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.contiguous().view(-1),
            ignore_index=ignore_index,
        )


def _partition_dimension(name: str) -> Optional[int]:
    column_suffixes = (
        ".attention.wq.weight",
        ".attention.wk.weight",
        ".attention.wv.weight",
        ".w1.weight",
        ".w3.weight",
    )
    row_suffixes = (
        ".attention.wo.weight",
        ".w2.weight",
    )
    if name == "output.weight" or name.endswith(column_suffixes):
        return 0
    if name.endswith(row_suffixes):
        return 1
    return None


@torch.no_grad()
def load_qwen3_moe_checkpoint(model: Transformer, model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
        shard_names = list(dict.fromkeys(weight_map.values()))
    else:
        shard_names = [path.name for path in sorted(model_dir.glob("*.safetensors"))]
        if not shard_names:
            raise FileNotFoundError(
                f"No safetensors checkpoint found under {model_dir}"
            )

    parameters = dict(model.named_parameters())
    missing = set(parameters)
    tp_size = dutils.get_tensor_parallel_world_size()
    tp_rank = dutils.get_tensor_parallel_rank()

    for shard_index, shard_name in enumerate(shard_names, start=1):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
        if tp_rank == 0:
            print(f"Loading shard {shard_index}/{len(shard_names)}: {shard_name}")

        with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
            for source_name in checkpoint.keys():
                target_name = map_qwen3_moe_name(source_name)
                parameter = parameters.get(target_name)
                if parameter is None:
                    raise KeyError(
                        f"Checkpoint key {source_name!r} maps to unknown "
                        f"parameter {target_name!r}"
                    )

                tensor = checkpoint.get_tensor(source_name)
                partition_dim = _partition_dimension(target_name)
                if partition_dim is not None and tp_size > 1:
                    dutils.ensure_divisibility(
                        tensor.size(partition_dim), tp_size
                    )
                    tensor = tensor.chunk(tp_size, dim=partition_dim)[tp_rank]

                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise ValueError(
                        f"Shape mismatch for {source_name}: checkpoint "
                        f"{tuple(tensor.shape)}, model {tuple(parameter.shape)}"
                    )
                parameter.copy_(
                    tensor.to(device=parameter.device, dtype=parameter.dtype)
                )
                missing.discard(target_name)

    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise KeyError(f"Checkpoint is missing {len(missing)} parameters: {preview}")


def sample_next_token(
    logits: Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    probabilities = F.softmax(logits / temperature, dim=-1)
    if 0 < top_k < probabilities.size(-1):
        top_k_probs, top_k_indices = torch.topk(
            probabilities, k=top_k, dim=-1
        )
        filtered = torch.zeros_like(probabilities)
        probabilities = filtered.scatter(-1, top_k_indices, top_k_probs)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(
            probabilities, descending=True, dim=-1
        )
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative_probs - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled)
    return torch.multinomial(probabilities, num_samples=1)


class WorkerProc:
    def __init__(
        self,
        model_dir: str,
        world_size: int,
        tp_size: int,
        max_seq_len: Optional[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        dtype: Optional[torch.dtype],
        init_url: Optional[str],
        backend: Optional[str],
    ):
        self.model_dir = model_dir
        self.world_size = world_size
        self.tp_size = tp_size
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.dtype = dtype
        self.init_url = init_url
        self.backend = backend

    @torch.inference_mode()
    def generate(self, rank: int, model: Transformer, input_text: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=False
        )
        messages = [{"role": "user", "content": input_text}]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        tokens = encoded["input_ids"].to(model.device)
        prompt_length = tokens.size(1)
        if prompt_length + self.max_new_tokens > model.max_seq_len:
            raise ValueError(
                f"Prompt ({prompt_length}) + output ({self.max_new_tokens}) "
                f"exceeds max_seq_len={model.max_seq_len}"
            )

        model.assign_kv_cache(
            max_batch_size=1,
            cache_seq_len=prompt_length + self.max_new_tokens,
        )
        input_pos = torch.arange(tokens.size(1), device=model.device)
        generated: list[int] = []

        if torch.cuda.is_available():
            torch.cuda.synchronize(model.device)
        start_time = time.perf_counter()

        for step in range(self.max_new_tokens):
            logits = model(tokens, input_pos, logits_to_keep=1)[:, -1, :]
            if rank == 0:
                next_token = sample_next_token(
                    logits,
                    self.temperature,
                    self.top_p,
                    self.top_k,
                )
            else:
                next_token = torch.empty(
                    (1, 1), dtype=torch.long, device=model.device
                )
            if self.world_size > 1:
                dist.broadcast(next_token, src=0)

            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id in model.args.eos_token_ids:
                break

            tokens = next_token
            input_pos = torch.tensor(
                [prompt_length + step],
                dtype=torch.long,
                device=model.device,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - start_time

        if rank == 0:
            text = tokenizer.decode(generated, skip_special_tokens=True)
            print(text)
            print(
                f"\nGenerated {len(generated)} tokens in {elapsed:.3f}s "
                f"({len(generated) / elapsed:.2f} tokens/s)"
            )

    def __call__(self, rank: int, input_text: str) -> None:
        dutils.init_tensor_parallel(
            rank,
            self.world_size,
            self.tp_size,
            backend=self.backend,
            init_url=self.init_url,
        )
        device: str | torch.device
        if torch.cuda.is_available():
            device = f"cuda:{rank}"
        else:
            device = "cpu"

        with open(
            os.path.join(self.model_dir, "config.json"),
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)
        model_args = ModelArgs.from_hf_config(
            config,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
        )
        generation_config_path = os.path.join(
            self.model_dir, "generation_config.json"
        )
        if os.path.exists(generation_config_path):
            with open(generation_config_path, "r", encoding="utf-8") as file:
                generation_config = json.load(file)
            eos_token_ids = generation_config.get("eos_token_id")
            if isinstance(eos_token_ids, int):
                eos_token_ids = [eos_token_ids]
            if eos_token_ids:
                model_args.eos_token_ids = tuple(eos_token_ids)
        model = Transformer(model_args, device=device)
        load_qwen3_moe_checkpoint(model, self.model_dir)
        model.eval()

        if rank == 0:
            local_billions = model.size() / 1_000_000_000
            print(
                f"Qwen3-30B-A3B-Instruct-2507: "
                f"{local_billions:.3f}B parameters on each TP rank"
            )
        self.generate(rank, model, input_text)
        dutils.destroy_tensor_parallel()


def parse_dtype(value: str) -> Optional[torch.dtype]:
    if value == "auto":
        return None
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal Qwen3-30B-A3B-Instruct-2507 TP inference"
    )
    parser.add_argument("model_dir", help="Local Hugging Face model directory")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--init-url")
    parser.add_argument("--backend", choices=("nccl", "gloo"))
    cli_args = parser.parse_args()

    world_size = cli_args.world_size or cli_args.tp_size
    if world_size != cli_args.tp_size:
        raise ValueError(
            "This minimal runner currently requires world_size == tp_size"
        )

    worker = WorkerProc(
        model_dir=cli_args.model_dir,
        world_size=world_size,
        tp_size=cli_args.tp_size,
        max_seq_len=cli_args.max_seq_len,
        max_new_tokens=cli_args.max_new_tokens,
        temperature=cli_args.temperature,
        top_p=cli_args.top_p,
        top_k=cli_args.top_k,
        dtype=parse_dtype(cli_args.dtype),
        init_url=cli_args.init_url,
        backend=cli_args.backend,
    )
    if world_size == 1:
        worker(0, cli_args.prompt)
    else:
        mp.spawn(
            worker,
            args=(cli_args.prompt,),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
