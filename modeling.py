import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor
from transformers import AutoTokenizer

from distributed_utils import get_tensor_parallel_world_size
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
    max_seq_len: int = 4096
    dropout_prob: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    num_experts: int = 128
    num_activated_experts: int = 8
    norm_experts_prob: bool = True
    eos_token_ids: tuple[int, ...] = (151645, 151643)

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
        config_dtype = dtype_by_name.get(
            config.get("torch_dtype", config.get("dtype")),
            torch.bfloat16,
        )
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
            max_seq_len=(
                max_seq_len
                if max_seq_len is not None
                else min(config["max_position_embeddings"], 4096)
            ),
            dropout_prob=config.get("attention_dropout", 0.0),
            dtype=dtype or config_dtype,
            num_experts=config["num_experts"],
            num_activated_experts=config["num_experts_per_tok"],
            norm_experts_prob=config.get("norm_topk_prob", True),
            eos_token_ids=eos_token_ids,
        )


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_size, dtype, device):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_size)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, input_pos: Tensor, k_val, v_val):
        # input_pos: L[t]
        # k_val, v_val: F[b, nh, t, hs]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class RMSNorm(nn.Module):
    def __init__(self, dim, norm_eps, dtype, device):
        super().__init__()
        self.eps = norm_eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(input_dtype)
        return self.weight * x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, device: str):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.dropout_prob = args.dropout_prob
        self.n_kv_heads = args.n_kv_heads
        parallel_size = get_tensor_parallel_world_size()
        self.n_local_heads = args.n_heads // parallel_size
        self.n_local_kv_heads = self.n_kv_heads // parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim
        # weights
        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, dtype=args.dtype, device=device, input_is_parallel=True)
        self.q_norm = RMSNorm(self.head_dim, args.norm_eps, args.dtype, device)
        self.k_norm = RMSNorm(self.head_dim, args.norm_eps, args.dtype, device)
        self.kv_cache = None

    @staticmethod
    def precompute_freqs_cis(head_dim: int, max_position_embeddings: int, theta: float = 10000.0):
        inv_freqs = 1.0 / (theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64)[: (head_dim // 2)].float() / head_dim))
        t = torch.arange(max_position_embeddings, device=inv_freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freqs)  # F(max_position_embeddings, head_dim/2)
        return freqs.cos(), freqs.sin()

    def apply_rotary_emb(self, xq, xk, cos, sin):
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)  # F[t, 1, head_dim/2]

        def apply(x):
            x1, x2 = torch.chunk(x.float(), 2, dim=-1)
            y1 = x1 * cos - x2 * sin
            y2 = x2 * cos + x1 * sin
            return torch.cat((y1, y2), dim=-1)

        xq_out = apply(xq)
        xk_out = apply(xk)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, x, attention_mask, cos, sin, input_pos: Optional[Tensor] = None, xa: Optional[Tensor] = None):
        batch_size, seq_length, _ = x.size()
        # Infer xq, xk and xv
        xq = self.wq(x)
        x_for_kv = x if xa is None else xa
        xk = self.wk(x_for_kv)
        xv = self.wv(x_for_kv)
        xq = xq.view(batch_size, seq_length, -1, self.head_dim)
        xk = xk.view(batch_size, seq_length, -1, self.head_dim)
        xv = xv.view(batch_size, seq_length, -1, self.head_dim)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        # Apply RoPE
        xq, xk = self.apply_rotary_emb(xq, xk, cos, sin)
        # Refine xq, xk and xv shape
        xq, xk, xv = [item.transpose(1, 2).contiguous() for item in [xq, xk, xv]]
        if self.kv_cache is not None:
            xk, xv = self.kv_cache.update(input_pos, xk, xv)
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)
        # DSPA
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=attention_mask, dropout_p=self.dropout_prob
        )
        # Infer output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs, device: str):
        super().__init__()
        hidden_dim = args.moe_intermediate_size
        self.w1 = ColumnParallelLinear(args.dim, hidden_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)
        self.w2 = RowParallelLinear(hidden_dim, args.dim, bias=False, dtype=args.dtype, device=device, input_is_parallel=True)
        self.w3 = ColumnParallelLinear(args.dim, hidden_dim, bias=False, dtype=args.dtype, device=device, gather_output=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MOEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs, device: str):
        super().__init__()
        self.num_experts = args.num_experts
        self.num_activated_experts = args.num_activated_experts
        self.norm_experts_prob = args.norm_experts_prob
        self.gate = nn.Linear(
            args.dim,
            args.num_experts,
            bias=False,
            dtype=args.dtype,
            device=device,
        )
        self.experts = nn.ModuleList(
            [FeedForward(args, device) for _ in range(args.num_experts)]
        )

    def forward(self, x):
        batch_size, seq_length, dim = x.shape
        x = x.view(-1, dim)
        routing_weights = F.softmax(self.gate(x), dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_activated_experts, dim=-1)
        if self.norm_experts_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)

        output = torch.zeros((batch_size * seq_length, dim), dtype=x.dtype, device=x.device)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = x[None, top_x].reshape(-1, dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            output.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        output = output.reshape(batch_size, seq_length, dim)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, device: str):
        super().__init__()
        self.attention = Attention(args, device)
        if getattr(args, "enable_visual", None):
            self.visual_attention = Attention(args, device)
            self.visual_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
            self.enable_visual = True
        else:
            self.enable_visual = False
        if getattr(args, "enable_audio", None):
            self.audio_attention = Attention(args, device)
            self.audio_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
            self.enable_audio = True
        else:
            self.enable_audio = False
        self.layer_id = layer_id
        self.feed_forward = MOEFeedForward(args, device)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)

    def forward(
        self,
        x,
        attention_mask,
        cos,
        sin,
        input_pos: Optional[Tensor] = None,
        xa: Optional[Tensor] = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), attention_mask, cos, sin, input_pos
        )
        if self.enable_visual:
            x = x + self.visual_attention(
                self.visual_norm(x), attention_mask, cos, sin, input_pos, xa
            )
        if self.enable_audio:
            x = x + self.audio_attention(
                self.audio_norm(x), attention_mask, cos, sin, input_pos, xa
            )
        out = x + self.feed_forward(self.ffn_norm(x))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, device="cuda:0"):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(
            args.vocab_size, args.dim, dtype=args.dtype, device=device
        )
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id, args, device)
            for layer_id in range(args.n_layers)
        ])
        self.norm = RMSNorm(args.dim, args.norm_eps, args.dtype, device)
        self.output = ColumnParallelLinear(
            args.dim,
            args.vocab_size,
            bias=False,
            dtype=args.dtype,
            device=device,
            gather_output=True,
        )
        self.head_dim = args.head_dim
        self.cos, self.sin = Attention.precompute_freqs_cis(
            self.head_dim,
            self.max_seq_len * 2,
            args.rope_theta,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    self.max_seq_len,
                    self.max_seq_len,
                    dtype=torch.bool,
                    device=device,
                )
            ),
            persistent=False,
        )

    def device(self):
        return self.output.weight.device

    def dtype(self):
        return self.args.dtype

    def size(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def assign_kv_cache(self, max_batch_size):
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size,
                self.max_seq_len,
                self.args.n_kv_heads,
                self.head_dim,
                self.dtype(),
                self.device(),
            )

    def forward(
        self,
        tokens,
        input_pos: Optional[Tensor] = None,
        images: Optional[Tensor] = None,
    ):
        # tokens: L[b, t]
        # images: F[b, c, h, w]
        _, seq_length = tokens.size()
        h = self.tok_embeddings(tokens)
        self.cos = self.cos.to(self.device())
        self.sin = self.sin.to(self.device())

        if input_pos is None:
            input_pos = torch.arange(0, seq_length, device=self.device())
            cos, sin = self.cos[input_pos], self.sin[input_pos]
            causal_mask = self.causal_mask[None, None, :seq_length, :seq_length]
        else:
            cos, sin = self.cos[input_pos], self.sin[input_pos]
            causal_mask = self.causal_mask[None, None, input_pos]
        for layer in self.layers:
            h = layer(h, causal_mask, cos, sin, input_pos)
        h = self.norm(h)
        return self.output(h)

    @staticmethod
    def post_loss(logits: Tensor, target_ids: Tensor, ignore_index: int = -100):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.contiguous().view(-1),
            ignore_index=ignore_index,
        )

    @staticmethod
    def post_pred(h, temperature):
        h = h[:, -1, :] / temperature
        return torch.multinomial(F.softmax(h, dim=-1), num_samples=1)


@torch.no_grad()
def load_qwen3_moe_checkpoint(model: Transformer, model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = [path.name for path in sorted(model_dir.glob("*.safetensors"))]
        if not shard_names:
            raise FileNotFoundError(
                f"No safetensors checkpoint found under {model_dir}"
            )

    parameters = dict(model.named_parameters())
    missing = set(parameters)

    for shard_index, shard_name in enumerate(shard_names, start=1):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
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
        max_seq_len: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        dtype: Optional[torch.dtype],
    ):
        self.model_dir = model_dir
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.dtype = dtype

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
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        tokens = torch.as_tensor(
            input_ids,
            dtype=torch.long,
            device=model.device(),
        )
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        prompt_length = tokens.size(1)
        if prompt_length + self.max_new_tokens > model.max_seq_len:
            raise ValueError(
                f"Prompt ({prompt_length}) + output ({self.max_new_tokens}) "
                f"exceeds max_seq_len={model.max_seq_len}"
            )

        model.assign_kv_cache(1)
        input_pos = torch.arange(tokens.size(1), device=model.device())
        generated: list[int] = []
        streamed_text = ""

        if torch.cuda.is_available():
            torch.cuda.synchronize(model.device())
        start_time = time.perf_counter()

        for step in range(self.max_new_tokens):
            logits = model(tokens, input_pos)[:, -1, :]
            next_token = sample_next_token(
                logits,
                self.temperature,
                self.top_p,
                self.top_k,
            )

            token_id = int(next_token.item())
            generated.append(token_id)
            decoded_text = tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).rstrip("\ufffd")
            if decoded_text.startswith(streamed_text):
                print(
                    decoded_text[len(streamed_text) :],
                    end="",
                    flush=True,
                )
                streamed_text = decoded_text
            if token_id in model.args.eos_token_ids:
                break

            tokens = next_token
            input_pos = torch.tensor(
                [prompt_length + step],
                dtype=torch.long,
                device=model.device(),
            )

        final_text = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if final_text.startswith(streamed_text):
            print(final_text[len(streamed_text) :], end="", flush=True)
        print(flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize(model.device())
        elapsed = time.perf_counter() - start_time

        if rank == 0:
            print(
                f"\nGenerated {len(generated)} tokens in {elapsed:.3f}s "
                f"({len(generated) / elapsed:.2f} tokens/s)"
            )

    def __call__(self, rank: int, input_text: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA or ROCm PyTorch build with an available GPU is required"
            )
        device = f"cuda:{rank}"

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
            billions = model.size() / 1_000_000_000
            print(
                f"Qwen3-30B-A3B-Instruct-2507: "
                f"{billions:.3f}B parameters"
            )
        self.generate(rank, model, input_text)


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
        description="Minimal single-GPU Qwen3-30B-A3B-Instruct-2507 inference"
    )
    parser.add_argument("model_dir", help="Local Hugging Face model directory")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    cli_args = parser.parse_args()

    worker = WorkerProc(
        model_dir=cli_args.model_dir,
        max_seq_len=cli_args.max_seq_len,
        max_new_tokens=cli_args.max_new_tokens,
        temperature=cli_args.temperature,
        top_p=cli_args.top_p,
        top_k=cli_args.top_k,
        dtype=parse_dtype(cli_args.dtype),
    )
    worker(0, cli_args.prompt)


if __name__ == "__main__":
    main()
