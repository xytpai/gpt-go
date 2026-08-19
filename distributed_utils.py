import os
from typing import Any, Tuple

import torch
import torch.distributed as dist


_TENSOR_PARALLEL_GROUP = None
_TENSOR_PARALLEL_RANK = 0
_TENSOR_PARALLEL_WORLD_SIZE = 1


def ensure_divisibility(numerator: int, denominator: int) -> None:
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Tuple[torch.Tensor, ...]:
    chunk_size = divide_and_check_no_remainder(tensor.size(-1), num_partitions)
    chunks = torch.split(tensor, chunk_size, dim=-1)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in chunks)
    return chunks


def init_tensor_parallel(
    rank: int,
    world_size: int,
    tp_size: int,
    backend: str | None = None,
    init_url: str | None = None,
) -> None:
    global _TENSOR_PARALLEL_GROUP
    global _TENSOR_PARALLEL_RANK
    global _TENSOR_PARALLEL_WORLD_SIZE

    if world_size % tp_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by tp_size={tp_size}")
    if tp_size > 1 and not torch.cuda.is_available():
        raise RuntimeError("GPU tensor parallelism requires CUDA or ROCm")

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if world_size == 1:
        _TENSOR_PARALLEL_GROUP = None
        _TENSOR_PARALLEL_RANK = 0
        _TENSOR_PARALLEL_WORLD_SIZE = 1
        return

    if backend is None:
        if torch.cuda.is_available() and dist.is_nccl_available():
            backend = "nccl"
        elif torch.cuda.is_available():
            raise RuntimeError(
                "TP>1 requires an NCCL/RCCL-enabled PyTorch build. "
                "Use Linux or WSL for GPU tensor parallelism."
            )
        else:
            backend = "gloo"
    if backend == "nccl" and not dist.is_nccl_available():
        raise RuntimeError("This PyTorch build does not provide NCCL/RCCL")
    init_url = init_url or os.environ.get("MASTER_ADDR")
    if init_url is None:
        init_url = "tcp://127.0.0.1:29500"
    elif not init_url.startswith(("tcp://", "env://", "file://")):
        port = os.environ.get("MASTER_PORT", "29500")
        init_url = f"tcp://{init_url}:{port}"

    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        rank=rank,
        world_size=world_size,
    )

    for group_index in range(world_size // tp_size):
        ranks = list(range(group_index * tp_size, (group_index + 1) * tp_size))
        group = dist.new_group(ranks, backend=backend)
        if rank in ranks:
            _TENSOR_PARALLEL_GROUP = group
            _TENSOR_PARALLEL_RANK = ranks.index(rank)
            _TENSOR_PARALLEL_WORLD_SIZE = tp_size


def destroy_tensor_parallel() -> None:
    global _TENSOR_PARALLEL_GROUP
    global _TENSOR_PARALLEL_RANK
    global _TENSOR_PARALLEL_WORLD_SIZE

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _TENSOR_PARALLEL_GROUP = None
    _TENSOR_PARALLEL_RANK = 0
    _TENSOR_PARALLEL_WORLD_SIZE = 1


def get_tensor_parallel_group():
    return _TENSOR_PARALLEL_GROUP


def get_tensor_parallel_world_size() -> int:
    return _TENSOR_PARALLEL_WORLD_SIZE


def get_tensor_parallel_rank() -> int:
    return _TENSOR_PARALLEL_RANK


def _reduce(ctx: Any, input_: torch.Tensor) -> torch.Tensor:
    if get_tensor_parallel_world_size() == 1:
        return input_
    if ctx is not None:
        ctx.mark_dirty(input_)
    dist.all_reduce(input_, group=get_tensor_parallel_group())
    return input_


def _split(input_: torch.Tensor) -> torch.Tensor:
    world_size = get_tensor_parallel_world_size()
    if world_size == 1:
        return input_
    chunks = split_tensor_along_last_dim(input_, world_size)
    return chunks[get_tensor_parallel_rank()].contiguous()


def _gather(input_: torch.Tensor) -> torch.Tensor:
    world_size = get_tensor_parallel_world_size()
    if world_size == 1:
        return input_
    tensors = [torch.empty_like(input_) for _ in range(world_size)]
    dist.all_gather(tensors, input_, group=get_tensor_parallel_group())
    return torch.cat(tensors, dim=-1).contiguous()


class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(None, grad_output)


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _reduce(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


def copy_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToTensorParallelRegion.apply(input_)


def reduce_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromTensorParallelRegion.apply(input_)


def scatter_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToTensorParallelRegion.apply(input_)


def gather_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromTensorParallelRegion.apply(input_)
