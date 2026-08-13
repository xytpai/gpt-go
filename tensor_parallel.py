import torch
import torch.nn as nn
import torch.nn.functional as F

import distributed_utils as dutils


class ColumnParallelLinear(nn.Module):
    """Y = XA where A is sharded along its output dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        bias: bool = True,
        gather_output: bool = True,
        device: str | torch.device = "cuda:0",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        world_size = dutils.get_tensor_parallel_world_size()
        self.output_size_per_partition = dutils.divide_and_check_no_remainder(
            out_features, world_size
        )
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                in_features,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = dutils.copy_to_tensor_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            return dutils.gather_from_tensor_parallel_region(output_parallel)
        return output_parallel


class RowParallelLinear(nn.Module):
    """Y = XA where A and X are sharded along their input dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        bias: bool = True,
        input_is_parallel: bool = False,
        reduce_output: bool = True,
        device: str | torch.device = "cuda:0",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.reduce_output = reduce_output

        world_size = dutils.get_tensor_parallel_world_size()
        self.input_size_per_partition = dutils.divide_and_check_no_remainder(
            in_features, world_size
        )
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                self.input_size_per_partition,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = dutils.scatter_to_tensor_parallel_region(input_)

        output = F.linear(input_parallel, self.weight)
        if self.reduce_output:
            output = dutils.reduce_from_tensor_parallel_region(output)
        if self.bias is not None:
            output = output + self.bias
        return output
