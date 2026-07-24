# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
import vllm.v1.kv_cache_interface
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import FixedCacheSpec


@dataclass(frozen=True)
class AscendFixedCacheSpec(FixedCacheSpec):
    """Ascend-specific FixedCacheSpec with NPU cache layout."""

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    cache_sparse_c8: bool = False
    c8_k_cache_dtype: torch.dtype = torch.int8
    c8_k_scale_cache_dtype: torch.dtype = torch.float16
    fixed_token_lengths: int = 2048

    @property
    def page_size_bytes(self) -> int:
        if self.cache_sparse_c8:
            num_heads_per_page = self.block_size * self.num_kv_heads
            index_head_dim = self.sparse_head_dim[-1]
            indexer_k_bytes = num_heads_per_page * index_head_dim * get_dtype_size(self.c8_k_cache_dtype)
            index_scale_head_dim = 1
            indexer_k_scale_bytes = (
                num_heads_per_page * index_scale_head_dim * get_dtype_size(self.c8_k_scale_cache_dtype)
            )
            return indexer_k_bytes + indexer_k_scale_bytes

        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        return cdiv(self.fixed_token_lengths, self.block_size * self.compress_ratio) * self.page_size_bytes


vllm.v1.kv_cache_interface.FixedCacheSpec = AscendFixedCacheSpec
