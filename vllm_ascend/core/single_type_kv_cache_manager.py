# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec


class CompressAttentionManager(FullAttentionManager):
    def __init__(self, kv_cache_spec: "AscendMLAAttentionSpec", block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.compress_ratio = kv_cache_spec.compress_ratio
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        num_tokens //= self.compress_ratio
        num_tokens_main_model //= self.compress_ratio

        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        if request_id in self.num_cached_block:
            assert len(new_computed_blocks) == 0
            return

        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
        num_total_computed_tokens //= self.compress_ratio
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), "Computed blocks should be empty when prefix caching is disabled"

        req_blocks.extend([self._null_block] * num_skipped_blocks)
        req_blocks.extend(new_computed_blocks)
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        num_tokens //= self.compress_ratio
        num_tokens_main_model //= self.compress_ratio

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        alignment_tokens: int | None = None,
    ) -> None:
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // (self.block_size * self.compress_ratio)

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size * self.compress_ratio,
            kv_cache_group_id=self.kv_cache_group_id,
        )
        self.num_cached_block[request.request_id] = num_full_blocks

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        drop_eagle_block: bool = False,
    ) -> tuple[list[KVCacheBlock], ...]:
        eagle_drop = drop_eagle_block
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple([] for _ in range(len(kv_cache_group_ids)))
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        logical_block_size = block_size * kv_cache_spec.compress_ratio
        logical_block_hashes = BlockHashListWithBlockSize(block_hashes, block_size, logical_block_size)
        max_num_blocks = max_length // logical_block_size
        for block_hash in itertools.islice(logical_block_hashes, max_num_blocks):
            if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if eagle_drop and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()

        while (
            logical_block_size != alignment_tokens
            and len(computed_blocks[0]) * logical_block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks


class FixedCacheManager(SingleTypeKVCacheManager):
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        assert self.dcp_world_size == 1
        assert self.pcp_world_size == 1
        self.compress_ratio = kv_cache_spec.compress_ratio
        self.fixed_token_lengths = kv_cache_spec.fixed_token_lengths
        self.num_fixed_blocks = self.fixed_token_lengths // self.block_size

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        assert len(new_computed_blocks) == 0
        if request_id in self.req_to_blocks and len(self.req_to_blocks[request_id]) > 0:
            return 0
        return self.num_fixed_blocks

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        if len(req_blocks) > 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(self.num_fixed_blocks)
            req_blocks.extend(new_blocks)
            self.new_block_ids.extend(b.block_id for b in new_blocks)
            return new_blocks

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        return

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        return

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return 0

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        return

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        return tuple([] for _ in range(len(kv_cache_group_ids)))

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0


# Register FixedCacheManager into upstream spec_manager_map
import vllm.v1.core.single_type_kv_cache_manager as _kvmgr

from vllm_ascend.core.kv_cache_interface import FixedCacheSpec

_kvmgr.spec_manager_map[FixedCacheSpec] = FixedCacheManager  # type: ignore[attr-defined]


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    **kwargs,
) -> SingleTypeKVCacheManager:
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry  # type: ignore[import-not-found]

    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    assert manager_class is not None, f"No KV cache manager registered for {type(kv_cache_spec).__name__}"
    if isinstance(kv_cache_spec, AscendMLAAttentionSpec) and kv_cache_spec.compress_ratio > 1:
        manager_class = CompressAttentionManager
        if max_model_len is not None:
            compress_ratio = kv_cache_spec.compress_ratio
            block_size = kv_cache_spec.block_size
            max_compressed_tokens = max_model_len // compress_ratio
            kwargs["max_admission_blocks_per_request"] = cdiv(max_compressed_tokens, block_size) + 1
    elif isinstance(kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec)):
        if max_num_batched_tokens is not None and max_model_len is not None:
            kwargs["max_admission_blocks_per_request"] = kv_cache_spec.max_admission_blocks_per_request(
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
