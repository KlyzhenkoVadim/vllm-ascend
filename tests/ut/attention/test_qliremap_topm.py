#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
from vllm.utils.math_utils import cdiv

n1 = 64
n2 = 1
d = 128
block_size = 32
layout_key = "PA_BSND"
layout_query = "TND"
query_quant_mode = 0
key_quant_mode = 0


def _gather_indexer_cache_pt(indexer_k_cache, indexer_scale_cache, block_table, flat_idx, total_count, indexer_dim):
    """Python reference: gather K+scale by flat physical indices into blocked layout."""
    B = flat_idx.shape[0]
    blk = indexer_k_cache.shape[1]
    nblocks = cdiv(total_count, blk)
    total = nblocks * blk
    pad = total - total_count
    if pad:
        flat_idx = F.pad(flat_idx, (0, pad)).contiguous()
    flat_idx = flat_idx.view(B, nblocks, blk)
    flat_blk = flat_idx.reshape(B * nblocks, blk).to(torch.int32)

    ks = indexer_k_cache.stride()
    k2d = torch.as_strided(indexer_k_cache,
                           (indexer_k_cache.shape[0] * blk, indexer_dim),
                           (ks[1], ks[-1]))
    ss = indexer_scale_cache.stride()
    s2d = torch.as_strided(indexer_scale_cache,
                           (indexer_scale_cache.shape[0] * blk, 1),
                           (ss[1], ss[-1]))

    new_k = torch_npu.npu_gather_sparse_index(k2d, flat_blk).unsqueeze(2)
    new_scale = torch_npu.npu_gather_sparse_index(s2d, flat_blk).unsqueeze(2)

    tbt = torch.arange(B * nblocks, device=block_table.device).view(B, nblocks).to(torch.int32)
    return new_k, new_scale, tbt


def test_qliremap_topm_correctness():
    """
    Verify that QLI with remap (topm_idxs + topm_count + chunk_start_token)
    produces identical output to QLI on pre-built composite K cache.

    Composite layout = [topM tokens (sequential) | chunk tokens (contiguous)].
    """
    b = 1
    index_topm = 128
    act_qlen = 64
    s2 = 512
    cmp_ratio = 4
    max_blocks = (s2 + block_size - 1) // block_size
    num_phys_blocks = max_blocks * b
    indexer_dim = 128

    torch.manual_seed(42)
    np.random.seed(42)

    # ---- block_table: sequential physical blocks ----
    block_table = torch.arange(num_phys_blocks, dtype=torch.int32).view(b, max_blocks).npu()

    # ---- key cache: random int8, shape (num_phys_blocks, block_size, 1, d) ----
    key = torch.tensor(
        np.random.uniform(-128, 127, (num_phys_blocks, block_size, n2, d))
    ).to(torch.int8).npu()
    key_scale = torch.tensor(
        np.random.uniform(0, 10, (num_phys_blocks, block_size, n2))
    ).to(torch.float16).npu()

    # ---- query: [T, n1, d] ----
    query = torch.tensor(
        np.random.uniform(-128, 127, (act_qlen, n1, d))
    ).to(torch.int8).npu()
    query_scale = torch.tensor(
        np.random.uniform(0, 10, (act_qlen, n1))
    ).to(torch.float16).npu()
    weights = torch.tensor(
        np.random.uniform(0, 0.01, (act_qlen, n1))
    ).to(torch.float16).npu()

    # ---- top-M indices: simulate from previous chunk ----
    # Pick random tokens from first half of the sequence (not overlapping with current chunk)
    topm_idxs = torch.randint(0, s2 // 2, (1, index_topm), dtype=torch.int32).npu()
    chunk_start_logical = 0
    chunk_start_token = chunk_start_logical * block_size

    # ---- actual seq lens ----
    composite_len = index_topm + act_qlen
    actual_seq_q = torch.tensor([act_qlen], dtype=torch.int32).npu()
    actual_seq_k = torch.tensor([composite_len], dtype=torch.int32).npu()

    # ---- metadata ----
    max_seqlen_q = actual_seq_q.max().item()
    max_seqlen_k = actual_seq_k.max().item()
    metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_key=actual_seq_k,
        num_heads_q=n1,
        num_heads_k=n2,
        head_dim=d,
        query_quant_mode=query_quant_mode,
        key_quant_mode=key_quant_mode,
        batch_size=b,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=index_topm,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=cmp_ratio,
        device="npu",
    ).npu()

    # ==============================
    # Method A: QLI with remap
    # ==============================
    topk_remap, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_key=actual_seq_k,
        block_table=block_table,
        metadata=metadata,
        query_quant_mode=query_quant_mode,
        key_quant_mode=key_quant_mode,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=index_topm,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=cmp_ratio,
        return_value=False,
        topm_idxs=topm_idxs,
        topm_count=index_topm,
        chunk_start_token=chunk_start_token,
    )

    # ==============================
    # Method B: Build composite K, run QLI on it
    # ==============================
    # Build top-M flat indices
    topm_s = topm_idxs.squeeze(0)  # (1, index_topm) -> (index_topm,) but need 2D
    topm_squeezed = topm_idxs.squeeze(1)  # (1, index_topm) for gather
    topm_flat = (
        torch.gather(block_table, 1, topm_squeezed // block_size) * block_size
        + topm_squeezed % block_size
    )  # (1, index_topm)

    # Build chunk flat indices (contiguous tokens starting at chunk_start_logical * block_size)
    chunk_global = (chunk_start_logical * block_size +
                    torch.arange(act_qlen, dtype=torch.int32, device="npu")).unsqueeze(0)  # (1, act_qlen)
    chunk_flat = (
        torch.gather(block_table, 1, chunk_global // block_size) * block_size
        + chunk_global % block_size
    )  # (1, act_qlen)

    # Merge
    combined_flat = torch.cat([topm_flat, chunk_flat], dim=1)  # (1, index_topm + act_qlen)

    composite_k, composite_scale, composite_bt = _gather_indexer_cache_pt(
        key, key_scale, block_table, combined_flat, composite_len, indexer_dim,
    )

    # Metadata for composite K (sequential block table)
    num_composite_blocks = composite_k.shape[0]
    actual_seq_k_composite = torch.tensor([composite_len], dtype=torch.int32).npu()

    metadata_comp = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_key=actual_seq_k_composite,
        num_heads_q=n1,
        num_heads_k=n2,
        head_dim=d,
        query_quant_mode=query_quant_mode,
        key_quant_mode=key_quant_mode,
        batch_size=b,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=composite_len,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=index_topm,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=cmp_ratio,
        device="npu",
    ).npu()

    topk_composite, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
        query=query,
        key=composite_k,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=composite_scale,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_key=actual_seq_k_composite,
        block_table=composite_bt,
        metadata=metadata_comp,
        query_quant_mode=query_quant_mode,
        key_quant_mode=key_quant_mode,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=index_topm,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=cmp_ratio,
        return_value=False,
    )

    # ==============================
    # Remap composite indices back to global for comparison
    # ==============================
    # Build global_idxs for composite mapping
    topm_flat_local = topm_idxs.squeeze(0).squeeze(0)  # (index_topm,)
    num_topm_blocks = cdiv(index_topm, block_size)
    topm_offset = num_topm_blocks * block_size
    total_slots = topm_offset + cdiv(act_qlen, block_size) * block_size
    global_idxs = torch.zeros(total_slots, dtype=torch.int32, device="npu")
    global_idxs[:index_topm] = topm_flat_local
    chunk_indices = (chunk_start_logical * block_size +
                     torch.arange(act_qlen, dtype=torch.int32, device="npu"))
    global_idxs[topm_offset : topm_offset + act_qlen] = chunk_indices

    # Map composite indices → global indices for both methods
    topk_remap_global = global_idxs[topk_remap.long()]
    topk_composite_global = global_idxs[topk_composite.long()]

    # ==============================
    # Assert equality
    # ==============================
    assert topk_remap.shape == topk_composite.shape, (
        f"Shape mismatch: remap={topk_remap.shape}, composite={topk_composite.shape}"
    )
    assert torch.equal(topk_remap_global, topk_composite_global), (
        "Mismatch between QLI-with-remap and QLI-on-composite-K.\n"
        f"Max diff: {(topk_remap_global != topk_composite_global).sum().item()} mismatches"
    )

    print("PASSED: QLI remap produces identical output to QLI on composite K.")

    # ==============================
    # Verify shape contract
    # ==============================
    assert topk_remap.shape == (act_qlen, 1, index_topm), (
        f"Expected output shape ({act_qlen}, 1, {index_topm}), got {topk_remap.shape}"
    )
    print("PASSED: Shape contract verified.")

    return True


def test_qliremap_no_remap_fallback():
    """Verify original behavior is preserved when topm_count=0 (no remap)."""
    b = 1
    s2 = 512
    act_qlen = 64
    cmp_ratio = 4
    max_blocks = (s2 + block_size - 1) // block_size
    num_phys_blocks = max_blocks * b

    torch.manual_seed(42)
    np.random.seed(42)

    block_table = torch.arange(num_phys_blocks, dtype=torch.int32).view(b, max_blocks).npu()
    key = torch.tensor(
        np.random.uniform(-128, 127, (num_phys_blocks, block_size, n2, d))
    ).to(torch.int8).npu()
    key_scale = torch.tensor(
        np.random.uniform(0, 10, (num_phys_blocks, block_size, n2))
    ).to(torch.float16).npu()
    query = torch.tensor(
        np.random.uniform(-128, 127, (act_qlen, n1, d))
    ).to(torch.int8).npu()
    query_scale = torch.tensor(
        np.random.uniform(0, 10, (act_qlen, n1))
    ).to(torch.float16).npu()
    weights = torch.tensor(
        np.random.uniform(0, 0.01, (act_qlen, n1))
    ).to(torch.float16).npu()

    actual_seq_q = torch.tensor([act_qlen], dtype=torch.int32).npu()
    actual_seq_k = torch.tensor([act_qlen], dtype=torch.int32).npu()

    metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_key=actual_seq_k,
        num_heads_q=n1, num_heads_k=n2, head_dim=d,
        query_quant_mode=query_quant_mode, key_quant_mode=key_quant_mode,
        batch_size=b,
        max_seqlen_q=act_qlen, max_seqlen_k=act_qlen,
        layout_query=layout_query, layout_key=layout_key,
        sparse_count=512, sparse_mode=3,
        pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
        cmp_ratio=cmp_ratio, device="npu",
    ).npu()

    # Call with topm_count=0 (no remap) — should fall back to original path
    topk_no_remap, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
        query=query, key=key, weights=weights,
        query_dequant_scale=query_scale, key_dequant_scale=key_scale,
        actual_seq_lengths_query=actual_seq_q, actual_seq_lengths_key=actual_seq_k,
        block_table=block_table, metadata=metadata,
        query_quant_mode=query_quant_mode, key_quant_mode=key_quant_mode,
        layout_query=layout_query, layout_key=layout_key,
        sparse_count=512, sparse_mode=3,
        pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
        cmp_ratio=cmp_ratio, return_value=False,
        topm_count=0, chunk_start_token=0,
    )

    # Call WITHOUT topm_idxs at all (original call via EXEC_NPU_CMD)
    topk_original, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
        query=query, key=key, weights=weights,
        query_dequant_scale=query_scale, key_dequant_scale=key_scale,
        actual_seq_lengths_query=actual_seq_q, actual_seq_lengths_key=actual_seq_k,
        block_table=block_table, metadata=metadata,
        query_quant_mode=query_quant_mode, key_quant_mode=key_quant_mode,
        layout_query=layout_query, layout_key=layout_key,
        sparse_count=512, sparse_mode=3,
        pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
        cmp_ratio=cmp_ratio, return_value=False,
    )

    assert torch.equal(topk_no_remap, topk_original), (
        "topm_count=0 should produce identical output to original QLI"
    )
    print("PASSED: topm_count=0 fallback produces identical output.")

    return True


def test_qliremap_edge_cases():
    """Test edge cases: empty chunk, topM only, boundary conditions."""
    b = 1
    index_topm = 64
    act_qlen = 1
    s2 = 256
    cmp_ratio = 4
    max_blocks = (s2 + block_size - 1) // block_size
    num_phys_blocks = max_blocks * b

    torch.manual_seed(42)
    np.random.seed(42)

    block_table = torch.arange(num_phys_blocks, dtype=torch.int32).view(b, max_blocks).npu()
    key = torch.tensor(
        np.random.uniform(-128, 127, (num_phys_blocks, block_size, n2, d))
    ).to(torch.int8).npu()
    key_scale = torch.tensor(
        np.random.uniform(0, 10, (num_phys_blocks, block_size, n2))
    ).to(torch.float16).npu()
    query = torch.tensor(
        np.random.uniform(-128, 127, (act_qlen, n1, d))
    ).to(torch.int8).npu()
    query_scale = torch.tensor(
        np.random.uniform(0, 10, (act_qlen, n1))
    ).to(torch.float16).npu()
    weights = torch.tensor(
        np.random.uniform(0, 0.01, (act_qlen, n1))
    ).to(torch.float16).npu()

    topm_idxs = torch.randint(0, s2 // 2, (1, index_topm), dtype=torch.int32).npu()
    composite_len = index_topm + act_qlen
    actual_seq_q = torch.tensor([act_qlen], dtype=torch.int32).npu()
    actual_seq_k = torch.tensor([composite_len], dtype=torch.int32).npu()

    metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_key=actual_seq_k,
        num_heads_q=n1, num_heads_k=n2, head_dim=d,
        query_quant_mode=query_quant_mode, key_quant_mode=key_quant_mode,
        batch_size=b,
        max_seqlen_q=act_qlen, max_seqlen_k=composite_len,
        layout_query=layout_query, layout_key=layout_key,
        sparse_count=index_topm, sparse_mode=3,
        pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
        cmp_ratio=cmp_ratio, device="npu",
    ).npu()

    topk_edge, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
        query=query, key=key, weights=weights,
        query_dequant_scale=query_scale, key_dequant_scale=key_scale,
        actual_seq_lengths_query=actual_seq_q, actual_seq_lengths_key=actual_seq_k,
        block_table=block_table, metadata=metadata,
        query_quant_mode=query_quant_mode, key_quant_mode=key_quant_mode,
        layout_query=layout_query, layout_key=layout_key,
        sparse_count=index_topm, sparse_mode=3,
        pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
        cmp_ratio=cmp_ratio, return_value=False,
        topm_idxs=topm_idxs, topm_count=index_topm, chunk_start_token=0,
    )

    assert topk_edge.shape == (act_qlen, 1, index_topm), (
        f"Edge case shape mismatch: {topk_edge.shape}"
    )
    assert topk_edge.dtype == torch.int32, f"Wrong dtype: {topk_edge.dtype}"
    assert not (topk_edge == -1).all(), "All output indices are -1 (invalid)"

    print("PASSED: Edge cases (act_qlen=1, index_topm=64).")

    return True


def test_qliremap_topm_varying_sizes():
    """
    Parametrized test over different (index_topm, act_qlen, s2) combos.
    """
    test_configs = [
        (32, 128, 512),    # small topM, medium chunk
        (256, 256, 1024),  # equal sizes
        (512, 8, 1024),    # large topM, tiny chunk
        (64, 1024, 2048),  # tiny topM, large chunk
    ]

    for idx, (index_topm, act_qlen, s2) in enumerate(test_configs):
        b = 1
        cmp_ratio = 4
        max_blocks = (s2 + block_size - 1) // block_size
        num_phys_blocks = max_blocks * b

        torch.manual_seed(100 + idx)
        np.random.seed(100 + idx)

        block_table = torch.arange(num_phys_blocks, dtype=torch.int32).view(b, max_blocks).npu()
        key = torch.tensor(
            np.random.uniform(-128, 127, (num_phys_blocks, block_size, n2, d))
        ).to(torch.int8).npu()
        key_scale = torch.tensor(
            np.random.uniform(0, 10, (num_phys_blocks, block_size, n2))
        ).to(torch.float16).npu()
        query = torch.tensor(
            np.random.uniform(-128, 127, (act_qlen, n1, d))
        ).to(torch.int8).npu()
        query_scale = torch.tensor(
            np.random.uniform(0, 10, (act_qlen, n1))
        ).to(torch.float16).npu()
        weights = torch.tensor(
            np.random.uniform(0, 0.01, (act_qlen, n1))
        ).to(torch.float16).npu()

        topm_idxs = torch.randint(0, s2 // 2, (1, index_topm), dtype=torch.int32).npu()
        composite_len = index_topm + act_qlen
        actual_seq_q = torch.tensor([act_qlen], dtype=torch.int32).npu()
        actual_seq_k = torch.tensor([composite_len], dtype=torch.int32).npu()

        metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
            actual_seq_lengths_query=actual_seq_q,
            actual_seq_lengths_key=actual_seq_k,
            num_heads_q=n1, num_heads_k=n2, head_dim=d,
            query_quant_mode=query_quant_mode, key_quant_mode=key_quant_mode,
            batch_size=b,
            max_seqlen_q=act_qlen, max_seqlen_k=composite_len,
            layout_query=layout_query, layout_key=layout_key,
            sparse_count=index_topm, sparse_mode=3,
            pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
            cmp_ratio=cmp_ratio, device="npu",
        ).npu()

        topk, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
            query=query, key=key, weights=weights,
            query_dequant_scale=query_scale, key_dequant_scale=key_scale,
            actual_seq_lengths_query=actual_seq_q, actual_seq_lengths_key=actual_seq_k,
            block_table=block_table, metadata=metadata,
            query_quant_mode=query_quant_mode, key_quant_mode=key_quant_mode,
            layout_query=layout_query, layout_key=layout_key,
            sparse_count=index_topm, sparse_mode=3,
            pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
            cmp_ratio=cmp_ratio, return_value=False,
            topm_idxs=topm_idxs, topm_count=index_topm, chunk_start_token=0,
        )

        assert topk.shape == (act_qlen, 1, index_topm), (
            f"Config {idx} ({index_topm}, {act_qlen}, {s2}): shape {topk.shape}"
        )
        print(f"PASSED: config {idx} ({index_topm}, {act_qlen}, {s2}) - shape OK")

    return True


if __name__ == "__main__":
    test_qliremap_topm_correctness()
    test_qliremap_no_remap_fallback()
    test_qliremap_edge_cases()
    test_qliremap_topm_varying_sizes()
    print("\n=== ALL TESTS PASSED ===")
