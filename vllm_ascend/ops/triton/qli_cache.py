from vllm.triton_utils import HAS_TRITON, tl, triton


if HAS_TRITON:
    @triton.jit(do_not_specialize=["num_blocks", "num_blocks_per_req", "index_topm"])
    def prepare_qli_cache_kernel(
        # k_cache: (num_slots, block_size, 1, head_dim)
        k_cache_ptr, k_stride_slot, k_stride_block,
        # scale_cache: (num_slots, block_size, 1, 1)
        scale_cache_ptr, s_stride_slot, s_stride_block,
        # block_table: (B, num_blocks_per_req)
        block_table_ptr, bt_stride_req,
        # topm_idxs: (B, index_topm)
        topm_idxs_ptr, ti_stride_req, ti_stride_pos,
        # new_k: (num_blocks, block_size, 1, head_dim)
        new_k_ptr, nk_stride_blk, nk_stride_block,
        # new_scale: (num_blocks, block_size, 1, 1)
        new_scale_ptr, ns_stride_blk, ns_stride_block,
        num_blocks, num_blocks_per_req, index_topm,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_blocks:
            return

        i = pid // num_blocks_per_req
        b = pid % num_blocks_per_req
        start = b * BLOCK_SIZE
        end = tl.minimum(start + BLOCK_SIZE, index_topm)

        offsets_d = tl.arange(0, HEAD_DIM)
        for s in range(BLOCK_SIZE):
            pos = start + s
            if pos < end:
                topm_idx = tl.load(topm_idxs_ptr + i * ti_stride_req + pos * ti_stride_pos)
                logical_block = topm_idx // BLOCK_SIZE
                physical_block = tl.load(block_table_ptr + i * bt_stride_req + logical_block)
                offset = topm_idx % BLOCK_SIZE

                k_data = tl.load(
                    k_cache_ptr + physical_block * k_stride_slot
                    + offset * k_stride_block + offsets_d
                )
                scale_data = tl.load(
                    scale_cache_ptr + physical_block * s_stride_slot
                    + offset * s_stride_block
                )

                tl.store(
                    new_k_ptr + pid * nk_stride_blk + s * nk_stride_block + offsets_d,
                    k_data
                )
                tl.store(
                    new_scale_ptr + pid * ns_stride_blk + s * ns_stride_block,
                    scale_data
                )