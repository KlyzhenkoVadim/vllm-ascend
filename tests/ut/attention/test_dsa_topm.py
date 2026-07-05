import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder


class TestDSAMetadataBuilderTopMBuffers(unittest.TestCase):
    """Test that build_decode_metadata creates temp buffers from input_batch.topm_state."""

    def setUp(self):
        # Patch parent __init__ to avoid complex MLA init
        def mock_parent_init(self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, **kw):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.kv_cache_spec = kv_cache_spec

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__",
            mock_parent_init,
        )
        self.parent_init_patcher.start()

        # Patch get_tensor_model_parallel_world_size to avoid TP dependency
        self.tp_patcher = patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size", return_value=1)
        self.tp_patcher.start()

        # Patch vllm_config for hadamard init check
        self.cfg_patcher = patch("vllm.config.get_current_vllm_config", return_value=MagicMock())
        self.cfg_patcher.start()

        # Patch hadamard creation to skip scipy import
        hf_config = MagicMock()
        hf_config.model_type = "deepseek_v4"
        hf_config.index_head_dim = 128
        hf_config.index_topk = 64
        hf_config.index_n_heads = 64
        hf_config.num_attention_heads = 64

        model_config = MagicMock()
        model_config.hf_config = hf_config
        model_config.get_head_size.return_value = 128
        model_config.dtype = torch.float16
        model_config.enable_sleep_mode = False

        vllm_config = MagicMock()
        vllm_config.model_config = model_config
        vllm_config.speculative_config = None
        vllm_config.compilation_config.pass_config.enable_sp = False
        vllm_config.scheduler_config.max_num_seqs = 64

        kv_cache_spec = MagicMock()
        kv_cache_spec.compress_ratio = 4

        self.builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
        self.builder.kv_cache_spec = kv_cache_spec
        self.builder.model_config = model_config
        self.builder.vllm_config = vllm_config
        self.builder.device = torch.device("cpu")
        self.builder.slot_mapping_shape = (0,)
        self.builder.hadamard = None
        self.builder.seqused_q = torch.zeros(0, device="cpu")
        self.builder._zero_i32 = torch.tensor([0], dtype=torch.int32)
        self.builder.decode_sas_metadata = torch.zeros(1024, dtype=torch.int32)
        self.builder.decode_qli_metadata = torch.zeros(1024, dtype=torch.int32)
        self.builder.start_pos_decode = torch.zeros(64, dtype=torch.int32)
        self.builder.cu_seqlens_ori_kv = torch.tensor([], dtype=torch.int32)
        self.builder.cu_seqlens_cmp_kv = torch.tensor([], dtype=torch.int32)
        self.builder.num_decodes = 3
        self.builder.num_decode_tokens = 3
        self.builder.compressor_ratio = 4
        self.builder.decode_threshold = 1
        self.builder.attn_mask_builder = MagicMock()

    def tearDown(self):
        self.parent_init_patcher.stop()
        self.tp_patcher.stop()
        self.cfg_patcher.stop()

    def _make_common_attn_metadata(self, num_reqs=3):
        meta = MagicMock()
        meta.num_reqs = num_reqs
        meta.num_input_tokens = num_reqs
        meta.num_actual_tokens = num_reqs
        meta.query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        meta.query_start_loc_cpu = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        meta.positions = torch.tensor([100, 200, 300], dtype=torch.int32)
        meta.positions_cpu = torch.tensor([100, 200, 300], dtype=torch.int32)
        meta.seq_lens = torch.tensor([500, 600, 700], dtype=torch.int32)
        meta._seq_lens_cpu = torch.tensor([500, 600, 700], dtype=torch.int32)
        meta.seq_lens_cpu = torch.tensor([500, 600, 700], dtype=torch.int32)
        meta.max_query_len = 1
        meta.max_seq_len = 700
        meta.causal = True
        meta.block_table_tensor = torch.arange(30, dtype=torch.int32).view(3, 10)
        return meta

    def test_no_temp_buffers_without_input_batch(self):
        """When input_batch is not provided, temp buffers should be None."""
        meta = self._make_common_attn_metadata()
        self.builder.seq_lens = meta.seq_lens
        self.builder.block_table = meta.block_table_tensor
        self.builder.decode_ratio_to_sas_metadata = {}

        # Patch _build_topm_subgroups to avoid CUDA ops
        with patch.object(AscendDSAMetadataBuilder, "_build_topm_subgroups", return_value=(None, None, None)):
            with patch("vllm_ascend.attention.dsa_v1.DeviceOperator") as mock_dev:
                mock_dev.get_dsa_sparse_attn_metadata_op.return_value = lambda **kw: torch.zeros(1024, dtype=torch.int32)
                mock_dev.get_dsa_sparse_attn_metadata_kwargs.return_value = {}
                mock_dev.get_dsa_decode_cu_seqlens_cmp_kv.return_value = {}
                result = self.builder.build_decode_metadata(
                    common_prefix_len=0, common_attn_metadata=meta, num_reqs_actual=3, input_batch=None
                )

        self.assertIsNone(result.topm_ustep)
        self.assertIsNone(result.topm_start_cache)
        self.assertIsNone(result.topm_idxs)
        self.assertIsNone(result.default_metadata)
        self.assertIsNone(result.compute_topm_metadata)
        self.assertIsNone(result.reuse_topm_metadata)

    def test_temp_buffers_created_with_input_batch(self):
        """When input_batch is provided, temp buffers should be created."""
        meta = self._make_common_attn_metadata()
        self.builder.seq_lens = meta.seq_lens
        self.builder.block_table = meta.block_table_tensor
        self.builder.decode_ratio_to_sas_metadata = {}

        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", "req_1", "req_2"]
        input_batch.topm_state = {
            "req_0": {"c4": MagicMock(ustep=5, start_cache=True, topm_idxs=torch.arange(550, dtype=torch.int32).view(1, 550))},
            "req_1": {"c4": MagicMock(ustep=0, start_cache=False, topm_idxs=None)},
        }

        with patch.object(AscendDSAMetadataBuilder, "_build_topm_subgroups", return_value=(None, None, None)):
            with patch("vllm_ascend.attention.dsa_v1.DeviceOperator") as mock_dev:
                mock_dev.get_dsa_sparse_attn_metadata_op.return_value = lambda **kw: torch.zeros(1024, dtype=torch.int32)
                mock_dev.get_dsa_sparse_attn_metadata_kwargs.return_value = {}
                mock_dev.get_dsa_decode_cu_seqlens_cmp_kv.return_value = {}
                result = self.builder.build_decode_metadata(
                    common_prefix_len=0, common_attn_metadata=meta, num_reqs_actual=3,
                    input_batch=input_batch, index_topm=550, micro_step_num=4,
                )

        self.assertIsNotNone(result.topm_ustep)
        self.assertIsNotNone(result.topm_start_cache)
        self.assertIsNotNone(result.topm_idxs)
        self.assertEqual(result.topm_ustep.shape, (3,))
        self.assertEqual(result.topm_start_cache.shape, (3,))
        self.assertEqual(result.topm_idxs.shape, (3, 1, 550))

    def test_temp_buffers_populated_from_topm_state(self):
        """Temp buffers should correctly reflect input_batch.topm_state values."""
        meta = self._make_common_attn_metadata()
        self.builder.seq_lens = meta.seq_lens
        self.builder.block_table = meta.block_table_tensor
        self.builder.decode_ratio_to_sas_metadata = {}

        expected_idxs = torch.arange(550, dtype=torch.int32).view(1, 550)

        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", "req_1", "req_2"]
        input_batch.topm_state = {
            "req_0": {"c4": MagicMock(ustep=5, start_cache=True, topm_idxs=expected_idxs)},
            "req_1": {"c4": MagicMock(ustep=0, start_cache=False, topm_idxs=None)},
            "req_2": {"c4": MagicMock(ustep=3, start_cache=True, topm_idxs=expected_idxs)},
        }

        with patch.object(AscendDSAMetadataBuilder, "_build_topm_subgroups", return_value=(None, None, None)):
            with patch("vllm_ascend.attention.dsa_v1.DeviceOperator") as mock_dev:
                mock_dev.get_dsa_sparse_attn_metadata_op.return_value = lambda **kw: torch.zeros(1024, dtype=torch.int32)
                mock_dev.get_dsa_sparse_attn_metadata_kwargs.return_value = {}
                mock_dev.get_dsa_decode_cu_seqlens_cmp_kv.return_value = {}
                result = self.builder.build_decode_metadata(
                    common_prefix_len=0, common_attn_metadata=meta, num_reqs_actual=3,
                    input_batch=input_batch, index_topm=550, micro_step_num=4,
                )

        self.assertEqual(result.topm_ustep[0].item(), 5)
        self.assertEqual(result.topm_start_cache[0].item(), 1)
        self.assertTrue(torch.equal(result.topm_idxs[0], expected_idxs))

        self.assertEqual(result.topm_ustep[1].item(), 0)
        self.assertEqual(result.topm_start_cache[1].item(), 0)

        self.assertEqual(result.topm_ustep[2].item(), 3)
        self.assertEqual(result.topm_start_cache[2].item(), 1)

    def test_temp_buffers_none_for_missing_req(self):
        """Missing request in topm_state should result in zero/default values."""
        meta = self._make_common_attn_metadata()
        self.builder.seq_lens = meta.seq_lens
        self.builder.block_table = meta.block_table_tensor
        self.builder.decode_ratio_to_sas_metadata = {}

        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", "req_1", "req_2"]
        input_batch.topm_state = {}

        with patch.object(AscendDSAMetadataBuilder, "_build_topm_subgroups", return_value=(None, None, None)):
            with patch("vllm_ascend.attention.dsa_v1.DeviceOperator") as mock_dev:
                mock_dev.get_dsa_sparse_attn_metadata_op.return_value = lambda **kw: torch.zeros(1024, dtype=torch.int32)
                mock_dev.get_dsa_sparse_attn_metadata_kwargs.return_value = {}
                mock_dev.get_dsa_decode_cu_seqlens_cmp_kv.return_value = {}
                result = self.builder.build_decode_metadata(
                    common_prefix_len=0, common_attn_metadata=meta, num_reqs_actual=3,
                    input_batch=input_batch, index_topm=550, micro_step_num=4,
                )

        self.assertEqual(result.topm_ustep.sum().item(), 0)
        self.assertEqual(result.topm_start_cache.sum().item(), 0)

    def test_temp_buffers_not_created_when_num_decodes_zero(self):
        """When num_decodes == 0, no temp buffers should be created."""
        self.builder.num_decodes = 0
        self.builder.num_decode_tokens = 0
        meta = self._make_common_attn_metadata()
        self.builder.seq_lens = meta.seq_lens
        self.builder.block_table = meta.block_table_tensor
        self.builder.decode_ratio_to_sas_metadata = {}

        input_batch = MagicMock()
        input_batch.req_ids = []
        input_batch.topm_state = {}

        with patch("vllm_ascend.attention.dsa_v1.DeviceOperator") as mock_dev:
            mock_dev.get_dsa_sparse_attn_metadata_op.return_value = lambda **kw: torch.zeros(1024, dtype=torch.int32)
            mock_dev.get_dsa_sparse_attn_metadata_kwargs.return_value = {}
            mock_dev.get_dsa_decode_cu_seqlens_cmp_kv.return_value = {}
            mock_dev.prepare_dsa_indexer_weights.side_effect = lambda x: x
            mock_dev.prepare_dsa_indexer_query_scale.side_effect = lambda x: x
            mock_dev.prepare_dsa_indexer_key_scale.side_effect = lambda x: x
            result = self.builder.build_decode_metadata(
                common_prefix_len=0, common_attn_metadata=meta, num_reqs_actual=0,
                input_batch=input_batch, index_topm=550, micro_step_num=4,
            )

        self.assertIsNone(result.topm_ustep)
        self.assertIsNone(result.topm_start_cache)
        self.assertIsNone(result.topm_idxs)

    def test_build_topm_subgroups_called_with_correct_params(self):
        """_build_topm_subgroups should be called with temp buffers when compress_ratio=4."""
        meta = self._make_common_attn_metadata()
        self.builder.seq_lens = meta.seq_lens
        self.builder.block_table = meta.block_table_tensor
        self.builder.decode_ratio_to_sas_metadata = {}

        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", "req_1", "req_2"]
        input_batch.topm_state = {
            "req_0": {"c4": MagicMock(ustep=5, start_cache=True, topm_idxs=None)},
        }

        with patch.object(AscendDSAMetadataBuilder, "_build_topm_subgroups",
                          return_value=(MagicMock(), MagicMock(), MagicMock())) as mock_subgroups:
            with patch("vllm_ascend.attention.dsa_v1.DeviceOperator") as mock_dev:
                mock_dev.get_dsa_sparse_attn_metadata_op.return_value = lambda **kw: torch.zeros(1024, dtype=torch.int32)
                mock_dev.get_dsa_sparse_attn_metadata_kwargs.return_value = {}
                mock_dev.get_dsa_decode_cu_seqlens_cmp_kv.return_value = {}
                self.builder.build_decode_metadata(
                    common_prefix_len=0, common_attn_metadata=meta, num_reqs_actual=3,
                    input_batch=input_batch, index_topm=550, micro_step_num=4,
                )

        mock_subgroups.assert_called_once()
        call_kwargs = mock_subgroups.call_args[1]
        self.assertIn("ustep", call_kwargs)
        self.assertIn("start_topm_cache", call_kwargs)
        self.assertIn("topm_idxs", call_kwargs)
        self.assertEqual(call_kwargs["index_topm"], 550)
        self.assertEqual(call_kwargs["micro_step_num"], 4)


class TestDSABuildTopMSubgroupsMaskLogic(unittest.TestCase):
    """Test _build_topm_subgroups mask computation (no NPU/custom ops needed)."""

    def setUp(self):
        self.index_topm = 550
        self.micro_step_num = 4
        self.index_topk = 64

    def _make_builder(self):
        hf_config = MagicMock()
        hf_config.model_type = "deepseek_v4"
        hf_config.index_head_dim = 128
        hf_config.index_topk = self.index_topk
        hf_config.index_n_heads = 64
        hf_config.num_attention_heads = 64

        model_config = MagicMock()
        model_config.hf_config = hf_config

        builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
        builder.model_config = model_config
        builder.compressor_ratio = 4
        return builder

    def test_all_default_mask_short_seqs(self):
        """Requests with kvlens < 4*index_topm should all be in default mask."""
        builder = self._make_builder()
        B = 3
        kvlens = torch.tensor([500, 800, 1000], dtype=torch.int32)
        qlens = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        block_table = torch.zeros(B, 10, dtype=torch.int32)
        start_topm_cache = torch.zeros(B, dtype=torch.bool)
        ustep = torch.zeros(B, dtype=torch.int32)
        topm_idxs = torch.zeros(B, 1, self.index_topm, dtype=torch.int32)

        with patch.object(builder, "model_config") as mc:
            mc.hf_config.index_n_heads = 64
            mc.hf_config.index_head_dim = 128
            with patch("torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata",
                       return_value=torch.zeros(1024, dtype=torch.int32)):
                default, compute, reuse = builder._build_topm_subgroups(
                    kvlens=kvlens, qlens=qlens, block_table=block_table,
                    start_topm_cache=start_topm_cache, ustep=ustep,
                    topm_idxs=topm_idxs, index_topm=self.index_topm,
                    micro_step_num=self.micro_step_num, index_topk=self.index_topk,
                )

        self.assertIsNotNone(default)
        self.assertIsNone(compute)
        self.assertIsNone(reuse)
        self.assertTrue(default.mask.all())

    def test_crossed_to_compute(self):
        """Requests crossing the threshold should transition from default to compute."""
        builder = self._make_builder()
        B = 3
        kvlens = torch.tensor([500, 3000, 4000], dtype=torch.int32)
        qlens = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        block_table = torch.zeros(B, 10, dtype=torch.int32)
        start_topm_cache = torch.tensor([False, False, False], dtype=torch.bool)
        ustep = torch.zeros(B, dtype=torch.int32)
        topm_idxs = torch.zeros(B, 1, self.index_topm, dtype=torch.int32)

        with patch("torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata",
                   side_effect=[
                       torch.zeros(1024, dtype=torch.int32),
                       torch.zeros(1024, dtype=torch.int32),
                   ]):
            default, compute, reuse = builder._build_topm_subgroups(
                kvlens=kvlens, qlens=qlens, block_table=block_table,
                start_topm_cache=start_topm_cache, ustep=ustep,
                topm_idxs=topm_idxs, index_topm=self.index_topm,
                micro_step_num=self.micro_step_num, index_topk=self.index_topk,
            )

        self.assertIsNotNone(default)
        self.assertIsNotNone(compute)
        self.assertIsNone(reuse)

        self.assertTrue(default.mask[0].item())
        self.assertFalse(default.mask[1].item())
        self.assertFalse(default.mask[2].item())

        self.assertFalse(compute.mask[0].item())
        self.assertTrue(compute.mask[1].item())
        self.assertTrue(compute.mask[2].item())

    def test_compute_and_reuse_masks(self):
        """Started-caching requests should split into compute (ustep % ms == 0) or reuse."""
        builder = self._make_builder()
        B = 4
        kvlens = torch.tensor([3000, 3000, 3000, 3000], dtype=torch.int32)
        qlens = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        block_table = torch.zeros(B, 10, dtype=torch.int32)
        start_topm_cache = torch.tensor([True, True, True, True], dtype=torch.bool)
        ustep = torch.tensor([0, 1, 3, 4], dtype=torch.int32)
        topm_idxs = torch.zeros(B, 1, self.index_topm, dtype=torch.int32)

        with patch("torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata",
                   side_effect=[
                       torch.zeros(1024, dtype=torch.int32),
                       torch.zeros(1024, dtype=torch.int32),
                   ]):
            default, compute, reuse = builder._build_topm_subgroups(
                kvlens=kvlens, qlens=qlens, block_table=block_table,
                start_topm_cache=start_topm_cache, ustep=ustep,
                topm_idxs=topm_idxs, index_topm=self.index_topm,
                micro_step_num=self.micro_step_num, index_topk=self.index_topk,
            )

        self.assertIsNone(default)
        self.assertIsNotNone(compute)
        self.assertIsNotNone(reuse)

        self.assertTrue(compute.mask[0].item())
        self.assertFalse(compute.mask[1].item())
        self.assertFalse(compute.mask[2].item())
        self.assertTrue(compute.mask[3].item())

        self.assertFalse(reuse.mask[0].item())
        self.assertTrue(reuse.mask[1].item())
        self.assertTrue(reuse.mask[2].item())
        self.assertFalse(reuse.mask[3].item())

    def test_ustep_incremented_on_crossed(self):
        """Crossed requests should have ustep reset to 0 and start_cache set to True."""
        builder = self._make_builder()
        B = 2
        kvlens = torch.tensor([3000, 500], dtype=torch.int32)
        qlens = torch.tensor([0, 1, 2], dtype=torch.int32)
        block_table = torch.zeros(B, 10, dtype=torch.int32)
        start_topm_cache = torch.tensor([False, False], dtype=torch.bool)
        ustep = torch.tensor([0, 0], dtype=torch.int32)
        topm_idxs = torch.zeros(B, 1, self.index_topm, dtype=torch.int32)

        with patch("torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata",
                   side_effect=[
                       torch.zeros(1024, dtype=torch.int32),
                       torch.zeros(1024, dtype=torch.int32),
                   ]):
            builder._build_topm_subgroups(
                kvlens=kvlens, qlens=qlens, block_table=block_table,
                start_topm_cache=start_topm_cache, ustep=ustep,
                topm_idxs=topm_idxs, index_topm=self.index_topm,
                micro_step_num=self.micro_step_num, index_topk=self.index_topk,
            )

        self.assertTrue(start_topm_cache[0].item())  # crossed
        self.assertEqual(ustep[0].item(), 0)           # reset to 0
        self.assertFalse(start_topm_cache[1].item())  # not crossed
        self.assertEqual(ustep[1].item(), 0)

    def test_reuse_metadata_has_topm_idxs(self):
        """Reuse subgroup metadata should contain topm_idxs filtered by reuse_mask."""
        builder = self._make_builder()
        B = 3
        kvlens = torch.tensor([3000, 3000, 3000], dtype=torch.int32)
        qlens = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        block_table = torch.zeros(B, 10, dtype=torch.int32)
        start_topm_cache = torch.tensor([True, True, True], dtype=torch.bool)
        ustep = torch.tensor([1, 0, 2], dtype=torch.int32)
        topm_idxs = torch.arange(B * self.index_topm, dtype=torch.int32).view(B, 1, self.index_topm)

        with patch("torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata",
                   side_effect=[
                       torch.zeros(1024, dtype=torch.int32),
                       torch.zeros(1024, dtype=torch.int32),
                   ]):
            default, compute, reuse = builder._build_topm_subgroups(
                kvlens=kvlens, qlens=qlens, block_table=block_table,
                start_topm_cache=start_topm_cache, ustep=ustep,
                topm_idxs=topm_idxs, index_topm=self.index_topm,
                micro_step_num=self.micro_step_num, index_topk=self.index_topk,
            )

        self.assertIsNotNone(reuse)
        self.assertIsNotNone(reuse.topm_idxs)
        self.assertEqual(reuse.topm_idxs.shape[0], reuse.mask.sum().item())
        # req 0 (ustep=1) -> reuse, req 1 (ustep=0) -> compute, req 2 (ustep=2) -> reuse
        self.assertTrue(torch.equal(reuse.topm_idxs[0], topm_idxs[0]))
        self.assertTrue(torch.equal(reuse.topm_idxs[1], topm_idxs[2]))
