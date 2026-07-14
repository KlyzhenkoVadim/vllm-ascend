import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder


class TestDSAMetadataBuilderTopMBuffers(unittest.TestCase):
    """Test that build_decode_metadata creates topm_idxs and delegates to _build_topm_subgroups."""

    def setUp(self):
        def mock_parent_init(self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, **kw):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__",
            mock_parent_init,
        )
        self.parent_init_patcher.start()

        self.tp_patcher = patch("vllm_ascend.attention.dsa_v1.get_tensor_model_parallel_world_size", return_value=1)
        self.tp_patcher.start()

        self.cfg_patcher = patch("vllm.config.get_current_vllm_config", return_value=MagicMock())
        self.cfg_patcher.start()

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
        vllm_config.additional_config = {
            "local_k_cache_config": {"index_topm": 550, "micro_step_num": 4}
        }

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

    def _make_common_attn_metadata(self, num_reqs=3, seq_lens=None):
        meta = MagicMock()
        meta.num_reqs = num_reqs
        meta.num_input_tokens = num_reqs
        meta.num_actual_tokens = num_reqs
        meta.query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        meta.query_start_loc_cpu = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        meta.positions = torch.tensor([100, 200, 300], dtype=torch.int32)
        meta.positions_cpu = torch.tensor([100, 200, 300], dtype=torch.int32)
        if seq_lens is None:
            seq_lens = torch.tensor([500, 600, 700], dtype=torch.int32)
        meta.seq_lens = seq_lens
        meta._seq_lens_cpu = seq_lens.clone()
        meta.seq_lens_cpu = seq_lens.clone()
        meta.max_query_len = 1
        meta.max_seq_len = seq_lens.max().item()
        meta.causal = True
        meta.block_table_tensor = torch.arange(30, dtype=torch.int32).view(3, 10)
        return meta

    def test_no_topm_without_input_batch(self):
        meta = self._make_common_attn_metadata()
        self.builder.seq_lens = meta.seq_lens
        self.builder.block_table = meta.block_table_tensor
        self.builder.decode_ratio_to_sas_metadata = {}

        with patch.object(AscendDSAMetadataBuilder, "_build_topm_subgroups", return_value=(None, None, None)):
            with patch("vllm_ascend.attention.dsa_v1.DeviceOperator") as mock_dev:
                mock_dev.get_dsa_sparse_attn_metadata_op.return_value = lambda **kw: torch.zeros(1024, dtype=torch.int32)
                mock_dev.get_dsa_sparse_attn_metadata_kwargs.return_value = {}
                mock_dev.get_dsa_decode_cu_seqlens_cmp_kv.return_value = {}
                result = self.builder.build_decode_metadata(
                    common_prefix_len=0, common_attn_metadata=meta, num_reqs_actual=3, input_batch=None
                )

        self.assertIsNone(result.topm_idxs)
        self.assertIsNone(result.default_metadata)
        self.assertIsNone(result.compute_topm_metadata)
        self.assertIsNone(result.reuse_topm_metadata)

    def test_topm_created_with_input_batch(self):
        seq_lens = torch.tensor([3000, 4000, 5000], dtype=torch.int32)
        meta = self._make_common_attn_metadata(seq_lens=seq_lens)
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
                    input_batch=input_batch,
                )

        self.assertIsNotNone(result.topm_idxs)
        self.assertEqual(result.topm_idxs.shape, (3, 1, 550))

    def test_build_topm_subgroups_called_with_correct_args(self):
        seq_lens = torch.tensor([3000, 4000, 5000], dtype=torch.int32)
        meta = self._make_common_attn_metadata(seq_lens=seq_lens)
        self.builder.seq_lens = meta.seq_lens
        self.builder.block_table = meta.block_table_tensor
        self.builder.decode_ratio_to_sas_metadata = {}

        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", "req_1", "req_2"]
        input_batch.topm_state = {}

        with patch.object(AscendDSAMetadataBuilder, "_build_topm_subgroups",
                          return_value=(MagicMock(), MagicMock(), MagicMock())) as mock_subgroups:
            with patch("vllm_ascend.attention.dsa_v1.DeviceOperator") as mock_dev:
                mock_dev.get_dsa_sparse_attn_metadata_op.return_value = lambda **kw: torch.zeros(1024, dtype=torch.int32)
                mock_dev.get_dsa_sparse_attn_metadata_kwargs.return_value = {}
                mock_dev.get_dsa_decode_cu_seqlens_cmp_kv.return_value = {}
                self.builder.build_decode_metadata(
                    common_prefix_len=0, common_attn_metadata=meta, num_reqs_actual=3,
                    input_batch=input_batch,
                )

        mock_subgroups.assert_called_once()
        call_kwargs = mock_subgroups.call_args[1]
        self.assertIn("input_batch", call_kwargs)
        self.assertIn("layer_name", call_kwargs)
        self.assertIn("seq_lens_np", call_kwargs)
        self.assertIn("topm_idxs", call_kwargs)
        self.assertEqual(call_kwargs["index_topm"], 550)
        self.assertEqual(call_kwargs["index_topk"], 64)
        self.assertEqual(call_kwargs["micro_step_num"], 4)

    def test_topm_not_created_when_num_decodes_zero(self):
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
            result = self.builder.build_decode_metadata(
                common_prefix_len=0, common_attn_metadata=meta, num_reqs_actual=0,
                input_batch=input_batch,
            )

        self.assertIsNone(result.topm_idxs)


class TestDSABuildTopMSubgroups(unittest.TestCase):
    """Test _build_topm_subgroups: mask computation from CPU state + subgroup building."""

    def setUp(self):
        self.index_topm = 550
        self.index_topk = 64
        self.micro_step_num = 4
        self.layer_name = "c4"

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

    def _call_with_topm_state(self, builder, topm_state, kvlens, seq_lens_np, B=3):
        req_ids = list(topm_state.keys())[:B]
        input_batch = MagicMock()
        input_batch.req_ids = req_ids
        input_batch.topm_state = topm_state

        qlens = torch.arange(B + 1, dtype=torch.int32)
        block_table = torch.zeros(B, 10, dtype=torch.int32)
        topm_idxs = torch.zeros(B, 1, self.index_topm, dtype=torch.int32)

        with patch("torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata",
                    return_value=torch.zeros(1024, dtype=torch.int32)):
            default, compute, reuse = builder._build_topm_subgroups(
                input_batch=input_batch,
                layer_name=self.layer_name,
                seq_lens_np=seq_lens_np,
                kvlens=kvlens,
                qlens=qlens,
                block_table=block_table,
                topm_idxs=topm_idxs,
                index_topm=self.index_topm,
                index_topk=self.index_topk,
                micro_step_num=self.micro_step_num,
            )
        return default, compute, reuse, topm_idxs

    def test_all_default_when_no_state(self):
        builder = self._make_builder()
        B = 3
        kvlens = torch.tensor([500, 800, 1000], dtype=torch.int32)
        seq_lens_np = kvlens.numpy()

        default, compute, reuse, _ = self._call_with_topm_state(
            builder, topm_state={}, kvlens=kvlens, seq_lens_np=seq_lens_np, B=B,
        )

        self.assertIsNotNone(default)
        self.assertIsNone(compute)
        self.assertIsNone(reuse)

    def test_short_seq_is_default(self):
        builder = self._make_builder()
        B = 1
        kvlens = torch.tensor([100], dtype=torch.int32)
        seq_lens_np = kvlens.numpy()

        state = MagicMock(ustep=0, start_cache=False, topm_idxs=None)
        default, compute, reuse, _ = self._call_with_topm_state(
            builder, topm_state={"req_0": {self.layer_name: state}},
            kvlens=kvlens, seq_lens_np=seq_lens_np, B=B,
        )

        self.assertIsNotNone(default)
        self.assertIsNone(compute)
        self.assertIsNone(reuse)

    def test_crosses_threshold_sets_start_cache(self):
        builder = self._make_builder()
        B = 1
        kvlens = torch.tensor([3000], dtype=torch.int32)
        seq_lens_np = kvlens.numpy()

        state = MagicMock(ustep=0, start_cache=False, topm_idxs=None)
        self._call_with_topm_state(
            builder, topm_state={"req_0": {self.layer_name: state}},
            kvlens=kvlens, seq_lens_np=seq_lens_np, B=B,
        )

        self.assertTrue(state.start_cache)
        self.assertEqual(state.ustep, 1)

    def test_compute_when_mod_zero(self):
        builder = self._make_builder()
        B = 1
        kvlens = torch.tensor([3000], dtype=torch.int32)
        seq_lens_np = kvlens.numpy()

        state = MagicMock(ustep=0, start_cache=True, topm_idxs=None)
        default, compute, reuse, _ = self._call_with_topm_state(
            builder, topm_state={"req_0": {self.layer_name: state}},
            kvlens=kvlens, seq_lens_np=seq_lens_np, B=B,
        )

        self.assertIsNone(default)
        self.assertIsNotNone(compute)
        self.assertIsNone(reuse)
        self.assertEqual(state.ustep, 1)

    def test_reuse_when_mod_nonzero(self):
        builder = self._make_builder()
        B = 1
        kvlens = torch.tensor([3000], dtype=torch.int32)
        seq_lens_np = kvlens.numpy()

        state = MagicMock(ustep=5, start_cache=True, topm_idxs=None)
        default, compute, reuse, _ = self._call_with_topm_state(
            builder, topm_state={"req_0": {self.layer_name: state}},
            kvlens=kvlens, seq_lens_np=seq_lens_np, B=B,
        )

        self.assertIsNone(default)
        self.assertIsNone(compute)
        self.assertIsNotNone(reuse)
        self.assertEqual(state.ustep, 6)

    def test_mixed_requests(self):
        builder = self._make_builder()
        B = 3
        kvlens = torch.tensor([3000, 4000, 5000], dtype=torch.int32)
        seq_lens_np = kvlens.numpy()

        expected_idxs = torch.arange(self.index_topm, dtype=torch.int32).view(1, self.index_topm)
        topm_state = {
            "req_0": {self.layer_name: MagicMock(ustep=5, start_cache=True, topm_idxs=expected_idxs)},
            "req_1": {self.layer_name: MagicMock(ustep=0, start_cache=False, topm_idxs=None)},
            "req_2": {self.layer_name: MagicMock(ustep=4, start_cache=True, topm_idxs=expected_idxs)},
        }

        default, compute, reuse, _ = self._call_with_topm_state(
            builder, topm_state=topm_state, kvlens=kvlens, seq_lens_np=seq_lens_np, B=B,
        )

        self.assertIsNone(default)
        self.assertIsNotNone(compute)
        self.assertIsNotNone(reuse)

    def test_ustep_incremented_on_cpu(self):
        builder = self._make_builder()
        B = 2
        kvlens = torch.tensor([3000, 3000], dtype=torch.int32)
        seq_lens_np = kvlens.numpy()

        state_0 = MagicMock(ustep=0, start_cache=True, topm_idxs=None)
        state_1 = MagicMock(ustep=3, start_cache=True, topm_idxs=None)
        topm_state = {
            "req_0": {self.layer_name: state_0},
            "req_1": {self.layer_name: state_1},
        }

        self._call_with_topm_state(
            builder, topm_state=topm_state, kvlens=kvlens, seq_lens_np=seq_lens_np, B=B,
        )

        self.assertEqual(state_0.ustep, 1)
        self.assertEqual(state_1.ustep, 4)

    def test_topm_idxs_copied_from_state(self):
        builder = self._make_builder()
        B = 1
        kvlens = torch.tensor([3000], dtype=torch.int32)
        seq_lens_np = kvlens.numpy()

        expected_idxs = torch.arange(self.index_topm, dtype=torch.int32).view(1, self.index_topm)
        state = MagicMock(ustep=0, start_cache=True, topm_idxs=expected_idxs)

        _, _, reuse, topm_idxs = self._call_with_topm_state(
            builder, topm_state={"req_0": {self.layer_name: state}},
            kvlens=kvlens, seq_lens_np=seq_lens_np, B=B,
        )

        self.assertIsNotNone(reuse)
        self.assertTrue(torch.equal(topm_idxs[0, 0], expected_idxs[0]))

    def test_reuse_metadata_has_correct_topm_idxs(self):
        builder = self._make_builder()
        B = 3
        kvlens = torch.tensor([3000, 3000, 3000], dtype=torch.int32)
        seq_lens_np = kvlens.numpy()

        expected_idxs_0 = torch.arange(self.index_topm, dtype=torch.int32).view(1, self.index_topm)
        expected_idxs_2 = torch.arange(self.index_topm, 2 * self.index_topm, dtype=torch.int32).view(1, self.index_topm)
        topm_state = {
            "req_0": {self.layer_name: MagicMock(ustep=5, start_cache=True, topm_idxs=expected_idxs_0)},
            "req_1": {self.layer_name: MagicMock(ustep=0, start_cache=False, topm_idxs=None)},
            "req_2": {self.layer_name: MagicMock(ustep=5, start_cache=True, topm_idxs=expected_idxs_2)},
        }

        default, compute, reuse, _ = self._call_with_topm_state(
            builder, topm_state=topm_state, kvlens=kvlens, seq_lens_np=seq_lens_np, B=B,
        )

        self.assertIsNone(default)
        self.assertIsNotNone(compute)
        self.assertIsNotNone(reuse)
        self.assertEqual(reuse.topm_idxs.shape[0], 2)
