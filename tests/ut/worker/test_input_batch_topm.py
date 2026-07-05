import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.worker.npu_input_batch import NPUInputBatch, TopMReqState


class TestTopMReqState(unittest.TestCase):
    def test_default_values(self):
        state = TopMReqState()
        self.assertEqual(state.ustep, 0)
        self.assertFalse(state.start_cache)
        self.assertIsNone(state.topm_idxs)

    def test_custom_values(self):
        idxs = torch.zeros(1, 550, dtype=torch.int32)
        state = TopMReqState(ustep=5, start_cache=True, topm_idxs=idxs)
        self.assertEqual(state.ustep, 5)
        self.assertTrue(state.start_cache)
        self.assertIs(state.topm_idxs, idxs)

    def test_ustep_increment(self):
        state = TopMReqState(ustep=3)
        state.ustep += 1
        self.assertEqual(state.ustep, 4)

    def test_start_cache_toggle(self):
        state = TopMReqState()
        state.start_cache = True
        self.assertTrue(state.start_cache)
        state.start_cache = False
        self.assertFalse(state.start_cache)


class TestNPUInputBatchTopmState(unittest.TestCase):
    def setUp(self):
        patcher_multi = patch("vllm_ascend.worker.npu_input_batch.MultiGroupBlockTable")
        self.mock_multi = patcher_multi.start()
        self.addCleanup(patcher_multi.stop)

        self.input_batch = NPUInputBatch.__new__(NPUInputBatch)
        self.input_batch.max_num_reqs = 4
        self.input_batch.max_model_len = 1024
        self.input_batch.max_num_batched_tokens = 1024
        self.input_batch.device = torch.device("cpu")
        self.input_batch.pin_memory = False
        self.input_batch.vocab_size = 32000
        self.input_batch._req_ids = [None] * 4
        self.input_batch.req_id_to_index = {}
        self.input_batch.topm_state = {}

    def test_init_topm_state_empty(self):
        self.assertEqual(self.input_batch.topm_state, {})

    def test_add_request_creates_topm_entry(self):
        request = MagicMock()
        request.req_id = "req_0"

        with patch.object(NPUInputBatch, "add_request", wraps=self.input_batch.add_request) as wrapped:
            with patch("vllm.v1.worker.gpu_input_batch.InputBatch.add_request", return_value=0):
                idx = self.input_batch.add_request(request)

        self.assertEqual(idx, 0)
        self.assertIn("req_0", self.input_batch.topm_state)
        self.assertEqual(self.input_batch.topm_state["req_0"], {})

    def test_add_request_overrides_setdefault(self):
        self.input_batch.topm_state["req_0"] = {"existing": "data"}
        request = MagicMock()
        request.req_id = "req_0"

        with patch("vllm.v1.worker.gpu_input_batch.InputBatch.add_request", return_value=0):
            idx = self.input_batch.add_request(request)

        self.assertIn("req_0", self.input_batch.topm_state)
        self.assertNotIn("existing", self.input_batch.topm_state["req_0"])

    def test_remove_request_removes_topm_entry(self):
        self.input_batch.topm_state["req_0"] = {}
        self.input_batch.topm_state["req_1"] = {}

        with patch("vllm.v1.worker.gpu_input_batch.InputBatch.remove_request", return_value=0):
            idx = self.input_batch.remove_request("req_0")

        self.assertNotIn("req_0", self.input_batch.topm_state)
        self.assertIn("req_1", self.input_batch.topm_state)

    def test_remove_request_unknown_id_no_error(self):
        self.input_batch.topm_state["req_0"] = {}

        with patch("vllm.v1.worker.gpu_input_batch.InputBatch.remove_request", return_value=None):
            idx = self.input_batch.remove_request("nonexistent")

        self.assertIn("req_0", self.input_batch.topm_state)

    def test_add_request_preserves_existing_topm_state(self):
        self.input_batch.topm_state["req_0"] = {"c4": TopMReqState(ustep=3)}
        request = MagicMock()
        request.req_id = "req_1"

        with patch("vllm.v1.worker.gpu_input_batch.InputBatch.add_request", return_value=1):
            self.input_batch.add_request(request)

        self.assertIn("req_0", self.input_batch.topm_state)
        self.assertIn("c4", self.input_batch.topm_state["req_0"])
        self.assertEqual(self.input_batch.topm_state["req_0"]["c4"].ustep, 3)
        self.assertIn("req_1", self.input_batch.topm_state)
        self.assertEqual(self.input_batch.topm_state["req_1"], {})

    def test_remove_request_preserves_other_entries(self):
        self.input_batch.topm_state["req_0"] = {"c4": TopMReqState(ustep=3)}
        self.input_batch.topm_state["req_1"] = {"c4": TopMReqState(ustep=7)}

        with patch("vllm.v1.worker.gpu_input_batch.InputBatch.remove_request", return_value=0):
            self.input_batch.remove_request("req_0")

        self.assertNotIn("req_0", self.input_batch.topm_state)
        self.assertIn("req_1", self.input_batch.topm_state)
        self.assertEqual(self.input_batch.topm_state["req_1"]["c4"].ustep, 7)
