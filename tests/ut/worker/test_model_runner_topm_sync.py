import unittest
from unittest.mock import MagicMock

import torch

from vllm_ascend.worker.npu_input_batch import TopMReqState


def _run_sync_loop(attn_metadata, input_batch, is_spec_decode=False):
    """Extracted sync logic from model_runner_v1.py:2316-2332."""
    if is_spec_decode or attn_metadata is None:
        return
    for ub_meta in attn_metadata:
        for layer_name, meta in ub_meta.items():
            decode = getattr(meta, 'decode', None)
            if decode is not None and decode.topm_ustep is not None:
                B = decode.topm_ustep.shape[0]
                for i, rid in enumerate(input_batch.req_ids[:B]):
                    if rid is None:
                        continue
                    layer_state = input_batch.topm_state.setdefault(
                        rid, {}
                    ).setdefault(layer_name, TopMReqState())
                    layer_state.ustep = decode.topm_ustep[i].item()
                    layer_state.start_cache = decode.topm_start_cache[i].item()
                    if decode.topm_idxs is not None:
                        layer_state.topm_idxs = decode.topm_idxs[i].clone()


class TestModelRunnerTopMSync(unittest.TestCase):
    """Test the sync loop that writes decode temp buffers back to input_batch.topm_state."""

    def test_sync_updates_ustep_and_start_cache(self):
        """Basic sync: ustep and start_cache values should be copied to topm_state."""
        attn_metadata = [{
            "layer_0": MagicMock(
                decode=MagicMock(
                    topm_ustep=torch.tensor([3, 7, 0], dtype=torch.int32),
                    topm_start_cache=torch.tensor([True, False, True], dtype=torch.bool),
                    topm_idxs=None,
                )
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", "req_1", "req_2"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertEqual(input_batch.topm_state["req_0"]["layer_0"].ustep, 3)
        self.assertEqual(input_batch.topm_state["req_0"]["layer_0"].start_cache, True)
        self.assertEqual(input_batch.topm_state["req_1"]["layer_0"].ustep, 7)
        self.assertEqual(input_batch.topm_state["req_1"]["layer_0"].start_cache, False)
        self.assertEqual(input_batch.topm_state["req_2"]["layer_0"].ustep, 0)
        self.assertEqual(input_batch.topm_state["req_2"]["layer_0"].start_cache, True)

    def test_sync_updates_topm_idxs(self):
        """topm_idxs should be cloned to topm_state when present."""
        idxs = torch.arange(6, dtype=torch.int32).view(2, 1, 3)
        attn_metadata = [{
            "layer_0": MagicMock(
                decode=MagicMock(
                    topm_ustep=torch.tensor([1, 2], dtype=torch.int32),
                    topm_start_cache=torch.tensor([True, False], dtype=torch.bool),
                    topm_idxs=idxs,
                )
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", "req_1"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertTrue(torch.equal(
            input_batch.topm_state["req_0"]["layer_0"].topm_idxs, idxs[0]
        ))
        self.assertTrue(torch.equal(
            input_batch.topm_state["req_1"]["layer_0"].topm_idxs, idxs[1]
        ))

    def test_sync_creates_new_topm_req_state_on_setdefault(self):
        """When topm_state has no entry for a req_id, sync should create TopMReqState."""
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(
                    topm_ustep=torch.tensor([5], dtype=torch.int32),
                    topm_start_cache=torch.tensor([True], dtype=torch.bool),
                    topm_idxs=None,
                )
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["new_req"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertIn("new_req", input_batch.topm_state)
        self.assertIn("c4", input_batch.topm_state["new_req"])
        self.assertIsInstance(input_batch.topm_state["new_req"]["c4"], TopMReqState)
        self.assertEqual(input_batch.topm_state["new_req"]["c4"].ustep, 5)

    def test_sync_updates_existing_topm_req_state(self):
        """Existing TopMReqState should be overwritten with new values."""
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(
                    topm_ustep=torch.tensor([10], dtype=torch.int32),
                    topm_start_cache=torch.tensor([True], dtype=torch.bool),
                    topm_idxs=None,
                )
            ),
        }]
        existing_idxs = torch.zeros(1, 550, dtype=torch.int32)
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {
            "req_0": {"c4": TopMReqState(ustep=3, start_cache=False, topm_idxs=existing_idxs)},
        }

        _run_sync_loop(attn_metadata, input_batch)

        self.assertEqual(input_batch.topm_state["req_0"]["c4"].ustep, 10)
        self.assertEqual(input_batch.topm_state["req_0"]["c4"].start_cache, True)
        self.assertIsNone(input_batch.topm_state["req_0"]["c4"].topm_idxs)

    def test_sync_handles_multiple_ub_slices(self):
        """Sync should process all UB slices in attn_metadata list."""
        attn_metadata = [
            {
                "c4": MagicMock(
                    decode=MagicMock(
                        topm_ustep=torch.tensor([1], dtype=torch.int32),
                        topm_start_cache=torch.tensor([False], dtype=torch.bool),
                        topm_idxs=None,
                    )
                ),
            },
            {
                "c4": MagicMock(
                    decode=MagicMock(
                        topm_ustep=torch.tensor([2], dtype=torch.int32),
                        topm_start_cache=torch.tensor([True], dtype=torch.bool),
                        topm_idxs=None,
                    )
                ),
            },
        ]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", "req_1"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertEqual(input_batch.topm_state["req_0"]["c4"].ustep, 1)
        self.assertEqual(input_batch.topm_state["req_0"]["c4"].start_cache, False)
        self.assertEqual(input_batch.topm_state["req_1"]["c4"].ustep, 2)
        self.assertEqual(input_batch.topm_state["req_1"]["c4"].start_cache, True)

    def test_sync_handles_multiple_layers(self):
        """Sync should process all layers in each metadata dict."""
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(
                    topm_ustep=torch.tensor([3], dtype=torch.int32),
                    topm_start_cache=torch.tensor([True], dtype=torch.bool),
                    topm_idxs=None,
                )
            ),
            "swa": MagicMock(
                decode=MagicMock(
                    topm_ustep=torch.tensor([7], dtype=torch.int32),
                    topm_start_cache=torch.tensor([False], dtype=torch.bool),
                    topm_idxs=None,
                )
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertEqual(input_batch.topm_state["req_0"]["c4"].ustep, 3)
        self.assertEqual(input_batch.topm_state["req_0"]["swa"].ustep, 7)

    def test_sync_skips_metadata_without_decode(self):
        """Metadata entries without decode attribute should be skipped."""
        attn_metadata = [{
            "layer_0": MagicMock(spec=[]),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertEqual(input_batch.topm_state, {})

    def test_sync_skips_decode_without_topm_ustep(self):
        """Decode metadata without topm_ustep (None) should be skipped."""
        attn_metadata = [{
            "layer_0": MagicMock(
                decode=MagicMock(
                    topm_ustep=None,
                )
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertEqual(input_batch.topm_state, {})

    def test_sync_skips_none_req_ids(self):
        """Requests with None in req_ids should be skipped (padding)."""
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(
                    topm_ustep=torch.tensor([10, 20, 30], dtype=torch.int32),
                    topm_start_cache=torch.tensor([True, False, True], dtype=torch.bool),
                    topm_idxs=None,
                )
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", None, "req_2"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertIn("req_0", input_batch.topm_state)
        self.assertNotIn(None, input_batch.topm_state)
        self.assertIn("req_2", input_batch.topm_state)
        self.assertEqual(input_batch.topm_state["req_0"]["c4"].ustep, 10)
        self.assertEqual(input_batch.topm_state["req_2"]["c4"].ustep, 30)

    def test_skips_spec_decode_mode(self):
        """Sync should be skipped when is_spec_decode is True."""
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(
                    topm_ustep=torch.tensor([5], dtype=torch.int32),
                    topm_start_cache=torch.tensor([True], dtype=torch.bool),
                    topm_idxs=None,
                )
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch, is_spec_decode=True)

        self.assertEqual(input_batch.topm_state, {})

    def test_skips_none_attn_metadata(self):
        """Sync should be skipped when attn_metadata is None."""
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(None, input_batch)

        self.assertEqual(input_batch.topm_state, {})

    def test_topm_idxs_clone_is_independent(self):
        """Cloned topm_idxs tensor should be independent of the original."""
        orig = torch.tensor([[[1, 2, 3]]], dtype=torch.int32)
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(
                    topm_ustep=torch.tensor([0], dtype=torch.int32),
                    topm_start_cache=torch.tensor([False], dtype=torch.bool),
                    topm_idxs=orig,
                )
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        cloned = input_batch.topm_state["req_0"]["c4"].topm_idxs
        self.assertTrue(torch.equal(cloned, orig))
        orig[0, 0, 0] = 999
        self.assertNotEqual(cloned[0, 0, 0].item(), 999)
