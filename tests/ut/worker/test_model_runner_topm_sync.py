import unittest
from unittest.mock import MagicMock

import torch

from vllm_ascend.worker.npu_input_batch import TopMReqState


def _run_sync_loop(attn_metadata, input_batch, is_spec_decode=False):
    """Extracted sync logic from model_runner_v1.py — topm_idxs only."""
    if is_spec_decode or attn_metadata is None:
        return
    for ub_meta in attn_metadata:
        for layer_name, meta in ub_meta.items():
            decode = getattr(meta, 'decode', None)
            if decode is not None and decode.topm_idxs is not None:
                B = decode.topm_idxs.shape[0]
                for i, rid in enumerate(input_batch.req_ids[:B]):
                    if rid is None:
                        continue
                    layer_state = input_batch.topm_state.setdefault(
                        rid, {}
                    ).setdefault(layer_name, TopMReqState())
                    layer_state.topm_idxs = decode.topm_idxs[i].clone()


class TestModelRunnerTopMSync(unittest.TestCase):
    """Test the sync loop: topm_idxs clone only (ustep/start_cache on CPU already)."""

    def test_sync_updates_topm_idxs(self):
        idxs = torch.arange(6, dtype=torch.int32).view(2, 1, 3)
        attn_metadata = [{
            "layer_0": MagicMock(
                decode=MagicMock(topm_idxs=idxs)
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
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(topm_idxs=torch.zeros(1, 1, 550, dtype=torch.int32))
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["new_req"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertIn("new_req", input_batch.topm_state)
        self.assertIn("c4", input_batch.topm_state["new_req"])
        self.assertIsInstance(input_batch.topm_state["new_req"]["c4"], TopMReqState)

    def test_sync_updates_existing_topm_req_state(self):
        new_idxs = torch.tensor([[[5]]], dtype=torch.int32)
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(topm_idxs=new_idxs)
            ),
        }]
        existing_idxs = torch.zeros(1, 1, 550, dtype=torch.int32)
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {
            "req_0": {"c4": TopMReqState(ustep=3, start_cache=False, topm_idxs=existing_idxs)},
        }

        _run_sync_loop(attn_metadata, input_batch)

        self.assertTrue(torch.equal(
            input_batch.topm_state["req_0"]["c4"].topm_idxs, new_idxs[0]
        ))

    def test_sync_handles_multiple_ub_slices(self):
        attn_metadata = [
            {
                "c4": MagicMock(
                    decode=MagicMock(topm_idxs=torch.zeros(1, 1, 550, dtype=torch.int32))
                ),
            },
            {
                "c4": MagicMock(
                    decode=MagicMock(topm_idxs=torch.ones(1, 1, 550, dtype=torch.int32))
                ),
            },
        ]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", "req_1"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertEqual(
            input_batch.topm_state["req_0"]["c4"].topm_idxs.sum().item(), 0
        )
        self.assertEqual(
            input_batch.topm_state["req_1"]["c4"].topm_idxs.sum().item(), 550
        )

    def test_sync_handles_multiple_layers(self):
        idxs_0 = torch.zeros(1, 1, 550, dtype=torch.int32)
        idxs_1 = torch.ones(1, 1, 550, dtype=torch.int32)
        attn_metadata = [{
            "c4": MagicMock(decode=MagicMock(topm_idxs=idxs_0)),
            "swa": MagicMock(decode=MagicMock(topm_idxs=idxs_1)),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertTrue(torch.equal(
            input_batch.topm_state["req_0"]["c4"].topm_idxs, idxs_0[0]
        ))
        self.assertTrue(torch.equal(
            input_batch.topm_state["req_0"]["swa"].topm_idxs, idxs_1[0]
        ))

    def test_sync_skips_metadata_without_decode(self):
        attn_metadata = [{
            "layer_0": MagicMock(spec=[]),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertEqual(input_batch.topm_state, {})

    def test_sync_skips_decode_without_topm_idxs(self):
        attn_metadata = [{
            "layer_0": MagicMock(
                decode=MagicMock(topm_idxs=None)
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertEqual(input_batch.topm_state, {})

    def test_sync_skips_none_req_ids(self):
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(topm_idxs=torch.zeros(3, 1, 550, dtype=torch.int32))
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0", None, "req_2"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch)

        self.assertIn("req_0", input_batch.topm_state)
        self.assertNotIn(None, input_batch.topm_state)
        self.assertIn("req_2", input_batch.topm_state)

    def test_skips_spec_decode_mode(self):
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(topm_idxs=torch.zeros(1, 1, 550, dtype=torch.int32))
            ),
        }]
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(attn_metadata, input_batch, is_spec_decode=True)

        self.assertEqual(input_batch.topm_state, {})

    def test_skips_none_attn_metadata(self):
        input_batch = MagicMock()
        input_batch.req_ids = ["req_0"]
        input_batch.topm_state = {}

        _run_sync_loop(None, input_batch)

        self.assertEqual(input_batch.topm_state, {})

    def test_topm_idxs_clone_is_independent(self):
        orig = torch.tensor([[[1, 2, 3]]], dtype=torch.int32)
        attn_metadata = [{
            "c4": MagicMock(
                decode=MagicMock(topm_idxs=orig)
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
