import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, TypeAlias

import torch
import torch.nn.functional as F
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, split_decodes_and_prefills
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
    npu_stream_switch,
    olora_tp_enable,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch, TopMReqState

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

if HAS_TRITON:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F811
else:
    triton_q_rms = None  # type: ignore

BUILD_METADATA_STEP_PREFILL = 0
BUILD_METADATA_STEP_DECODE = 1

_DSV4_DSA_OVERLAP_STREAM = None


def dsv4_dsa_overlap_stream() -> torch.npu.Stream:
    global _DSV4_DSA_OVERLAP_STREAM
    if _DSV4_DSA_OVERLAP_STREAM is None:
        _DSV4_DSA_OVERLAP_STREAM = torch_npu.npu.Stream()
    return _DSV4_DSA_OVERLAP_STREAM


# mypy: disable-error-code="has-type"


def hadamard_transform_ref(
    x: torch.Tensor,
    hadamard: torch.Tensor,
    scale: float = 1.0,
):
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, hadamard=hadamard, scale=hidden_size**-0.5)


def hadamard_linear(x: torch.Tensor, hadamard: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int]:
    """
    Part 1 of rotate_activation: Execute F.linear (matrix multiplication).
    This runs in main stream, parallel with aux_stream kv_scatter.

    Returns:
        Tuple of (linear_output, original_shape, original_dim)
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard)
    return out, x_shape, dim


def hadamard_scale(out: torch.Tensor, x_shape: tuple[int, ...], dim: int, scale: float = 1.0) -> torch.Tensor:
    """
    Part 2 of rotate_activation: Execute scale multiplication and reshape.
    This runs in main stream after aux_stream completes.
    """
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def _is_w8a8_dynamic(linear) -> bool:
    """True iff ``linear`` is wired up with ``AscendW8A8DynamicLinearMethod``."""
    qm = getattr(linear, "quant_method", None)
    if qm is None or isinstance(qm, AscendUnquantizedLinearMethod):
        return False
    inner = getattr(qm, "quant_method", None)
    return isinstance(inner, AscendW8A8DynamicLinearMethod)


def pad_to_blocks(x: torch.Tensor, length_list: torch.Tensor, block_size: int = 128):
    """
    Pads a ragged/packed tensor into fixed-size blocks.

    Args:
        x: Input tensor of shape [t, n, d] where t = sum(length_list).
        length_list: Tensor of shape [bs] containing valid sequence lengths.
        block_size: The size of each block (default 128).

    Returns:
        padded_blocks: Tensor of shape [total_blocks, block_size, n, d].
    """
    # 1. Validation
    if x.shape[0] != length_list.sum():
        raise ValueError(f"Input dimension 0 ({x.shape[0]}) does not match sum of length_list ({length_list.sum()})")

    bs = length_list.shape[0]
    n, d = x.shape[1], x.shape[2]

    # 2. Calculate how many blocks are needed for each request
    # Formula: ceil(length / block_size) -> (length + block_size - 1) // block_size
    blocks_per_req = (length_list + block_size - 1) // block_size
    total_blocks = blocks_per_req.sum() + 1

    # 3. Allocate output tensor with zeros (this handles the padding automatically)
    # Shape: [total_blocks, block_size, n, d]
    out = torch.zeros((total_blocks, block_size, n, d), dtype=x.dtype, device=x.device)

    # 4. Fill data
    input_offset = 0
    block_offset = 1

    for i in range(bs):
        length = length_list[i]
        num_blocks = blocks_per_req[i]

        if length > 0:
            # Slice the valid data for this request from the packed input
            # Shape: [length, n, d]
            req_data = x[input_offset : input_offset + length]

            # Select the assigned blocks in the output
            # Shape: [num_blocks, block_size, n, d]
            target_blocks = out[block_offset : block_offset + num_blocks]

            # View as a flat sequence to easily copy the data
            # Shape: [num_blocks * block_size, n, d]
            target_flat = target_blocks.view(-1, n, d)

            # Copy valid data into the beginning of the allocated blocks
            # The rest remains zeros
            target_flat[:length] = req_data

        # Update pointers
        input_offset += length
        block_offset += num_blocks

    return out


class AscendDSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "ASCEND_DSA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_builder_cls():
        from vllm_ascend.utils import enable_dsa_cp

        if enable_dsa_cp():
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder

            return AscendDSACPMetadataBuilder
        return AscendDSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_scale_shape(num_blocks: int, block_size: int, scale_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, scale_size

    @staticmethod
    def get_impl_cls() -> type["DSAAttentionImpl"]:
        from vllm_ascend.utils import enable_dsa_cp

        if enable_dsa_cp():
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl

            return AscendDSACPImpl
        return AscendDSAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [2, 4, 8, 16, 32, 64, 128]


@dataclass
class AscendDSAPrefillMetadata:
    """Prefill Specific Metadata for Ascend"""

    attn_mask: torch.Tensor
    query_lens: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    input_positions: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_query_len: int
    max_seq_lens: int

    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    start_pos: torch.Tensor | None = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    cu_c4_cmp_seqlen_list: torch.Tensor = None
    cu_c128_cmp_seqlen_list: torch.Tensor = None
    topm_idxs: torch.Tensor = None
    topm_num_blocks: int = 0
    topm_chunk_start_logical: int = 0
    act_qlen: int = 0


@dataclass
class AscendDSADecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen_kv: int
    max_seqlen_q: int
    seq_lens_list: list[int]
    max_seq_lens: int
    slot_mapping: torch.Tensor

    query_start_loc: torch.tensor = None
    query_start_loc_cpu: torch.tensor = None
    attn_mask: torch.Tensor | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    cp_seq_len: torch.Tensor = None
    batch_seq_mask: torch.Tensor = None
    start_pos: torch.Tensor = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None


@dataclass
class AscendDSAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    query_lens: list[int] | None = None
    # The dimension of the attention heads
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: AscendDSADecodeMetadata | None = None
    prefill: AscendDSAPrefillMetadata | None = None
    reshape_cache_event: torch.npu.Event = None

    # metadata for dsv4 indexer

    hadamard: torch.Tensor | None = None

    start_pos: torch.Tensor | None = None

    def __post_init__(self):
        pass


DSAMetadataList: TypeAlias = list[AscendDSAMetadata]
DSAPrepareResult: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    AscendDSAMetadata,
    bool,
]


def _require_prefill_metadata(metadata: AscendDSAMetadata) -> AscendDSAPrefillMetadata:
    assert metadata.prefill is not None
    return metadata.prefill


def _require_decode_metadata(metadata: AscendDSAMetadata) -> AscendDSADecodeMetadata:
    assert metadata.decode is not None
    return metadata.decode

#SECTION - AscendDSAMetadataBuilder
class AscendDSAMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    hadamard = None
    start_pos_prefill: torch.Tensor | None = None
    start_pos_decode: torch.Tensor | None = None
    decode_sas_metadata: torch.Tensor | None = None
    decode_qli_metadata: torch.Tensor | None = None
    prefill_ratio_to_sas_metadata: dict | None = None
    decode_ratio_to_sas_metadata: dict | None = None
    block_size: int | None = 128
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec: AscendMLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendDSAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.metadata_cls = metadata_cls if metadata_cls is not None else AscendDSAMetadata
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        scheduler_config = vllm_config.scheduler_config
        # self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        self.spec_slot_mapping = None
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens,)  # type: ignore
        else:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens, 2)  # type: ignore
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.spec_slot_mapping = [
                torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
                for _ in range(spec_token_num)
            ]
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.cos_cache = None
        self.sin_cache = None

        self.cu_seq_lens_cpu: torch.Tensor = None
        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.context_lens_cpu: torch.Tensor = None
        self.num_actual_tokens: int | None = None
        self.block_table: torch.Tensor = None
        self.slot_mapping: torch.Tensor = None
        self.graph_pad_size = 0
        self.query_lens: torch.Tensor = None
        self.seq_lens: torch.Tensor = None
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

        self.compressor_ratio = getattr(kv_cache_spec, "compress_ratio", 0)
        hf_config = self.model_config.hf_config

        if AscendDSAMetadataBuilder.hadamard is None:
            if hf_config.model_type == "deepseek_v4":
                indexer_head_dim = hf_config.index_head_dim
                try:
                    from scipy.linalg import hadamard  # type: ignore[import-untyped]
                except ImportError as e:
                    raise ImportError("Please install scipy") from e
                log_dim = math.ceil(math.log2(indexer_head_dim))
                dim_padded = 2**log_dim
                if self.vllm_config.model_config.enable_sleep_mode:
                    # Sleep mode allocates KV inside CaMemAllocator; tag Hadamard so
                    # sleep/wake does not treat it as KV cache.
                    from vllm_ascend.device_allocator.camem import CaMemAllocator

                    allocator = CaMemAllocator.get_instance()
                    with allocator.use_allocation_tag(CaMemAllocator.sleep_persistent_tag):
                        AscendDSAMetadataBuilder.hadamard = torch.tensor(
                            hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device
                        ).to(torch.bfloat16)
                else:
                    AscendDSAMetadataBuilder.hadamard = torch.tensor(
                        hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device
                    ).to(torch.bfloat16)
        self.start_pos_prefill = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
        self.start_pos_decode = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
        self.decode_sas_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.decode_qli_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.cu_seqlens_ori_kv = torch.tensor([], device=self.device)
        self.cu_seqlens_cmp_kv = torch.tensor([], device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        self._zero_i32 = torch.tensor([0], device=self.device, dtype=torch.int32)
        # Note(qcs): we use two dimension slot_mapping for kvcache with shape
        # [block_nums, block_size, head_num, head_dim]
        self.slot_mapping = torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch", scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if num_tokens <= self.decode_threshold:
                decodes.append(i)
            else:
                prefills.append(i)

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill], decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        return modified_batch

    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    #SECTION - build
    # model_runner中调用接口
    # 模型前向传播过程中，每层都调用一次
    #LINK - /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:3015
    # 基于model_runner构造和传入的AscendCommonAttentionMetadata，构建AscendDSADecodeMetadata、AscendDSAPrefillMetadata
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        num_reqs_actual = kwargs.get("num_reqs_actual")
        self.prefill_ratio_to_sas_metadata = kwargs.get("prefill_ratio_to_sas_metadata")
        self.decode_ratio_to_sas_metadata = kwargs.get("decode_ratio_to_sas_metadata")
        assert self.prefill_ratio_to_sas_metadata is not None
        assert self.decode_ratio_to_sas_metadata is not None
        self.block_size = kwargs.get("block_size", 128)

        self.common_ratio_to_sas_metadata = kwargs.get("common_ratio_to_sas_metadata")
        assert self.common_ratio_to_sas_metadata is not None

        if self.common_ratio_to_sas_metadata.get("num_decodes", None) is None:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
            )
            self.common_ratio_to_sas_metadata["num_decodes"] = self.num_decodes
            self.common_ratio_to_sas_metadata["num_prefills"] = self.num_prefills
            self.common_ratio_to_sas_metadata["num_decode_tokens"] = self.num_decode_tokens
            self.common_ratio_to_sas_metadata["num_prefill_tokens"] = self.num_prefill_tokens
            self.set_num_actual_tokens(common_attn_metadata)
            assert self.num_decodes + self.num_prefills == num_reqs
            assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens
            num_input_tokens = common_attn_metadata.num_input_tokens
            input_positions = common_attn_metadata.positions[:num_input_tokens].long()
            self.common_ratio_to_sas_metadata["input_positions"] = input_positions
            if self.num_prefills:
                cos, sin = get_cos_and_sin_dsa(input_positions)
            else:
                cos, sin = get_cos_and_sin_dsa(input_positions, True)
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            self.common_ratio_to_sas_metadata["seq_lens"] = self.seq_lens

            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            self.query_lens = query_seq_lens_cpu[:num_reqs]
            self.common_ratio_to_sas_metadata["query_lens"] = self.query_lens
        else:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                self.common_ratio_to_sas_metadata["num_decodes"],
                self.common_ratio_to_sas_metadata["num_prefills"],
                self.common_ratio_to_sas_metadata["num_decode_tokens"],
                self.common_ratio_to_sas_metadata["num_prefill_tokens"],
            )
            self.set_num_actual_tokens(common_attn_metadata)
            num_input_tokens = common_attn_metadata.num_input_tokens
            input_positions = self.common_ratio_to_sas_metadata["input_positions"]
            cos, sin = self.common_ratio_to_sas_metadata["cos"], self.common_ratio_to_sas_metadata["sin"]
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]
            self.query_lens = self.common_ratio_to_sas_metadata["query_lens"]

        # Currently, MTP-fullgraph is incompatibility pcp
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        self.slot_mapping[:num_input_tokens] = DeviceOperator.format_dsa_slot_mapping(slot_mapping, self.block_size)

        self.graph_pad_size = common_attn_metadata.graph_pad_size
        block_table_size = self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_PREFILL)
        self.block_table = common_attn_metadata.block_table_tensor[:block_table_size]

        prefill_metadata = None
        if self.num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata(
                common_prefix_len, common_attn_metadata,
                input_batch=kwargs.get("input_batch", None))

        decode_metadata = None

        if self.num_decodes > 0:
            decode_metadata = self.build_decode_metadata(common_prefix_len, common_attn_metadata, num_reqs_actual)

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            query_lens=self.query_lens,
            slot_mapping=None,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_mask=None,
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=None,
            seq_lens=self.seq_lens,
            cos=cos,
            sin=sin,
            hadamard=AscendDSAMetadataBuilder.hadamard,
        )
    #!SECTION

    def build_prefill_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        input_batch: Optional["NPUInputBatch"] = None,
    ) -> AscendDSAPrefillMetadata:
        assert self.prefill_ratio_to_sas_metadata is not None
        assert self.decode_ratio_to_sas_metadata is not None
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        # reqs_start: the start request position of prefill request
        reqs_start = self.num_decodes
        # reqs_start: the start token position of prefill request
        tokens_start = self.num_decode_tokens

        if self.prefill_ratio_to_sas_metadata.get("prefill_input_positions", None) is None:
            input_positions = common_attn_metadata.positions[: self.num_actual_tokens].long()
            max_query_len = self.query_lens[reqs_start:].max().item()
            # Prefer _seq_lens_cpu (always available, updated during draft
            # iterations) over seq_lens_cpu (None in async spec decode mode).
            if common_attn_metadata._seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            elif common_attn_metadata.seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            else:
                _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
            max_seq_lens = _seq_lens_cpu[reqs_start:].max().item()
            self.prefill_ratio_to_sas_metadata["input_positions"] = input_positions
            self.prefill_ratio_to_sas_metadata["max_query_len"] = max_query_len
            self.prefill_ratio_to_sas_metadata["max_seq_lens"] = max_seq_lens

            prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]
            prefill_input_positions = input_positions[tokens_start:]
            self.prefill_ratio_to_sas_metadata["prefill_input_positions"] = prefill_input_positions
            self.prefill_ratio_to_sas_metadata["prefill_query_start_loc"] = prefill_query_start_loc

            prefill_query_start_loc_cpu = query_start_loc_cpu[reqs_start:] - query_start_loc_cpu[reqs_start]
            self.prefill_ratio_to_sas_metadata["prefill_query_start_loc_cpu"] = prefill_query_start_loc_cpu

            cos, sin = get_cos_and_sin_dsa(prefill_input_positions)
            self.prefill_ratio_to_sas_metadata["cos"] = cos
            self.prefill_ratio_to_sas_metadata["sin"] = sin

            prefill_seq_lens = self.seq_lens[reqs_start:]
            num_prefill = prefill_seq_lens.shape[0]
            self.prefill_ratio_to_sas_metadata["prefill_seq_lens"] = prefill_seq_lens
            self.prefill_ratio_to_sas_metadata["num_prefill"] = num_prefill
        else:
            input_positions = self.prefill_ratio_to_sas_metadata["input_positions"]
            max_query_len = self.prefill_ratio_to_sas_metadata["max_query_len"]
            max_seq_lens = self.prefill_ratio_to_sas_metadata["max_seq_lens"]
            prefill_input_positions = self.prefill_ratio_to_sas_metadata["prefill_input_positions"]
            prefill_query_start_loc = self.prefill_ratio_to_sas_metadata["prefill_query_start_loc"]
            prefill_query_start_loc_cpu = self.prefill_ratio_to_sas_metadata["prefill_query_start_loc_cpu"]
            cos = self.prefill_ratio_to_sas_metadata["cos"]
            sin = self.prefill_ratio_to_sas_metadata["sin"]
            prefill_seq_lens = self.prefill_ratio_to_sas_metadata["prefill_seq_lens"]
            num_prefill = self.prefill_ratio_to_sas_metadata["num_prefill"]

        def _get_padded_compressed_position(prefill_input_positions, compress_ratio):
            if compress_ratio <= 1:
                return prefill_input_positions
            mask = ((prefill_input_positions + 1) % compress_ratio) == 0
            input_positions = prefill_input_positions[mask]
            input_positions = (input_positions + 1) - compress_ratio
            target_shape = (
                min(self.num_prefill_tokens, self.num_prefill_tokens // compress_ratio + self.num_prefills),
            )
            pad_right = target_shape[0] - input_positions.shape[0]
            pad_positions = F.pad(input_positions, (0, pad_right), value=0.0)
            return pad_positions

        def _get_cmp_seq_lens(prefill_seq_lens, compress_ratio):
            # Note(qcs): some models use compress_ratio=0 as non-compression tag.
            _cmp_seq_lens = prefill_seq_lens // compress_ratio if compress_ratio >= 1 else prefill_seq_lens
            return torch.concat(
                (torch.tensor([0], device=_cmp_seq_lens.device), torch.cumsum(_cmp_seq_lens, -1)), dim=-1
            )

        def _get_compressed_decode_token_start_and_end(decode_input_positions, compress_ratio):
            # Note(qcs): some models use compress_ratio=0 as non-compression tag.
            if compress_ratio == 0:
                compress_ratio = 1
            # TODO(yilin): decode_input_positions is a device tensor,
            # this will introduce sync operation. Refactor me to torch.where instead
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            compressed_decode_num = mask.sum()

            end = min(self.num_prefill_tokens, self.num_prefill_tokens // compress_ratio + self.num_prefills)
            return compressed_decode_num, end

        if self.prefill_ratio_to_sas_metadata.get(f"c{self.compressor_ratio}_cos", None) is None:
            compress_cos, compress_sin = get_cos_and_sin_dsa(
                _get_padded_compressed_position(prefill_input_positions, self.compressor_ratio)
            )
            self.prefill_ratio_to_sas_metadata[f"c{self.compressor_ratio}_cos"] = compress_cos
            self.prefill_ratio_to_sas_metadata[f"c{self.compressor_ratio}_sin"] = compress_sin
        else:
            compress_cos = self.prefill_ratio_to_sas_metadata[f"c{self.compressor_ratio}_cos"]
            compress_sin = self.prefill_ratio_to_sas_metadata[f"c{self.compressor_ratio}_sin"]

        if self.prefill_ratio_to_sas_metadata.get(f"compressed_c{self.compressor_ratio}_tokens_start", None) is None:
            decode_input_positions = input_positions[:tokens_start]
            compressed_tokens_start, compressed_tokens_end = _get_compressed_decode_token_start_and_end(
                decode_input_positions, self.compressor_ratio
            )
            self.prefill_ratio_to_sas_metadata[f"compressed_c{self.compressor_ratio}_tokens_start"] = (
                compressed_tokens_start
            )
            self.prefill_ratio_to_sas_metadata[f"compressed_c{self.compressor_ratio}_tokens_ebd"] = (
                compressed_tokens_end
            )
        else:
            compressed_tokens_start = self.prefill_ratio_to_sas_metadata[
                f"compressed_c{self.compressor_ratio}_tokens_start"
            ]
            compressed_tokens_end = self.prefill_ratio_to_sas_metadata[
                f"compressed_c{self.compressor_ratio}_tokens_ebd"
            ]

        prefill_slot_mapping = self.slot_mapping[
            compressed_tokens_start : compressed_tokens_end + compressed_tokens_start
        ]

        assert self.start_pos_prefill is not None
        self.start_pos_prefill.fill_(0)
        seq_lens_q = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
        self.start_pos_prefill[:num_prefill] = self.seq_lens[reqs_start:] - seq_lens_q

        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = self.model_config.hf_config.index_topk

        # --- topM prefill: config ---
        index_topm = None
        micro_step_num = None
        local_k_cache_config = self.vllm_config.additional_config.get("local_k_cache_config", {})
        index_topm = local_k_cache_config.get("index_topm", None)
        micro_step_num = local_k_cache_config.get("micro_step_num", None)
        has_topm = (input_batch is not None and self.compressor_ratio == 4
                    and index_topm is not None and micro_step_num is not None)
        topm_idxs_prefill = None
        num_topm_blocks = 0
        chunk_start_logical = 0
        act_qlen = 0

        cu_c4_cmp_seqlen_list = None
        cu_c128_cmp_seqlen_list = None

        layer_name = f"c{self.compressor_ratio}"
        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        if self.compressor_ratio <= 1:
            if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
                self.prefill_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=prefill_query_start_loc,
                    cu_seqlens_ori_kv=prefill_query_start_loc,
                    cu_seqlens_cmp_kv=None,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[reqs_start:],
                    max_seqlen_q=seq_lens_q.max(),
                    max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                    batch_size=len(self.seq_lens[reqs_start:]),
                    cmp_ratio=1,
                    ori_mask_mode=4,  # 4:sliding window
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=False,
                )
            sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
        elif self.compressor_ratio == 4:
            if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
                self.prefill_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=prefill_query_start_loc,
                    cu_seqlens_ori_kv=prefill_query_start_loc,
                    cu_seqlens_cmp_kv=cu_c4_cmp_seqlen_list,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[reqs_start:],
                    max_seqlen_q=seq_lens_q.max(),
                    max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                    batch_size=len(self.seq_lens[reqs_start:]),
                    cmp_topk=index_topk,
                    # topk=index_topk,
                    cmp_ratio=4,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                )
            sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
        else:
            if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
                self.prefill_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=prefill_query_start_loc,
                    cu_seqlens_ori_kv=prefill_query_start_loc,
                    cu_seqlens_cmp_kv=cu_c128_cmp_seqlen_list,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[reqs_start:],
                    max_seqlen_q=seq_lens_q.max(),
                    max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                    batch_size=len(self.seq_lens[reqs_start:]),
                    cmp_ratio=128,  #
                    ori_mask_mode=4,  # 4:sliding window
                    cmp_mask_mode=3,  # 3:causal
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                )
            sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
        if self.prefill_ratio_to_sas_metadata.get("qli") is None:
            kvlens_key = self.seq_lens[reqs_start:].clone()
            max_seqlen_k = self.seq_lens[reqs_start:].max().item()

            if has_topm:
                B = num_prefill
                if common_attn_metadata._seq_lens_cpu is not None:
                    _sl_cpu = common_attn_metadata._seq_lens_cpu
                elif common_attn_metadata.seq_lens_cpu is not None:
                    _sl_cpu = common_attn_metadata.seq_lens_cpu
                else:
                    _sl_cpu = common_attn_metadata.seq_lens.cpu()
                seq_lens_np = _sl_cpu[reqs_start : reqs_start + B].numpy()
                layer_name_topm = self.layer_names[0]
                actual_qlens = prefill_query_start_loc_cpu[1:] - prefill_query_start_loc_cpu[:-1]

                topm_idxs_prefill, kvlens_key, max_seqlen_k, _has_cached, num_topm_blocks, chunk_start_logical, act_qlen = \
                    self._build_topm_subgroups_prefill(
                        input_batch=input_batch,
                        layer_name=layer_name_topm,
                        seq_lens_np=seq_lens_np,
                        kvlens=self.seq_lens[reqs_start:],
                        actual_qlens=actual_qlens,
                        block_table=self.block_table[reqs_start:, ...],
                        index_topm=index_topm,
                        index_topk=index_topk,
                        micro_step_num=micro_step_num,
                    )

            self.prefill_ratio_to_sas_metadata["qli"] = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=prefill_query_start_loc[1:].clone(),
                actual_seq_lengths_key=kvlens_key,
                num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,  # 128
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=len(self.seq_lens[reqs_start:]),
                max_seqlen_q=seq_lens_q.max().item(),
                max_seqlen_k=max_seqlen_k,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=index_topm if has_topm else index_topk,
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                device=str(self.seqused_q.device),
            )
        qli_metadata = self.prefill_ratio_to_sas_metadata.get("qli")

        return AscendDSAPrefillMetadata(
            attn_mask=None,
            query_lens=self.query_lens[reqs_start:].to(torch.int32),
            seq_lens=self.seq_lens[reqs_start:],
            context_lens=self.seq_lens[reqs_start:],
            input_positions=prefill_input_positions,
            block_table=self.block_table[reqs_start:, ...],
            slot_mapping=prefill_slot_mapping,
            max_query_len=max_query_len,
            max_seq_lens=max_seq_lens,
            query_start_loc=prefill_query_start_loc,
            sin=sin,
            cos=cos,
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            start_pos=self.start_pos_prefill[:num_prefill],
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            cu_c4_cmp_seqlen_list=cu_c4_cmp_seqlen_list,
            cu_c128_cmp_seqlen_list=cu_c128_cmp_seqlen_list,
            topm_idxs=topm_idxs_prefill,
            topm_num_blocks=num_topm_blocks,
            topm_chunk_start_logical=chunk_start_logical,
            act_qlen=act_qlen,
        )

    #SECTION - build_decode_metadata
    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        num_reqs_actual: int | None,
    ) -> AscendDSADecodeMetadata:
        assert self.decode_ratio_to_sas_metadata is not None
        if self.decode_ratio_to_sas_metadata.get("query_start_loc", None) is None:
            query_start_loc = common_attn_metadata.query_start_loc[: self.num_decodes + 1]
            self.decode_ratio_to_sas_metadata["query_start_loc"] = query_start_loc
            input_positions = common_attn_metadata.positions[: self.num_decode_tokens].long()
            self.decode_ratio_to_sas_metadata["input_positions"] = input_positions
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True)
            self.decode_ratio_to_sas_metadata["cos"] = cos
            self.decode_ratio_to_sas_metadata["sin"] = sin

            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: self.num_decodes + 1]
            input_positions_cpu = common_attn_metadata.positions_cpu[: self.num_decode_tokens].long()

            # Prefer _seq_lens_cpu (always available, updated during draft
            # iterations) over seq_lens_cpu (None in async spec decode mode).
            if common_attn_metadata._seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            elif common_attn_metadata.seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            else:
                _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
            max_seq_lens = _seq_lens_cpu[: self.num_decodes].max().item()
            decode_input_positions = input_positions_cpu
            seq_lens_list = _seq_lens_cpu[: self.num_decodes].tolist()
            self.decode_ratio_to_sas_metadata["query_start_loc_cpu"] = query_start_loc_cpu
            self.decode_ratio_to_sas_metadata["decode_input_positions"] = decode_input_positions
            self.decode_ratio_to_sas_metadata["max_seq_lens"] = max_seq_lens
            self.decode_ratio_to_sas_metadata["seq_lens_list"] = seq_lens_list

            max_seqlen_kv = torch.max(_seq_lens_cpu[: self.num_decodes]).item()
            max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
            self.decode_ratio_to_sas_metadata["max_seqlen_kv"] = max_seqlen_kv
            self.decode_ratio_to_sas_metadata["max_seqlen_q"] = max_seqlen_q

            seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
            start_pos_decode = self.seq_lens[: self.num_decodes] - seq_lens_q
            self.decode_ratio_to_sas_metadata["start_pos_decode"] = start_pos_decode
        else:
            query_start_loc = self.decode_ratio_to_sas_metadata["query_start_loc"]
            input_positions = self.decode_ratio_to_sas_metadata["input_positions"]
            cos = self.decode_ratio_to_sas_metadata["cos"]
            sin = self.decode_ratio_to_sas_metadata["sin"]
            query_start_loc_cpu = self.decode_ratio_to_sas_metadata["query_start_loc_cpu"]
            decode_input_positions = self.decode_ratio_to_sas_metadata["decode_input_positions"]
            max_seq_lens = self.decode_ratio_to_sas_metadata["max_seq_lens"]
            seq_lens_list = self.decode_ratio_to_sas_metadata["seq_lens_list"]
            max_seqlen_kv = self.decode_ratio_to_sas_metadata["max_seqlen_kv"]
            max_seqlen_q = self.decode_ratio_to_sas_metadata["max_seqlen_q"]
            start_pos_decode = self.decode_ratio_to_sas_metadata["start_pos_decode"]

        block_table_size = self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_DECODE)

        cp_seq_len, batch_seq_mask = None, None

        def _get_padded_compressed_position(decode_input_positions, compress_ratio, device):
            if compress_ratio <= 1:
                return decode_input_positions
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            input_positions = decode_input_positions[mask]
            input_positions = (input_positions + 1) - compress_ratio
            target_shape = (min(self.num_decode_tokens, self.num_decode_tokens // compress_ratio + self.num_decodes),)
            pad_right = target_shape[0] - input_positions.shape[0]
            pad_positions = F.pad(input_positions, (0, pad_right), value=0.0)
            gpu_pad_positions = pad_positions.pin_memory().to(device, non_blocking=True)
            return gpu_pad_positions

        layer_name = f"c{self.compressor_ratio}"
        if self.decode_ratio_to_sas_metadata.get(layer_name + "_cos", None) is None:
            compress_cos, compress_sin = get_cos_and_sin_dsa(
                {
                    layer_name: _get_padded_compressed_position(
                        decode_input_positions, self.compressor_ratio, input_positions.device
                    )
                },
                use_cache=True,
            )
            self.decode_ratio_to_sas_metadata[layer_name + "_cos"] = compress_cos
            self.decode_ratio_to_sas_metadata[layer_name + "_sin"] = compress_sin
        else:
            compress_cos = self.decode_ratio_to_sas_metadata[layer_name + "_cos"]
            compress_sin = self.decode_ratio_to_sas_metadata[layer_name + "_sin"]

        #NOTE - compressed_decode_num
        # 实际是计算本轮decode中，合法的到边界的compressed_token，看返回值compressed_decode_num
        # 这个值主要用于compressed_tokens_start，用于下方找slot_mapping的切片
        # 由于无效压缩token不占slot_mapping，有效压缩token被compact到buffer连续前缀，所以compressed_decode_num可以直接用于切片，得到合法slot_mapping
        #LINK - /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1074
        # 例子：
        # decode_input_positions=[9, 11], compressor_ratio=4
        # mask=[False, True], compressed_decode_num=1
        # slot_mapping[:1]
        def _get_compressed_decode_token_start(decode_input_positions, compress_ratio):
            # Note(qcs): some models use compress_ratio=0 as non-compression tag.
            if compress_ratio == 0:
                compress_ratio = 1
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            compressed_decode_num = mask.sum().item()
            return compressed_decode_num

        if self.decode_ratio_to_sas_metadata.get("compressed_tokens_start_" + str(self.compressor_ratio), None) is None:
            compressed_tokens_start = _get_compressed_decode_token_start(decode_input_positions, self.compressor_ratio)
            self.decode_ratio_to_sas_metadata["compressed_tokens_start_" + str(self.compressor_ratio)] = (
                compressed_tokens_start
            )
        else:
            compressed_tokens_start = self.decode_ratio_to_sas_metadata[
                "compressed_tokens_start_" + str(self.compressor_ratio)
            ]

        slot_mapping = DeviceOperator.pad_dsa_decode_slot_mapping(
            self.slot_mapping[:compressed_tokens_start], self.num_decode_tokens, self.compressor_ratio, self.num_decodes
        )

        assert self.start_pos_decode is not None
        self.start_pos_decode.fill_(0)
        self.start_pos_decode[: self.num_decodes] = start_pos_decode

        if num_reqs_actual is not None and num_reqs_actual < self.num_decodes:
            self.start_pos_decode[num_reqs_actual:].fill_(0)
            self.block_table[num_reqs_actual : self.num_decodes, ...].fill_(0)

        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = self.model_config.hf_config.index_topk

        assert self.decode_sas_metadata is not None

        cu_seqlens_ori_kv = DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
            self.decode_ratio_to_sas_metadata,
            "cu_seqlens_ori_kv",
            self.seq_lens,
            self.num_decodes,
            self._zero_i32,
            self.cu_seqlens_ori_kv,
        )
        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        cu_seqlens_cmp_kv = DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
        #NOTE - sas_metadata
        # 每轮decode都会进入if分支重建self.decode_ratio_to_sas_metadata
        if self.compressor_ratio <= 1:
            if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
                self.decode_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=query_start_loc,  # cached
                    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[: self.num_decodes],  # cached
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=len(self.seq_lens[: self.num_decodes]),  # cached
                    cmp_ratio=1,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=False,
                )
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
        elif self.compressor_ratio == 4:
            if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
                self.decode_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=query_start_loc,  # cached
                    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[: self.num_decodes],  # cached
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=len(self.seq_lens[: self.num_decodes]),  # cached
                    cmp_topk=index_topk,
                    # topk=index_topk,
                    cmp_ratio=4,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                )
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
        else:
            if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
                self.decode_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=query_start_loc,
                    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[: self.num_decodes],
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=len(self.seq_lens[: self.num_decodes]),
                    cmp_ratio=128,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                )
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
        assert self.decode_qli_metadata is not None
        if self.decode_ratio_to_sas_metadata.get("qli") is None:
            #LINK - /vllm-workspace/vllm-ascend/csrc/attention/quant_lightning_indexer_metadata/README.md
            # 每轮decode都会进入if分支重建self.decode_ratio_to_sas_metadata["qli"]，同一轮内会层间复用
            # npu_quant_lightning_indexer_metadata构造和返回一个形状为[1024]的tensor，然后下方再写入qli_metadata中
            # qli_metadata构造函数中的变量已经在函数开头处理好了，都是decode requests的相关参数，基本都是长度、数量等
            # TND、BSND是指tensor物理排布：batch_size (B) / seq_len (S) / total_tokens (T) / head (N) / head_dim (D)。TND即vllm拼接的一维flat batch布局，BSND即 Transformer通用基于batch的输入布局。PA代表paged attention，按block页表存储kv cache，PA_BSND逻辑上BSND操作但物理上按block来寻址。
            self.decode_ratio_to_sas_metadata["qli"] = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=query_start_loc[1:].clone(),
                actual_seq_lengths_key=self.seq_lens[: self.num_decodes].clone(),
                num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,  # 128
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=len(self.seq_lens[: self.num_decodes]),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=index_topk,
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                device=str(self.seqused_q.device),
            )
        self.decode_qli_metadata[:1024] = self.decode_ratio_to_sas_metadata.get("qli")
        decode_metadata = AscendDSADecodeMetadata(
            input_positions=input_positions,
            block_table=self.block_table[:block_table_size, ...],
            slot_mapping=slot_mapping,
            seq_lens=self.seq_lens[: self.num_decodes],  # cached
            seq_lens_list=seq_lens_list,
            max_seq_lens=max_seq_lens,
            max_seqlen_kv=max_seqlen_kv,
            max_seqlen_q=max_seqlen_q,
            attn_mask=None,
            query_start_loc=query_start_loc,  # cached
            query_start_loc_cpu=query_start_loc_cpu,
            sin=sin[: self.num_decode_tokens, ...],
            cos=cos[: self.num_decode_tokens, ...],
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            cp_seq_len=cp_seq_len,
            batch_seq_mask=batch_seq_mask,
            start_pos=self.start_pos_decode[: self.num_decodes],  # cached
            sas_metadata=self.decode_sas_metadata,
            qli_metadata=self.decode_qli_metadata,
        )
        return decode_metadata
    #!SECTION

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSADecodeMetadata:
        assert self.compressor_ratio <= 1, "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )
        num_input_tokens = common_attn_metadata.num_input_tokens
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        if num_prefills:
            cos, sin = get_cos_and_sin_dsa(input_positions)
        else:
            # disable use_cache, otherwise, draft_index>0 will override draft_index=0
            # take care of this, if full graph is needed then rope cache is inevitable
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=False)

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        self.spec_slot_mapping[draft_index - 1][:num_input_tokens] = DeviceOperator.format_dsa_slot_mapping(  # type: ignore[index]
            slot_mapping, self.block_size
        )

        prefill_metadata = None
        if num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata_for_drafting(
                draft_index=draft_index,
                common_attn_metadata=common_attn_metadata,
                reqs_start=num_decodes,
                tokens_start=num_decode_tokens,
                num_prefill_tokens=num_prefill_tokens,
            )

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = self.build_decode_metadata_for_drafting(
                draft_index=draft_index,
                common_attn_metadata=common_attn_metadata,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
            )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_lens=None,
            slot_mapping=None,
            head_dim=self.model_config.get_head_size(),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            attn_mask=None,
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=None,
            block_tables=None,
            seq_lens=None,
            cos=cos,
            sin=sin,
            hadamard=None,
        )

    def build_prefill_metadata_for_drafting(
        self,
        draft_index: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendDSAPrefillMetadata:
        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size

        reqs_start = kwargs.get("reqs_start")
        tokens_start = kwargs.get("tokens_start")
        num_prefill_tokens = kwargs.get("num_prefill_tokens")
        query_start_loc = common_attn_metadata.query_start_loc
        prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]
        seq_lens_q = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
        seq_lens = common_attn_metadata.seq_lens[reqs_start:]

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        input_positions = common_attn_metadata.positions[:num_actual_tokens].long()
        prefill_input_positions = input_positions[tokens_start:]
        cos, sin = get_cos_and_sin_dsa(prefill_input_positions)

        prefill_slot_mapping = self.spec_slot_mapping[draft_index - 1][tokens_start:num_prefill_tokens]  # type: ignore[index]
        block_table = common_attn_metadata.block_table_tensor[: common_attn_metadata.num_reqs]

        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        sas_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=prefill_query_start_loc,
            cu_seqlens_ori_kv=prefill_query_start_loc,
            cu_seqlens_cmp_kv=None,
            seqused_q=self.seqused_q,
            seqused_kv=seq_lens,
            max_seqlen_q=seq_lens_q.max(),
            max_seqlen_kv=seq_lens.max(),
            batch_size=len(seq_lens),
            cmp_ratio=1,
            ori_mask_mode=4,
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
        )

        return AscendDSAPrefillMetadata(
            attn_mask=None,
            query_lens=None,
            seq_lens=seq_lens,
            context_lens=None,
            input_positions=None,  # type: ignore[arg-type]
            block_table=block_table[reqs_start:, ...],
            slot_mapping=prefill_slot_mapping,
            max_query_len=None,  # type: ignore[arg-type]
            max_seq_lens=None,  # type: ignore[arg-type]
            query_start_loc=prefill_query_start_loc,
            sin=sin,
            cos=cos,
            compress_sin=None,
            compress_cos=None,
            start_pos=None,
            sas_metadata=sas_metadata,
            qli_metadata=None,
            cu_c4_cmp_seqlen_list=None,
            cu_c128_cmp_seqlen_list=None,
        )

    def build_decode_metadata_for_drafting(
        self,
        draft_index: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendDSADecodeMetadata:
        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size

        num_decodes = kwargs.get("num_decodes")
        num_decode_tokens = kwargs.get("num_decode_tokens")
        num_decodes_typed = num_decodes or 0
        num_decode_tokens_typed = num_decode_tokens or 0
        query_start_loc = common_attn_metadata.query_start_loc[: num_decodes_typed + 1]
        seq_lens = common_attn_metadata.seq_lens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_decodes_typed + 1]
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()

        if common_attn_metadata._seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        elif common_attn_metadata.seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        max_seqlen_kv = torch.max(_seq_lens_cpu[:num_decodes]).item()

        input_positions = common_attn_metadata.positions[:num_decode_tokens_typed].long()
        # disable use_cache, otherwise, draft_index>0 will override draft_index=0
        # take care of this, if full graph is needed then rope cache is inevitable
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=False)

        slot_mapping = self.spec_slot_mapping[draft_index - 1][:num_decode_tokens_typed]  # type: ignore[index]
        block_table = common_attn_metadata.block_table_tensor

        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)

        decode_sas_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,
            cu_seqlens_ori_kv=self.cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=self.cu_seqlens_cmp_kv,
            seqused_q=self.seqused_q,
            seqused_kv=seq_lens[:num_decodes],
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=len(seq_lens[:num_decodes]),
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
        )

        decode_metadata = AscendDSADecodeMetadata(
            input_positions=None,
            block_table=block_table[:num_decodes, ...],
            slot_mapping=slot_mapping,
            seq_lens=seq_lens[:num_decodes],
            seq_lens_list=None,  # type: ignore[arg-type]
            max_seq_lens=None,  # type: ignore[arg-type]
            max_seqlen_kv=None,  # type: ignore[arg-type]
            max_seqlen_q=None,  # type: ignore[arg-type]
            attn_mask=None,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=None,
            sin=sin[:num_decode_tokens, ...],
            cos=cos[:num_decode_tokens, ...],
            compress_sin=None,
            compress_cos=None,
            cp_seq_len=None,
            batch_seq_mask=None,
            start_pos=None,
            sas_metadata=decode_sas_metadata,
            qli_metadata=None,
        )
        return decode_metadata

    def get_block_table_size(self, common_attn_metadata: AscendCommonAttentionMetadata, build_metadata_step: int):
        if build_metadata_step == BUILD_METADATA_STEP_PREFILL:
            # If graph_pad_size > -1, mean is running in fullgraph mode.
            # Maybe this block_table change can be removed when graph_pad_size > 1.
            # if self.graph_pad_size > common_attn_metadata.num_reqs and \
            #         self.speculative_config.disable_padded_drafter_batch:
            #     return self.graph_pad_size
            return common_attn_metadata.num_reqs
        return self.num_decodes

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        **kwargs,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state"
            )

        assert attn_metadata is not None
        attn_metadata.attn_state = attn_state
        return attn_metadata

    def _build_topm_subgroups_prefill(self, input_batch, layer_name, seq_lens_np,
                                       kvlens, actual_qlens, block_table,
                                       index_topm, index_topk, micro_step_num):
        B = len(kvlens)
        device = kvlens.device
        req_ids = input_batch.req_ids[:B]
        block_size = 32

        topm_idxs = torch.zeros(B, 1, index_topm, dtype=torch.int32, device=device)
        actual_seq_lengths_key = kvlens.clone()
        max_seqlen_k = int(actual_seq_lengths_key.max().item())
        has_cached = False

        num_topm_blocks = 0
        chunk_start_logical = 0
        act_qlen_return = 0

        for i, rid in enumerate(req_ids):
            if rid is None:
                continue
            state = input_batch.topm_state.get(rid, {}).get(layer_name)
            kvlen = int(seq_lens_np[i])
            act_qlen = int(actual_qlens[i])
            short = kvlen < 4 * index_topm

            if state is None:
                if short:
                    continue
                state = input_batch.topm_state.setdefault(rid, {}).setdefault(
                    layer_name, TopMReqState())

            if not state.start_cache and not short:
                state.start_cache = True
                state.ustep = 0

            if state.topm_idxs is not None:
                has_cached = True
                topm_idxs[i] = state.topm_idxs

                topm_logical = state.topm_idxs // block_size
                topm_unique_blocks = torch.unique(topm_logical)
                num_topm_blocks = topm_unique_blocks.numel()
                num_prev_blocks = cdiv(kvlen, block_size)
                chunk_start_logical = num_prev_blocks - cdiv(act_qlen, block_size)
                act_qlen_return = act_qlen

                new_kvlen = num_topm_blocks * block_size + act_qlen
                actual_seq_lengths_key[i] = new_kvlen

            state.ustep += 1
            max_seqlen_k = max(max_seqlen_k, int(actual_seq_lengths_key[i].item()))

        return topm_idxs, actual_seq_lengths_key, max_seqlen_k, has_cached, num_topm_blocks, chunk_start_logical, act_qlen_return
#!SECTION

#SECTION - AscendDSAImpl
class AscendDSAImpl(DSAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        **kwargs,
    ):
        self.num_heads = n_heads
        self.n_local_heads = n_local_heads
        self.scale = scale
        self.o_lora_rank = o_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.head_dim = head_dim
        self.n_group = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5

        # MLA Args
        self.wq_a = kwargs["wq_a"]
        self.wq_b = kwargs["wq_b"]
        self.wkv = kwargs["wkv"]
        self.q_norm = kwargs["q_norm"]
        self.q_norm_without_weight = kwargs["q_norm_without_weight"]
        self.kv_norm = kwargs["kv_norm"]

        # CV wrapper: split wq_a/wkv/wq_b into quantize(Vector) + matmul(Cube)
        self.cv_wq_a = CVLinearWrapper(self.wq_a)
        self.cv_wkv = CVLinearWrapper(self.wkv)
        self.cv_wq_b = CVLinearWrapper(self.wq_b)

        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")

        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]

        self.eps = kwargs["eps"]

        self.attn_sink = kwargs["attn_sink"]

        ascend_config = get_ascend_config()
        self.multistream_dsv4_dsa_overlap = ascend_config.multistream_dsv4_dsa_overlap
        self.vllm_config = get_current_vllm_config()

        self.enable_local_k_cache = ascend_config.enable_local_k_cache

        # indexer param
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads
            self.inderxer_dim: int = self.indexer.head_dim
            self.inderxer_wq_b = self.indexer.wq_b
            self.cv_inderxer_wq_b = CVLinearWrapper(self.inderxer_wq_b)
            self.weights_proj = self.indexer.weights_proj
            self.indexer_softmax_scale = self.inderxer_dim**-0.5

            self.indexer_compress = self.indexer.compressor

            # indexer_compressor
            self.indexcom_ape = self.indexer.compressor.ape
            self.indexcom_wkv = self.indexer.compressor.wkv
            self.indexcom_wgate = self.indexer.compressor.wgate
            self.indexcom_norm = self.indexer.compressor.norm

            self.indexcom_head_dim = self.indexer.compressor.head_dim
            self.indexcom_rotate = self.indexer.compressor.rotate
            self.index_topk = self.indexer.index_topk
            self.index_topm = kwargs.get("index_topm", None)
            self.micro_step_num = kwargs.get("micro_step_num", None)
            self.indexer_dim: int = self.indexer.head_dim
            #ANCHOR - DSAImpl定义indexcache相关字段
            # layer_name -> req_id -> full topm idxs
            # anchor轮保存原始topm idx，reuse轮用它把local topk映射回全局compressed idx
            self.indexcache_topm_idx_cache: dict[str, dict[str, torch.Tensor]] = {}

        # compress param
        if self.compressor is not None:
            self.compressor_head_dim = self.compressor.head_dim
            self.compressor_overlap = self.compressor.overlap
            self.compressor_rotate = self.compressor.rotate

            self.compressor_ape = self.compressor.ape
            self.compressor_wkv = self.compressor.wkv
            self.compressor_wgate = self.compressor.wgate
            self.compressor_norm = self.compressor.norm
            self.compressor_norm_eps = self.compressor.norm_eps

        # IndexCache: skip_topk indicates this layer reuses topk from a previous
        # indexer-bearing layer; use_index_cache marks whether the buffer must
        # be kept fresh on non-skip layers so downstream skip layers can read.
        self.skip_topk = kwargs.get("skip_topk", False)
        self.topk_indices_buffer = kwargs.get("topk_indices_buffer")
        self.use_index_cache = self.skip_topk or getattr(
            self.vllm_config.model_config.hf_config,
            "use_index_cache",
            False,
        )

    def _get_indexcache_topk_indices(self, num_tokens: int, offset: int = 0) -> torch.Tensor:
        if self.topk_indices_buffer is None:
            raise RuntimeError("IndexCache requires topk_indices_buffer when skip_topk is enabled.")
        topk_indices = self.topk_indices_buffer[offset : offset + num_tokens]
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(1)
        return topk_indices

    def _update_indexcache_topk_indices(self, topk_indices: torch.Tensor, offset: int = 0) -> None:
        if self.topk_indices_buffer is None:
            return
        num_tokens = topk_indices.shape[0]
        topk_tokens = topk_indices.shape[-1]
        topk_indices_to_cache = topk_indices
        topk_indices_buffer = self.topk_indices_buffer[offset : offset + num_tokens, :topk_tokens]
        if topk_indices_to_cache.dim() == 3 and topk_indices_buffer.dim() == 2:
            assert topk_indices_to_cache.shape[1] == 1
            topk_indices_to_cache = topk_indices_to_cache.squeeze(1)
        topk_indices_buffer.copy_(topk_indices_to_cache)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass

    # cast to bfloat16 to speed up
    def rope_single(self, x, cos, sin, inverse=False):
        if inverse:
            sin = -sin
        tnd_layout = 1
        if len(x.shape) == 3:
            num_tokens, num_heads, rotary_dim = x.shape
        else:
            tnd_layout = 0
            _, num_tokens, num_heads, rotary_dim = x.shape
        x_rot = torch_npu.npu_rotary_mul(
            x.reshape(num_tokens, num_heads, 1, rotary_dim), cos, sin, rotary_mode="interleave"
        )
        if tnd_layout:
            x = x_rot.reshape(num_tokens, -1, rotary_dim)
        else:
            x = x_rot.reshape(1, num_tokens, -1, rotary_dim)
        return x

    def forward(  # type: ignore[override]
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: DSAMetadataList,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            return output.fill_(0)
        if not isinstance(attn_metadata, list):
            attn_metadata = [attn_metadata]
        output_padded = output
        # Process for Flash Comm V1
        has_prefill = attn_metadata[0].num_prefills > 0
        has_decode = attn_metadata[0].num_decodes > 0
        decode_tokens = attn_metadata[0].num_decode_tokens
        actual_tokens = attn_metadata[0].num_actual_tokens

        # Process for Flash Comm V1
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, need_gather_q_kv)
        prefill_hidden_states = hidden_states[decode_tokens:actual_tokens]
        decode_hidden_states = hidden_states[:decode_tokens]

        forward_context = get_forward_context()
        o_proj_input_shape = (forward_context.num_tokens, self.n_local_heads, self.head_dim)
        o_proj_input = torch.empty(o_proj_input_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        assert kv_cache is not None, "kv_cache tensor tuple must be provided."
        if has_prefill:
            assert attn_metadata[0].prefill is not None
            output_prefill = self._forward_prefill(
                layer_name,
                prefill_hidden_states,
                kv_cache,
                attn_metadata,
            )  # type: ignore[arg-type]
            o_proj_input[decode_tokens:actual_tokens] = output_prefill
            cos = attn_metadata[0].prefill.cos[layer_name]
            sin = attn_metadata[0].prefill.sin[layer_name]

        if has_decode:
            assert attn_metadata[0].decode is not None
            output_decode = self._forward_decode(layer_name, decode_hidden_states, kv_cache, attn_metadata)
            o_proj_input[:decode_tokens] = output_decode
            cos = attn_metadata[0].decode.cos[layer_name]
            sin = attn_metadata[0].decode.sin[layer_name]

        cos = attn_metadata[0].cos[layer_name]
        sin = attn_metadata[0].sin[layer_name]
        num_tokens = o_proj_input.shape[0]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            o_proj_input.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # o
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            o = o_proj_input.view(num_tokens, self.n_local_groups, -1)
            o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
            o = torch_npu.npu_transpose_quant_batchmatmul(
                o,
                self.wo_a.weight,
                dtype=torch.bfloat16,
                bias=None,
                group_sizes=(0, 0, 32),
                x1_scale=swiglu_out_scale.view(torch.float8_e8m0fnu),
                x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )
            o = o.reshape(num_tokens, -1)
            output[...] = self.wo_b(o)
        else:
            o_proj_input = o_proj_input.view(num_tokens, self.n_local_groups, -1)
            if olora_tp_enable():
                o_proj_input = self.wo_a(o_proj_input)
            else:
                # wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
                # o = torch.einsum("tgd,grd->tgr", o, wo_a)
                o_proj_input = torch_npu.npu_transpose_batchmatmul(
                    o_proj_input,
                    self.wo_a.weight,
                    bias=None,
                    scale=None,
                    perm_x1=(1, 0, 2),
                    perm_x2=(0, 1, 2),
                    perm_y=(1, 0, 2),
                    batch_split_factor=1,
                )
                o_proj_input = o_proj_input.reshape(num_tokens, -1)
            output[...] = self.wo_b(o_proj_input)

        return output_padded

    def _mla_prolog_multistream(self, hidden_states, cos, sin, swa_kv_cache, slot_mapping, is_prefill=False):
        """3-block multi-stream: 3-stage CV parallel + serial tail

        Block partition (V: Vector, C: Cube, AIV: AI Vector):
          Part1: q_quant[V] -> q_a_down[C]  ||  kv_quant[V]
          Part2: q_norm[V] + q_b_quant[V]  ||  kv_matmul[C]
          Part3: q_b_matmul[C]             ||  kv_norm[V] + rope[V] + scatter[AIV]
          Tail:  q_rms[V] + rope[V] (wait for auxiliary stream to complete)

        Each stream's data is self-contained; no cross-stream sync is needed between blocks.
        Only the tail wait_stream ensures scatter is complete.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()

        is_w8a8 = _is_w8a8_dynamic(self.wq_b)

        # Part1: q_quant[V] -> q_a_down[C]  ||  kv_quant[V]
        q_quant, q_pertoken_scale = self.cv_wq_a.quantize(hidden_states)

        e_q_quant_done = main_stream.record_event()

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_q_quant_done)
            kv_quant, kv_pertoken_scale = self.cv_wkv.quantize(hidden_states)

        wq_a_result = self.cv_wq_a.matmul(q_quant, q_pertoken_scale)
        main_stream.wait_stream(aux_stream)

        # Part2: q_norm[V] + q_b_quant[V]  ||  kv_matmul[C]
        e_part2_start = main_stream.record_event()

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_part2_start)
            kv = self.cv_wkv.matmul(kv_quant, kv_pertoken_scale)

        if is_prefill:
            qr = self.q_norm(wq_a_result)
            q_b_quant, q_b_scale = self.cv_wq_b.quantize(qr)
            qr_pertoken_scale = None
        elif is_w8a8:
            qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                wq_a_result, self.q_norm.weight, epsilon=self.eps
            )
            q_b_quant, q_b_scale = qr, qr_pertoken_scale
        else:
            qr = self.q_norm(wq_a_result)
            q_b_quant, q_b_scale = qr, None
            qr_pertoken_scale = None

        main_stream.wait_stream(aux_stream)

        # Part3: q_b_matmul[C]  ||  kv_norm[V] + rope[V] + scatter[AIV]
        e_part3_start = main_stream.record_event()

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_part3_start)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)

        if is_prefill:
            q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))
        elif is_w8a8:
            q = torch_npu.npu_quant_matmul(
                q_b_quant,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=q_b_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (self.n_local_heads, self.head_dim))
        else:
            q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))

        # Serial tail: wait for auxiliary stream then execute q_rms[V] + rope[V]
        main_stream.wait_stream(aux_stream)

        q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        return q, qr, qr_pertoken_scale

    def _forward_prefill(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
    ):
        compress_common_attn_metadata = None
        (compress_kv_cache,
         swa_kv_cache,
         state_cache,
         indexer_k_cache,
         indexer_scale_cache,
         indexer_full_cache,
         indexer_local_k_cache,
         indexer_local_scale_cache) = (
            DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)
        )
        if not self.enable_local_k_cache:
            if self.compress_ratio == 4:
                (compress_kv_cache, swa_kv_cache, state_cache, _, indexer_k_cache, indexer_scale_cache) = kv_cache
                # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
                (compressor_attn_metadata, compressor_kv_state_metadata, _, indexer_kv_scale_metadata, swa_metadata) = (
                    attn_metadata
                )
                compress_common_attn_metadata = compressor_attn_metadata
            elif self.compress_ratio == 128:
                (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = kv_cache
                # sorted keys: [attn, compressor.state_cache, swa_cache]
                (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
                compress_common_attn_metadata = compressor_attn_metadata
            else:
                (
                    _,
                    swa_kv_cache,
                    _,
                    _,
                    _,
                    _,
                ) = kv_cache
                # sorted keys: [swa_cache]
                (swa_metadata,) = attn_metadata
                compress_common_attn_metadata = swa_metadata
        else:
            if self.compress_ratio == 4:
                (compressor_attn_metadata,
                compressor_kv_state_metadata,
                _,
                indexer_kv_scale_metadata,
                indexer_local_kv_scale_metadata,
                swa_metadata) = (
                    attn_metadata
                )                
                # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
                compress_common_attn_metadata = compressor_attn_metadata
            elif self.compress_ratio == 128:
                # sorted keys: [attn, compressor.state_cache, swa_cache]
                (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
                compress_common_attn_metadata = compressor_attn_metadata
            else:
                # sorted keys: [swa_cache]
                (swa_metadata,) = attn_metadata
                compress_common_attn_metadata = swa_metadata

        common_prefill_metadata = _require_prefill_metadata(compress_common_attn_metadata)
        swa_prefill_metadata = _require_prefill_metadata(swa_metadata)
        cos = common_prefill_metadata.cos[layer_name]
        sin = common_prefill_metadata.sin[layer_name]
        actual_seq_lengths_query = common_prefill_metadata.query_start_loc
        actual_seq_lengths_key = common_prefill_metadata.seq_lens

        if self.multistream_dsv4_dsa_overlap:
            # mla prolog: q + kv dual-stream parallel
            q, qr, _ = self._mla_prolog_multistream(
                hidden_states, cos, sin, swa_kv_cache, swa_prefill_metadata.slot_mapping, is_prefill=True
            )
        else:
            # mlaprolog
            share_hs_quant = _is_w8a8_dynamic(self.wq_a) and _is_w8a8_dynamic(self.wkv)
            if share_hs_quant:
                hs_int8, hs_pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                q_a = torch_npu.npu_quant_matmul(
                    hs_int8,
                    self.wq_a.weight,
                    self.wq_a.weight_scale,
                    pertoken_scale=hs_pertoken_scale,
                    bias=self.wq_a.bias,
                    output_dtype=hidden_states.dtype,
                )
            else:
                q_a = self.wq_a(hidden_states)

            # q
            if _is_w8a8_dynamic(self.wq_b):
                qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                    q_a, self.q_norm.weight, epsilon=self.eps
                )
                q = torch_npu.npu_quant_matmul(
                    qr,
                    self.wq_b.weight,
                    self.wq_b.weight_scale,
                    pertoken_scale=qr_pertoken_scale,
                    bias=self.wq_b.bias,
                    output_dtype=hidden_states.dtype,
                ).unflatten(-1, (self.n_local_heads, self.head_dim))
            else:
                qr = self.q_norm(q_a)
                q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
                qr_pertoken_scale = None
            q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                q.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            # win kv & tok_dis
            if share_hs_quant:
                kv = torch_npu.npu_quant_matmul(
                    hs_int8,
                    self.wkv.weight,
                    self.wkv.weight_scale,
                    pertoken_scale=hs_pertoken_scale,
                    bias=self.wkv.bias,
                    output_dtype=hidden_states.dtype,
                )
            else:
                kv = self.wkv(hidden_states)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )

            # swa exec kv
            DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_prefill_metadata.slot_mapping)

        compress_cos = common_prefill_metadata.compress_cos[layer_name]
        compress_sin = common_prefill_metadata.compress_sin[layer_name]

        attn_op = DeviceOperator.get_dsa_sparse_attn_op()
        extra_attn_kwargs: dict = DeviceOperator.get_dsa_sparse_attn_base_kwargs()
        DeviceOperator.add_dsa_sparse_attn_extra_kwargs(extra_attn_kwargs, cu_seqlens_ori_kv=actual_seq_lengths_query)

        if self.compress_ratio <= 1:
            return attn_op(
                q,
                ori_kv=swa_kv_cache,
                ori_block_table=swa_prefill_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=common_prefill_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=max(self.compress_ratio, 1),
                ori_mask_mode=4,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                **extra_attn_kwargs,
            )[0]

        if self.compress_ratio > 1:
            compressor_prefill_metadata = _require_prefill_metadata(compressor_attn_metadata)
            compressor_state_prefill_metadata = _require_prefill_metadata(compressor_kv_state_metadata)
            compress_topk_idxs = None
            # Only call indexer_select_qli when compress_ratio == 4 (requires 5 elements in attn_metadata)
            if self.compress_ratio == 4:
                # IndexCache: prefill segment lives at buffer[num_decode_tokens:]
                # because dsa_v1 forward splits hidden_states as
                # [decode | prefill]. See AscendDSAImpl.forward.
                prefill_offset = attn_metadata[0].num_decode_tokens
                prefill_num_tokens = hidden_states.shape[0]
                if self.skip_topk:
                    compress_topk_idxs = self._get_indexcache_topk_indices(prefill_num_tokens, offset=prefill_offset)
                else:
                    if self.multistream_dsv4_dsa_overlap:
                        indexer_q = self.cv_indexer_select_qli(  # multistream version
                            x=hidden_states,
                            qr=qr,
                            kv_cache=kv_cache,
                            attn_metadata=attn_metadata,
                            cos=cos,
                            sin=sin,
                            compressed_cos=compress_cos,
                            compressed_sin=compress_sin,
                            actual_seq_lengths_query=actual_seq_lengths_query,
                            with_prefill=True,
                        )
                    else:
                        compress_topk_idxs = self.indexer_select_qli(  # original version
                            x=hidden_states,
                            qr=qr,
                            kv_cache=kv_cache,
                            attn_metadata=attn_metadata,
                            cos=cos,
                            sin=sin,
                            compressed_cos=compress_cos,
                            compressed_sin=compress_sin,
                            actual_seq_lengths_query=actual_seq_lengths_query,
                            actual_seq_lengths_key=actual_seq_lengths_key,
                            with_prefill=True,
                            qr_pertoken_scale=qr_pertoken_scale,
                        )

            coff = 2 if self.compressor_overlap else 1

            # Inline compressor + scatter (c128, c4 non-dual)
            compressed_kv = torch.ops._C_ascend.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                state_cache.squeeze(-2),
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                state_block_table=compressor_state_prefill_metadata.block_table,
                cu_seqlens=actual_seq_lengths_query,
                seqused=None,
                start_pos=common_prefill_metadata.start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )

            if compressed_kv.numel() == 0:
                compressed_kv = None

            # For multistream_dsv4_dsa_overlap with compress_ratio=4:
            # aux_stream: indexer_weights_proj (parallel with main q_quant + kv_scatter)
            # main stream: compressed_kv -> q_quant -> kv_scatter -> wait aux_stream -> lightning_indexer
            if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
                main_stream = torch.npu.current_stream()
                aux_stream = dsv4_dsa_overlap_stream()
                e_compressed_kv_done = main_stream.record_event()
                with npu_stream_switch(aux_stream, enabled=True):
                    torch.npu.current_stream().wait_event(e_compressed_kv_done)
                    weights_proj_output = self.weights_proj(hidden_states)
                # Main stream: q_quant (between compressed_kv and kv_scatter)
                q_quant, q_scale = DeviceOperator.indexer_quantize_query(indexer_q)

            DeviceOperator.dsa_kv_compress_scatter(
                compress_kv_cache, compressed_kv, compressor_prefill_metadata.slot_mapping
            )

            if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
                # Wait aux_stream weights_proj done, then compute dot
                main_stream.wait_stream(aux_stream)
                weights = weights_proj_output * (self.indexer_softmax_scale * self.indexer_heads**-0.5)
                # lightning_indexer
                indexer_scale_prefill_metadata = _require_prefill_metadata(indexer_kv_scale_metadata)
                qlens = indexer_scale_prefill_metadata.query_start_loc[1:]
                kvlens = indexer_scale_prefill_metadata.seq_lens
                block_table = indexer_scale_prefill_metadata.block_table
                qli_metadata = indexer_scale_prefill_metadata.qli_metadata
                compress_topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
                    query=q_quant,
                    key=indexer_k_cache,
                    weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
                    query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale),
                    key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(indexer_scale_cache),
                    actual_seq_lengths_query=qlens,
                    actual_seq_lengths_key=kvlens,
                    block_table=block_table,
                    metadata=qli_metadata,
                    query_quant_mode=0,
                    key_quant_mode=0,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topk,
                    sparse_mode=3,
                    pre_tokens=(1 << 63) - 1,
                    next_tokens=(1 << 63) - 1,
                    cmp_ratio=4,
                    return_value=False,
                )

            if self.compress_ratio == 4 and self.use_index_cache:
                self._update_indexcache_topk_indices(compress_topk_idxs, offset=prefill_offset)

            if self.compress_ratio == 4:
                DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                    extra_attn_kwargs, cu_seqlens_cmp_kv=common_prefill_metadata.cu_c4_cmp_seqlen_list
                )
                attn_output = attn_op(
                    q,
                    ori_kv=swa_kv_cache,
                    cmp_kv=compress_kv_cache,
                    cmp_sparse_indices=compress_topk_idxs,
                    ori_block_table=swa_prefill_metadata.block_table,
                    cmp_block_table=compressor_prefill_metadata.block_table,
                    cu_seqlens_q=actual_seq_lengths_query,
                    seqused_kv=actual_seq_lengths_key,
                    sinks=self.attn_sink,
                    metadata=common_prefill_metadata.sas_metadata,
                    softmax_scale=self.softmax_scale,
                    cmp_ratio=self.compress_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    **extra_attn_kwargs,
                )[0]
            else:
                DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                    extra_attn_kwargs, cu_seqlens_cmp_kv=common_prefill_metadata.cu_c128_cmp_seqlen_list
                )
                attn_output = attn_op(
                    q,
                    ori_kv=swa_kv_cache,
                    cmp_kv=compress_kv_cache,
                    ori_block_table=swa_prefill_metadata.block_table,
                    cmp_block_table=compressor_prefill_metadata.block_table,
                    cu_seqlens_q=actual_seq_lengths_query,
                    seqused_kv=actual_seq_lengths_key,
                    sinks=self.attn_sink,
                    metadata=common_prefill_metadata.sas_metadata,
                    softmax_scale=self.softmax_scale,
                    cmp_ratio=self.compress_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    **extra_attn_kwargs,
                )[0]
        return attn_output

    #ANCHOR - save topm_idxs
    def _save_indexcache_anchor_topm_idxs(
        self,
        layer_name: str,
        req_ids: tuple[str, ...] | None,
        topm_idxs: torch.Tensor,
    ) -> None:
        if req_ids is None:
            return

        # QLI output shape [num_decode_tokens, 1, topM]。seq_len < topM时尾部补-1。
        topm_idxs_2d = topm_idxs[:, 0, :]
        num_decode_rows = topm_idxs_2d.shape[0]

        layer_topm_idx_cache = self.indexcache_topm_idx_cache.setdefault(layer_name, {})

        # 当前未开MTP，decode row和req_id一一对应；MTP需要额外的token-row到req映射。
        # req_id才是跨decode轮稳定的请求标识，row_idx只是当前batch内的位置。
        for row_idx, req_id in enumerate(req_ids):
            layer_topm_idx_cache[req_id] = topm_idxs_2d[row_idx].clone()

    #ANCHOR - gather topm kv and save in indexer_local_k_cache
    def _save_indexcache_anchor_topm_local_kv(
        self,
        topm_idxs: torch.Tensor,
        indexer_k_cache: torch.Tensor,
        indexer_scale_cache: torch.Tensor,
        indexer_local_k_cache: torch.Tensor,
        indexer_local_scale_cache: torch.Tensor,
        global_block_table: torch.Tensor,
        local_block_table: torch.Tensor,
    ) -> None:
        # QLI 输出 topm_idxs: [num_decode_rows, 1, topM]，当前 topM 通常为 2048。
        # 去掉中间长度为 1 的维度后，topm_idxs_2d: [num_decode_rows, topM]。
        topm_idxs_2d = topm_idxs[:, 0, :].to(torch.long)
        num_decode_rows, topm_count = topm_idxs_2d.shape
        block_size = indexer_k_cache.shape[1]   # 128
        assert topm_count % block_size == 0

        # Phase 1: 从 global indexer K/scale cache 收集 topM K/scale
        # 短 prompt 选不满 topM 时，topm_idxs 尾部使用 -1 表示 padding。
        # valid_topm_mask: [num_decode_rows, topM]，后面用它将 padding 对应的 K/scale 清零。
        valid_topm_mask = topm_idxs_2d >= 0
        # gather 不接受负索引，因此临时将 -1 替换为 0；真实 index 0 仍由 valid_topm_mask 区分。
        # safe_topm_idxs 形状仍为 [num_decode_rows, topM]。
        safe_topm_idxs = topm_idxs_2d.clamp_min(0)
        # idx // 128 得到 block_table 的逻辑 block id（序号），查询 block_table 后得到全局 cache 中的物理 block id。
        # topm_block_idxs/topm_block_offsets: [num_decode_rows, topM]。
        topm_block_idxs = torch.gather(
            global_block_table.to(torch.long)[:num_decode_rows], 
            1, 
            torch.div(safe_topm_idxs, block_size, rounding_mode="floor")
        )
        topm_block_offsets = safe_topm_idxs % block_size

        # 为了使用 torch_npu.npu_gather_sparse_index 收集 topm_k_cache
        # 需要 flatten 拉平，torch_npu.npu_gather_sparse_index 仅 支持 1D 索引
        # indexer_k_cache (int8) 和 indexer_scale_cache (float16)
        #   shape:   [num_blocks, 128, 1, 128]  [num_blocks, 128, 1, 1]
        #   stride:  [16640, 128, 128, 1]       [8320, 1, 1, 1]
        # 一个 K slot 包含 1 * 128 = 128 个 int8，一个 K block 包含 128 个slot，共 16384 字节。
        # stride(0)=16640 是完整 page 大小，其中多出的 256 字节保存该 block 的 float16 scale，因此不能直接合并前两维。
        # 16640 // 128 = 130 行：前 128 行是 K，最后 2 行是 scale。
        # 内存布局：[K0][Scale0][K1][Scale1]...
        k_slot_width = math.prod(indexer_k_cache.shape[2:])             # 128
        k_rows_per_page = indexer_k_cache.stride(0) // k_slot_width     # 130，每行对应一个 K + Scale Block，取代原来每行一个 K Block 128 元素的语义
        # 将底层 storage 构造成后续仅用于读取的二维 table，torch.as_strided 仅改元数据：
        #   shape  = [num_blocks * 130, 128]
        #   stride = [128, 1]
        flat_indexer_k_cache = torch.as_strided(
            indexer_k_cache,
            size=(indexer_k_cache.shape[0] * k_rows_per_page, k_slot_width),
            stride=(k_slot_width, 1),
            storage_offset=indexer_k_cache.storage_offset(),
        )
        # flat_topm_k_idxs: [num_decode_rows, topM]。
        flat_topm_k_idxs = topm_block_idxs * k_rows_per_page + topm_block_offsets

        # scale cache 的 storage_offset 指向第 0 个 page 末尾的 scale 区域，
        # stride(0) 是相邻 page 的 FP16 元素间距。将这段 storage 构造为
        # 宽度为 1 的二维 table，使每个 scale slot 可以用一个线性索引读取。
        scale_slot_width = math.prod(indexer_scale_cache.shape[2:])          # 1
        scale_rows_per_page = (                                             # 8320
            indexer_scale_cache.stride(0) // scale_slot_width
        )
        # 从第 0 个 page 的 scale 起点延伸到最后一个 scale block 末尾，
        # 避免从非零 storage_offset 创建越过底层 storage 的 view。
        flat_scale_row_count = (
            (indexer_scale_cache.shape[0] - 1) * scale_rows_per_page
            + indexer_scale_cache.shape[1]
        )
        flat_indexer_scale_cache = torch.as_strided(
            indexer_scale_cache,
            size=(flat_scale_row_count, scale_slot_width),
            stride=(scale_slot_width, 1),
            storage_offset=indexer_scale_cache.storage_offset(),
        )
        # flat_topm_scale_idxs: [num_decode_rows, topM]。
        flat_topm_scale_idxs = (
            topm_block_idxs * scale_rows_per_page + topm_block_offsets
        )

        # sparse gather 输出 [num_decode_rows, topM, 128]
        # unsqueeze(-2) 恢复为 [num_decode_rows, topM, 1, 128]
        topm_k_cache = torch_npu.npu_gather_sparse_index(
            flat_indexer_k_cache,
            flat_topm_k_idxs,
        ).unsqueeze(-2)
        # sparse gather 输出 [num_decode_rows, topM, 1]，
        # unsqueeze(-2) 恢复为 [num_decode_rows, topM, 1, 1]。
        topm_scale_cache = torch_npu.npu_gather_sparse_index(
            flat_indexer_scale_cache,
            flat_topm_scale_idxs,
        ).unsqueeze(-2)

        # padding 位置此前安全读取了 token 0，现在统一覆盖为全 0；所有 shape 保持不变。
        topm_k_cache = torch.where(
            valid_topm_mask[..., None, None],
            topm_k_cache,
            torch.zeros_like(topm_k_cache),
        )
        topm_scale_cache = torch.where(
            valid_topm_mask[..., None, None],
            topm_scale_cache,
            torch.zeros_like(topm_scale_cache),
        )

        # Phase 2: 将 topm K/scale 写入 indexer local cache
        # topM=2048、block_size=128 时，每个 request 使用 16 个 local block
        num_topm_blocks = topm_count // block_size

        # 取出 local cache 中保存 topM 的物理 block id
        # local_block_table: [num_decode_rows, max_num_blocks]
        # local_topm_block_mapping: [num_decode_rows * num_topm_blocks, 1]
        local_topm_block_mapping = local_block_table[
            :num_decode_rows, :num_topm_blocks
        ].reshape(-1, 1).to(torch.int32)

        # 每个连续的 block_size 个 topM 项组成一个完整 local block
        # topm_k_blocks: [num_decode_rows * num_topm_blocks, 128, 1, 128]
        # topm_scale_blocks: [num_decode_rows * num_topm_blocks, 128, 1, 1]
        topm_k_blocks = topm_k_cache.reshape(
            num_decode_rows * num_topm_blocks,
            block_size,
            *topm_k_cache.shape[2:],
        )
        topm_scale_blocks = topm_scale_cache.reshape(
            num_decode_rows * num_topm_blocks,
            block_size,
            *topm_scale_cache.shape[2:],
        )

        # mapping 的每一行只包含物理 block id，scatter 将对应的完整 K/scale block
        # 写入 indexer_local_k_cache 和 indexer_local_scale_cache
        torch.ops._C_ascend.npu_scatter_nd_update_v2(
            indexer_local_k_cache, local_topm_block_mapping, topm_k_blocks
        )
        torch.ops._C_ascend.npu_scatter_nd_update_v2(
            indexer_local_scale_cache, local_topm_block_mapping, topm_scale_blocks
        )

    #ANCHOR - local topk_idxs -> global topk_idxs
    # local_topk_idxs的值域是[0, topM)，对应anchor轮保存的topM局部位置。
    def _restore_indexcache_reuse_topk_idxs(
        self,
        layer_name: str,
        req_ids: tuple[str, ...] | None,
        local_topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        assert req_ids is not None

        # QLI output shape [num_decode_tokens, 1, topk].
        local_topk_idxs_2d = local_topk_idxs[:, 0, :].to(torch.long)
        layer_topm_idx_cache = self.indexcache_topm_idx_cache[layer_name]
        # 根据 req_id 对齐 topk 和 topm
        saved_topm_idxs = torch.stack(
            [layer_topm_idx_cache[req_id].to(local_topk_idxs_2d.device) for req_id in req_ids],
            dim=0,
        )
        # QLI可能返回-1表示无效位置，gather不支持负数索引
        valid_topk_mask = local_topk_idxs_2d >= 0
        safe_local_topk_idxs = local_topk_idxs_2d.clamp_min(0)
        # torch.gather 恢复映射
        global_topk_idxs = torch.gather(saved_topm_idxs, 1, safe_local_topk_idxs)
        global_topk_idxs = torch.where(
            valid_topk_mask,
            global_topk_idxs,
            torch.full_like(global_topk_idxs, -1),
        )
        return global_topk_idxs[:, None, :].to(local_topk_idxs.dtype)

    #SECTION - forward_decode
    # CSA-token-indexcache
    # 1. 怎么判断anchor
    # 加个字段req_id，为每个req维护anchor状态表
    # qlens=[1,2,3] anchor_mask=[Y,N,N]

    # 2. anchor和非anchor qli逻辑

    # 2.1 anchor和非anchor用不同indexer_k_cache，分别调用qli，然后拼接
    # 但这样动态分组，在decode graph内部，anchor和非anchor组tenshor shape会变化，图不友好
    # 如果想图友好，就得在model_runner重排batch和metadata：anchor decode/non-anchor decode/prefill，然后分段跑不同QLI，过于复杂

    # 2.2 统一anchor：存在anchor即全走full_qli，全是非anchor再topm_qli，单轮decode单类qli，图友好，性能差一点
    # HOST侧决策anchor/non-anchor，forward_decode里if else消费HOST侧变量，不动态判断，不会破坏图
    # anchor/non-anchor 最好形成两套计算图 ACL graph，需要将这个变量定义如 graph descriptor
    #LINK - /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2679
    # 这个决策点需要记录每个req的last_anchor_position，然后只要有一个新请求/老请求到了新anchor点，就全部走anchor
    # build_decode_metadata中没有req信息，最好在model_runner的metadata中计算，只需要增加少量metadata字段就行
    # 待确认：在什么位置计算
    #LINK - /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1810
    
    # 3. topm_indexer_k_cache
    # topm_indexer_k_cache
    # topm_indexer_scale_decode_metadata: 重点是block_table（读），slot_mapping（写）好像不用操心，有专门的算子基于topk_idxs gather和写入，其他可以仿indexer_scale_decode_metadata
    def _forward_decode(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
    ):
        assert attn_metadata[0].decode is not None
        compress_common_attn_metadata = None

        #NOTE - kv cache
        # num_blocks, block_size, num_kv_heads, head_dim
        # compress_kv_cache: [78763, 128, 1, 512]
        # swa_kv_cache: [78763, 128, 1, 512]
        # state_cache: [78763, 8, 1, 512] compressor压缩需要的中间状态
        # _
        # indexer_k_cache: [78763, 128, 1, 128]
        # indexer_scale_cache: [78763, 128, 1, 1] 量化scale
        # indexer_local_k_cache: [78763, 128, 1, 128] indexcache topm cache 
        # indexer_local_scale_cache: [78763, 128, 1, 1] indexcache topm scale cache
        (compress_kv_cache,
         swa_kv_cache,
         state_cache,
         indexer_k_cache,
         indexer_scale_cache,
         indexer_full_cache,
         indexer_local_k_cache,
         indexer_local_scale_cache) = (
            DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)
        )
        if not self.enable_local_k_cache:
            if self.compress_ratio == 4:
                (compress_kv_cache, swa_kv_cache, state_cache, _, indexer_k_cache, indexer_scale_cache) = kv_cache
                # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
                (compressor_attn_metadata, compressor_kv_state_metadata, _, indexer_kv_scale_metadata, swa_metadata) = (
                    attn_metadata
                )
                compress_common_attn_metadata = compressor_attn_metadata
            elif self.compress_ratio == 128:
                (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = kv_cache
                # sorted keys: [attn, compressor.state_cache, swa_cache]
                (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
                compress_common_attn_metadata = compressor_attn_metadata
            else:
                (_, swa_kv_cache, _, _, _, _) = kv_cache
                # sorted keys: [swa_cache]
                (swa_metadata,) = attn_metadata
                compress_common_attn_metadata = swa_metadata
        else:
            if self.compress_ratio == 4:
                # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
                (compressor_attn_metadata,
                compressor_kv_state_metadata,
                _,
                indexer_kv_scale_metadata,
                indexer_local_kv_scale_metadata,
                swa_metadata) = (
                    attn_metadata
                )
                compress_common_attn_metadata = compressor_attn_metadata
            elif self.compress_ratio == 128:
                # sorted keys: [attn, compressor.state_cache, swa_cache]
                (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
                compress_common_attn_metadata = compressor_attn_metadata
            else:
                # sorted keys: [swa_cache]
                (swa_metadata,) = attn_metadata
                compress_common_attn_metadata = swa_metadata
        common_decode_metadata = _require_decode_metadata(compress_common_attn_metadata)
        swa_decode_metadata = _require_decode_metadata(swa_metadata)
        cos = common_decode_metadata.cos[layer_name]
        sin = common_decode_metadata.sin[layer_name]
        actual_seq_lengths_query = common_decode_metadata.query_start_loc
        actual_seq_lengths_key = common_decode_metadata.seq_lens

        if self.multistream_dsv4_dsa_overlap:
            # mla prolog: q + kv dual-stream parallel
            q, qr, qr_pertoken_scale = self._mla_prolog_multistream(
                hidden_states, cos, sin, swa_kv_cache, swa_decode_metadata.slot_mapping, is_prefill=False
            )
        else:
            # Share one dynamic-quant of hidden_states between wq_a (main stream)
            # and wkv (attention stream) when both sides are W8A8 dynamic.
            share_hs_quant = _is_w8a8_dynamic(self.wq_a) and _is_w8a8_dynamic(self.wkv)
            if share_hs_quant:
                hs_int8, hs_pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

            # q
            if _is_w8a8_dynamic(self.wq_b):
                if share_hs_quant:
                    q_a = torch_npu.npu_quant_matmul(
                        hs_int8,
                        self.wq_a.weight,
                        self.wq_a.weight_scale,
                        pertoken_scale=hs_pertoken_scale,
                        bias=self.wq_a.bias,
                        output_dtype=hidden_states.dtype,
                    )
                else:
                    q_a = self.wq_a(hidden_states)
                qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                    q_a, self.q_norm.weight, epsilon=self.eps
                )
                q = torch_npu.npu_quant_matmul(
                    qr,
                    self.wq_b.weight,
                    self.wq_b.weight_scale,
                    pertoken_scale=qr_pertoken_scale,
                    bias=self.wq_b.bias,
                    output_dtype=hidden_states.dtype,
                ).unflatten(-1, (self.n_local_heads, self.head_dim))
            else:
                if share_hs_quant:
                    q_a = torch_npu.npu_quant_matmul(
                        hs_int8,
                        self.wq_a.weight,
                        self.wq_a.weight_scale,
                        pertoken_scale=hs_pertoken_scale,
                        bias=self.wq_a.bias,
                        output_dtype=hidden_states.dtype,
                    )
                    qr = q = self.q_norm(q_a)
                else:
                    qr = q = self.q_norm(self.wq_a(hidden_states))
                q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
                qr_pertoken_scale = None

            q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                q.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )

            # win kv & tok_dis
            if share_hs_quant:
                kv = torch_npu.npu_quant_matmul(
                    hs_int8,
                    self.wkv.weight,
                    self.wkv.weight_scale,
                    pertoken_scale=hs_pertoken_scale,
                    bias=self.wkv.bias,
                    output_dtype=hidden_states.dtype,
                )
            else:
                kv = self.wkv(hidden_states)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )

            # swa exec kv
            DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_decode_metadata.slot_mapping)

        if self.compress_ratio > 1:
            compressor_decode_metadata = _require_decode_metadata(compressor_attn_metadata)
            compressor_state_decode_metadata = _require_decode_metadata(compressor_kv_state_metadata)
            compress_cos = common_decode_metadata.compress_cos[layer_name]
            compress_sin = common_decode_metadata.compress_sin[layer_name]
            compress_topk_idxs = None
            if self.compress_ratio == 4:
                # IndexCache: decode segment occupies buffer[:num_decode_tokens]
                decode_num_tokens = hidden_states.shape[0]
                if self.skip_topk:
                    compress_topk_idxs = self._get_indexcache_topk_indices(decode_num_tokens, offset=0)
                else:
                    if self.multistream_dsv4_dsa_overlap:
                        #NOTE - multistream indexer_q
                        # cv指CV wrapper: split wq_a/wkv/wq_b into quantize(Vector) + matmul(Cube)，cube/vector多流并行，定义于1392行
                        indexer_q = self.cv_indexer_select_qli(  # multistream version
                            x=hidden_states,
                            qr=qr,
                            kv_cache=kv_cache,
                            attn_metadata=attn_metadata,
                            cos=cos,
                            sin=sin,
                            compressed_cos=compress_cos,
                            compressed_sin=compress_sin,
                            actual_seq_lengths_query=actual_seq_lengths_query,
                            with_prefill=False,
                            qr_pertoken_scale=qr_pertoken_scale,
                        )
                    else:
                        compress_topk_idxs = self.indexer_select_qli(  # original version
                            x=hidden_states,
                            qr=qr,
                            kv_cache=kv_cache,
                            attn_metadata=attn_metadata,
                            cos=cos,
                            sin=sin,
                            compressed_cos=compress_cos,
                            compressed_sin=compress_sin,
                            actual_seq_lengths_query=actual_seq_lengths_query,
                            actual_seq_lengths_key=actual_seq_lengths_key,
                            with_prefill=False,
                            qr_pertoken_scale=qr_pertoken_scale,
                        )

            coff = 2 if self.compressor_overlap else 1

            # Inline compressor + scatter (c128, c4 non-dual)
            #SECTION - cmp_kv
            #LINK - /vllm-workspace/vllm-ascend/csrc/attention/compressor/README.md
            # 每轮decode都调用，更新state_cache，存储当前compressor中间状态，计算cmp_kv
            # cmp_kv只有在压缩边界才有意义，才会构造slot_mapping写入cache
            # cmp_kv不在压缩边界时不是有效结果，即使计算也不会写入cache
            compressed_kv = torch.ops._C_ascend.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                state_cache.squeeze(-2),
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                state_block_table=compressor_state_decode_metadata.block_table,
                cu_seqlens=actual_seq_lengths_query,
                seqused=None,
                start_pos=common_decode_metadata.start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )

            # For multistream_dsv4_dsa_overlap with compress_ratio=4:
            # aux_stream: indexer_weights_proj (parallel with main q_quant + kv_scatter)
            # main stream: compressed_kv -> q_quant -> kv_scatter -> wait aux_stream -> lightning_indexer
            if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
                main_stream = torch.npu.current_stream()
                aux_stream = dsv4_dsa_overlap_stream()
                e_compressed_kv_done = main_stream.record_event()
                with npu_stream_switch(aux_stream, enabled=True):
                    torch.npu.current_stream().wait_event(e_compressed_kv_done)
                    weights_proj_output = self.weights_proj(hidden_states)
                # Main stream: q_quant (between compressed_kv and kv_scatter)
                q_quant, q_scale = DeviceOperator.indexer_quantize_query(indexer_q)

            DeviceOperator.dsa_kv_compress_scatter(
                compress_kv_cache, compressed_kv, compressor_decode_metadata.slot_mapping
            )

            #NOTE - cmp_kv -> cache
            #LINK - https://gitcode.com/cann/ops-nn/tree/master/index/scatter_nd_update
            #LINK - /vllm-workspace/vllm-ascend/csrc/moe/scatter_nd_update_v2/op_host/op_api/aclnn_scatter_nd_update_v2.cpp:106
            # kv计算后就通过slot_mapping写入cache，attention计算时传入cache，就不会遗漏当前q可能对应的kv
            # slot_mapping：[] shape=(0, 2) 0是因为压缩边界，每4/128个token计算和写入一次，需要写入是会大于0；2维是 (block_id, offset)。
            # 文档106行，当算子传入slot_mapping为空，算子直接返回。所以可以每轮都调用，但只有压缩边界，slot_mapping存在时才真正写入。
            # 该逻辑适用于compress_kv_cache和indexer_k_cache
            torch.ops._C_ascend.npu_scatter_nd_update_v2(
                compress_kv_cache, compressor_decode_metadata.slot_mapping, compressed_kv
            )
            #!SECTION

            if self.compress_ratio == 4 and self.use_index_cache:
                self._update_indexcache_topk_indices(compress_topk_idxs, offset=0)

        attn_op = DeviceOperator.get_dsa_sparse_attn_op()
        extra_attn_kwargs: dict = DeviceOperator.get_dsa_sparse_attn_base_kwargs()

        if self.compress_ratio <= 1:
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                ori_block_table=swa_decode_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=swa_decode_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=max(self.compress_ratio, 1),
                ori_mask_mode=4,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                **extra_attn_kwargs,
            )[0]
        elif self.compress_ratio == 4:
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_decode_metadata.block_table,
                cmp_block_table=compressor_decode_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_decode_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                **extra_attn_kwargs,
            )[0]
        else:
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                ori_block_table=swa_decode_metadata.block_table,
                cmp_block_table=compressor_decode_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_decode_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                **extra_attn_kwargs,
            )[0]
        return attn_output
    #!SECTION

    def _indexer_qkv_prepare(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        with_prefill: bool = False,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        if not self.enable_local_k_cache:
            (
                _,
                _,
                _,
                indexer_state_cache,
                indexer_k_cache,
                indexer_scale_cache,
            ) = kv_cache
            (
                _,
                _,
                indexer_kv_state_metadata,
                indexer_kv_scale_metadata,
                _,
            ) = attn_metadata
        else:
            (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache, indexer_local_k_cache, indexer_local_scale_cache) = (
                DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
            )
            (
                _,
                _,
                indexer_kv_state_metadata,
                indexer_kv_scale_metadata,
                indexer_local_kv_scale_metadata,
                _,
            ) = attn_metadata

           
        if (
            _is_w8a8_dynamic(self.inderxer_wq_b)
            and qr_pertoken_scale is not None
            and get_ascend_device_type() not in {AscendDeviceType.A5}
        ):
            q = torch_npu.npu_quant_matmul(
                qr,
                self.inderxer_wq_b.weight,
                self.inderxer_wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.inderxer_wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            q = self.inderxer_wq_b(qr)
        q = q.view(-1, self.indexer_heads, self.indexcom_head_dim)  # [T, N, D]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.indexcom_head_dim - self.rope_head_dim, self.indexcom_head_dim],
        )

        q = rotate_activation(q, indexer_kv_scale_metadata.hadamard)
        coff = 2 if self.compressor_overlap else 1

        if with_prefill:
            indexer_state_prefill_metadata = _require_prefill_metadata(indexer_kv_state_metadata)
            indexer_scale_prefill_metadata = _require_prefill_metadata(indexer_kv_scale_metadata)
            kv_block_table = indexer_state_prefill_metadata.block_table
            start_pos = indexer_scale_prefill_metadata.start_pos
        else:
            indexer_state_decode_metadata = _require_decode_metadata(indexer_kv_state_metadata)
            indexer_scale_decode_metadata = _require_decode_metadata(indexer_kv_scale_metadata)
            kv_block_table = indexer_state_decode_metadata.block_table
            start_pos = indexer_scale_decode_metadata.start_pos

        kv = torch.ops._C_ascend.compressor(
            x,
            self.indexcom_wkv.weight,
            self.indexcom_wgate.weight,
            indexer_state_cache.squeeze(-2),
            self.indexcom_ape,
            self.indexcom_norm.weight,
            compressed_sin.view(-1, compressed_sin.shape[-1]),
            compressed_cos.view(-1, compressed_cos.shape[-1]),
            state_block_table=kv_block_table,
            cu_seqlens=actual_seq_lengths_query,
            seqused=None,
            start_pos=start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )

        if kv.numel() == 0:
            kv = None
        elif self.indexcom_rotate:
            kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

        return (
            q,
            kv,
            indexer_k_cache,
            indexer_scale_cache,
            indexer_full_cache,
            indexer_kv_state_metadata,
            indexer_kv_scale_metadata,
            with_prefill,
        )

    def _indexer_qli_finish(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None,
        weights: torch.Tensor,
        indexer_k_cache: torch.Tensor,
        indexer_scale_cache: torch.Tensor,
        indexer_full_cache: torch.Tensor | None,
        indexer_kv_state_metadata,
        indexer_kv_scale_metadata,
        with_prefill: bool,
    ):
        q, q_scale, kv, kv_scale = self._indexer_quant_scatter(
            q,
            kv,
            indexer_k_cache,
            indexer_scale_cache,
            indexer_full_cache,
            indexer_kv_scale_metadata,
            with_prefill,
        )
        return self._indexer_qli(
            q,
            weights,
            q_scale,
            indexer_k_cache,
            indexer_scale_cache,
            indexer_kv_scale_metadata,
            with_prefill,
        )

    def _indexer_quant_scatter(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None,
        indexer_k_cache: torch.Tensor,
        indexer_scale_cache: torch.Tensor,
        indexer_full_cache: torch.Tensor | None,
        indexer_kv_scale_metadata,
        with_prefill: bool,
    ):
        slot_mapping = (
            indexer_kv_scale_metadata.prefill.slot_mapping
            if with_prefill
            else indexer_kv_scale_metadata.decode.slot_mapping
        )
        return DeviceOperator.indexer_quant_scatter(
            q, kv, indexer_k_cache, indexer_scale_cache, indexer_full_cache, slot_mapping
        )

    def _indexer_qli(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        q_scale: torch.Tensor,
        indexer_k_cache: torch.Tensor,
        indexer_scale_cache: torch.Tensor,
        indexer_kv_scale_metadata,
        with_prefill: bool,
    ):
        if with_prefill:
            assert indexer_kv_scale_metadata.prefill is not None
            prefill_meta = indexer_kv_scale_metadata.prefill
            qlens = prefill_meta.query_start_loc[1:]
            qli_metadata = prefill_meta.qli_metadata

            use_topm = self.index_topm is not None and self.compress_ratio == 4

            if use_topm and prefill_meta.topm_num_blocks > 0:
                composite_k, composite_scale, composite_bt, composite_kvlen, num_topm_blocks = \
                    self._prepare_k_cache_for_qli_prefill(
                        indexer_k_cache,
                        indexer_scale_cache,
                        prefill_meta.block_table,
                        prefill_meta.topm_idxs,
                        prefill_meta.topm_chunk_start_logical,
                        prefill_meta.act_qlen,
                    )

                topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
                    query=q,
                    key=composite_k,
                    weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
                    query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale),
                    key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(composite_scale),
                    actual_seq_lengths_query=qlens,
                    actual_seq_lengths_key=composite_kvlen,
                    block_table=composite_bt,
                    metadata=qli_metadata,
                    query_quant_mode=0,
                    key_quant_mode=0,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topm,
                    sparse_mode=3,
                    pre_tokens=(1 << 63) - 1,
                    next_tokens=(1 << 63) - 1,
                    cmp_ratio=4,
                    return_value=False,
                )

                topk_idxs = self._remap_composite_to_original(
                    topk_idxs,
                    prefill_meta.topm_idxs,
                    prefill_meta.topm_chunk_start_logical,
                    prefill_meta.act_qlen,
                    num_topm_blocks,
                )
                prefill_meta.topm_idxs = topk_idxs
                topk_idxs = topk_idxs[:, :, :self.index_topk]
            else:
                kvlens = prefill_meta.seq_lens
                block_table = prefill_meta.block_table
                topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
                    query=q,
                    key=indexer_k_cache,
                    weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
                    query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale),
                    key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(indexer_scale_cache),
                    actual_seq_lengths_query=qlens,
                    actual_seq_lengths_key=kvlens,
                    block_table=block_table,
                    metadata=qli_metadata,
                    query_quant_mode=0,
                    key_quant_mode=0,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topm,
                    sparse_mode=3,
                    pre_tokens=(1 << 63) - 1,
                    next_tokens=(1 << 63) - 1,
                    cmp_ratio=4,
                    return_value=False,
                )
                if use_topm:
                    prefill_meta.topm_idxs = topk_idxs
                    topk_idxs = topk_idxs[:, :, :self.index_topk]
        else:
            assert indexer_kv_scale_metadata.decode is not None
            qlens = indexer_kv_scale_metadata.decode.query_start_loc[1:]
            kvlens = indexer_kv_scale_metadata.decode.seq_lens
            block_table = indexer_kv_scale_metadata.decode.block_table
            qli_metadata = indexer_kv_scale_metadata.decode.qli_metadata

            topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
                query=q,
                key=indexer_k_cache,
                weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
                query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale),
                key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(indexer_scale_cache),
                actual_seq_lengths_query=qlens,
                actual_seq_lengths_key=kvlens,
                block_table=block_table,
                metadata=qli_metadata,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                return_value=False,
            )
        return topk_idxs

    def indexer_select_qli(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor | None = None,
        with_prefill: bool = False,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        q, kv, ik, isc, ifc, indexer_kv_state_meta, isc_meta, wp = self._indexer_qkv_prepare(
            x,
            qr,
            kv_cache,
            attn_metadata,
            cos,
            sin,
            compressed_cos,
            compressed_sin,
            actual_seq_lengths_query,
            with_prefill,
            qr_pertoken_scale,
        )

        weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexer_heads**-0.5)

        return self._indexer_qli_finish(q, kv, weights, ik, isc, ifc, indexer_kv_state_meta, isc_meta, wp)

    def cv_indexer_select_qli(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        with_prefill: bool = False,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        """
        Multistream version: 4-block segmentation, main stream and aux stream
        alternate submission to achieve V/C engine parallel

        Core strategy:
        - Part0: Main pre-compute qr_quant[V] + compressor[C/mixed] + kv_hadamard[V]
        - Part1: Main matmul[C] ∥ Aux kv_quant[V] + scatter_k_cache[AIV]
        - Part2: Main rope[V] (serial)
        - Part3: Main q_hadamard[C] ∥ Aux scatter_scale_cache[AIV]
        - Part4: Caller runs weights_proj + q_quant + indexer
        """
        """
        > 00 = ('model. layers.0. self_attn. swa_cache', AscendDSAMetadata(n
        > 01 = ('model. layers.1.self_attn.swa_cache', AscendDSAMetadata(nu
        > 02 = ( 'model. layers.2.self_attn.attn', AscendDSAMetadata(num_act
        > 03 = ( 'model. layers.2. self_attn. compressor.state_cache', Ascend[
        > 04 = ( 'model. layers.2. self_attn. indexer . compressor.state_cache'
        > 05 = ( 'model. layers.2.self_attn. indexer.k_cache', AscendDSAMetac
        > 06 = ( 'model. layers.2.self_attn. indexer.local_k_cache', AscendDS
        > 07 = ('model. layers.2.self_attn. swa_cache', AscendDSAMetadata(nu
        > 08 = ('model. layers.3.self_attn.attn', AscendDSAMetadata(num_act
        > 09 = ( 'model. layers.3.self_attn. compressor.state_cache', Ascendl
        > 10 = ('model. layers.3.self_attn. swa_cache', AscendDSAMetadata(nu
        """
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache, indexer_local_k_cache, indexer_local_scale_cache) = (
                    DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
                )
        # if not self.enable_local_k_cache:
        #     (_, _, _, indexer_state_cache, indexer_k_cache, indexer_scale_cache) = kv_cache
        #     (_, _, indexer_kv_state_metadata, indexer_kv_scale_metadata, _) = attn_metadata
        # else:
        #     (_, _, _, indexer_state_cache, indexer_k_cache, indexer_scale_cache, _, _) = kv_cache
        #     (_, _, indexer_kv_state_metadata, indexer_kv_scale_metadata, _, _) = attn_metadata
        (_,
        _,
        indexer_kv_state_metadata,
        indexer_kv_scale_metadata,
        indexer_local_kv_scale_metadata,
        _,
        ) = attn_metadata
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()

        # ===== Part0: Pre-compute on main =====
        if _is_w8a8_dynamic(self.inderxer_wq_b) and qr_pertoken_scale is not None:
            qr_quant_ready = qr
            qr_scale_ready = qr_pertoken_scale
        else:
            qr_quant_ready, qr_scale_ready = self.cv_inderxer_wq_b.quantize(qr)

        coff = 2 if self.compressor_overlap else 1

        if with_prefill:
            indexer_state_prefill_metadata = _require_prefill_metadata(indexer_kv_state_metadata)
            indexer_scale_prefill_metadata = _require_prefill_metadata(indexer_kv_scale_metadata)
            kv_block_table = indexer_state_prefill_metadata.block_table
            start_pos = indexer_scale_prefill_metadata.start_pos
        else:
            indexer_state_decode_metadata = _require_decode_metadata(indexer_kv_state_metadata)
            indexer_scale_decode_metadata = _require_decode_metadata(indexer_kv_scale_metadata)
            kv_block_table = indexer_state_decode_metadata.block_table
            start_pos = indexer_scale_decode_metadata.start_pos

        #NOTE - multistream indexer_k
        # indexer_k 逻辑同 cmp_kv
        kv = torch.ops._C_ascend.compressor(
            x,
            self.indexcom_wkv.weight,
            self.indexcom_wgate.weight,
            indexer_state_cache.squeeze(-2),
            self.indexcom_ape,
            self.indexcom_norm.weight,
            compressed_sin.view(-1, compressed_sin.shape[-1]),
            compressed_cos.view(-1, compressed_cos.shape[-1]),
            state_block_table=kv_block_table,
            cu_seqlens=actual_seq_lengths_query,
            seqused=None,
            start_pos=start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )

        if kv.numel() == 0:
            kv = None
        elif self.indexcom_rotate:
            kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

        # ===== Part1: matmul[C] ∥ kv_quant[V] + scatter_k_cache[AIV] =====
        # Record event before main stream operations for aux_stream to wait
        e_kv_ready = main_stream.record_event()

        # Aux: kv_quant + scatter_k_cache (parallel with main matmul + rope)
        if kv is not None:
            slot_mapping_indexer = (
                indexer_scale_prefill_metadata.slot_mapping
                if with_prefill
                else indexer_scale_decode_metadata.slot_mapping
            )
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_kv_ready)
                kv, kv_scale = DeviceOperator.indexer_quant_scatter_part1(
                    kv, indexer_k_cache, indexer_full_cache, slot_mapping_indexer
                )

        # Main: matmul q from qr (directly submit, V/C different engines dispatch naturally)
        if _is_w8a8_dynamic(self.inderxer_wq_b) and qr_pertoken_scale is not None:
            q = torch_npu.npu_quant_matmul(
                qr_quant_ready,
                self.inderxer_wq_b.weight,
                self.inderxer_wq_b.weight_scale,
                pertoken_scale=qr_scale_ready,
                bias=self.inderxer_wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            q = self.cv_inderxer_wq_b.matmul(qr_quant_ready, qr_scale_ready)  # qr_matmul

        if kv is not None:
            main_stream.wait_stream(aux_stream)

        q = q.view(-1, self.indexer_heads, self.indexcom_head_dim)

        # ===== Part2: rope[V] (main only) =====
        torch.ops._C_ascend.inplace_partial_rotary_mul(  # rope
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.indexcom_head_dim - self.rope_head_dim, self.indexcom_head_dim],
        )

        # Wait for aux_stream kv_scatter to complete before proceeding
        if kv is not None:
            main_stream.wait_stream(aux_stream)

        e_rope_done = main_stream.record_event()

        # ===== Part3: q_hadamard[C] ∥ scatter_scale_cache[AIV] =====
        # Note: On A5, indexer_compress_epilog_v2 in Part1 handles both k_cache
        # and scale_cache in one fused operation, so Part3 is skipped
        # (kv_scale is None on A5 from indexer_quant_scatter_part1).
        if kv is not None and kv_scale is not None:
            slot_mapping_indexer_part3 = (
                indexer_scale_prefill_metadata.slot_mapping
                if with_prefill
                else indexer_scale_decode_metadata.slot_mapping
            )
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_rope_done)
                DeviceOperator.dsa_indexer_scatter_scale_part3(
                    kv_scale, indexer_scale_cache, slot_mapping_indexer_part3
                )

        # Main: q_hadamard[Part1 - linear] (directly submit, C/AIV different engines dispatch naturally)
        # Part1: F.linear - parallel with aux_stream kv_scatter
        hidden_size = q.size(-1)
        q_linear, q_shape, q_dim = hadamard_linear(q, indexer_kv_scale_metadata.hadamard)

        if kv is not None:
            main_stream.wait_stream(aux_stream)

        # Main: q_hadamard[Part2 - scale] (after aux_stream completes)
        # Part2: scale * reshape - dot multiplication
        q = hadamard_scale(q_linear, q_shape, q_dim, scale=hidden_size**-0.5)

        return q

    def _remap_composite_to_original(
        self,
        topk_idxs_comp: torch.Tensor,
        topm_idxs: torch.Tensor,
        chunk_start_logical: int,
        act_qlen: int,
        num_topm_blocks: int,
        block_size: int = 32,
    ) -> torch.Tensor:
        topm_flat = topm_idxs.squeeze(0).squeeze(0)
        topm_len = topm_flat.numel()
        topm_offset = num_topm_blocks * block_size
        total_slots = topm_offset + cdiv(act_qlen, block_size) * block_size

        global_idxs = torch.zeros(total_slots, dtype=torch.int32, device=topm_flat.device)
        global_idxs[:topm_len] = topm_flat
        chunk_indices = chunk_start_logical * block_size + torch.arange(act_qlen, device=topm_flat.device)
        global_idxs[topm_offset : topm_offset + act_qlen] = chunk_indices

        return global_idxs[topk_idxs_comp.long()]

    def _gather_indexer_cache(
        self,
        flat_idx: torch.Tensor,
        total_count: int,
        k2d: torch.Tensor,
        s2d: torch.Tensor,
        blk: int,
    ):
        B = flat_idx.shape[0]
        nblocks = cdiv(total_count, blk)
        total = nblocks * blk
        pad = total - total_count
        if pad:
            flat_idx = F.pad(flat_idx, (0, pad)).contiguous()
        fs = flat_idx.stride()
        flat_idx = torch.as_strided(flat_idx, (B * nblocks, blk),
                                     (fs[1] * blk, fs[1])).to(torch.int32)
        new_k = torch_npu.npu_gather_sparse_index(k2d, flat_idx).unsqueeze(2)
        new_scale = torch_npu.npu_gather_sparse_index(s2d, flat_idx).unsqueeze(2)
        return new_k, new_scale, nblocks

    def _prepare_k_cache_for_qli(
        self,
        indexer_k_cache,
        indexer_scale_cache,
        block_table,
        topm_idxs
    ):
        B = topm_idxs.shape[0]
        blk = indexer_k_cache.shape[1]

        flat_idx = torch.gather(block_table, 1, topm_idxs // blk) * blk + topm_idxs % blk

        ks = indexer_k_cache.stride()
        k2d = torch.as_strided(indexer_k_cache,
                            (indexer_k_cache.shape[0] * blk, self.indexer_dim),
                            (ks[1], ks[-1]))
        ss = indexer_scale_cache.stride()
        s2d = torch.as_strided(indexer_scale_cache,
                            (indexer_scale_cache.shape[0] * blk, 1),
                            (ss[1], ss[-1]))

        new_k, new_scale, nblocks = self._gather_indexer_cache(
            flat_idx, self.index_topm, k2d, s2d, blk)

        tbt = torch.arange(B * nblocks, device=block_table.device).view(B, nblocks).to(torch.int32)
        return (new_k, new_scale, tbt)

    def _prepare_k_cache_for_qli_prefill(
        self,
        indexer_k_cache,
        indexer_scale_cache,
        block_table,
        topm_idxs,
        chunk_start_logical,
        act_qlen,
    ):
        block_size = 32
        blk = block_size
        B = block_table.shape[0]
        device = block_table.device
        topm_squeezed = topm_idxs.squeeze(1)

        ks = indexer_k_cache.stride()
        k2d = torch.as_strided(indexer_k_cache,
                            (indexer_k_cache.shape[0] * blk, self.indexer_dim),
                            (ks[1], ks[-1]))
        ss = indexer_scale_cache.stride()
        s2d = torch.as_strided(indexer_scale_cache,
                            (indexer_scale_cache.shape[0] * blk, 1),
                            (ss[1], ss[-1]))

        topm_flat = torch.gather(block_table[:1], 1, topm_squeezed // blk) * blk + topm_squeezed % blk
        chunk_global = chunk_start_logical * block_size + torch.arange(act_qlen, dtype=torch.int32, device=device)
        chunk_flat = torch.gather(block_table[:1], 1, chunk_global // blk) * blk + chunk_global % blk
        combined_flat = torch.cat([topm_flat, chunk_flat], dim=1)

        num_topm_blocks = cdiv(self.index_topm, blk)
        total_count = self.index_topm + act_qlen
        composite_k, composite_scale, nblocks = self._gather_indexer_cache(
            combined_flat, total_count, k2d, s2d, blk)

        composite_bt = torch.arange(nblocks, device=device, dtype=block_table.dtype).expand(B, -1)
        composite_kvlen = torch.tensor([num_topm_blocks * block_size + act_qlen],
                                        device=device, dtype=torch.int32)
        return composite_k, composite_scale, composite_bt, composite_kvlen, num_topm_blocks
#!SECTION
