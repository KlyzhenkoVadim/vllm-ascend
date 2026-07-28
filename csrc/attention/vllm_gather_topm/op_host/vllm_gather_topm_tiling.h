#ifndef VLLM_GATHER_TOPM_TILING_H
#define VLLM_GATHER_TOPM_TILING_H

#include "err/ops_err.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

// Inputs Index
constexpr uint32_t KEY_INDEX = 0;
constexpr uint32_t KEY_SCALE_INDEX = 1;
constexpr uint32_t BLOCK_TABLE_INDEX = 2;
constexpr uint32_t TOPM_IDXS_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_KEY_INDEX = 4;
constexpr uint32_t GATHERED_KEY_OUT_INDEX = 0;
constexpr uint32_t GATHERED_SCALE_OUT_INDEX = 1;

constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t BLOCK_SIZE_LIMIT = 1024;
constexpr uint32_t BLOCK_SIZE_FACTOR = 16;
constexpr uint32_t TOPM_MAX = 2048;
constexpr uint32_t TOPM_COUNT_IDX = 0;
constexpr uint32_t BLOCK_SIZE_IDX = 0;

BEGIN_TILING_DATA_DEF(GatherTopmTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, topmCount)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, numGatherBlocksPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, totalGatherElements)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(int64_t, stride)
TILING_DATA_FIELD_DEF(int64_t, scaleStride)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(VllmGatherTopm, GatherTopmTilingData)

}  // namespace optiling
#endif
