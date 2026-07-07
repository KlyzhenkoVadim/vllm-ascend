#include "vllm_gather_topm_tiling.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {

static ge::graphStatus TilingPrepareForVllmGatherTopm(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForVllmGatherTopm(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("VllmGatherTopm", "Tiling context is null."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // Input shapes
    auto keyShape = context->GetInputShape(KEY_INDEX)->GetStorageShape();
    auto topmIdxsShape = context->GetInputShape(TOPM_IDXS_INDEX)->GetStorageShape();
    auto blockTableShape = context->GetInputShape(BLOCK_TABLE_INDEX)->GetStorageShape();

    uint32_t batchSize = static_cast<uint32_t>(topmIdxsShape.GetDim(DIM_IDX_ZERO));
    uint32_t topmCount = static_cast<uint32_t>(topmIdxsShape.GetDim(DIM_IDX_ONE));
    uint32_t blockSize = static_cast<uint32_t>(keyShape.GetDim(DIM_IDX_ONE));
    uint32_t headDim = static_cast<uint32_t>(keyShape.GetDim(DIM_IDX_THREE));
    uint32_t maxBlockNumPerBatch = static_cast<uint32_t>(blockTableShape.GetDim(DIM_IDX_ONE));
    uint32_t numGatherBlocksPerBatch =
        (topmCount + blockSize - 1) / blockSize;
    uint32_t totalGatherElements = batchSize * topmCount;

    auto keyDesc = context->GetInputDesc(KEY_INDEX);
    int64_t stride = keyDesc->GetOriginShape().GetDim(DIM_IDX_ZERO) > 1
                         ? keyDesc->GetStorageShape().GetDim(DIM_IDX_ONE) *
                               keyDesc->GetStorageShape().GetDim(DIM_IDX_TWO) *
                               keyDesc->GetStorageShape().GetDim(DIM_IDX_THREE)
                         : 1;

    auto keyScaleDesc = context->GetInputDesc(KEY_SCALE_INDEX);
    int64_t scaleStride = keyScaleDesc->GetOriginShape().GetDim(DIM_IDX_ZERO) > 1
                              ? keyScaleDesc->GetStorageShape().GetDim(DIM_IDX_ONE) *
                                    keyScaleDesc->GetStorageShape().GetDim(DIM_IDX_TWO)
                              : 1;

    // Workspace: single buffer, no workspace needed
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = 0;

    // Tiling data
    GatherTopmTilingData tilingData;
    tilingData.set_batchSize(batchSize);
    tilingData.set_topmCount(topmCount);
    tilingData.set_blockSize(blockSize);
    tilingData.set_headDim(headDim);
    tilingData.set_maxBlockNumPerBatch(maxBlockNumPerBatch);
    tilingData.set_numGatherBlocksPerBatch(numGatherBlocksPerBatch);
    tilingData.set_totalGatherElements(totalGatherElements);
    tilingData.set_usedCoreNum(blockDim);
    tilingData.set_stride(stride);
    tilingData.set_scaleStride(scaleStride);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    uint32_t inputKType = static_cast<uint32_t>(keyDesc->GetDataType());
    uint32_t outputType = static_cast<uint32_t>(
        context->GetOutputDesc(GATHERED_KEY_OUT_INDEX)->GetDataType());
    uint32_t tilingKey = (inputKType << 0) | (outputType << 4);
    context->SetTilingKey(tilingKey);
    context->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(VllmGatherTopm)
    .Tiling(TilingForVllmGatherTopm)
    .TilingParse<int32_t>(TilingPrepareForVllmGatherTopm);

}  // namespace optiling
