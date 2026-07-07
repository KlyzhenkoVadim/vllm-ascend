#ifndef GATHER_TOPM_KERNEL_H
#define GATHER_TOPM_KERNEL_H

#include "kernel_operator.h"
#include "../vllm_gather_topm_common.h"

namespace GatherTopmKernel {
using namespace GatherTopmCommon;

template <typename K_T, typename S_T>
class GatherTopm {
public:
    __aicore__ inline GatherTopm() {};
    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *keyScale,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *topmIdxs,
                                __gm__ uint8_t *actualSeqLengthsKey,
                                __gm__ uint8_t *gatheredKey, __gm__ uint8_t *gatheredScale,
                                const GatherTopmTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

protected:
    TPipe *pipe = nullptr;

    GlobalTensor<K_T> keyGm;
    GlobalTensor<S_T> keyScaleGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topmIdxsGm;
    GlobalTensor<K_T> gatheredKeyGm;
    GlobalTensor<S_T> gatheredScaleGm;

    uint32_t batchSize;
    uint32_t topmCount;
    uint32_t blockSize;
    uint32_t headDim;
    uint32_t maxBlockNumPerBatch;
    uint32_t numGatherBlocksPerBatch;
    uint32_t totalGatherElements;
    uint32_t usedCoreNum;
    int64_t stride;
    int64_t scaleStride;
};

template <typename K_T, typename S_T>
__aicore__ inline void GatherTopm<K_T, S_T>::Init(
    __gm__ uint8_t *key, __gm__ uint8_t *keyScale,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *topmIdxs,
    __gm__ uint8_t *actualSeqLengthsKey,
    __gm__ uint8_t *gatheredKey, __gm__ uint8_t *gatheredScale,
    const GatherTopmTilingData *__restrict tiling, TPipe *tPipe)
{
    (void)actualSeqLengthsKey;
    pipe = tPipe;

    batchSize = tiling->batchSize;
    topmCount = tiling->topmCount;
    blockSize = tiling->blockSize;
    headDim = tiling->headDim;
    maxBlockNumPerBatch = tiling->maxBlockNumPerBatch;
    numGatherBlocksPerBatch = tiling->numGatherBlocksPerBatch;
    totalGatherElements = tiling->totalGatherElements;
    usedCoreNum = tiling->usedCoreNum;
    stride = tiling->stride;
    scaleStride = tiling->scaleStride;

    uint32_t totalBlocks = batchSize * numGatherBlocksPerBatch;

    keyGm.SetGlobalBuffer((__gm__ K_T *)key);
    keyScaleGm.SetGlobalBuffer((__gm__ S_T *)keyScale);
    blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    topmIdxsGm.SetGlobalBuffer((__gm__ int32_t *)topmIdxs);

    gatheredKeyGm.SetGlobalBuffer((__gm__ K_T *)gatheredKey, totalBlocks * blockSize * 1 * headDim);
    gatheredScaleGm.SetGlobalBuffer((__gm__ S_T *)gatheredScale, totalBlocks * blockSize * 1 * 1);
}

template <typename K_T, typename S_T>
__aicore__ inline void GatherTopm<K_T, S_T>::Process()
{
    uint32_t coreIdx = GetBlockIdx();
    // Split work across cores: each core handles a contiguous range of (batch, topm_pos) pairs
    uint32_t elementsPerCore = (totalGatherElements + usedCoreNum - 1) / usedCoreNum;
    uint32_t start = coreIdx * elementsPerCore;
    uint32_t end = Min(start + elementsPerCore, totalGatherElements);

    if (start >= end) {
        return;
    }

    // Local buffer for one key vector: headDim * sizeof(K_T)
    // Use DataCopy with single-element granularity
    uint32_t vecBytes = headDim * sizeof(K_T);
    uint32_t scaleBytes = sizeof(S_T);

    for (uint32_t i = start; i < end; i++) {
        uint32_t batchIdx = i / topmCount;
        uint32_t topmPos = i % topmCount;
        int32_t topmIdx = topmIdxsGm.GetValue(batchIdx * topmCount + topmPos);

        if (topmIdx < 0) {
            continue;
        }

        uint32_t logicalBlock = static_cast<uint32_t>(topmIdx) / blockSize;
        uint32_t offset = static_cast<uint32_t>(topmIdx) % blockSize;

        uint32_t physBlock = static_cast<uint32_t>(
            blockTableGm.GetValue(batchIdx * maxBlockNumPerBatch + logicalBlock));

        // Source: key[physBlock, offset, 0, :]
        uint32_t srcVecOffset = physBlock * blockSize * headDim + offset * headDim;
        uint32_t srcScaleOffset = physBlock * blockSize + offset;

        // Destination in block layout:
        // outBlockIdx = topmPos / blockSize
        // outPos = topmPos % blockSize
        uint32_t outBlockIdx = topmPos / blockSize;
        uint32_t outPos = topmPos % blockSize;
        uint32_t globalOutBlock = batchIdx * numGatherBlocksPerBatch + outBlockIdx;
        uint32_t dstVecOffset = globalOutBlock * blockSize * headDim + outPos * headDim;
        uint32_t dstScaleOffset = globalOutBlock * blockSize + outPos;

        // Copy key vector
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = vecBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        AscendC::DataCopy(gatheredKeyGm[dstVecOffset], keyGm[srcVecOffset], copyParams);

        // Copy scale
        copyParams.blockLen = scaleBytes;
        AscendC::DataCopy(gatheredScaleGm[dstScaleOffset], keyScaleGm[srcScaleOffset], copyParams);
    }
}

}  // namespace GatherTopmKernel
#endif
