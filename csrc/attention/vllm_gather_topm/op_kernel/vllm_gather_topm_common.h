#ifndef VLLM_GATHER_TOPM_COMMON_H
#define VLLM_GATHER_TOPM_COMMON_H

using namespace AscendC;

namespace GatherTopmCommon {

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

struct RunInfo {
    uint32_t batchIdx;
    uint32_t topmCount;
    uint32_t blockSize;
    uint32_t headDim;
    uint32_t numGatherBlocks;
    uint32_t maxBlockNumPerBatch;
};

struct ConstInfo {
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_128B = 128;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 65536;

    uint32_t batchSize;
    uint32_t topmCount;
    uint32_t blockSize;
    uint32_t headDim;
    uint32_t maxBlockNumPerBatch;
    uint32_t numGatherBlocksPerBatch;
    uint32_t totalGatherBlocks;
    uint32_t stride;
    uint32_t scaleStride;
};

}  // namespace GatherTopmCommon
#endif
