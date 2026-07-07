#include "kernel_operator.h"
#if (__CCE_AICORE__ == 310)
    #include "arch35/gather_topm_kernel.h"
#else
    #include "arch32/gather_topm_kernel.h"
#endif
#include "vllm_gather_topm_template_tiling_key.h"
using namespace GatherTopmKernel;
using namespace optiling::detail;

template <typename K_T, typename S_T>
__global__ __aicore__ void vllm_gather_topm(__gm__ uint8_t *key, __gm__ uint8_t *keyScale,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *topmIdxs,
                                            __gm__ uint8_t *actualSeqLengthsKey,
                                            __gm__ uint8_t *gatheredKey, __gm__ uint8_t *gatheredScale,
                                            __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    GET_TILING_DATA_WITH_STRUCT(GatherTopmTilingData, tiling_data_in, tiling);
    const GatherTopmTilingData *__restrict tiling_data = &tiling_data_in;

    GatherTopm<K_T, S_T> op;
    op.Init(key, keyScale, blockTable, topmIdxs, actualSeqLengthsKey,
            gatheredKey, gatheredScale, tiling_data, &tPipe);
    op.Process();
}

#define GATHER_TOPM_IMPL(K_T, S_T)                                                                     \
    template __global__ __aicore__ void vllm_gather_topm<K_T, S_T>(                                    \
        __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *,                         \
        __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *);

#if (__CCE_AICORE__ == 310)
GATHER_TOPM_IMPL(fp8_e4m3fn_t, float)
#else
GATHER_TOPM_IMPL(int8_t, half)
#endif
