#ifndef VLLM_GATHER_TOPM_TEMPLATE_TILING_KEY_H
#define VLLM_GATHER_TOPM_TEMPLATE_TILING_KEY_H

#include "kernel_tiling/kernel_tiling.h"

#define GET_GATHER_TOPM_TILING_KEY(K_T, OUT_T) \
    (K_T << 0 | OUT_T << 4)

#endif
