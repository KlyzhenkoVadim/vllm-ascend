import torch
import torch_npu
import numpy as np
import torch.nn as nn
import math
# 从vllm_ascend中导出，这两个很关键
import vllm_ascend.vllm_ascend_C
import vllm_ascend.meta_registration


indexer_k_cache=torch.zeros((6,128,1,128),dtype=torch.float16).npu()
slot_mapping=torch.tensor([[0,0],[0,1]],dtype=torch.int32).npu()
kv=torch.ones((2,128),dtype=torch.float16).npu()


print(f"indexer_k_cache:{indexer_k_cache}")

torch.ops._C_ascend.npu_scatter_nd_update_v2(
                    indexer_k_cache, slot_mapping, kv
                )

print(f"indexer_k_cache:{indexer_k_cache}")

# source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash