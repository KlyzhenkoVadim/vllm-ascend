add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        vllm_gather_topm_def.cpp
     )
endif()

add_ops_compile_options(
    OP_NAME VllmGatherTopm
    OPTIONS --cce-auto-sync=off
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE vllm_gather_topm ACLNNTYPE aclnn)
endif()
