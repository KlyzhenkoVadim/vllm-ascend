#include "register/op_def_registry.h"

namespace ops {

class VllmGatherTopmInferShape : public OpInferShape {
public:
    explicit VllmGatherTopmInferShape(const char *name) : OpInferShape(name) {}

    ge::graphStatus InferShape(ge::Operator &op) override
    {
        auto keyDesc = op.GetInputDescByName("key");
        auto topmIdxsDesc = op.GetInputDescByName("topm_idxs");

        auto keyShape = keyDesc.GetShape();
        auto topmIdxsShape = topmIdxsDesc.GetShape();

        int64_t batchSize = topmIdxsShape.GetDim(0);
        int64_t topmCount = topmIdxsShape.GetDim(1);
        int64_t blockSize = keyShape.GetDim(1);
        int64_t headDim = keyShape.GetDim(3);

        int64_t numGatherBlocks = (topmCount + blockSize - 1) / blockSize * batchSize;

        // gathered_key: [numGatherBlocks, blockSize, 1, headDim]
        ge::Shape keyOutShape(std::vector<int64_t>{numGatherBlocks, blockSize, 1, headDim});
        op.UpdateOutputDesc("gathered_key", ge::GeTensorDesc(keyOutShape, keyDesc.GetFormat(), keyDesc.GetDataType()));

        // gathered_scale: [numGatherBlocks, blockSize, 1, 1]
        auto keyScaleDesc = op.GetInputDescByName("key_scale");
        ge::Shape scaleOutShape(std::vector<int64_t>{numGatherBlocks, blockSize, 1, 1});
        op.UpdateOutputDesc("gathered_scale",
                            ge::GeTensorDesc(scaleOutShape, keyScaleDesc.GetFormat(), keyScaleDesc.GetDataType()));

        return ge::GRAPH_SUCCESS;
    }
};

OP_ADD(VllmGatherTopmInferShape);

}  // namespace ops
