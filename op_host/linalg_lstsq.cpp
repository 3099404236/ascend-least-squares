#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_lstsq_tiling.h"
#include "../op_kernel/tiling_key_linalg_lstsq.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        if (num_cores_aiv <= 0) num_cores_aiv = 24;

        uint32_t m = 1;
        uint32_t n = 1;
        uint32_t k = 1;
        uint32_t batch_size = 1;
        uint32_t b_has_batch = 0;

        const gert::StorageShape *shape_A_storage = context->GetInputShape(0);
        const gert::StorageShape *shape_B_storage = context->GetInputShape(1);
        if (shape_A_storage != nullptr && shape_B_storage != nullptr) {
            const auto &shape_A = shape_A_storage->GetStorageShape();
            const auto &shape_B = shape_B_storage->GetStorageShape();
            size_t dimA = shape_A.GetDimNum();
            size_t dimB = shape_B.GetDimNum();
            int64_t total_A = shape_A.GetShapeSize();
            int64_t total_B = shape_B.GetShapeSize();

            if (dimA >= 2) {
                m = static_cast<uint32_t>(shape_A.GetDim(dimA - 2));
                n = static_cast<uint32_t>(shape_A.GetDim(dimA - 1));
                if (total_A > 0 && m > 0 && n > 0) {
                    batch_size = static_cast<uint32_t>(total_A / (m * n));
                }
            }
            if (dimB == dimA) {
                k = static_cast<uint32_t>(shape_B.GetDim(dimB - 1));
                b_has_batch = (batch_size > 1 && total_B > m * k) ? 1 : 0;
            } else if (dimB == dimA - 1 && dimA > 2 && shape_B.GetDim(dimB - 1) == m && total_B == batch_size * m) {
                k = 1;
                b_has_batch = (batch_size > 1) ? 1 : 0;
            } else if (dimB == 2) {
                k = static_cast<uint32_t>(shape_B.GetDim(1));
                b_has_batch = 0;
            } else if (dimB == 1) {
                k = 1;
                b_has_batch = 0;
            } else {
                k = static_cast<uint32_t>((dimB >= 1) ? shape_B.GetDim(dimB - 1) : 1);
                b_has_batch = (batch_size > 1) ? 1 : 0;
            }
        }

        const gert::Tensor *tensor_A = context->GetRequiredInputTensor(0);
        ge::DataType dtype_A = (tensor_A != nullptr) ? tensor_A->GetDataType() : ge::DT_FLOAT;
        uint32_t DT_A = static_cast<uint32_t>(dtype_A);
        ASCENDC_TPL_SEL_PARAM(context, DT_A);

        LinalgLstsqTilingData *tiling = context->GetTilingData<LinalgLstsqTilingData>();
        tiling->m = m;
        tiling->n = n;
        tiling->k = k;
        tiling->batch_size = batch_size;
        tiling->has_residuals = (m > n) ? 1 : 0;
        tiling->b_has_batch = b_has_batch;

        context->SetBlockDim(num_cores_aiv);

        // Workspace allocation: allocate scratch memory for intermediate QR factors and transposes
        size_t ws_floats = static_cast<size_t>(num_cores_aiv) * (m * n + m * k + n * k) * 2;
        if (ws_floats < 2 * 1024 * 1024) ws_floats = 2 * 1024 * 1024;
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = ws_floats * sizeof(float);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *shape_A = context->GetInputShape(0);
        const gert::Shape *shape_B = context->GetInputShape(1);
        gert::Shape *shape_sol = context->GetOutputShape(0);
        gert::Shape *shape_res = context->GetOutputShape(1);
        if (shape_A == nullptr || shape_B == nullptr || shape_sol == nullptr) {
            return GRAPH_SUCCESS;
        }

        size_t dimA = shape_A->GetDimNum();
        size_t dimB = shape_B->GetDimNum();
        if (dimA < 2) return GRAPH_FAILED;

        int64_t m = shape_A->GetDim(dimA - 2);
        int64_t n = shape_A->GetDim(dimA - 1);
        int64_t total_A = shape_A->GetShapeSize();
        int64_t total_B = shape_B->GetShapeSize();
        int64_t batch_size = (total_A > 0 && m > 0 && n > 0) ? (total_A / (m * n)) : 1;
        bool b_is_vector = (dimB == 1) || (dimB == dimA - 1 && shape_B->GetDim(dimB - 1) == m && total_B == batch_size * m);
        int64_t k = b_is_vector ? 1 : ((dimB >= 1) ? shape_B->GetDim(dimB - 1) : 1);

        if (b_is_vector) {
            if (dimA == 2) {
                shape_sol->SetDimNum(1);
                shape_sol->SetDim(0, n);
            } else {
                shape_sol->SetDimNum(dimA - 1);
                for (size_t i = 0; i < dimA - 2; ++i) shape_sol->SetDim(i, shape_A->GetDim(i));
                shape_sol->SetDim(dimA - 2, n);
            }
        } else {
            shape_sol->SetDimNum(dimA);
            for (size_t i = 0; i < dimA - 2; ++i) shape_sol->SetDim(i, shape_A->GetDim(i));
            shape_sol->SetDim(dimA - 2, n);
            shape_sol->SetDim(dimA - 1, k);
        }

        if (shape_res != nullptr) {
            if (m > n) {
                if (dimA == 2) {
                    shape_res->SetDimNum(1);
                    shape_res->SetDim(0, k);
                } else {
                    shape_res->SetDimNum(dimA - 1);
                    for (size_t i = 0; i < dimA - 2; ++i) shape_res->SetDim(i, shape_A->GetDim(i));
                    shape_res->SetDim(dimA - 2, k);
                }
            } else {
                shape_res->SetDimNum(1);
                shape_res->SetDim(0, 0);
            }
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_FLOAT);
        context->SetOutputDataType(1, ge::DT_FLOAT);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class LinalgLstsq : public OpDef {
    public:
        explicit LinalgLstsq(const char *name) : OpDef(name) {
            this->Input("A")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Input("B")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Output("solution")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Output("residuals")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Attr("driver").AttrType(OPTIONAL).String("gels");
            this->Attr("rcond").AttrType(OPTIONAL).Float(-1.0);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LinalgLstsq);
}  // namespace ops
