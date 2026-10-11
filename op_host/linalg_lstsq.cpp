#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "error_util.h"

#include "../op_kernel/linalg_lstsq_tiling.h"
#include "../op_kernel/tiling_key_linalg_lstsq.h"

namespace {
// Usable Unified Buffer per AIV core on Ascend 910B (~192 KB) minus headroom.
constexpr size_t kUbBudgetBytes = 176 * 1024;
// Number of semi-normal iterative-refinement passes.  3 is enough to take a
// x0 with ~1e-1 relative error down to ~1e-7 at cond(A)=1e3 (see docs).
constexpr uint32_t kRefineIters = 3;
}  // namespace

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
        if (m == 0) m = 1;
        if (n == 0) n = 1;
        if (k == 0) k = 1;
        if (batch_size == 0) batch_size = 1;

        // ------------------------------------------------------------------
        // Size the row-block so the kernel's UB footprint fits the budget.
        //   fixed (floats) = 2*n*ns + 3*n*ks + 3*ks + 10*max(ns,ks)
        //   per-row (floats) = ns + 3*ks
        // ------------------------------------------------------------------
        const uint32_t ns = (n + 7u) / 8u * 8u;
        const uint32_t ks = (k + 7u) / 8u * 8u;
        const uint32_t tmp_len = (ns > ks) ? ns : ks;

        const size_t fixed_floats =
            static_cast<size_t>(2) * n * ns + static_cast<size_t>(3) * n * ks +
            static_cast<size_t>(3) * ks + static_cast<size_t>(10) * tmp_len;
        const size_t per_row_floats = static_cast<size_t>(ns) + 3u * static_cast<size_t>(ks);
        const size_t budget_floats = kUbBudgetBytes / sizeof(float);

        if (fixed_floats >= budget_floats) {
            OP_LOGE(context->GetNodeName(),
                    "linalg_lstsq: n=%u too large for the AIV Unified Buffer "
                    "(need %zu floats for the n x n Gram factor alone, budget %zu). "
                    "Reduce n or tile the Cholesky.",
                    n, fixed_floats, budget_floats);
            return ge::GRAPH_FAILED;
        }

        size_t block_rows = (budget_floats - fixed_floats) / per_row_floats;
        if (block_rows > 512) block_rows = 512;
        block_rows = (block_rows / 8) * 8;   // keep it tidy / 32B friendly
        if (block_rows < 8) block_rows = 8;
        if (block_rows > m) block_rows = m;

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
        tiling->block_rows = static_cast<uint32_t>(block_rows);
        tiling->nref = kRefineIters;
        tiling->core_num = static_cast<uint32_t>(num_cores_aiv);
        tiling->reserved = 0;

        context->SetBlockDim(num_cores_aiv);

        // The kernel streams A/B straight from GM and keeps every intermediate in
        // UB, so no user workspace is required.
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
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
