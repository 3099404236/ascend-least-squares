/**
 * linalg_lstsq -- Ascend C device kernel (Ascend 910B / CANN 9.0)
 *
 * Computes, for every batch b:
 *     X_b = argmin_X || A_b X - B_b ||_2        (A_b is m x n, m >= n)
 *     residuals_b[j] = || A_b X_b[:, j] - B_b[:, j] ||_2^2   (only when m > n)
 *
 * ---------------------------------------------------------------------------
 * NUMERICAL STRATEGY
 * ---------------------------------------------------------------------------
 * Plain float32 normal equations lose ~cond(A)^2 * eps digits, which at
 * cond(A) = 1e3 is ~0.1 relative -- far outside the 1e-4 budget.  Instead:
 *
 *   1. Form G = A^T A (n x n) and h = A^T B (n x k) with *double-float*
 *      (hi/lo, Dekker TwoProd + Knuth TwoSum) compensated accumulation.
 *      Without this the Cholesky factor is perturbed enough that the
 *      refinement iteration matrix has spectral radius ~1 and refinement
 *      stalls.
 *   2. R = chol(G), G = R^T R   (outer-product form -> contiguous vector ops).
 *   3. x0 = R^-1 R^-T h.
 *   4. Semi-normal-equations iterative refinement (Skeel), NREF passes:
 *         r  = B - A x        (double-float, so the residual is exact to ~eps^2)
 *         g  = A^T r          (double-float)
 *         dx = R^-1 R^-T g
 *         x  = x + dx
 *      Measured convergence: ~2 digits per pass from a x0 that is already
 *      ~1e-1 accurate; 3 passes land at ~1e-7 (prototype, cond = 1e3).
 *   5. residuals = sum_i (r_hi + r_lo)^2, compensated; the 2*r_hi*r_lo cross
 *      term is kept because it is ~eps * |r|^2 and dropping it costs ~4 digits
 *      of absolute accuracy in the output.
 *
 * ---------------------------------------------------------------------------
 * MEMORY / EXECUTION MODEL
 * ---------------------------------------------------------------------------
 * * Vector core (AIV) only.  Batches are split across GetBlockNum() cores.
 * * A and B are streamed in row-blocks of `block_rows` rows; A is never held
 *   whole in UB, so m is unbounded.
 * * All 2-D UB buffers use 32-byte padded row strides (ns / ks) so that every
 *   vector operand starts on a 32-byte boundary.
 * * Requires n to fit: 2*n*ns + 3*n*ks + ... floats.  The host sizes
 *   `block_rows` against the UB budget and rejects oversized n.
 */

#include "kernel_operator.h"
#include "linalg_lstsq_tiling.h"
#include "tiling_key_linalg_lstsq.h"

using namespace AscendC;

namespace {
constexpr float kSplit = 4097.0f;  // 2^12 + 1 : Dekker splitter for 24-bit float32

__aicore__ inline uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }
}  // namespace

template <typename DT_A>
class KernelLinalgLstsq {
public:
    __aicore__ inline KernelLinalgLstsq() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR solution, GM_ADDR residuals,
                                GM_ADDR workspace, const LinalgLstsqTilingData &tiling) {
        (void)workspace;
        m_ = tiling.m;
        n_ = tiling.n;
        k_ = tiling.k;
        batch_ = tiling.batch_size;
        hasResiduals_ = tiling.has_residuals;
        bHasBatch_ = tiling.b_has_batch;
        blockRows_ = tiling.block_rows;
        nref_ = tiling.nref;

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_A *>(a));
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_A *>(b));
        solGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_A *>(solution));
        if (hasResiduals_ != 0) {
            resGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_A *>(residuals));
        }

        // 32-byte padded row strides (8 float32 per 32B block)
        ns_ = (n_ + 7u) / 8u * 8u;
        ks_ = (k_ + 7u) / 8u * 8u;
        tmpLen_ = ns_ > ks_ ? ns_ : ks_;

        pipe_.InitBuffer(gBuf_, n_ * ns_ * sizeof(float));       // G, later R in upper tri
        pipe_.InitBuffer(gLoBuf_, n_ * ns_ * sizeof(float));     // G compensation
        pipe_.InitBuffer(xBuf_, n_ * ks_ * sizeof(float));       // h, later X
        pipe_.InitBuffer(ggBuf_, n_ * ks_ * sizeof(float));      // refinement rhs g
        pipe_.InitBuffer(ggLoBuf_, n_ * ks_ * sizeof(float));    // g compensation
        pipe_.InitBuffer(aBuf_, blockRows_ * ns_ * sizeof(float));
        pipe_.InitBuffer(bBuf_, blockRows_ * ks_ * sizeof(float));
        pipe_.InitBuffer(rHiBuf_, blockRows_ * ks_ * sizeof(float));
        pipe_.InitBuffer(rLoBuf_, blockRows_ * ks_ * sizeof(float));
        pipe_.InitBuffer(resBuf_, 3u * ks_ * sizeof(float));
        pipe_.InitBuffer(tmpBuf_, 10u * tmpLen_ * sizeof(float));

        g_ = gBuf_.Get<float>();
        gLo_ = gLoBuf_.Get<float>();
        x_ = xBuf_.Get<float>();
        gg_ = ggBuf_.Get<float>();
        ggLo_ = ggLoBuf_.Get<float>();
        aBlk_ = aBuf_.Get<float>();
        bBlk_ = bBuf_.Get<float>();
        rHiBlk_ = rHiBuf_.Get<float>();
        rLoBlk_ = rLoBuf_.Get<float>();

        LocalTensor<float> resAll = resBuf_.Get<float>();
        resHi_ = resAll;
        resLo_ = resAll[ks_];
        resCr_ = resAll[2u * ks_];

        LocalTensor<float> tmpAll = tmpBuf_.Get<float>();
        t0_ = tmpAll;
        t1_ = tmpAll[tmpLen_];
        t2_ = tmpAll[2u * tmpLen_];
        t3_ = tmpAll[3u * tmpLen_];
        t4_ = tmpAll[4u * tmpLen_];
        t5_ = tmpAll[5u * tmpLen_];
        t6_ = tmpAll[6u * tmpLen_];
        t7_ = tmpAll[7u * tmpLen_];
        t8_ = tmpAll[8u * tmpLen_];
        t9_ = tmpAll[9u * tmpLen_];
    }

    __aicore__ inline void Process() {
        if (m_ == 0u || n_ == 0u || k_ == 0u || batch_ == 0u) return;
        uint32_t coreNum = GetBlockNum();
        if (coreNum == 0u) coreNum = 1u;
        const uint32_t coreId = GetBlockIdx();
        for (uint32_t b = coreId; b < batch_; b += coreNum) {
            ProcessBatch(b);
        }
    }

private:
    // =====================================================================
    // double-float primitives (vectorised, float32 only)
    // =====================================================================

    // (hi, lo) = exact split of a, i.e. a == hi + lo.   t is scratch.
    __aicore__ inline void DfSplit(LocalTensor<float> &hi, LocalTensor<float> &lo,
                                   const LocalTensor<float> &a, LocalTensor<float> &t, uint32_t L) {
        Muls(t, a, kSplit, L);
        Sub(hi, t, a, L);
        Sub(hi, t, hi, L);
        Sub(lo, a, hi, L);
    }

    // acc += a * s,  a = vector of length L, s = float32 scalar, double-float accurate.
    __aicore__ inline void DfAxpyScalar(LocalTensor<float> &accHi, LocalTensor<float> &accLo,
                                        const LocalTensor<float> &a, float s, uint32_t L) {
        if (L == 0u) return;
        const float cs = s * kSplit;
        const float shi = cs - (cs - s);
        const float slo = s - shi;

        DfSplit(t0_, t1_, a, t2_, L);          // t0 = ahi, t1 = alo
        Muls(t2_, a, s, L);                    // t2 = p = fl(a*s)
        // e = ((ahi*shi - p) + ahi*slo + alo*shi) + alo*slo
        Muls(t3_, t0_, shi, L);
        Sub(t3_, t3_, t2_, L);
        Muls(t4_, t0_, slo, L);
        Add(t3_, t3_, t4_, L);
        Muls(t4_, t1_, shi, L);
        Add(t3_, t3_, t4_, L);
        Muls(t4_, t1_, slo, L);
        Add(t3_, t3_, t4_, L);
        // exact Knuth TwoSum on (accHi, p): s == sum + err
        Add(t4_, accHi, t2_, L);               // sum
        Sub(t5_, t4_, accHi, L);               // bb = sum - a
        Sub(t6_, t4_, t5_, L);                 // sum - bb
        Sub(t6_, accHi, t6_, L);               // a - (sum - bb)
        Sub(t0_, t2_, t5_, L);                 // b - bb
        Add(t6_, t6_, t0_, L);                 // TwoSum error
        Add(accLo, accLo, t6_, L);
        Add(accLo, accLo, t3_, L);             // + TwoProd error
        Muls(accHi, t4_, 1.0f, L);             // accHi = sum
    }

    // acc += a * b (elementwise, both vectors), double-float accurate.
    __aicore__ inline void DfMacAccum(LocalTensor<float> &accHi, LocalTensor<float> &accLo,
                                      const LocalTensor<float> &a, const LocalTensor<float> &b,
                                      uint32_t L) {
        if (L == 0u) return;
        DfSplit(t0_, t1_, a, t2_, L);          // ahi, alo
        DfSplit(t3_, t4_, b, t2_, L);          // bhi, blo
        Mul(t2_, a, b, L);                     // p = fl(a*b)
        Mul(t5_, t0_, t3_, L);                 // ahi*bhi
        Sub(t5_, t5_, t2_, L);                 //   - p
        Mul(t6_, t0_, t4_, L);                 // ahi*blo
        Add(t5_, t5_, t6_, L);
        Mul(t6_, t1_, t3_, L);                 // alo*bhi
        Add(t5_, t5_, t6_, L);
        Mul(t6_, t1_, t4_, L);                 // alo*blo
        Add(t5_, t5_, t6_, L);                 // e
        // exact Knuth TwoSum on (accHi, p)
        Add(t6_, accHi, t2_, L);               // sum
        Sub(t7_, t6_, accHi, L);               // bb = sum - a
        Sub(t0_, t6_, t7_, L);                 // sum - bb
        Sub(t0_, accHi, t0_, L);               // a - (sum - bb)
        Sub(t3_, t2_, t7_, L);                 // b - bb
        Add(t0_, t0_, t3_, L);                 // TwoSum error
        Add(accLo, accLo, t0_, L);
        Add(accLo, accLo, t5_, L);             // + TwoProd error
        Muls(accHi, t6_, 1.0f, L);
    }

    // =====================================================================
    // global <-> UB movement (per row, so any `cols` is legal)
    // =====================================================================
    __aicore__ inline void LoadRows(LocalTensor<float> &dst, GlobalTensor<DT_A> &src,
                                    uint32_t srcElemOff, uint32_t rows, uint32_t cols,
                                    uint32_t dstStride) {
        if (rows == 0u || cols == 0u) return;
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = cols * static_cast<uint32_t>(sizeof(float));  // bytes
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPadExtParams<float> pad;
        pad.isPad = false;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            DataCopyPad(dst[r * dstStride], src[srcElemOff + r * cols], params, pad);
        }
    }

    __aicore__ inline void StoreRows(GlobalTensor<DT_A> &dst, uint32_t dstElemOff,
                                     LocalTensor<float> &src, uint32_t rows, uint32_t cols,
                                     uint32_t srcStride) {
        if (rows == 0u || cols == 0u) return;
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = cols * static_cast<uint32_t>(sizeof(float));  // bytes
        params.srcStride = 0;
        params.dstStride = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            DataCopyPad(dst[dstElemOff + r * cols], src[r * srcStride], params);
        }
    }

    // =====================================================================
    // Cholesky: upper R with G = R^T R, in place in the upper triangle of g_.
    // Outer-product form keeps every vector operand row-aligned and contiguous.
    // =====================================================================
    __aicore__ inline void Cholesky() {
        for (uint32_t j = 0; j < n_; ++j) {
            PipeBarrier<PIPE_ALL>();  // V -> S: make the updated G[j,j] visible
            t0_.SetValue(0, g_.GetValue(j * ns_ + j));
            PipeBarrier<PIPE_ALL>();  // S -> V
            Sqrt(t1_, t0_, 1);
            PipeBarrier<PIPE_ALL>();  // V -> S
            const float d = t1_.GetValue(0);
            const float invd = (d > 0.0f) ? (1.0f / d) : 0.0f;
            // normalise the whole row (columns < j hold unused lower-triangle junk)
            Muls(g_[j * ns_], g_[j * ns_], invd, n_);
            PipeBarrier<PIPE_ALL>();  // V -> S: row j normalised, safe to read scalars
            for (uint32_t i = j + 1; i < n_; ++i) {
                PipeBarrier<PIPE_ALL>();  // V -> S before each scalar read
                const float rji = g_.GetValue(j * ns_ + i);
                Muls(t1_, g_[j * ns_], rji, n_);
                Sub(g_[i * ns_], g_[i * ns_], t1_, n_);
            }
        }
    }

    // R^T y = b  (forward substitution), in place on a length-k row layout.
    __aicore__ inline void SolveLowerT(LocalTensor<float> &b) {
        for (uint32_t i = 0; i < n_; ++i) {
            Duplicate(t5_, 0.0f, k_);
            for (uint32_t p = 0; p < i; ++p) {
                PipeBarrier<PIPE_ALL>();  // V -> S before the scalar read of R[p,i]
                const float rpi = g_.GetValue(p * ns_ + i);
                Muls(t1_, b[p * ks_], rpi, k_);
                Add(t5_, t5_, t1_, k_);
            }
            PipeBarrier<PIPE_ALL>();  // V -> S before the scalar read of R[i,i]
            const float rii = g_.GetValue(i * ns_ + i);
            Sub(t1_, b[i * ks_], t5_, k_);
            Muls(b[i * ks_], t1_, (rii != 0.0f) ? (1.0f / rii) : 0.0f, k_);
        }
    }

    // R x = b  (back substitution).
    __aicore__ inline void SolveUpper(LocalTensor<float> &b) {
        for (int32_t i = static_cast<int32_t>(n_) - 1; i >= 0; --i) {
            const uint32_t ii = static_cast<uint32_t>(i);
            Duplicate(t5_, 0.0f, k_);
            for (uint32_t p = ii + 1u; p < n_; ++p) {
                PipeBarrier<PIPE_ALL>();  // V -> S before the scalar read of R[i,p]
                const float rip = g_.GetValue(ii * ns_ + p);
                Muls(t1_, b[p * ks_], rip, k_);
                Add(t5_, t5_, t1_, k_);
            }
            PipeBarrier<PIPE_ALL>();  // V -> S before the scalar read of R[i,i]
            const float rii = g_.GetValue(ii * ns_ + ii);
            Sub(t1_, b[ii * ks_], t5_, k_);
            Muls(b[ii * ks_], t1_, (rii != 0.0f) ? (1.0f / rii) : 0.0f, k_);
        }
    }

    // =====================================================================
    // fused pass:  r = B - A X   (double-float)  and  g += A^T r
    //              optionally also accumulate residuals = sum_i r_i^2
    // =====================================================================
    __aicore__ inline void ResidualAndGram(uint32_t aOff, uint32_t bOff, bool withResiduals) {
        if (withResiduals) {
            Duplicate(resHi_, 0.0f, ks_);
            Duplicate(resLo_, 0.0f, ks_);
            Duplicate(resCr_, 0.0f, ks_);
        }
        for (uint32_t r0 = 0; r0 < m_; r0 += blockRows_) {
            const uint32_t rb = umin(blockRows_, m_ - r0);
            LoadRows(aBlk_, aGm_, aOff + r0 * n_, rb, n_, ns_);
            LoadRows(bBlk_, bGm_, bOff + r0 * k_, rb, k_, ks_);
            PipeBarrier<PIPE_ALL>();  // MTE2 -> S/V

            for (uint32_t rr = 0; rr < rb; ++rr) {
                LocalTensor<float> rHi = rHiBlk_[rr * ks_];
                LocalTensor<float> rLo = rLoBlk_[rr * ks_];
                LocalTensor<float> bRow = bBlk_[rr * ks_];

                // acc = sum_p A[i,p] * X[p,:]
                // NB: use t8_/t9_ as the accumulator pair.  DfAxpyScalar uses
                // t0..t6 internally, so passing t5_/t6_ here would let the helper
                // clobber accLo before it is read back.
                Duplicate(t8_, 0.0f, k_);
                Duplicate(t9_, 0.0f, k_);
                for (uint32_t p = 0; p < n_; ++p) {
                    PipeBarrier<PIPE_ALL>();  // V -> S before the scalar read of A[i,p]
                    const float av = aBlk_.GetValue(rr * ns_ + p);
                    DfAxpyScalar(t8_, t9_, x_[p * ks_], av, k_);
                }
                // r = b - acc, kept as a double-float pair
                Sub(rHi, bRow, t8_, k_);
                Muls(t0_, t8_, -1.0f, k_);          // b = -acc_hi
                Sub(t1_, rHi, bRow, k_);            // bb = s - a
                Sub(t2_, rHi, t1_, k_);             // s - bb
                Sub(t2_, bRow, t2_, k_);            // a - (s - bb)
                Sub(t1_, t0_, t1_, k_);             // b - bb
                Add(t2_, t2_, t1_, k_);             // TwoSum error
                Sub(rLo, t2_, t9_, k_);             // r_lo = err - acc_lo

                if (withResiduals) {
                    // res += r^2 (exact) + 2*r_hi*r_lo (cross term)
                    DfMacAccum(resHi_, resLo_, rHi, rHi, k_);
                    Muls(t0_, rHi, 2.0f, k_);
                    Mul(t0_, t0_, rLo, k_);
                    Add(resCr_, resCr_, t0_, k_);
                }

                // g[p,:] += A[i,p] * r[i,:]
                for (uint32_t p = 0; p < n_; ++p) {
                    PipeBarrier<PIPE_ALL>();  // V -> S before the scalar read of A[i,p]
                    const float av = aBlk_.GetValue(rr * ns_ + p);
                    DfAxpyScalar(gg_[p * ks_], ggLo_[p * ks_], rHi, av, k_);
                    DfAxpyScalar(gg_[p * ks_], ggLo_[p * ks_], rLo, av, k_);
                }
            }
        }
    }

    // =====================================================================
    // per-batch driver
    // =====================================================================
    __aicore__ inline void ProcessBatch(uint32_t b) {
        const uint32_t aOff = b * m_ * n_;
        const uint32_t bOff = (bHasBatch_ != 0u ? b : 0u) * m_ * k_;
        const uint32_t xOff = b * n_ * k_;

        // ---- 1. G = A^T A, h = A^T B, double-float accumulation -------------
        Duplicate(g_, 0.0f, n_ * ns_);
        Duplicate(gLo_, 0.0f, n_ * ns_);
        Duplicate(x_, 0.0f, n_ * ks_);
        Duplicate(ggLo_, 0.0f, n_ * ks_);   // reused as h_lo

        for (uint32_t r0 = 0; r0 < m_; r0 += blockRows_) {
            const uint32_t rb = umin(blockRows_, m_ - r0);
            LoadRows(aBlk_, aGm_, aOff + r0 * n_, rb, n_, ns_);
            LoadRows(bBlk_, bGm_, bOff + r0 * k_, rb, k_, ks_);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t rr = 0; rr < rb; ++rr) {
                LocalTensor<float> aRow = aBlk_[rr * ns_];
                LocalTensor<float> bRow = bBlk_[rr * ks_];
                for (uint32_t p = 0; p < n_; ++p) {
                    PipeBarrier<PIPE_ALL>();  // V -> S before the scalar read of A[i,p]
                    const float av = aBlk_.GetValue(rr * ns_ + p);
                    DfAxpyScalar(g_[p * ns_], gLo_[p * ns_], aRow, av, n_);
                    DfAxpyScalar(x_[p * ks_], ggLo_[p * ks_], bRow, av, k_);
                }
            }
        }
        // fold compensation, then h lives in x_
        Add(g_, g_, gLo_, n_ * ns_);
        Add(x_, x_, ggLo_, n_ * ks_);
        PipeBarrier<PIPE_ALL>();

        // ---- 2. R = chol(G), then x = R^-1 R^-T h --------------------------
        Cholesky();
        PipeBarrier<PIPE_ALL>();
        SolveLowerT(x_);
        SolveUpper(x_);
        PipeBarrier<PIPE_ALL>();

        // ---- 3. semi-normal iterative refinement ---------------------------
        for (uint32_t it = 0; it < nref_; ++it) {
            Duplicate(gg_, 0.0f, n_ * ks_);
            Duplicate(ggLo_, 0.0f, n_ * ks_);
            ResidualAndGram(aOff, bOff, false);
            Add(gg_, gg_, ggLo_, n_ * ks_);
            PipeBarrier<PIPE_ALL>();
            SolveLowerT(gg_);
            SolveUpper(gg_);
            Add(x_, x_, gg_, n_ * ks_);
            PipeBarrier<PIPE_ALL>();
        }

        // ---- 4. residuals of the final x -----------------------------------
        if (hasResiduals_ != 0u) {
            ResidualAndGram(aOff, bOff, true);
            Add(resHi_, resHi_, resLo_, k_);
            Add(resHi_, resHi_, resCr_, k_);
            StoreRows(resGm_, b * k_, resHi_, 1u, k_, ks_);
        }

        // ---- 5. store solution ---------------------------------------------
        StoreRows(solGm_, xOff, x_, n_, k_, ks_);
    }

    // ---- buffers ---------------------------------------------------------
    TPipe pipe_;
    TBuf<TPosition::VECCALC> gBuf_, gLoBuf_, xBuf_, ggBuf_, ggLoBuf_;
    TBuf<TPosition::VECCALC> aBuf_, bBuf_, rHiBuf_, rLoBuf_, resBuf_, tmpBuf_;
    GlobalTensor<DT_A> aGm_, bGm_, solGm_, resGm_;

    LocalTensor<float> g_, gLo_, x_, gg_, ggLo_;
    LocalTensor<float> aBlk_, bBlk_, rHiBlk_, rLoBlk_;
    LocalTensor<float> resHi_, resLo_, resCr_;
    LocalTensor<float> t0_, t1_, t2_, t3_, t4_, t5_, t6_, t7_, t8_, t9_;

    uint32_t m_ = 0, n_ = 0, k_ = 0, batch_ = 0;
    uint32_t hasResiduals_ = 0, bHasBatch_ = 0;
    uint32_t blockRows_ = 1, nref_ = 3;
    uint32_t ns_ = 0, ks_ = 0, tmpLen_ = 0;
};

extern "C" __global__ __aicore__ void linalg_lstsq(GM_ADDR a, GM_ADDR b, GM_ADDR solution,
                                                   GM_ADDR residuals, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLinalgLstsq<float> op;
    op.Init(a, b, solution, residuals, workspace, tiling_data);
    op.Process();
}
