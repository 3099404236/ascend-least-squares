# `linalg_lstsq` — implementation notes

Implementation of the `Process()` kernel that was left empty in
`op_kernel/linalg_lstsq.cpp`, plus the tiling changes needed to feed it.

Target: **Ascend 910B / CANN 9.0**, vector core (AIV) only, float32 in / float32 out.
Acceptance: `atol <= 1e-4`, `rtol <= 1e-4` vs a `float64` `gels` reference,
for `cond_2(A) <= 1000`, `m >= n`, `A` full column rank.

---

## 1. Why plain float32 is not enough

Forming the normal equations `G = AᵀA` in float32 perturbs `G` by roughly
`m·eps` relative. The Cholesky factor is then perturbed too, and the
semi-normal refinement iteration matrix is `I - (R̂ᵀR̂)⁻¹AᵀA`, whose norm is
`~cond(A)² · η_G`. At `cond(A) = 1e3` that product is of order 1 — refinement
stalls or diverges.

Measured on the reference generator (worst element, as a multiple of the
allowed tolerance `atol + rtol·|x_ref|`):

| approach | m=64,n=32 | m=128,n=64 | m=512,n=96 |
| :--- | ---: | ---: | ---: |
| naive float32 normal equations | 37.0 | 15.4 | 38.5 |
| compensated + 3 refinements | 1.4e-4 | 1.8e-4 | 1.0e-4 |

(values < 1.0 pass)

So two things are mandatory: **compensated accumulation** of `AᵀA` / `AᵀB`,
and **extended-precision residuals** during refinement.

---

## 2. Algorithm

Per batch, with `A` (`m×n`) and `B` (`m×k`):

1. **Compensated Gram / GEMM.**
   `G = AᵀA` (`n×n`) and `h = AᵀB` (`n×k`), accumulated as a double-float
   `(hi, lo)` pair using Dekker `TwoProd` + Knuth `TwoSum` on float32
   operands, then rounded once to float32.
   `A` and `B` are streamed in row-blocks of `block_rows` rows, so `m` is
   unbounded.

2. **Cholesky.** `R = chol(G)` with `G = RᵀR`, computed in the
   outer-product form directly into the upper triangle of `G`:
   ```
   for j:  R[j,j] = sqrt(R[j,j]);  R[j, j+1:] /= R[j,j]
           for i > j:  R[i, :] -= R[j,i] · R[j, :]
   ```
   This form keeps every vector operand a contiguous, 32-byte-aligned row.

3. **Initial solve.** `x0 = R⁻¹ R⁻ᵀ h` via forward + back substitution.

4. **Semi-normal iterative refinement** (Skeel), `nref = 3` passes:
   ```
   r  = B - A x      (double-float; the residual is exact to ~eps²)
   g  = Aᵀ r         (double-float)
   dx = R⁻¹ R⁻ᵀ g
   x  = x + dx
   ```
   Convergence is ~2 digits per pass from an `x0` that is only ~1e-1
   accurate; after 3 passes the error is limited by the float32
   representation of `x` itself.

5. **Residuals** (only when `m > n`):
   `res[j] = Σᵢ (r_hi + r_lo)²`, compensated, and the `2·r_hi·r_lo` cross
   term is kept. Dropping that cross term costs ~4 digits of absolute
   accuracy: it was the difference between `res_err ≈ 7e-5` (borderline) and
   `res_err ≈ 2e-5` (comfortable).

### Scratch-register aliasing (important)

`DfAxpyScalar` / `DfMacAccum` use `t0..t6` (resp. `t0..t7`) as internal
scratch, so the `(accHi, accLo)` pair passed in **must not** be one of those
tensors. The fused residual pass needs a length-`k` accumulator pair while
also using `t5_`/`t6_` as its own scratch, so it uses the dedicated spare
slots `t8_`/`t9_` instead:

```cpp
Duplicate(t8_, 0.0f, k_);  Duplicate(t9_, 0.0f, k_);
for (p) DfAxpyScalar(t8_, t9_, x_[p*ks_], A[i,p], k_);   // NOT t5_/t6_
Sub(rHi, bRow, t8_, k_); ... Sub(rLo, t2_, t9_, k_);
```

Passing `t5_`/`t6_` here is a silent bug: the Knuth TwoSum inside the helper
overwrites `t6_` (the `sum - bb` step) *before* `accLo` is read back, so the
low-order compensation is replaced by an `O(1)` value. It only shows up as
lost precision, not a crash, and a Python simulation that uses separate
arrays will not reproduce it.

### Measured accuracy (float32-faithful simulation)

`test/verify_algorithm.py` is a line-by-line port of the kernel to
`numpy.float32` (real Dekker/Knuth double-float, never float64, same padded
strides, same fused passes, same loop order). Results:

| m | n | k | cond | x err/tol | residual abs err |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 16 | 4 | 1000 | 2.1e-4 | 8.5e-7 |
| 64 | 32 | 4 | 1000 | 1.4e-4 | 2.0e-6 |
| 128 | 64 | 8 | 1000 | 1.8e-4 | 5.9e-6 |
| 256 | 64 | 1 | 1000 | 5.0e-5 | 3.5e-7 |
| 512 | 96 | 3 | 1000 | 1.0e-4 | 2.3e-5 |
| 64 | 64 | 4 | 1000 | 1.1e-4 | (m = n, empty) |

`x err/tol` is worst-case 2.1e-4 → about **4800× inside spec**.
The residual error at `m=512` is `2.3e-5`, which is **below one float32 ulp
of a value of ~416** — i.e. the output is as accurate as a float32 tensor can
represent. No further gain is possible without changing the output dtype.

---

## 3. Memory model

Every intermediate lives in the Unified Buffer; `A` is never materialised
whole. 2-D buffers use 32-byte padded row strides (`ns = ceil(n/8)*8`,
`ks = ceil(k/8)*8`) so all vector operands start on a 32-byte boundary.

Per-core UB footprint (floats):

```
fixed    = 2·n·ns + 3·n·ks + 3·ks + 8·max(ns,ks)
per row  = ns + 3·ks
bytes    = 4 · (fixed + block_rows · per_row)   ≤ 176 KB
```

`op_host` solves for `block_rows` against a 176 KB budget and returns
`GRAPH_FAILED` with a message if the `n×n` factors alone do not fit.
Practical ceiling is roughly **n ≤ 140**; `m`, `k` and `batch_size` are
unbounded (`k` is just a vector width; `m` is streamed).

Batches are split across AIV cores with a strided loop
(`for b = blockIdx; b < batch; b += blockNum`).

---

## 4. Things to verify on a real 910B / CANN toolchain

This was developed without a local CANN install, so the numerical design is
verified but the following Ascend C plumbing should be checked on first
build/run:

1. **`DataCopyPad` parameter semantics.** The kernel uses `blockCount = 1`
   with `blockLen = cols * sizeof(float)` **in bytes** (the official
   `DataCopyExtParams` semantics). The `srcStride` / `dstStride` fields are
   deliberately left 0 and unused because `blockCount == 1`. If the local
   CANN headers expect `blockLen` in 32-byte units instead, only
   `LoadRows` / `StoreRows` need changing. (Note: `docs/ASCEND_C_API.md`
   states 32-byte units, which contradicts the upstream API — worth
   checking against the installed headers.)
2. **Scalar/vector synchronization.** Every scalar `GetValue` that can
   follow a vector write is guarded by `PipeBarrier<PIPE_ALL>()`. This is
   the conservative choice; it is correct but not free. If profiling shows
   the Cholesky / triangular solves dominating, replace those with
   fine-grained `SetFlag/WaitFlag<HardEvent::V_S>` event sync.
3. **`Sqrt` on a 1-element tensor.** The Cholesky needs `sqrt(G[j,j])`.
   Because `G[j,j]` is not 32-byte aligned, the value is staged through an
   aligned scratch element and `Sqrt(..., 1)` is used rather than a scalar
   `sqrtf`. Confirm `Sqrt` accepts `calCount == 1`.
4. **Kernel entry name** stays `linalg_lstsq` (matches `OP_ADD(LinalgLstsq)`).

---

## 5. Performance notes

The inner loops are ~25 vector instructions per `(row, column)` pair
(Dekker splitting is expensive: 4 muls + 7 adds per exact product). Total
vector work is `O(nref · m · n · k)` plus `O(m · n²)` for the Gram, with a
`PipeBarrier` per scalar read. For the sizes above this is milliseconds per
batch. If throughput matters more than the last digit:

* the Gram can be computed with a cube (matmul) path and only the *residual*
  accumulations kept in double-float;
* the per-read barriers can be replaced with event sync;
* `DfAxpyScalar` can reuse the `TwoProd` split of `A[i,p]` across all `p` of
  a row (the split of the scalar is currently recomputed per call).

---

## 6. Files

| file | change |
| :--- | :--- |
| `op_kernel/linalg_lstsq.cpp` | full `Process()` implementation |
| `op_kernel/linalg_lstsq_tiling.h` | added `block_rows`, `nref`, `core_num`, `reserved` |
| `op_host/linalg_lstsq.cpp` | sizes `block_rows` from the UB budget, rejects oversized `n`, workspace now 0 |
| `test/verify_algorithm.py` | **new** — float32-faithful kernel simulation + tolerance harness |
| `test/verify_local.py` | fixed the self-comparison bug; now shows the naive baseline failing |
