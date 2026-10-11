#!/usr/bin/env python3
"""
Numerical verification for the `linalg_lstsq` Ascend C kernel.

This is NOT a re-implementation of the maths "in spirit" -- it is a
line-by-line port of op_kernel/linalg_lstsq.cpp to numpy.float32:

  * every arithmetic step is float32 (numpy.float32),
  * extended precision is genuine Dekker/Knuth TwoProd/TwoSum on float32
    operands (a true hi/lo "double-float", never float64),
  * the same padded 32-byte row strides (ns / ks) as the kernel,
  * the same fused residual+Gram pass, the same loop order, the same
    3 semi-normal refinement passes.

If this script passes, the *algorithm* is sound; what remains untested
without a 910B + CANN toolchain is the Ascend C API plumbing itself.

Run:  python verify_algorithm.py
"""
import numpy as np

try:
    import scipy.linalg
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

F = np.float32
SPLIT = F(4097.0)          # 2^12 + 1
ATOL = 1e-4
RTOL = 1e-4


# ---------------------------------------------------------------------------
# double-float primitives (exactly what the kernel's DfSplit / TwoSum do)
# ---------------------------------------------------------------------------
def two_prod(a, b):
    a = np.asarray(a, dtype=F)
    b = np.asarray(b, dtype=F)
    p = (a * b).astype(F)
    ca = (a * SPLIT).astype(F)
    ahi = (ca - (ca - a).astype(F)).astype(F)
    alo = (a - ahi).astype(F)
    cb = (b * SPLIT).astype(F)
    bhi = (cb - (cb - b).astype(F)).astype(F)
    blo = (b - bhi).astype(F)
    e = (ahi * bhi).astype(F)
    e = (e - p).astype(F)
    e = (e + (ahi * blo).astype(F)).astype(F)
    e = (e + (alo * bhi).astype(F)).astype(F)
    e = (e + (alo * blo).astype(F)).astype(F)
    return p, e


def two_sum(a, b):
    a = np.asarray(a, dtype=F)
    b = np.asarray(b, dtype=F)
    s = (a + b).astype(F)
    bb = (s - a).astype(F)
    e = ((a - (s - bb).astype(F)).astype(F) + (b - bb).astype(F)).astype(F)
    return s, e


def df_axpy_scalar(acc_hi, acc_lo, a_vec, s):
    """Mirror of KernelLinalgLstsq::DfAxpyScalar -- acc += a_vec * s."""
    # split the scalar
    cs = F(s * SPLIT)
    s_hi = F(cs - F(cs - s))
    s_lo = F(s - s_hi)
    # split the vector
    t = (a_vec * SPLIT).astype(F)
    a_hi = (t - (t - a_vec).astype(F)).astype(F)
    a_lo = (a_vec - a_hi).astype(F)
    # p = fl(a*s)
    p = (a_vec * s).astype(F)
    # e = ((ahi*shi - p) + ahi*slo + alo*shi) + alo*slo
    e = (a_hi * s_hi).astype(F)
    e = (e - p).astype(F)
    e = (e + (a_hi * s_lo).astype(F)).astype(F)
    e = (e + (a_lo * s_hi).astype(F)).astype(F)
    e = (e + (a_lo * s_lo).astype(F)).astype(F)
    # exact Knuth TwoSum on (acc_hi, p)
    s_sum = (acc_hi + p).astype(F)
    bb = (s_sum - acc_hi).astype(F)
    u = (s_sum - bb).astype(F)
    u = (acc_hi - u).astype(F)
    v = (p - bb).astype(F)
    u = (u + v).astype(F)
    acc_lo = (acc_lo + u).astype(F)
    acc_lo = (acc_lo + e).astype(F)
    return s_sum, acc_lo


def df_mac_accum(acc_hi, acc_lo, a, b):
    """Mirror of DfMacAccum -- acc += a*b elementwise, double-float."""
    t = (a * SPLIT).astype(F)
    a_hi = (t - (t - a).astype(F)).astype(F)
    a_lo = (a - a_hi).astype(F)
    t = (b * SPLIT).astype(F)
    b_hi = (t - (t - b).astype(F)).astype(F)
    b_lo = (b - b_hi).astype(F)
    p = (a * b).astype(F)
    e = (a_hi * b_hi).astype(F)
    e = (e - p).astype(F)
    e = (e + (a_hi * b_lo).astype(F)).astype(F)
    e = (e + (a_lo * b_hi).astype(F)).astype(F)
    e = (e + (a_lo * b_lo).astype(F)).astype(F)
    # exact Knuth TwoSum on (acc_hi, p)
    s_sum = (acc_hi + p).astype(F)
    bb = (s_sum - acc_hi).astype(F)
    u = (s_sum - bb).astype(F)
    u = (acc_hi - u).astype(F)
    v = (p - bb).astype(F)
    u = (u + v).astype(F)
    acc_lo = (acc_lo + u).astype(F)
    acc_lo = (acc_lo + e).astype(F)
    return s_sum, acc_lo


# ---------------------------------------------------------------------------
# kernel port
# ---------------------------------------------------------------------------
class LstsqKernelSim:
    def __init__(self, m, n, k, block_rows=64, nref=3):
        self.m, self.n, self.k = m, n, k
        self.block_rows = max(1, min(block_rows, m))
        self.nref = nref
        self.ns = (n + 7) // 8 * 8          # padded row stride, n-wide
        self.ks = (k + 7) // 8 * 8          # padded row stride, k-wide

    # -- Cholesky (outer-product form), in place on the upper triangle of g --
    def _cholesky(self, g):
        n = self.n
        for j in range(n):
            d = F(np.sqrt(np.float32(g[j, j])))
            invd = F(1.0 / d) if d > 0 else F(0.0)
            g[j, :] = (g[j, :] * invd).astype(F)      # whole row (junk below j)
            for i in range(j + 1, n):
                rji = g[j, i]
                g[i, :] = (g[i, :] - (g[j, :] * rji).astype(F)).astype(F)

    def _solve_lower_t(self, g, b):
        n, k = self.n, self.k
        for i in range(n):
            acc = np.zeros(k, dtype=F)
            for p in range(i):
                acc = (acc + (b[p, :] * g[p, i]).astype(F)).astype(F)
            b[i, :] = (((b[i, :] - acc).astype(F)) / g[i, i]).astype(F)

    def _solve_upper(self, g, b):
        n, k = self.n, self.k
        for i in range(n - 1, -1, -1):
            acc = np.zeros(k, dtype=F)
            for p in range(i + 1, n):
                acc = (acc + (b[p, :] * g[i, p]).astype(F)).astype(F)
            b[i, :] = (((b[i, :] - acc).astype(F)) / g[i, i]).astype(F)

    # -- fused r = B - A X  and  g += A^T r  (kernel: ResidualAndGram) ------
    def _residual_and_gram(self, A, B, x, with_residuals):
        m, n, k = self.m, self.n, self.k
        g_hi = np.zeros((n, k), dtype=F)
        g_lo = np.zeros((n, k), dtype=F)
        res_hi = np.zeros(k, dtype=F)
        res_lo = np.zeros(k, dtype=F)
        res_cr = np.zeros(k, dtype=F)

        for r0 in range(0, m, self.block_rows):
            rb = min(self.block_rows, m - r0)
            a_blk = A[r0:r0 + rb]
            b_blk = B[r0:r0 + rb]
            for rr in range(rb):
                acc_hi = np.zeros(k, dtype=F)
                acc_lo = np.zeros(k, dtype=F)
                for p in range(n):
                    acc_hi, acc_lo = df_axpy_scalar(acc_hi, acc_lo, x[p, :], a_blk[rr, p])
                # r = b - acc, as a double-float pair
                r_hi = (b_blk[rr, :] - acc_hi).astype(F)
                neg = (-acc_hi).astype(F)
                bb = (r_hi - b_blk[rr, :]).astype(F)
                t2 = (r_hi - bb).astype(F)
                t2 = (b_blk[rr, :] - t2).astype(F)
                t1 = (neg - bb).astype(F)
                t2 = (t2 + t1).astype(F)
                r_lo = (t2 - acc_lo).astype(F)

                if with_residuals:
                    res_hi, res_lo = df_mac_accum(res_hi, res_lo, r_hi, r_hi)
                    res_cr = (res_cr + (F(2.0) * r_hi * r_lo).astype(F)).astype(F)

                for p in range(n):
                    av = a_blk[rr, p]
                    g_hi[p, :], g_lo[p, :] = df_axpy_scalar(g_hi[p, :], g_lo[p, :], r_hi, av)
                    g_hi[p, :], g_lo[p, :] = df_axpy_scalar(g_hi[p, :], g_lo[p, :], r_lo, av)
        return g_hi, g_lo, res_hi, res_lo, res_cr

    def run(self, A, B):
        m, n, k = self.m, self.n, self.k
        A = np.ascontiguousarray(A, dtype=F)
        B = np.ascontiguousarray(B, dtype=F)

        # ---- 1. G = A^T A, h = A^T B, double-float ------------------------
        g_hi = np.zeros((n, self.ns), dtype=F)
        g_lo = np.zeros((n, self.ns), dtype=F)
        h_hi = np.zeros((n, self.ks), dtype=F)
        h_lo = np.zeros((n, self.ks), dtype=F)

        for r0 in range(0, m, self.block_rows):
            rb = min(self.block_rows, m - r0)
            a_blk = A[r0:r0 + rb]
            b_blk = B[r0:r0 + rb]
            for rr in range(rb):
                for p in range(n):
                    av = a_blk[rr, p]
                    g_hi[p, :n], g_lo[p, :n] = df_axpy_scalar(
                        g_hi[p, :n], g_lo[p, :n], a_blk[rr, :], av)
                    h_hi[p, :k], h_lo[p, :k] = df_axpy_scalar(
                        h_hi[p, :k], h_lo[p, :k], b_blk[rr, :], av)
        g = (g_hi[:, :n] + g_lo[:, :n]).astype(F)
        x = (h_hi[:, :k] + h_lo[:, :k]).astype(F)

        # ---- 2. Cholesky + initial solve ----------------------------------
        self._cholesky(g)
        self._solve_lower_t(g, x)
        self._solve_upper(g, x)

        # ---- 3. semi-normal iterative refinement ---------------------------
        for _ in range(self.nref):
            g_hi, g_lo, _, _, _ = self._residual_and_gram(A, B, x, False)
            rhs = (g_hi[:, :k] + g_lo[:, :k]).astype(F)
            self._solve_lower_t(g, rhs)
            self._solve_upper(g, rhs)
            x = (x + rhs).astype(F)

        # ---- 4. residuals of the final x -----------------------------------
        _, _, res_hi, res_lo, res_cr = self._residual_and_gram(A, B, x, True)
        res = ((res_hi + res_lo).astype(F) + res_cr).astype(F)
        return x, res


# ---------------------------------------------------------------------------
# reference + harness
# ---------------------------------------------------------------------------
def lstsq_f64(A64, B64):
    if _HAVE_SCIPY:
        X, res, _, _ = scipy.linalg.lstsq(A64, B64, lapack_driver="gelsy")
        return X, res
    X, res, _, _ = np.linalg.lstsq(A64, B64, rcond=None)
    return X, res


def gen_case(m, n, k, cond, seed):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.linspace(cond, 1.0, n)
    A = (U @ np.diag(s) @ V.T).astype(F)
    B = rng.standard_normal((m, k)).astype(F)
    X_ref, res_ref = lstsq_f64(A.astype(np.float64), B.astype(np.float64))
    return A, B, X_ref, res_ref


def ratio_vs_tol(X, X_ref):
    diff = np.abs(X.astype(np.float64) - X_ref)
    tol = ATOL + RTOL * np.abs(X_ref)
    return float(np.max(diff / tol))


def naive_normal_equations(A, B):
    G = (A.T.astype(F) @ A.astype(F)).astype(F)
    h = (A.T.astype(F) @ B.astype(F)).astype(F)
    return np.linalg.solve(G.astype(np.float64), h.astype(np.float64)).astype(F)


def main():
    configs = [
        dict(m=32, n=16, k=4, cond=1000.0, seed=42),
        dict(m=64, n=32, k=4, cond=1000.0, seed=7),
        dict(m=128, n=64, k=8, cond=1000.0, seed=11),
        dict(m=256, n=64, k=1, cond=1000.0, seed=3),
        dict(m=512, n=96, k=3, cond=1000.0, seed=99),
        dict(m=64, n=64, k=4, cond=1000.0, seed=5),      # square -> residuals empty
        dict(m=32, n=1, k=1, cond=1.0, seed=8),
        dict(m=64, n=2, k=16, cond=100.0, seed=13),
    ]
    worst_x, worst_r = 0.0, 0.0
    all_pass = True
    for cfg in configs:
        A, B, X_ref, res_ref = gen_case(cfg["m"], cfg["n"], cfg["k"], cfg["cond"], cfg["seed"])
        cond_a = np.linalg.cond(A.astype(np.float64))
        print("=" * 78)
        print(f"m={cfg['m']} n={cfg['n']} k={cfg['k']} cond={cond_a:.1f}")

        r_naive = ratio_vs_tol(naive_normal_equations(A, B), X_ref)
        print(f"  naive float32 normal equations : err/tol={r_naive:9.3e}  "
              f"{'PASS' if r_naive <= 1 else 'FAIL'}")

        sim = LstsqKernelSim(cfg["m"], cfg["n"], cfg["k"], block_rows=64, nref=3)
        X, res = sim.run(A, B)
        r_x = ratio_vs_tol(X, X_ref)
        r_res = (float(np.max(np.abs(res.astype(np.float64) - res_ref)))
                 if len(res_ref) else 0.0)
        ok = (r_x <= 1.0) and (r_res <= ATOL)
        all_pass = all_pass and ok
        worst_x = max(worst_x, r_x)
        worst_r = max(worst_r, r_res)
        print(f"  kernel simulation (3 refine)   : err/tol={r_x:9.3e}  "
              f"res_err={r_res:9.3e}  {'PASS' if ok else 'FAIL'}")

    print("=" * 78)
    print(f"WORST: x err/tol = {worst_x:.3e} (limit 1.0)   "
          f"residual abs err = {worst_r:.3e} (limit {ATOL:g})")
    print(f"VERDICT: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
