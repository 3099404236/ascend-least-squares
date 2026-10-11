#!/usr/bin/env python3
"""Local reference validation script for linalg_lstsq.

This is a light-weight sanity check on the *problem generator* and on the
tolerance helper.  The real numerical validation of the kernel algorithm is
in `verify_algorithm.py`, which is a float32-faithful port of
`op_kernel/linalg_lstsq.cpp`.

Run:  python verify_local.py
"""
import numpy as np

try:
    import scipy.linalg as _sla
except ImportError:  # scipy is optional; numpy's SVD driver is fine here
    _sla = None


def lstsq_f64(A, B):
    """float64 least squares, scipy `gels` when available, else numpy SVD."""
    if _sla is not None:
        return _sla.lstsq(A, B, lapack_driver="gels")
    return np.linalg.lstsq(A, B, rcond=None)


def generate_case(m=32, n=16, k=4, batch_size=2, cond=1000.0, seed=42):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(batch_size):
        U, _ = np.linalg.qr(rng.standard_normal((m, n)))
        V, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = np.linspace(cond, 1.0, n)
        A = (U @ np.diag(s) @ V.T).astype(np.float32)
        B = rng.standard_normal((m, k)).astype(np.float32)

        # Ground truth using float64
        X_ref, res_ref, _, _ = lstsq_f64(A.astype(np.float64), B.astype(np.float64))
        cases.append((A, B, X_ref.astype(np.float32),
                      res_ref.astype(np.float32) if len(res_ref) > 0 else None))
    return cases


def verify_solution(X_pred, X_ref, atol=1e-4, rtol=1e-4):
    diff = np.abs(X_pred - X_ref)
    tol = atol + rtol * np.abs(X_ref)
    passed = bool(np.all(diff <= tol))
    max_err = float(np.max(diff))
    return passed, max_err


if __name__ == "__main__":
    cases = generate_case()
    driver = "scipy.linalg.lstsq(gels)" if _sla is not None else "numpy.linalg.lstsq(SVD)"
    print(f"Generated {len(cases)} test cases.")
    print(f"Reference: {driver} in float64.\n")
    for i, (A, B, X_ref, res_ref) in enumerate(cases):
        cond = np.linalg.cond(A.astype(np.float64))
        # naive float32 normal equations, the baseline the kernel must beat
        G = (A.T @ A).astype(np.float32)
        h = (A.T @ B).astype(np.float32)
        X_naive = np.linalg.solve(G.astype(np.float64), h.astype(np.float64)).astype(np.float32)
        passed, max_err = verify_solution(X_naive, X_ref)
        print(f"Case {i+1}: cond(A)={cond:8.1f}  naive float32 normal equations -> "
              f"passed={passed}, max_err={max_err:.2e}")
    print("\nNote: the naive baseline is expected to FAIL at cond(A) ~ 1e3;")
    print("      run `python verify_algorithm.py` for the real kernel check.")
