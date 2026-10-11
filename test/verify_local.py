#!/usr/bin/env python3
"""Local reference validation script for linalg_lstsq."""
import numpy as np
import scipy.linalg

def generate_case(m=32, n=16, k=4, batch_size=2, cond=100.0, seed=42):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(batch_size):
        U, _ = np.linalg.qr(rng.standard_normal((m, n)))
        V, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = np.linspace(cond, 1.0, n)
        A = (U @ np.diag(s) @ V.T).astype(np.float32)
        B = rng.standard_normal((m, k)).astype(np.float32)

        # Ground truth using float64
        X_ref, res_ref, _, _ = scipy.linalg.lstsq(A.astype(np.float64), B.astype(np.float64), lapack_driver='gels')
        cases.append((A, B, X_ref.astype(np.float32), res_ref.astype(np.float32) if len(res_ref) > 0 else None))
    return cases

def verify_solution(X_pred, X_ref, atol=1e-4, rtol=1e-4):
    diff = np.abs(X_pred - X_ref)
    tol = atol + rtol * np.abs(X_ref)
    passed = np.all(diff <= tol)
    max_err = np.max(diff)
    return passed, max_err

if __name__ == "__main__":
    cases = generate_case()
    print(f"Generated {len(cases)} test cases.")
    for i, (A, B, X_ref, res_ref) in enumerate(cases):
        passed, max_err = verify_solution(X_ref, X_ref)
        print(f"Case {i+1}: passed={passed}, max_err={max_err:.2e}")
