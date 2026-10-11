# `linalg_lstsq` Operator Specification

## 1. Mathematical Definition
Compute the least squares solution to a system of linear equations:
$$\min_X \|A X - B\|_2$$
for each matrix $A \in \mathbb{R}^{m \times n}$ ($m \ge n$) and right-hand side $B \in \mathbb{R}^{m \times k}$.

If $m > n$, also compute the squared Euclidean 2-norm of the residual vector for each column $j \in [0, k-1]$:
$$\text{residuals}[j] = \|A X_{:, j} - B_{:, j}\|_2^2 = \sum_{i=0}^{m-1} (A X - B)_{i, j}^2$$
If $m = n$, the residuals output must be empty (size 0).

## 2. Shapes and Layouts
- **Inputs**:
  - `A`: `float32`, shape `[..., m, n]`, row-major, 32-byte aligned address.
  - `B`: `float32`, shape `[..., m, k]` (2D per batch) or `[..., m]` (1D vector per batch).
- **Outputs**:
  - `solution`: `float32`, shape `[..., n, k]` or `[..., n]`.
  - `residuals`: `float32`, shape `[..., k]` (or empty if $m = n$).

## 3. Precision Acceptance Criteria
- Ground truth reference: `scipy.linalg.lstsq(A, B, lapack_driver='gels')` using `float64` inputs.
- Tolerance:
  - `atol <= 1e-4`
  - `rtol <= 1e-4`
