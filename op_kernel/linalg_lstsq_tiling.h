#pragma once
#include <cstdint>

// Tiling payload shared between op_host and op_kernel.
//
// Layout notes:
//   * m, n, k, batch_size, has_residuals, b_has_batch  -- problem shape (as before)
//   * block_rows   -- number of A/B rows streamed into UB per iteration
//   * nref         -- number of semi-normal iterative-refinement passes
//   * core_num     -- number of AIV cores the host launched (for batch split)
struct LinalgLstsqTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t batch_size;
    uint32_t has_residuals;
    uint32_t b_has_batch;
    uint32_t block_rows;
    uint32_t nref;
    uint32_t core_num;
    uint32_t reserved;
};
