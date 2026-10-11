# Ascend C Programming API Reference (CANN 9.0 / Ascend 910B)

This reference document summarizes standard Ascend C programming APIs for AI Core vector operators on the Ascend 910B NPU.

---

## 1. Memory Hierarchy & Types
- **Global Memory (GM)**:
  - Addressed via `GM_ADDR` (or `__gm__ T*`).
  - Wrapped as `GlobalTensor<T>`.
  - Accessed via DMA transfers (`DataCopy`, `DataCopyPad`).
- **Unified Buffer (UB)**:
  - On-chip high-speed scratchpad per AI Vector core (~192 KB available).
  - Wrapped as `LocalTensor<T>`.
  - All vector operations and intermediate buffers must reside in UB.
  - Operations must satisfy 32-byte alignment (8 `float` elements).

---

## 2. Queue & Buffer Management (`TPipe`)
```cpp
#include "kernel_operator.h"
using namespace AscendC;

TPipe pipe;
TQue<QuePosition::VECIN, 1> inQueueA;
TQue<QuePosition::VECOUT, 1> outQueueSol;
TBuf<TPosition::VECCALC> calcBuf;

// Initialize buffers (size must be 32-byte aligned)
pipe.InitBuffer(inQueueA, 1, 32 * 1024);
pipe.InitBuffer(outQueueSol, 1, 32 * 1024);
pipe.InitBuffer(calcBuf, 64 * 1024);

// Allocate and Enqueue
LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
inQueueA.EnQue(aLocal);

// Dequeue and Free
LocalTensor<float> aReady = inQueueA.DeQue<float>();
inQueueA.FreeTensor(aReady);
```

---

## 3. Data Movement APIs (`DataCopy`)

### 3.1 Contiguous Block Copy
```cpp
// GM -> UB
DataCopy(dstLocal, srcGlobal, count);

// UB -> GM
DataCopy(dstGlobal, srcLocal, count);
```

### 3.2 2D Strided Matrix Copy (`DataCopyPad`)
```cpp
DataCopyExtParams copyParams;
copyParams.blockCount = num_rows;
copyParams.blockLen   = row_bytes / 32;   // in units of 32 bytes (8 floats)
copyParams.srcStride  = gap_src / 32;     // gap between rows in units of 32 bytes
copyParams.dstStride  = gap_dst / 32;

DataCopyPadExtParams<float> padParams;
padParams.isPad = false;

DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
```

---

## 4. Vector Compute APIs

Operates on `LocalTensor<float>` operands in UB.

### 4.1 Arithmetic Operations
```cpp
Add(dst, a, b, count);       // dst[i] = a[i] + b[i]
Sub(dst, a, b, count);       // dst[i] = a[i] - b[i]
Mul(dst, a, b, count);       // dst[i] = a[i] * b[i]
Div(dst, a, b, count);       // dst[i] = a[i] / b[i]
Muls(dst, a, scalar, count); // dst[i] = a[i] * scalar
Adds(dst, a, scalar, count); // dst[i] = a[i] + scalar
Duplicate(dst, val, count);  // dst[i] = val
Sqrt(dst, a, count);         // dst[i] = sqrt(a[i])
Abs(dst, a, count);          // dst[i] = abs(a[i])
```

### 4.2 Reductions
```cpp
// Sum reduction of vector elements into a scalar/vector
ReduceSum(dst, src, workTensor, count);
```

---

## 5. Synchronization APIs
```cpp
// Synchronize vector execution pipeline
PipeBarrier<PIPE_V>();

// Synchronize all pipelines (MTE2, MTE3, V, S)
PipeBarrier<PIPE_ALL>();
```

---

## 6. Multi-Core Primitives
```cpp
int core_id = GetBlockIdx();   // Current AIV core ID (0 .. BlockDim - 1)
int core_num = GetBlockNum();  // Total number of allocated AIV cores
```
