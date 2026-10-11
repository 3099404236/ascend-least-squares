# Ascend C Standalone Operator: `linalg_lstsq` (Least Squares)

本仓库为华为 2026 电子设计竞赛 **`linalg_lstsq`（矩阵最小二乘解）** 算子的独立原生 Ascend C 开发工程。

---

## 1. 目标平台与评价指标

- **目标硬件**: 华为昇腾 910B (CANN 9.0, Ascend C 编程模型)
- **数学问题**: 计算超定或正方线性方程组的最小二乘解：
  $$\min_X \|A X - B\|_F^2$$
- **精度标准**:
  - 解向量 $X$: 与 `float64` 参考解相比，要求 `atol <= 1e-4` 且 `rtol <= 1e-4`。
  - 残差平方和（当 $m > n$ 时输出）: 要求 `atol <= 1e-4`。
  - 测试用例保证 $A$ 为列满秩矩阵，条件数 $\text{cond}_2(A) \le 1000$。
- **独立工程设计**:
  - 本仓库专为单算子独立开发打造，已剔除其他 8 个算子和外部无关模块干扰。
  - `op_kernel/linalg_lstsq.cpp` 提供最基础的单核/多核极简骨架，`Process()` 方法留空，供开发者/AI 从头设计。

---

## 2. 输入与输出张量定义

| 张量名称 | 属性 | 数据类型 | 格式 | 形状说明 |
| :--- | :--- | :--- | :--- | :--- |
| **`A`** | 输入 (必选) | `float32` | ND | `[..., m, n]` ($m \ge n \ge 1$，列满秩) |
| **`B`** | 输入 (必选) | `float32` | ND | `[..., m, k]` 或 `[..., m]` ($k=1$ 时的向量情况) |
| **`solution`** | 输出 (必选) | `float32` | ND | `[..., n, k]` 或 `[..., n]` |
| **`residuals`** | 输出 (可选) | `float32` | ND | $m > n$ 时为 `[..., k]` 或 `[..., 1]`；$m = n$ 时为空/不输出 |

---

## 3. 目录结构

```text
ascend-least-squares/
├── README.md                  # 本文档
├── specs/
│   └── lstsq_spec.md          # 详细数学规格与边界形状说明
├── docs/
│   └── ASCEND_C_API.md        # 常用 Ascend C 官方底层 API 速查手册
├── op_host/
│   └── linalg_lstsq.cpp       # Host 侧 Tiling 切分与 Workspace 内存分配
├── op_kernel/
│   ├── linalg_lstsq_tiling.h  # TilingData 结构体定义
│   ├── tiling_key_linalg_lstsq.h # TilingKey 宏选择器
│   └── linalg_lstsq.cpp       # Device 侧 Ascend C Kernel 实现入口
└── test/
    └── verify_local.py        # Python 本地数值基准对照验证脚本
```

---

## 4. 推荐数值解法思路

1. **Householder QR 分解**:
   - 对 $A$ 进行正交三角分解 $A = Q R$。
   - $Q^T B$ 变换：计算 $c = Q^T B$。
   - 回代求解：$R x_0 = c[0:n]$。
2. **残差与高精度迭代求精 (Iterative Refinement)**:
   - 单精度 `float32` 在条件数接近 $10^3$ 时，直接回代可能损失 $3 \sim 4$ 位精度（由于求解正规方程的等价误差放大为 $\text{cond}(A)^2 \epsilon$）。
   - 计算残差 $r = B - A x_0$ 以及更新量 $g = A^T r$ 时，必须采用扩展精度或补偿求和（如 Double-Float / Dekker TwoProd/TwoSum 累加）。
   - 通过一步校正半正规方程迭代求精求解 $R R^T \Delta x = g$，$x = x_0 + \Delta x$，即可轻松跨过 $10^{-4}$ 精度红线。

---

## 5. 实现状态（本次提交）

`Process()` 已实现，方案与上节思路一致但做了两处关键加强：

1. **$A^T A$ / $A^T B$ 必须用双精度补偿累加（Dekker TwoProd + Knuth TwoSum）**，
   否则 $G$ 的相对扰动 $\eta_G$ 会让迭代求精的迭代矩阵谱半径
   $\sim \mathrm{cond}(A)^2 \eta_G \ge 1$，求精直接停滞。
2. **残差 $r = B - AX$ 与 $g = A^T r$ 同样用双精度累加**，并保留
   $\sum (r_{hi}+r_{lo})^2$ 中的交叉项 $2 r_{hi} r_{lo}$（少这一项残差输出会
   损失约 4 位绝对精度）。

流程：补偿累加 $G,h$ → 外积形式 Cholesky $G=R^TR$ → $x_0=R^{-1}R^{-T}h$
→ 3 次半正规方程迭代求精 → 输出 $x$ 与残差平方和。

**实测精度**（`test/verify_algorithm.py`，float32 逐行仿真，含双精度补偿）：

| m | n | k | cond | x 误差/容差 | 残差绝对误差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 16 | 4 | 1000 | 2.1e-4 | 8.5e-7 |
| 64 | 32 | 4 | 1000 | 1.4e-4 | 2.0e-6 |
| 128 | 64 | 8 | 1000 | 1.8e-4 | 5.9e-6 |
| 512 | 96 | 3 | 1000 | 1.0e-4 | 2.3e-5 |
| 64 | 64 | 4 | 1000 | 1.1e-4 | （m=n，残差为空） |

误差/容差 < 1 即通过，最差 2.1e-4 ⇒ **约 4800 倍余量**；残差误差已低于
float32 在该量级（~416）上的 1 ulp，即输出精度的物理极限。

详细设计、内存模型、支持范围以及**需要在真实 CANN 工具链上确认的 API 细节**
见 [`docs/DESIGN.md`](docs/DESIGN.md)。

### 运行验证

```bash
python test/verify_algorithm.py   # 核函数算法仿真 + 容差判定（需要 numpy）
python test/verify_local.py       # 参考生成器 / 朴素 float32 基线对照（需要 scipy）
```

### 支持范围

- `m`、`k`、`batch_size` 无限制（`A` 按行分块流式读入，`k` 只是向量宽度）。
- `n` 受 UB 限制，约 **n ≤ 140**；`op_host` 会按 176 KB 预算自动计算
  `block_rows`，`n` 过大时直接返回 `GRAPH_FAILED` 并打印原因。
- 多核：按 batch 以 `for b = blockIdx; b < batch; b += blockNum` 分摊到各 AIV 核。
