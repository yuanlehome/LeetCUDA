#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define BLOCK_SIZE 256
#define BLOCK_SIZE_S 16
#define PAD 1
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

// FP32
// col2row means read x[row][col] and
// write y[col][row] row2col means read x[col][row] and write y[row][col]
__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y,
                                                 const int M, const int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = tid / N;
  const int col = tid % N;
  if (tid < M * N) {
    y[col * M + row] = x[tid];
  }
}

__global__ void mat_transpose_f32_row2col_kernel(float *x, float *y,
                                                 const int M, const int N) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = tid / M;
  const int row = tid % M;
  if (tid < M * N) {
    y[tid] = x[row * N + col];
  }
}

__global__ void mat_transpose_f32x4_col2row_kernel(float *x, float *y,
                                                   const int M, const int N) {
  int tid = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  int row = tid / N;
  int col = tid % N;

  if (row < M && (col + 3) < N) {
    float4 x_val = reinterpret_cast<float4 *>(x)[tid / 4];

    y[col * M + row] = x_val.x;
    y[(col + 1) * M + row] = x_val.y;
    y[(col + 2) * M + row] = x_val.z;
    y[(col + 3) * M + row] = x_val.w;
  }
}

__global__ void mat_transpose_f32x4_row2col_kernel(float *x, float *y,
                                                   const int M, const int N) {
  const int tid = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  const int col = tid / M;
  const int row = tid % M;

  if ((row + 3) < M && col < N) {
    float4 x_val;
    x_val.x = x[row * N + col];
    x_val.y = x[(row + 1) * N + col];
    x_val.z = x[(row + 2) * N + col];
    x_val.w = x[(row + 3) * N + col];
    reinterpret_cast<float4 *>(y)[tid / 4] = x_val;
  }
}

// work for row == col
__global__ void mat_transpose_f32_diagonal_2d_kernel(float *x, float *y,
                                                     int row, int col) {
  const int block_y = blockIdx.x;
  const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  const int global_col = threadIdx.x + blockDim.x * block_x;
  const int global_row = threadIdx.y + blockDim.y * block_y;
  if (global_col < col && global_row < row) {
    y[global_row * col + global_col] = x[global_col * row + global_row];
  }
}

__global__ void mat_transpose_f32_col2row_2d_kernel(float *x, float *y,
                                                    const int M, const int N) {
  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_x < N && tid_y < M) {
    y[tid_x * M + tid_y] = x[tid_y * N + tid_x];
  }
}

// equal to col2row because the cofig of grid dim
__global__ void mat_transpose_f32_row2col_2d_kernel(float *x, float *y,
                                                    const int M, const int N) {
  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_x < N && tid_y < M) {
    y[tid_x * M + tid_y] = x[tid_y * N + tid_x];
  }
}

__global__ void mat_transpose_f32x4_col2row_2d_kernel(float *x, float *y,
                                                      const int M,
                                                      const int N) {
  const int tid_x = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_x + 3 < N && tid_y < M) {
    float4 x_val = reinterpret_cast<float4 *>(x)[(tid_y * N + tid_x) / 4];
    y[tid_x * M + tid_y] = x_val.x;
    y[(tid_x + 1) * M + tid_y] = x_val.y;
    y[(tid_x + 2) * M + tid_y] = x_val.z;
    y[(tid_x + 3) * M + tid_y] = x_val.w;
  }
}

__global__ void mat_transpose_f32x4_row2col_2d_kernel(float *x, float *y,
                                                      const int M,
                                                      const int N) {
  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = 4 * (blockIdx.y * blockDim.y + threadIdx.y);
  if (tid_y + 3 < M && tid_x < N) {
    float4 x_val;
    x_val.x = x[tid_y * N + tid_x];
    x_val.y = x[(tid_y + 1) * N + tid_x];
    x_val.z = x[(tid_y + 2) * N + tid_x];
    x_val.w = x[(tid_y + 3) * N + tid_x];
    reinterpret_cast<float4 *>(y)[(tid_x * M + tid_y) / 4] = x_val;
  }
}

// clang-format off
// ### 1. **核函数的核心目标**
// - 将输入矩阵 `x`（尺寸 `M × N`）转置后写入输出矩阵 `y`（尺寸 `N × M`）。
// - 使用共享内存 `tile` 优化访问模式：先将数据从全局内存加载到共享内存，再以转置方式读取并写入全局内存。
// - 每个线程处理4个连续元素（通过 `float4` 向量化访问）。

// ### 2. **索引计算详解**
// #### (1) **输入数据的加载（`x → tile`）**
// - **全局索引**：
//   - `tid_x = blockIdx.x * blockDim.x + threadIdx.x`：当前线程处理的列索引（在 `x` 中）。
//   - `tid_y = 4 * (blockIdx.y * blockDim.y + threadIdx.y)`：当前线程处理的起始行索引（在 `x` 中），乘以4表示每个线程处理4行。
// - **共享内存写入位置**：
//   - 每个线程将 `x` 中同一列（`tid_x`）的4个连续行元素（`tid_y` 到 `tid_y+3`）写入共享内存 `tile`。
//   - 写入位置：
//     `tile[local_y*4 + k][local_x]`，其中 `k = 0, 1, 2, 3`。
//     `local_x = threadIdx.x`（块内列索引），
//     `local_y = threadIdx.y`（块内行索引）。

// #### (2) **转置数据的读取和写入（`tile → y`）**
// - **共享内存读取**：
//   - 每个线程从 `tile` 中读取4个元素，形成 `float4` 向量：
//     ```cpp
//     smem_val.x = tile[local_x*4][local_y];
//     smem_val.y = tile[local_x*4 + 1][local_y];
//     smem_val.z = tile[local_x*4 + 2][local_y];
//     smem_val.w = tile[local_x*4 + 3][local_y];
//     ```
//     - **关键点**：索引交换（`local_x` 和 `local_y` 角色互换）实现转置：
//       - 写入 `tile` 时：行 = `local_y*4 + k`，列 = `local_x`。
//       - 读取 `tile` 时：行 = `local_x*4 + k`，列 = `local_y`。
//     - 这等价于将共享内存中的子矩阵转置。

// - **全局内存写入位置**：
//   - `out_x` 和 `out_y` 计算转置后元素在 `y` 中的位置：
//     ```cpp
//     const int out_x = blockIdx.x * blockDim.x + local_y;  // 转置后的行索引
//     const int out_y = 4 * (blockIdx.y * blockDim.y + local_x); // 转置后的列起始索引
//     ```
//     - **`out_x`（行索引）**：
//       由**块ID.x × 块宽度** + **线程块内行索引 `local_y`** 构成。
//       - 物理意义：在转置矩阵 `y` 中，行索引对应原矩阵 `x` 的列索引。
//       - 为什么用 `local_y`？
//         因为转置后，原矩阵的行局部索引（`local_y`）变为列局部索引，但这里通过交换角色，`local_y` 直接映射到输出行索引的块内偏移。

//     - **`out_y`（列起始索引）**：
//       由**块ID.y × 块高度 × 4** + **线程块内列索引 `local_x × 4`** 构成。
//       - 物理意义：在转置矩阵 `y` 中，列索引对应原矩阵 `x` 的行索引。
//       - 乘以4：每个线程写入4个连续列元素（`out_y` 到 `out_y+3`）。

// ### 3. **写入逻辑**
// - **写入操作**：
//   ```cpp
//   reinterpret_cast<float4*>(y)[(out_x * M + out_y) / 4] = smem_val;
//   ```
//   - 将 `float4` 向量 `smem_val` 一次性写入全局内存 `y`。
//   - **地址计算**：`(out_x * M + out_y) / 4`
//     - `out_x * M`：计算行起始位置（`y` 是行优先存储）。
//     - `+ out_y`：在行内偏移到列起始位置。
//     - `/ 4`：因为 `float4` 指针的每个元素对应4个连续的 `float`，除法将字节偏移转换为 `float4` 索引。

// - **物理意义**：
//   线程将数据写入 `y` 的以下位置：
//   - **行**：`out_x`（对应原矩阵的列索引）。
//   - **列**：`out_y` 到 `out_y+3`（对应原矩阵的行索引 `tid_y` 到 `tid_y+3`）。

// ### 4. **设计优势**
// - **合并写入（Coalesced Access）**：
//   同一Warp中的线程按 `threadIdx.x` 连续递增，导致 `out_y` 以4为步长连续增加（例如 `out_y`, `out_y+4`, `out_y+8`, ...）。这使得全局内存写入是连续的（每个线程写4个连续元素），符合合并访问要求。
// - **共享内存转置**：
//   通过交换 `local_x` 和 `local_y` 的角色，避免全局内存的非合并访问。

// ### 总结
// `out_x` 和 `out_y` 的核心作用是**将共享内存中的转置数据映射到全局内存 `y` 的正确位置**：
// - `out_x` = 转置后的行索引（原矩阵列索引的映射）。
// - `out_y` = 转置后的列起始索引（原矩阵行索引的映射，并乘以4处理向量化写入）。
// 写入逻辑通过 `float4` 实现高效向量化存储，并依赖索引设计确保合并访问。
// clang-format on
__global__ void mat_transpose_f32x4_shared_bcf_merge_write_row2col_2d_kernel(
    float *x, float *y, const int M, const int N) {
  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = 4 * (blockIdx.y * blockDim.y + threadIdx.y);
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[BLOCK_SIZE_S * 4][BLOCK_SIZE_S + PAD];
  if (tid_y + 3 < M && tid_x < N) {
    // load value from x to shared memory
    tile[local_y * 4][local_x] = x[tid_y * N + tid_x];
    tile[local_y * 4 + 1][local_x] = x[(tid_y + 1) * N + tid_x];
    tile[local_y * 4 + 2][local_x] = x[(tid_y + 2) * N + tid_x];
    tile[local_y * 4 + 3][local_x] = x[(tid_y + 3) * N + tid_x];
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    smem_val.x = tile[local_x * 4][local_y];
    smem_val.y = tile[local_x * 4 + 1][local_y];
    smem_val.z = tile[local_x * 4 + 2][local_y];
    smem_val.w = tile[local_x * 4 + 3][local_y];

    const int out_x = blockIdx.x * blockDim.x + local_y;
    const int out_y = 4 * (blockIdx.y * blockDim.y + local_x);
    reinterpret_cast<float4 *>(y)[(out_x * M + out_y) / 4] = smem_val;
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack)        \
  void mat_transpose_##tag(torch::Tensor x, torch::Tensor y) {                 \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int M = x.size(0);                                                   \
    const int N = x.size(1);                                                   \
    dim3 block(BLOCK_SIZE);                                                    \
    dim3 grid(((N * M + BLOCK_SIZE - 1) / n_pack / BLOCK_SIZE));               \
    mat_transpose_##tag##_kernel<<<grid, block>>>(                             \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), M, N);                 \
  }

#define TORCH_BINDING_MAT_TRANSPOSE2D(tag, th_type, element_type,              \
                                      n_element_row, n_element_col)            \
  void mat_transpose_##tag##2d(torch::Tensor x, torch::Tensor y) {             \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int M = x.size(0);                                                   \
    const int N = x.size(1);                                                   \
    dim3 block(BLOCK_SIZE_S, BLOCK_SIZE_S);                                    \
    dim3 grid((N + BLOCK_SIZE_S - 1) / (BLOCK_SIZE_S * n_element_col),         \
              (M + BLOCK_SIZE_S - 1) / (BLOCK_SIZE_S * n_element_row));        \
    mat_transpose_##tag##_2d_kernel<<<grid, block>>>(                          \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), M, N);                 \
  }

// 1d index
TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32_row2col, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_col2row, torch::kFloat32, float, 4)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_row2col, torch::kFloat32, float, 4)
// 2d index. easier for diagonal
TORCH_BINDING_MAT_TRANSPOSE2D(f32_col2row, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32_row2col, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_col2row, torch::kFloat32, float, 1, 4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_row2col, torch::kFloat32, float, 4, 1)
// diagonal index method.
TORCH_BINDING_MAT_TRANSPOSE2D(f32_diagonal, torch::kFloat32, float, 1, 1)
// shared memory
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_merge_write_row2col,
                              torch::kFloat32, float, 4, 1)

// CuTe implentations
extern void mat_transpose_cute_col2row_reg(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row2col_reg(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_col_smem(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row_smem(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_col_smem_swizzled(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row_smem_swizzled(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row_cvectorized(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row_rvectorized(torch::Tensor, torch::Tensor);
extern void mat_transpose_cute_row_cvectorized_swizzled(torch::Tensor,
                                                        torch::Tensor);
extern void mat_transpose_cute_row_rvectorized_swizzled(torch::Tensor,
                                                        torch::Tensor);
extern void
    mat_transpose_cute_row_rvectorized_swizzled_optimized(torch::Tensor,
                                                          torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 1d index
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col)
  // 2d index. easier for diagonal
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col2d)
  // diagonal index method.
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_diagonal2d)
  // shared memory optimize
  TORCH_BINDING_COMMON_EXTENSION(
      mat_transpose_f32x4_shared_bcf_merge_write_row2col2d)
  // CuTe implentations
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col2row_reg)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row2col_reg)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_smem)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col_smem)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col_smem_swizzled)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_smem_swizzled)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_cvectorized)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_rvectorized)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_cvectorized_swizzled)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_rvectorized_swizzled)
  TORCH_BINDING_COMMON_EXTENSION(
      mat_transpose_cute_row_rvectorized_swizzled_optimized)
}
