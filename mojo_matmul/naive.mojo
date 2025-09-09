from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_equal
from python import PythonObject, Python
from config import MATMUL_DIM_SIZE, TPB
import sys

alias dtype = DType.float32

alias M = MATMUL_DIM_SIZE
alias N = MATMUL_DIM_SIZE

alias K = MATMUL_DIM_SIZE
alias NUM_BLOCKS_ROW = (M + TPB - 1) // TPB
alias NUM_BLOCKS_COL = (N + TPB - 1) // TPB

fn naive_matmul[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    M: Int,
    N: Int,
    K: Int,
    dtype: DType = DType.float32,
](
    a: LayoutTensor[mut=True, dtype, a_layout],
    b: LayoutTensor[mut=True, dtype, b_layout],
    c: LayoutTensor[mut=True, dtype, c_layout],
):
    m = block_dim.y * block_idx.y + thread_idx.y
    n = block_dim.x * block_idx.x + thread_idx.x

    if m < M and n <  N:
        tmp: c.element_type = 0
        for k in range(K):
            tmp += a[m, k] * b[k, n]

        c[m, n] = tmp

fn benchmark_naive[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
](ctx: DeviceContext, a_tensor: LayoutTensor, b_tensor: LayoutTensor, c_tensor: LayoutTensor) raises :
    time = Python.import_module("time")

    # Warmup
    ctx.enqueue_function[naive_matmul[
            a_layout,
            b_layout,
            c_layout,
            M, N, K
        ]](a_tensor, b_tensor, c_tensor, grid_dim = (NUM_BLOCKS_ROW, NUM_BLOCKS_COL), block_dim = (TPB, TPB))
    ctx.synchronize()


    start = time.monotonic()
    for _ in range(10):
        ctx.enqueue_function[naive_matmul[
                a_layout,
                b_layout,
                c_layout,
                M, N, K
        ]](a_tensor, b_tensor, c_tensor, grid_dim = (NUM_BLOCKS_ROW, NUM_BLOCKS_COL), block_dim = (TPB, TPB))
        ctx.synchronize()

    end = time.monotonic()

    elapsed = (end - start) / Float64(10)
    flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    gflops = (flops / elapsed) / 1e9

    print("Kernel avg:", elapsed * 1e3, "ms")
    print("Perf:", gflops, "GFLOP/s")