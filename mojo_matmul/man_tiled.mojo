from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.host.compile import get_gpu_target
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_equal
from python import PythonObject, Python
from config import MATMUL_DIM_SIZE, TPB
from sys import simd_width_of, argv, alignof

alias dtype = DType.float32

alias M = MATMUL_DIM_SIZE
alias N = MATMUL_DIM_SIZE
alias K = MATMUL_DIM_SIZE
alias BLOCK_DIM_SIZE: Int = TPB
alias TILE: Int = 16
alias NUM_BLOCKS_ROW = (M + TPB - 1) // TPB
alias NUM_BLOCKS_COL = (N + (TPB // TILE) - 1) // TPB


fn man_tiled[
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
    # Indexing
    m = block_dim.y * block_idx.y + thread_idx.y
    n = block_dim.x * block_idx.x + thread_idx.x
    loc_m = thread_idx.y
    loc_n = thread_idx.x

    # shared mem
    c_tile = c.tile[BLOCK_DIM_SIZE, BLOCK_DIM_SIZE](
        block_idx.y, block_idx.x
    ).tile[TILE, 1](loc_m, loc_n)

    a_smem = (
        tb[dtype]().row_major[BLOCK_DIM_SIZE, BLOCK_DIM_SIZE]().shared().alloc()
    )

    b_smem = (
        tb[dtype]().row_major[BLOCK_DIM_SIZE, BLOCK_DIM_SIZE]().shared().alloc()
    )

    alias simd = simd_width_of[dtype]()

    acc = tb[dtype]().layout[TILE]().local().alloc().fill(0)

    # loop over tiles with offset of BLOCK_DIM_SIZE
    for tile_idx_k in range(K // BLOCK_DIM_SIZE):
        k_base = tile_idx_k * BLOCK_DIM_SIZE
        tiled_m = (block_idx.y * BLOCK_DIM_SIZE) + loc_m * TILE

        @parameter
        for t in range(TILE):
            src_k = tiled_m + t
            a_smem[loc_m * TILE + t, loc_n] = a[src_k, k_base + loc_n]

            src_k = k_base + loc_m * TILE + t
            b_smem[loc_m * TILE + t, loc_n] = b[src_k, n]

        # Wait for all asynchronous copies to complete.
        barrier()

        for k in range(BLOCK_DIM_SIZE):
            var a_tile = a_smem.tile[TILE, 1](loc_m, k)
            var b_tile = b_smem.tile[1, BLOCK_DIM_SIZE](k, 0)
            var b_val = b_tile[0, loc_n]

            @parameter
            for t in range(TILE):
                acc[t] += a_tile[t, 0] * b_val

        barrier()

    c_tile.copy_from(acc)


fn benchmark_man_tiled[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
](
    ctx: DeviceContext,
    a_tensor: LayoutTensor,
    b_tensor: LayoutTensor,
    c_tensor: LayoutTensor,
) raises:
    time = Python.import_module("time")

    # Warmup
    ctx.enqueue_function[man_tiled[a_layout, b_layout, c_layout, M, N, K]](
        a_tensor,
        b_tensor,
        c_tensor,
        grid_dim=(NUM_BLOCKS_ROW, NUM_BLOCKS_COL),
        block_dim=(TPB, TPB // TILE),
    )
    ctx.synchronize()

    start = time.monotonic()
    for _ in range(10):
        ctx.enqueue_function[man_tiled[a_layout, b_layout, c_layout, M, N, K]](
            a_tensor,
            b_tensor,
            c_tensor,
            grid_dim=(NUM_BLOCKS_ROW, NUM_BLOCKS_COL),
            block_dim=(TPB, TPB // TILE),
        )
        ctx.synchronize()

    end = time.monotonic()

    elapsed = (end - start) / Float64(10)
    flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    gflops = (flops / elapsed) / 1e9

    print("Kernel avg:", elapsed * 1e3, "ms")
    print("Perf:", gflops, "GFLOP/s")
