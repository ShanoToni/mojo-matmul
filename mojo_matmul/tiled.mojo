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
alias NUM_BLOCKS_ROW = (M + TPB - 1) // TPB
alias NUM_BLOCKS_COL = (N + (TPB // 16) - 1) // TPB


fn tiled[
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
    alias BLOCK_DIM_SIZE: Int = 64
    alias TILE: Int = 16

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
    acc = tb[dtype]().layout[TILE]().local().alloc().fill(0)

    alias simd = simd_width_of[dtype]()
    alias num_threads: Int = TPB * TPB

    # loop over tiles with offset of BLOCK_DIM_SIZE
    for tile_idx_k in range(K // BLOCK_DIM_SIZE):
        # create tiles
        alias load_a_layout = Layout.row_major(1, BLOCK_DIM_SIZE)
        alias load_b_layout = Layout.row_major(BLOCK_DIM_SIZE, 1)
        var a_tile = a.tile[BLOCK_DIM_SIZE, BLOCK_DIM_SIZE](
            block_idx.y, tile_idx_k
        )
        var b_tile = b.tile[BLOCK_DIM_SIZE, BLOCK_DIM_SIZE](
            tile_idx_k, block_idx.x
        )
        copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)

        # Wait for all asynchronous copies to complete.
        async_copy_wait_all()
        barrier()

        @parameter
        for k in range(BLOCK_DIM_SIZE):
            var a_tile = a_smem.tile[TILE, 1](loc_m, k)
            var b_tile = b_smem.tile[1, BLOCK_DIM_SIZE](k, 0)
            var b_val = b_tile[0, loc_n]

            @parameter
            for t in range(TILE):
                acc[t] += a_tile[t, 0] * b_val

        barrier()

    c_tile.copy_from(acc)


fn benchmark_tiled[
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
    ctx.enqueue_function[tiled[a_layout, b_layout, c_layout, M, N, K]](
        a_tensor,
        b_tensor,
        c_tensor,
        grid_dim=(NUM_BLOCKS_ROW, NUM_BLOCKS_COL),
        block_dim=(TPB, TPB // 16),
    )
    ctx.synchronize()

    start = time.monotonic()
    for _ in range(10):
        ctx.enqueue_function[tiled[a_layout, b_layout, c_layout, M, N, K]](
            a_tensor,
            b_tensor,
            c_tensor,
            grid_dim=(NUM_BLOCKS_ROW, NUM_BLOCKS_COL),
            block_dim=(TPB, TPB // 16),
        )
        ctx.synchronize()

    end = time.monotonic()

    elapsed = (end - start) / Float64(10)
    flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    gflops = (flops / elapsed) / 1e9

    print("Kernel avg:", elapsed * 1e3, "ms")
    print("Perf:", gflops, "GFLOP/s")
