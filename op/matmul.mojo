from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb

from gpu.memory import async_copy_wait_all
from layout.layout_tensor import copy_dram_to_sram_async

alias TPB = 16

fn async_cpy_matmul[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    M: Int,
    N: Int,
    K: Int,
    dtype: DType = DType.float32,
](
    c: LayoutTensor[mut=True, dtype, c_layout],
    a: LayoutTensor[mut=True, dtype, a_layout],
    b: LayoutTensor[mut=True, dtype, b_layout],
):
    loc_m = thread_idx.y
    loc_n = thread_idx.x
    alias TILE_SIZE = TPB
    tiled_m = (block_idx.y * TILE_SIZE) + loc_m
    tiled_n = (block_idx.x * TILE_SIZE) + loc_n

    shared_a = tb[dtype]().row_major[TILE_SIZE, TILE_SIZE]().shared().alloc()
    shared_b = tb[dtype]().row_major[TILE_SIZE, TILE_SIZE]().shared().alloc()
    c_tile = c.tile[TILE_SIZE, TILE_SIZE](block_idx.y, block_idx.x)

    var acc: c.element_type = 0

    alias load_a_layout = Layout.row_major(1, TILE_SIZE)
    alias load_b_layout = Layout.row_major(1, TILE_SIZE)

    @parameter
    for tile_idx in range((M + TILE_SIZE-1) // TILE_SIZE):
        a_tile = a.tile[TILE_SIZE, TILE_SIZE](block_idx.y, tile_idx)
        b_tile = b.tile[TILE_SIZE, TILE_SIZE](tile_idx, block_idx.x)

        copy_dram_to_sram_async[
            thread_layout=load_a_layout,
            num_threads=TILE_SIZE*TILE_SIZE,
            block_dim_count=2,
        ](shared_a, a_tile)
        copy_dram_to_sram_async[
            thread_layout=load_b_layout,
            num_threads=TILE_SIZE*TILE_SIZE,
            block_dim_count=2,
        ](shared_b, b_tile)

        async_copy_wait_all()
        barrier()

        @parameter
        for k in range(min(TILE_SIZE, K - tile_idx * TILE_SIZE)):
            acc += shared_a[loc_m, k] * shared_b[k, loc_n]

        barrier()
    if tiled_m < M and tiled_n < N:
        c_tile[loc_m, loc_n] = acc

fn shared_matmul[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    M: Int,
    N: Int,
    K: Int,
    dtype: DType = DType.float32,
](
    c: LayoutTensor[mut=True, dtype, c_layout],
    a: LayoutTensor[mut=True, dtype, a_layout],
    b: LayoutTensor[mut=True, dtype, b_layout],
):
    loc_m = thread_idx.y
    loc_n = thread_idx.x
    alias TILE_SIZE = TPB
    tiled_m = (block_idx.y * TILE_SIZE) + thread_idx.y
    tiled_n = (block_idx.x * TILE_SIZE) + thread_idx.x

    shared_a = tb[dtype]().row_major[TILE_SIZE, TILE_SIZE]().shared().alloc()
    shared_b = tb[dtype]().row_major[TILE_SIZE, TILE_SIZE]().shared().alloc()


    var acc: c.element_type = 0
    @parameter
    for tile_idx in range((M + TILE_SIZE -1) // TILE_SIZE):
        if tiled_m < M and (tile_idx * TILE_SIZE + loc_n) < K:
            shared_a[loc_m, loc_n] = a[tiled_m, tile_idx * TILE_SIZE + loc_n]
        if (tile_idx * TILE_SIZE + loc_m) < K and tiled_n < N:
            shared_b[loc_m, loc_n] = a[tile_idx * TILE_SIZE + loc_m, tiled_n]

        barrier()

        if tiled_m < M and tiled_n < N:
            @parameter
            for k in range(min(TILE_SIZE, K - tile_idx * TILE_SIZE)):
                acc += shared_a[loc_m, k] * shared_b[k, loc_n]

        barrier()
    if tiled_m < M and tiled_n < N:
        c[tiled_m, tiled_n] = acc

fn naive_matmul[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    M: Int,
    N: Int,
    K: Int,
    dtype: DType = DType.float32,
](
    c: LayoutTensor[mut=True, dtype, c_layout],
    a: LayoutTensor[mut=True, dtype, a_layout],
    b: LayoutTensor[mut=True, dtype, b_layout],
):
    m = block_dim.y * block_idx.y + thread_idx.y
    n = block_dim.x * block_idx.x + thread_idx.x

    if m < M and n <  N:
        tmp: c.element_type = 0
        @parameter
        for k in range(K):
            tmp += a[m, k] * b[k, n]

        c[m, n] = tmp



import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

@compiler.register("matmul")
struct MatMulCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,
        implementation: StaticString,
        M: Int,
        N: Int,
        K: Int,
        dtype: DType = DType.float32,
    ](
        c: OutputTensor[rank=2],
        a: InputTensor[dtype = dtype, rank = 2],
        b: InputTensor[dtype = dtype, rank = 2],
        ctx: DeviceContextPtr,
    ) raises:
        c_tensor = c.to_layout_tensor()
        a_tensor = a.to_layout_tensor()
        b_tensor = b.to_layout_tensor()

        alias a_layout = a_tensor.layout
        alias b_layout = b_tensor.layout
        alias c_layout = c_tensor.layout

        alias NUM_BLOCKS_ROW = (M + TPB - 1) // TPB
        alias NUM_BLOCKS_COL = (N + TPB - 1) // TPB

        @parameter
        if target == "gpu":
            @parameter
            if implementation == "naive":
                gpu_ctx = ctx.get_device_context()
                gpu_ctx.enqueue_function[naive_matmul[
                    a_layout,
                    b_layout,
                    c_layout,
                    M, N, K
                ]](c_tensor, a_tensor, b_tensor, grid_dim = (NUM_BLOCKS_ROW, NUM_BLOCKS_COL), block_dim = (TPB, TPB))
            elif implementation == "shared":
                gpu_ctx = ctx.get_device_context()
                gpu_ctx.enqueue_function[shared_matmul[
                    a_layout,
                    b_layout,
                    c_layout,
                    M, N, K
                ]](c_tensor, a_tensor, b_tensor, grid_dim = (NUM_BLOCKS_ROW, NUM_BLOCKS_COL), block_dim = (TPB, TPB))
            elif implementation == "async_cpy":
                gpu_ctx = ctx.get_device_context()
                gpu_ctx.enqueue_function[async_cpy_matmul[
                    a_layout,
                    b_layout,
                    c_layout,
                    M, N, K
                ]](c_tensor, a_tensor, b_tensor, grid_dim = (NUM_BLOCKS_ROW, NUM_BLOCKS_COL), block_dim = (TPB, TPB))
            else:
                raise Error("Unsupported target: " + target)
        else:
            raise Error("Unsupported target: " + target)
