from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_equal
from python import PythonObject, Python
import sys

from config import MATMUL_DIM_SIZE, TPB
from naive import naive_matmul, benchmark_naive
from shared import shared_matmul, benchmark_shared
from tiled import tiled, benchmark_tiled

alias dtype = DType.float32

alias M = MATMUL_DIM_SIZE
alias N = MATMUL_DIM_SIZE

alias K = MATMUL_DIM_SIZE
alias NUM_BLOCKS_ROW = (M + 32 - 1) // 32
alias NUM_BLOCKS_COL = (N + 32 - 1) // 32


def init_data[M: Int, N: Int, K: Int]() -> (PythonObject, PythonObject):
    np = Python.import_module("numpy")

    np_a = np.random.randint(low=-32, high=32, size=(M * K), dtype=np.int32)
    np_a = np_a.astype(np.float32)

    np_b = np.random.randint(low=-32, high=32, size=(K * N), dtype=np.int32)
    np_b = np_b.astype(np.float32)

    return np_a, np_b


def main():
    if len(sys.argv()) != 2:
        print("Usage: pixi run mojo --[naive | shared | tiled]")
        return
    alias a_layout = Layout.row_major(M, K)
    alias b_layout = Layout.row_major(K, N)
    alias c_layout = Layout.row_major(M, N)

    np_a, np_b = init_data[M, N, K]()

    with DeviceContext() as ctx:
        c = ctx.enqueue_create_buffer[dtype](M * N)

        a = ctx.enqueue_create_buffer[dtype](M * K)
        with a.map_to_host() as a_host:
            for m in range(M):
                for k in range(K):
                    a_host[m * K + k] = Float32(np_a[m * K + k])

        b = ctx.enqueue_create_buffer[dtype](K * N)
        with b.map_to_host() as b_host:
            for k in range(K):
                for n in range(N):
                    b_host[k * N + n] = Float32(np_b[k * N + n])

        a_tensor = LayoutTensor[dtype, a_layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[dtype, b_layout](b.unsafe_ptr())
        c_tensor = LayoutTensor[dtype, c_layout](c.unsafe_ptr())

        if sys.argv()[1] == "naive":
            benchmark_naive[
                a_layout,
                b_layout,
                c_layout,
            ](ctx, a_tensor, b_tensor, c_tensor)
        elif sys.argv()[1] == "shared":
            benchmark_shared[
                a_layout,
                b_layout,
                c_layout,
            ](ctx, a_tensor, b_tensor, c_tensor)
        elif sys.argv()[1] == "tiled":
            benchmark_tiled[
                a_layout,
                b_layout,
                c_layout,
            ](ctx, a_tensor, b_tensor, c_tensor)
        else:
            print("Unknown commandline arg pased")
            return

        # Verify result
        expected = ctx.enqueue_create_buffer[dtype](M * N)
        exp_tensor = LayoutTensor[dtype, c_layout](expected.unsafe_ptr())
        ctx.enqueue_function[
            naive_matmul[a_layout, b_layout, c_layout, M, N, K]
        ](
            a_tensor,
            b_tensor,
            exp_tensor,
            grid_dim=(NUM_BLOCKS_ROW, NUM_BLOCKS_COL),
            block_dim=(32, 32),
        )
        ctx.synchronize()

        with c.map_to_host() as c_host:
            with expected.map_to_host() as exp_host:
                print("out:", c_host)
                print("expected:", exp_host)
                for i in range(M * N):
                    assert_equal(c_host[i], exp_host[i])
