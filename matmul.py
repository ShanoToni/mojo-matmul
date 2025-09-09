import torch
from time import perf_counter_ns
import argparse
import numpy as np
from pathlib import Path
import sys
from max.torch import CustomOpLibrary

def handle_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--M", type=int, default=1024, help="Rows of A and C")
    parser.add_argument("--N", type=int, default=1024, help="Cols of B and C")
    parser.add_argument("--K", type=int, default=1024, help="Cols of A / Rows of B")
    parser.add_argument("--impl", type=str, default="naive", help="Type of mojo kernel to run; naive, shared, async_cpy ...")

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--torch",
        action="store_true",
        help="Run computation on pure Python CPU"
    )
    group.add_argument(
        "--mojo",
        action="store_true",
        help="Run on Mojo Kernel GPU"
    )

    args = parser.parse_args()

    return args

def benchmark_matmul(matmul_fn, A: torch.Tensor, B: torch.Tensor, repeat: int = 10):
    # Warmup
    matmul_fn(A, B)

    start_time = perf_counter_ns()

    for _ in range(repeat):
        _ = matmul_fn(A, B)

    torch.cuda.synchronize()
    end_time = perf_counter_ns()

    elapsed = (end_time - start_time) / repeat
    elapsed_ms = elapsed / 1e6

    c = matmul_fn(A, B)

    return elapsed_ms

def mojo_matmul(matmul_impl, a_tensor: torch.Tensor, b_tensor: torch.Tensor) -> torch.Tensor:
    mojo_kernels = Path(__file__).parent / "op"
    ops = CustomOpLibrary(mojo_kernels)

    c_tensor = torch.empty(a_tensor.shape[0], b_tensor.shape[1], device=a_tensor.device, dtype=a_tensor.dtype)

    matmul = ops.matmul[{
        "implementation": matmul_impl,
        "M": a_tensor.shape[0],
        "N": b_tensor.shape[1],
        "K": a_tensor.shape[1]
    }]

    matmul(c_tensor, a_tensor, b_tensor)

    return c_tensor

# Add new impls here

def naive_mojo(a_tensor: torch.Tensor, b_tensor: torch.Tensor) -> torch.Tensor:
    return mojo_matmul("naive", a_tensor, b_tensor)

def shared_mojo(a_tensor: torch.Tensor, b_tensor: torch.Tensor) -> torch.Tensor:
    return mojo_matmul("shared", a_tensor, b_tensor)

def async_cpy_mojo(a_tensor: torch.Tensor, b_tensor: torch.Tensor) -> torch.Tensor:
    return mojo_matmul("async_cpy", a_tensor, b_tensor)


def get_mojo_func(args):
    if args.impl == "naive":
        return naive_mojo
    elif args.impl == "shared":
        return shared_mojo
    elif args.impl == "async_cpy":
        return async_cpy_mojo

def main():
    args = handle_args()
    M = args.M
    N = args.N
    K = args.K

    np_a = np.random.randint(low=-32, high=32, size=(M * K), dtype=np.int32)
    np_a = np_a.astype(np.float32)
    A = torch.from_numpy(np_a).reshape(M, K).to(device="cuda")

    np_b = np.random.randint(low=-32, high=32, size=(K * N), dtype=np.int32)
    np_b = np_b.astype(np.float32)
    B = torch.from_numpy(np_a).reshape(K, N).to(device="cuda")


    print("=" * 60)
    print("Running Ref Torch Matmul...")
    ref_out = torch.matmul(A, B)

    print("=" * 60)
    print("Ref Run Complete!")
    if args.torch == True:
        print("=" * 60)
        print("Benchmarking pyTorch Ref!")

        print("=" * 60)
        avg_time_ms, out = benchmark_matmul(torch.matmul, A, B, repeat=10)
        print("Benchmark results:")
        print(f"Average time: {avg_time_ms:.3f} ms")
        print(f"Result shape: {out.shape}")


    elif args.mojo == True:
        mojo_func = get_mojo_func(args)
        out = mojo_func(A, B)

        print("=" * 60)
        print(f"Checking Correctness!")
        compared_tensor = out == ref_out
        if compared_tensor.sum() != M * N:
            print("Verification Failed")
            print(out)
            print(ref_out)
            sys.exit("Stopping!")
        else:
            print("Verification Passed")

        print("=" * 60)
        print(f"Benchmarking Mojo {args.impl}!")
        print("=" * 60)
        avg_time_ms = benchmark_matmul(mojo_func, A, B, repeat=1)
        print("Benchmark results:")
        print(f"Average time: {avg_time_ms:.3f} ms")
        print(f"Result shape: {out.shape}")



if __name__ == "__main__":
    main()
