"""General Matrix Multiplication (GEMM) operator."""
import argparse
import time
import numpy as np

# How much COMUPUTE does matrix multiplication require?


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="GEMM")
    PARSER.add_argument("--N", type=int, default=4096, help="Matrix size.")
    args = PARSER.parse_args()
    N = args.N

    # Initialize two random matrices
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # Compute: N^2 memory loads output with 2N compute each
    print(f"{N*N*2*N/1e9:.1f} GFLOP.")
    st = time.monotonic()
    # Compute reference on the CPU to verify GPU computation
    C_ref = A @ B
    et = time.monotonic()
    print(f"time: {(et - st):.2f}s")
    # Floating point operations per second. (FLOPS)
    print(f"{N*N*2*N/((et-st) * 1e9):.1f} GFLOPS.")
