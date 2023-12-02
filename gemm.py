"""General Matrix Multiplication (GEMM) operator."""
import os
# Restrict numpy to single thread.
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import time
import numpy as np

# How much COMUPUTE does matrix multiplication require?


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="GEMM")
    PARSER.add_argument("--N", type=int, default=1024, help="Matrix size.")
    PARSER.add_argument("--save", action='store_true', help="Saves matrices for testing.")
    args = PARSER.parse_args()
    N = args.N

    # Initialize two random matrices
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    if args.save:
        if not os.path.exists("tests/mat"):
            os.makedirs("tests/mat")
        with open("tests/mat/matA", "wb") as f:
            f.write(A.data)
        with open("tests/mat/matB", "wb") as f:
            f.write(B.data)

    # Compute: N^2 memory loads output with 2N compute each
    print(f"{N*N*2*N/1e9:.1f} GFLOP.")
    st = time.monotonic()
    # Compute reference on the CPU to verify GPU computation
    C = A @ B
    et = time.monotonic()
    print(f"time: {(et - st):.2f}s")
    # Floating point operations per second. (FLOPS)
    print(f"{N*N*2*N/((et-st) * 1e9):.1f} GFLOPS.")

    if args.save:
        with open("tests/mat/matC", "wb") as f:
            f.write(C.data)
