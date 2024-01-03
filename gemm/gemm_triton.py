"""General Matrix Multiplication (GEMM) using Triton."""
import argparse
import time
import triton
import triton.language as tl
import numpy as np

# GPU: Grid -> Block -> Thread

# Define the block range along one dimension. 
# BLOCK_SIZE means to launch BLOCK_SIZE threads to do the computation.
BLOCK_SIZE = 512


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, N):
    """Matrix multiplication."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Load the data from VRAM to shared memory
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    # Run the computation
    out = tl.dot(a, b)
    # Store the data from shared memory to VRAM
    tl.store(c_ptr + offsets, out, mask=mask)


def matmul(a, b, N):
    """Matrix multiplication."""
    # Allocate an empty array to store the result
    c = np.zeros((N,N), dtype=np.float32)
    # Define the kernel grid
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
    # Define the kernel
    matmul_kernel[grid](a, b, c, N)
    return c


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="GEMM")
    PARSER.add_argument("--N", type=int, default=1024, help="Matrix size.")
    PARSER.add_argument("--save", action='store_true', help="Saves matrices for testing.")
    args = PARSER.parse_args()
    N = args.N

    # Initialize two random matrices
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # Compute: N^2 memory loads output with 2N compute each
    print(f"{N*N*2*N/1e9:.1f} GFLOP.")
    st = time.monotonic()
    # Compute the reference output
    C = matmul(A, B, N)
    et = time.monotonic()
    print(f"time: {(et - st):.2f}s")
    # Floating point operations per second. (FLOPS)
    print(f"{N*N*2*N/((et-st) * 1e9):.1f} GFLOPS.")
