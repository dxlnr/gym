"""General Matrix Multiplication (GEMM) using Triton."""
import argparse
import time
import triton
import triton.language as tl
import torch

# GPU: Grid -> Block -> Thread

# Define the block range along one dimension. 
# BLOCK_SIZE means to launch BLOCK_SIZE threads to do the computation.
BLOCK_SIZE = 512


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, N, K, BLOCK):
    """Matrix multiplication."""
    pid = tl.program_id(axis=0)
    tl.static_print(pid)


def matmul(a, b, N):
    """Matrix multiplication."""
    # Allocate an empty array to store the result
    c = torch.empty_like(a)
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
    A = torch.randn((N, N), device='cuda', dtype=torch.float32)
    B = torch.randn((N, N), device='cuda', dtype=torch.float32)

    # Compute: N^2 memory loads output with 2N compute each
    print(f"{N*N*2*N/1e9:.1f} GFLOP.")
    st = time.monotonic()
    # Compute the reference output
    C = matmul(A, B, N)
    et = time.monotonic()
    print(f"time: {(et - st):.2f}s")
    # Floating point operations per second. (FLOPS)
    print(f"{N*N*2*N/((et-st) * 1e9):.1f} GFLOPS.")
