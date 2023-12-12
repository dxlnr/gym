print("******** cuda runtime ***********")

from tinygrad.runtime.ops_cuda import CUDAProgram, compile_cuda, CUDAAllocator
from tinygrad.device import Device

# some intialization
device = Device["cuda"]
CUDAAllocator = CUDAAllocator(device)

# allocate some buffers
out = CUDAAllocator.alloc(4)
a = CUDAAllocator.alloc(4)
b = CUDAAllocator.alloc(4)

# load in some values (little endian)
CUDAAllocator.copyin(a, bytearray([2,0,0,0]))
CUDAAllocator.copyin(b, bytearray([3,0,0,0]))

# compile a program to a binary
lib = compile_cuda("__global__ void add_vectors(int *a, int *b, int *out) { int id = blockDim.x * blockIdx.x + threadIdx.x; out[id] = a[id] + b[id]; }")

# create a runtime for the program.
fxn = CUDAProgram("add_vectors", lib)

# # run the program
# fxn(out, a, b)

# check the data out
# print(val := MallocAllocator.as_buffer(out).cast("I").tolist()[0])
# assert val == 5
