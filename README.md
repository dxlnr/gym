<h1 align="center">
  <b>AI GYM</b><br>
</h1>


### Matrix Multiplication (GEMM)

Can be found in `gemm`. Currently `-DTILE` matmul is fastest and correct.
```sh
# Performance check (Cache)
valgrind --tool=cachegrind ./gemm
# Compiling (using tiling)
clang -O2 -DTILE -ffast-math -march=native gemm.c -o gemm
# Run
./gemm
```

Performance Metrics: 

Theoretical Max FLOPS: (Max) Clock Speed x FLOPs per cylce x number of cores

### Links

- [Tutorial](https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/): An Introduction to Reinforcement Learning Using OpenAI Gym
- [8+](https://www.gocoder.one/blog/reinforcement-learning-project-ideas/) Reinforcement Learning Project Ideas
