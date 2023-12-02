<h1 align="center">
  <b>AI GYM</b><br>
</h1>


### Matrix Multiplication

```sh
# Performance check (Cache)
valgrind --tool=cachegrind ./gemm
# Compiling (using tiling)
clang -O2 -DTILE -march=native -mavx gemm.c -o gemm
# Run
./gemm
```

### Links

- [Tutorial](https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/): An Introduction to Reinforcement Learning Using OpenAI Gym
- [8+](https://www.gocoder.one/blog/reinforcement-learning-project-ideas/) Reinforcement Learning Project Ideas
