# C++ GPT-2 Inference Engine (From Scratch)

A high-performance, educational C++ inference runtime for Large Language Models (LLMs), inspired by projects like `llm.c` and `llama.cpp`.

## Features
- Custom Tensor and Matrix Multiplication implementation
- 12-layer GPT-2 transformer forward pass
- Tokenizer with vocab loading
- Sampling strategies: greedy, top-k, top-p
- Interactive mode for text generation

## Quickstart

### Build (Windows)
Ensure Visual Studio Build Tools are installed, then run:
```bat
.\compile.bat
```

### Run Interactive Mode
```bat
.\llm_engine.exe
```
Enter prompts like "Once upon a time" and press Enter to generate continuations.

### Model Files
Place these in the project root (not included in repo):
- `gpt2_weights.bin`
- `gpt2_config.json`
- `vocab.json`

## Development Roadmap
<details>

Phase 1: Foundation
- Tensor class with strided memory
- Matrix multiplication kernels
- Basic tensor operations

Phase 2: Architecture
- LayerNorm implementation
- Linear layers
- GELU activation
- Multi-head attention mechanism
- KV cache support

Phase 3: Runtime
- BPE tokenizer improvements
- Greedy sampling
- Temperature-based sampling
- Top-k and top-p sampling

Phase 4: Optimization
- SIMD (AVX/NEON) intrinsics
- Multithreading with OpenMP
- Memory optimizations

Phase 5: Advanced Features
- Flash Attention
- Quantization (Int8/4-bit)
- CUDA/GPU backend
- Distributed inference

</details>

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Inspirations
- [Karpathy's llm.c](https://github.com/karpathy/llm.c)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)
- [PicoGPT](https://github.com/jaymody/picoGPT)

## License
MIT
  