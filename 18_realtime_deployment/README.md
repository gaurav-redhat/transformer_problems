# Problem 18: Real-Time Deployment

[← Back to Main](../README.md) | [← Previous](../17_hardware_inefficiency/README.md)

---

![Problem 18](./problem.png)

## What's the Problem?

Your voice assistant needs to respond in under 200ms to feel natural. Your game NPC needs to react in a single frame (16ms). Your autonomous car needs to make decisions in milliseconds.

Meanwhile, a single forward pass through GPT-4 takes 500ms-2000ms. That's not real-time. That's "please wait while I think."

Real-time applications need:
- Low latency (fast response)
- Consistent latency (no random spikes)  
- Low memory (for edge devices)
- Low power (for mobile/embedded)

Transformers struggle with all of these.

## The Latency Budget

| Application | Latency Requirement | LLM Reality |
|-------------|--------------------| ------------|
| Voice assistant | < 200ms | 500-2000ms |
| Game NPC | < 16ms | Way off |
| Autocomplete | < 100ms | Maybe |
| Autonomous driving | < 50ms | Not happening |

## Why So Slow?

Several factors compound:
1. **Model size**: Billions of parameters to load/compute
2. **Sequential generation**: One token at a time
3. **Memory bandwidth**: Can't feed the GPU fast enough
4. **Batch size 1**: Real-time means single requests, no batching efficiency

## How Do We Fix It?

| Approach | Latency Reduction |
|----------|-------------------|
| **Quantization** | INT8/INT4 runs 2-4x faster than FP16 |
| **Pruning** | Remove 50%+ of weights with minimal quality loss |
| **Distillation** | Smaller model, faster inference |
| **Speculative Decoding** | Guess multiple tokens, verify in parallel |
| **TensorRT/ONNX** | Optimized inference runtimes |
| **Edge Models** | Purpose-built small models (Phi, Gemma) |

## The Quantization Revolution

Moving from FP16 to INT4:
- 4x smaller model
- 2-4x faster computation
- Often <1% accuracy loss

This is why consumer apps can now run 7B models locally. GGML/llama.cpp made this accessible.

## The Real-Time Reality

For truly real-time applications today, you typically need:
- Smaller models (1-7B parameters)
- Heavy quantization (4-bit)
- Speculative decoding
- Optimized serving stack (vLLM, TensorRT-LLM)

Or... don't use transformers. For some applications, specialized smaller models or even classical algorithms still win on latency.

## Learn More

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA's optimized inference
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — CPU inference with quantization
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput serving

---

[← Back to Main](../README.md) | [← Previous](../17_hardware_inefficiency/README.md)
