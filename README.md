# gpt-go

Minimal PyTorch implementation of
`Qwen/Qwen3-30B-A3B-Instruct-2507` for learning single-GPU inference.

Implemented features:

- Qwen3-MoE attention with GQA, QK-Norm and RoPE
- 128 routed experts with top-8 normalized routing
- KV-cache prefill and autoregressive decode
- Streaming Hugging Face safetensors checkpoint loading
- Single-GPU CUDA or ROCm inference (TP=1)

## Setup

```bash
python -m pip install -r requirements.txt
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir ${MODEL_DIR}/Qwen3-30B-A3B-Instruct-2507
```

## Run

Single GPU:

```bash
python modeling.py /models/Qwen3-30B-A3B-Instruct-2507 \
  "Explain tensor parallelism." \
  --max-seq-len 4096 \
  --max-new-tokens 1024
```

Using the shared model directory:

```bash
python modeling.py ${MODEL_DIR}/Qwen3-30B-A3B-Instruct-2507 "写一篇科幻小说" --max-seq-len 4096 --max-new-tokens 1024
```

`--max-seq-len` defaults to 4096. Although the checkpoint supports 262144
tokens, this reference implementation allocates a square causal mask and a
full KV cache, so very large values require substantially more memory.

This is a readable reference implementation, not a throughput replacement for
vLLM or SGLang. MoE routing uses ordinary PyTorch operations and does not
implement tensor or expert parallelism.
