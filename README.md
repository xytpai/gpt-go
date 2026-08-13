# gpt-go

Minimal PyTorch implementation of
`Qwen/Qwen3-30B-A3B-Instruct-2507` for learning tensor parallel inference.
The model code mirrors the small, explicit style of
[`xytpai/gpt/modeling.py`](https://github.com/xytpai/gpt/blob/master/modeling.py)
instead of depending on the Transformers model implementation.

Implemented features:

- Qwen3-MoE attention with GQA, QK-Norm and RoPE
- 128 routed experts with top-8 normalized routing
- KV-cache prefill and autoregressive decode
- Hugging Face safetensors checkpoint loading
- Tensor parallel attention, experts and vocabulary projection
- TP=1, TP=2 and TP=4 on one Linux/WSL CUDA or ROCm node

## Setup

```bash
python -m pip install -r requirements.txt
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir /models/Qwen3-30B-A3B-Instruct-2507
```

## Run

Single GPU:

```bash
python modeling.py /models/Qwen3-30B-A3B-Instruct-2507 \
  "Explain tensor parallelism." \
  --max-seq-len 4096 \
  --max-new-tokens 128
```

Four-GPU tensor parallel:

```bash
python modeling.py /models/Qwen3-30B-A3B-Instruct-2507 \
  "Explain tensor parallelism." \
  --tp-size 4 \
  --max-seq-len 4096 \
  --max-new-tokens 128
```

`--max-seq-len` sets the context-length limit. The checkpoint supports 262144
tokens, while the KV cache is allocated only for the prompt plus
`--max-new-tokens`.

This is a readable reference implementation, not a throughput replacement for
vLLM or SGLang. MoE routing uses ordinary PyTorch operations and does not
implement expert parallelism.
