def map_qwen3_moe_name(name: str) -> str:
    """Map a Hugging Face Qwen3-MoE checkpoint key to this project."""

    if name == "lm_head.weight":
        return "output.weight"
    if name.startswith("model."):
        name = name[len("model.") :]

    name = name.replace("embed_tokens", "tok_embeddings")
    name = name.replace("self_attn.q_proj", "attention.wq")
    name = name.replace("self_attn.k_proj", "attention.wk")
    name = name.replace("self_attn.v_proj", "attention.wv")
    name = name.replace("self_attn.o_proj", "attention.wo")
    name = name.replace("self_attn.q_norm", "attention.q_norm")
    name = name.replace("self_attn.k_norm", "attention.k_norm")
    name = name.replace(".mlp.gate.", ".feed_forward.gate.")
    name = name.replace(".mlp.experts.", ".feed_forward.experts.")
    name = name.replace(".gate_proj.", ".w1.")
    name = name.replace(".down_proj.", ".w2.")
    name = name.replace(".up_proj.", ".w3.")
    name = name.replace("input_layernorm", "attention_norm")
    name = name.replace("post_attention_layernorm", "ffn_norm")
    return name


def run(state_dict):
    return {map_qwen3_moe_name(name): tensor for name, tensor in state_dict.items()}
