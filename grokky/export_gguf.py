"""
Export Grok MoE checkpoint to GGUF for Go inference engine.

Extends nanollama's GGUF export with MoE weight mapping:
  - Router: blk.N.ffn_gate_inp.weight
  - Experts: blk.N.ffn_{gate,up,down}.E.weight
  - Shared expert: blk.N.ffn_{gate,up,down}_shexp.weight
  - Double pre-norm: blk.N.{attn,ffn}_post_norm.weight
  - MoE metadata: expert_count, expert_used_count, etc.

Usage:
    python export_gguf.py --checkpoint ckpt.pt --tokenizer tok.model --output grok.gguf
"""

import os
import re
import sys
import struct
import argparse

sys.path.insert(0, '/home/ubuntu/nanollama')
sys.path.insert(0, '/home/ubuntu/grokky.go')

import torch
from scripts.export_gguf import (
    GGUFWriter, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
    Q4_BLOCK_SIZE, Q8_BLOCK_SIZE, load_tokenizer_metadata,
    tensor_to_bytes, tensor_to_q4_0, tensor_to_q8_0,
    compute_intermediate_size,
)


# Weight name mapping for Grok MoE
WEIGHT_MAP = {
    "tok_embeddings.weight": "token_embd.weight",
    "output.weight": "output.weight",
    "norm.weight": "output_norm.weight",
}

LAYER_WEIGHT_MAP = {
    # Standard attention + norms
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "attn.c_q.weight": "attn_q.weight",
    "attn.c_k.weight": "attn_k.weight",
    "attn.c_v.weight": "attn_v.weight",
    "attn.c_proj.weight": "attn_output.weight",
    # Grok double pre-norm
    "attn_post_norm.weight": "attn_post_norm.weight",
    "ffn_post_norm.weight": "ffn_post_norm.weight",
}


def map_name(name: str) -> str:
    """Map Grok weight name to GGUF tensor name."""
    if name in WEIGHT_MAP:
        return WEIGHT_MAP[name]
    if name.startswith("layers."):
        parts = name.split(".", 2)
        layer_idx = parts[1]
        rest = parts[2]
        # Standard layer weights
        if rest in LAYER_WEIGHT_MAP:
            return f"blk.{layer_idx}.{LAYER_WEIGHT_MAP[rest]}"
        # MoE router
        if rest == "ffn.router.gate.weight":
            return f"blk.{layer_idx}.ffn_gate_inp.weight"
        # MoE experts: ffn.experts.E.{gate,up,down}_proj.weight
        m = re.match(r"ffn\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", rest)
        if m:
            expert_idx = m.group(1)
            proj = m.group(2).replace("_proj", "")
            proj_map = {"gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}
            return f"blk.{layer_idx}.{proj_map[proj]}.{expert_idx}.weight"
        # Shared expert
        m = re.match(r"ffn\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight", rest)
        if m:
            proj = m.group(1).replace("_proj", "")
            proj_map = {"gate": "ffn_gate_shexp", "up": "ffn_up_shexp", "down": "ffn_down_shexp"}
            return f"blk.{layer_idx}.{proj_map[proj]}.weight"
    raise ValueError(f"Unknown weight name: {name}")


def main():
    parser = argparse.ArgumentParser(description="Export Grok MoE checkpoint to GGUF")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="f16", choices=["f32", "f16", "q4_0", "q8_0"])
    args = parser.parse_args()

    print("=" * 60)
    print("Grok MoE → GGUF converter")
    print("=" * 60)

    # Load checkpoint
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        config = ckpt.get("config", {})
    else:
        state = ckpt
        config = {}
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Infer config
    n_embd = config.get("n_embd", 0)
    n_layer = config.get("n_layer", 0)
    n_head = config.get("n_head", 0)
    n_kv_head = config.get("n_kv_head", 0)
    vocab_size = config.get("vocab_size", 32000)
    sequence_len = config.get("sequence_len", 2048)
    norm_eps = config.get("norm_eps", 1e-5)
    rope_theta = config.get("rope_theta", 10000.0)
    num_experts = config.get("num_experts", 0)
    num_experts_per_tok = config.get("num_experts_per_tok", 2)

    if n_embd == 0 and "tok_embeddings.weight" in state:
        n_embd = state["tok_embeddings.weight"].shape[1]
    if n_layer == 0:
        layers = {int(k.split(".")[1]) for k in state if k.startswith("layers.")}
        n_layer = max(layers) + 1 if layers else 0
    if n_head == 0 and "layers.0.attn.c_q.weight" in state:
        n_head = state["layers.0.attn.c_q.weight"].shape[0] // 64
    if n_kv_head == 0 and "layers.0.attn.c_k.weight" in state:
        n_kv_head = state["layers.0.attn.c_k.weight"].shape[0] // 64

    head_dim = n_embd // n_head if n_head else 64
    intermediate_size = compute_intermediate_size(n_embd)

    print(f"  Model: {n_layer}L / {n_embd}D / {n_head}H / {n_kv_head}KV / MoE:{num_experts}x{num_experts_per_tok}")

    # Write GGUF
    writer = GGUFWriter(args.output)

    # Architecture metadata
    writer.add_string("general.architecture", "grok")
    writer.add_string("general.name", f"grokky-{n_layer}L-{n_embd}D-moe{num_experts}")
    writer.add_uint32("llama.block_count", n_layer)
    writer.add_uint32("llama.embedding_length", n_embd)
    writer.add_uint32("llama.attention.head_count", n_head)
    writer.add_uint32("llama.attention.head_count_kv", n_kv_head)
    writer.add_uint32("llama.attention.key_length", head_dim)
    writer.add_uint32("llama.attention.value_length", head_dim)
    writer.add_uint32("llama.feed_forward_length", intermediate_size)
    writer.add_uint32("llama.context_length", sequence_len)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", norm_eps)
    writer.add_float32("llama.rope.freq_base", rope_theta)
    writer.add_uint32("llama.vocab_size", vocab_size)

    # MoE metadata
    if num_experts > 0:
        writer.add_uint32("llama.expert_count", num_experts)
        writer.add_uint32("llama.expert_used_count", num_experts_per_tok)
        writer.add_bool("grok.shared_expert", config.get("shared_expert", False))
        writer.add_bool("grok.use_gelu", config.get("use_gelu", False))
        writer.add_bool("grok.double_prenorm", config.get("use_double_prenorm", False))
        attn_clamp = config.get("attn_clamp", 0.0)
        if attn_clamp > 0:
            writer.add_float32("grok.attn_clamp", attn_clamp)

    # Tokenizer
    if args.tokenizer and os.path.exists(args.tokenizer):
        tok_meta = load_tokenizer_metadata(args.tokenizer)
        if tok_meta:
            writer.add_string("tokenizer.ggml.model", tok_meta["model"])
            writer.add_string_array("tokenizer.ggml.tokens", tok_meta["tokens"])
            writer.add_float32_array("tokenizer.ggml.scores", tok_meta["scores"])
            writer.add_int32_array("tokenizer.ggml.token_type", tok_meta["token_types"])
            writer.add_uint32("tokenizer.ggml.bos_token_id", tok_meta["bos_id"])
            writer.add_uint32("tokenizer.ggml.eos_token_id", tok_meta["eos_id"])

    # Convert weights
    dtype_map = {"f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16, "q4_0": GGML_TYPE_Q4_0, "q8_0": GGML_TYPE_Q8_0}
    ggml_type = dtype_map[args.dtype]

    print(f"\nConverting to {args.dtype}...")
    converted = 0
    for name in sorted(state.keys()):
        tensor = state[name]
        gguf_name = map_name(name)
        if tensor.dim() == 1:
            t_ggml = GGML_TYPE_F32
        else:
            block_size = Q4_BLOCK_SIZE if ggml_type == GGML_TYPE_Q4_0 else Q8_BLOCK_SIZE
            if ggml_type in (GGML_TYPE_Q4_0, GGML_TYPE_Q8_0) and tensor.numel() % block_size != 0:
                t_ggml = GGML_TYPE_F16
            else:
                t_ggml = ggml_type
        writer.add_tensor(gguf_name, tensor.float(), t_ggml)
        dtype_names = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0"}
        shape_str = "x".join(str(d) for d in tensor.shape)
        print(f"  {name:55s} → {gguf_name:40s} [{shape_str}] {dtype_names[t_ggml]}")
        converted += 1

    print(f"\nTotal: {converted} tensors")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer.write()
    print("Done!")


if __name__ == "__main__":
    main()
