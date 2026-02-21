"""
Grok MoE architecture for nanollama.

Extends nanollama's Llama with:
- MoE (Mixture of Experts) with top-k routing + load balancing
- GELU activation (Grok uses GELU, not SiLU)
- Double pre-norm (RMSNorm before AND after sub-layers, 4 norms/layer)
- Soft attention clamping: clamp * tanh(logits / clamp)
- Optional shared expert (always-on, Grok/DeepSeek-V2 style)

Imports from nanollama without modifying it.
"""

import sys
sys.path.insert(0, '/home/ubuntu/nanollama')

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanollama.llama import (
    LlamaConfig, RMSNorm, rms_norm, precompute_freqs_cis, apply_rotary_emb,
    CausalSelfAttention, Llama,
)
from nanollama.common import get_dist_info, print0
from nanollama.optim import MuonAdamW, DistMuonAdamW


# ── Grok Config ─────────────────────────────────────────────────────────────

@dataclass
class GrokConfig(LlamaConfig):
    """Grok MoE config — extends LlamaConfig with MoE + Grok-specific fields."""
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_aux_loss_coeff: float = 0.01
    shared_expert: bool = True
    use_gelu: bool = True              # Grok uses GELU, not SiLU
    use_double_prenorm: bool = True    # 4 RMSNorm per layer (Grok-1 style)
    attn_clamp: float = 30.0          # soft attention clamping (Grok-1 = 30)


# Named Grok configs: (depth, dim, heads, kv_heads, num_experts)
GROK_CONFIGS = {
    #  name        depth  dim   heads  kv  experts  ~total / ~active per tok
    "grok-nano":  (  12,   384,    6,   6,    4),   #  ~102M total / ~40M active
    "grok-micro": (  16,   512,    8,   4,    8),   # ~385M total / ~80M active
    "grok-mini":  (  20,   768,   12,   4,    8),   # ~850M total / ~175M active
    "grok-small": (  24,  1024,   16,   4,    8),   # ~1.7B total / ~340M active
    "grok-1.5b":  (  28,  1280,   20,   4,    8),   # ~3.7B total / ~1.4B active
}


def get_grok_config(name: str) -> GrokConfig:
    """Get a named Grok MoE config."""
    if name not in GROK_CONFIGS:
        raise ValueError(f"Unknown Grok config '{name}'. Available: {list(GROK_CONFIGS.keys())}")
    depth, n_embd, n_head, n_kv_head, num_experts = GROK_CONFIGS[name]
    return GrokConfig(
        n_layer=depth, n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head,
        tie_embeddings=False, num_experts=num_experts, num_experts_per_tok=2,
        shared_expert=True, use_gelu=True, use_double_prenorm=True, attn_clamp=30.0,
    )


# ── MoE Modules ─────────────────────────────────────────────────────────────

class GrokFFN(nn.Module):
    """Grok-style FFN: down(gelu(gate(x)) * up(x)). Uses GELU instead of SiLU."""

    def __init__(self, config: GrokConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class MoERouter(nn.Module):
    """Top-k expert router with load balancing auxiliary loss."""

    def __init__(self, config: GrokConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.aux_loss_coeff = config.moe_aux_loss_coeff
        self.gate = nn.Linear(config.n_embd, config.num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """x: (B*T, D). Returns weights, indices, aux_loss."""
        logits = self.gate(x).float()  # router always fp32 for stability
        top_k_logits, expert_indices = torch.topk(logits, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        # Load balancing loss (Switch Transformer style)
        gates = F.softmax(logits, dim=-1)
        expert_mask = F.one_hot(expert_indices[:, 0], self.num_experts).float()
        f = expert_mask.mean(dim=0)  # fraction of tokens per expert
        P = gates.mean(dim=0)        # avg probability per expert
        aux_loss = self.aux_loss_coeff * self.num_experts * (f * P).sum()
        return expert_weights, expert_indices, aux_loss


class MoEFFN(nn.Module):
    """Mixture of Experts FFN with optional shared expert (Grok-style)."""

    def __init__(self, config: GrokConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.router = MoERouter(config)
        self.experts = nn.ModuleList([GrokFFN(config) for _ in range(config.num_experts)])
        self.shared_expert = GrokFFN(config) if config.shared_expert else None

    def forward(self, x: torch.Tensor):
        """x: (B, T, D). Returns output, aux_loss."""
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        expert_weights, expert_indices, aux_loss = self.router(x_flat)

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e_idx in range(self.num_experts):
                mask = (expert_indices[:, k] == e_idx)
                if not mask.any():
                    continue
                expert_input = x_flat[mask]
                expert_output = self.experts[e_idx](expert_input.unsqueeze(1)).squeeze(1)
                output[mask] += expert_weights[mask, k:k+1] * expert_output

        output = output.view(B, T, D)
        if self.shared_expert is not None:
            output = output + self.shared_expert(x)
        return output, aux_loss


# ── Grok Transformer Block ──────────────────────────────────────────────────

class GrokTransformerBlock(nn.Module):
    """Grok transformer block: double pre-norm + MoE + attention clamping."""

    def __init__(self, config: GrokConfig, layer_idx: int):
        super().__init__()
        self.attn_clamp = config.attn_clamp
        # Pre-norms
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.ffn_norm = RMSNorm(config.n_embd, config.norm_eps)
        # Post-norms (Grok double pre-norm: norm before AND after sub-layer)
        self.attn_post_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.ffn_post_norm = RMSNorm(config.n_embd, config.norm_eps)
        # Attention
        self.attn = CausalSelfAttention(config, layer_idx)
        # MoE FFN
        self.ffn = MoEFFN(config)

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor],
                window_size: Tuple[int, int], kv_cache=None):
        # Attention with double pre-norm
        h_attn = self.attn(self.attn_norm(x), cos_sin, window_size, kv_cache)
        h_attn = self.attn_post_norm(h_attn)
        x = x + h_attn
        # MoE FFN with double pre-norm
        h_ffn, aux_loss = self.ffn(self.ffn_norm(x))
        h_ffn = self.ffn_post_norm(h_ffn)
        x = x + h_ffn
        return x, aux_loss


# ── Grok Model ──────────────────────────────────────────────────────────────

class Grok(nn.Module):
    """Grok MoE model. Same structure as Llama but with MoE + Grok-specific features."""

    def __init__(self, config: GrokConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        # Pad vocab for tensor core efficiency
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab {config.vocab_size} → {padded_vocab}")

        self.tok_embeddings = nn.Embedding(padded_vocab, config.n_embd)
        self.layers = nn.ModuleList([GrokTransformerBlock(config, i) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.output = nn.Linear(config.n_embd, padded_vocab, bias=False)  # never tied for Grok

        # RoPE
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_freqs_cis(head_dim, self.rotary_seq_len, theta=config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        windows = [(long_w, 0) if c == "L" else (short_w, 0) for c in pattern * config.n_layer]
        windows = windows[:config.n_layer]
        windows[-1] = (long_w, 0)
        return windows

    def get_device(self) -> torch.device:
        return self.tok_embeddings.weight.device

    @torch.no_grad()
    def init_weights(self):
        """Initialize weights following Grok conventions."""
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.001)
        s = (3 ** 0.5) * (self.config.n_embd ** -0.5)

        for layer in self.layers:
            # Norms
            layer.attn_norm.weight.fill_(1.0)
            layer.ffn_norm.weight.fill_(1.0)
            layer.attn_post_norm.weight.fill_(1.0)
            layer.ffn_post_norm.weight.fill_(1.0)
            # Attention
            nn.init.uniform_(layer.attn.c_q.weight, -s, s)
            nn.init.uniform_(layer.attn.c_k.weight, -s, s)
            nn.init.uniform_(layer.attn.c_v.weight, -s, s)
            nn.init.zeros_(layer.attn.c_proj.weight)
            # Router
            nn.init.normal_(layer.ffn.router.gate.weight, mean=0.0, std=0.01)
            # Experts
            for expert in layer.ffn.experts:
                nn.init.uniform_(expert.gate_proj.weight, -s, s)
                nn.init.uniform_(expert.up_proj.weight, -s, s)
                nn.init.zeros_(expert.down_proj.weight)
            # Shared expert
            if layer.ffn.shared_expert is not None:
                nn.init.uniform_(layer.ffn.shared_expert.gate_proj.weight, -s, s)
                nn.init.uniform_(layer.ffn.shared_expert.up_proj.weight, -s, s)
                nn.init.zeros_(layer.ffn.shared_expert.down_proj.weight)

        self.norm.weight.fill_(1.0)

        head_dim = self.config.n_embd // self.config.n_head
        device = self.tok_embeddings.weight.device
        self.cos, self.sin = precompute_freqs_cis(
            head_dim, self.rotary_seq_len, theta=self.config.rope_theta, device=device)
        if device.type == "cuda":
            self.tok_embeddings.to(dtype=torch.bfloat16)

    def estimate_flops(self) -> int:
        """Estimate FLOPS based on active params per token (MoE-aware)."""
        expert_ratio = self.config.num_experts_per_tok / self.config.num_experts
        active_params = 0
        for name, p in self.named_parameters():
            if '.experts.' in name:
                active_params += int(p.numel() * expert_ratio)
            elif name == 'tok_embeddings.weight':
                continue
            else:
                active_params += p.numel()
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = sum(12 * h * q * min(w[0], t) for w in self.window_sizes)
        return 6 * active_params + attn_flops

    def num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        expert_ratio = self.config.num_experts_per_tok / self.config.num_experts
        active = 0
        for name, p in self.named_parameters():
            if '.experts.' in name:
                active += int(p.numel() * expert_ratio)
            elif name == 'tok_embeddings.weight':
                active += p.numel()  # embedding lookup counted for memory
            else:
                active += p.numel()
        return {'total': total, 'active_per_token': active}

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95)):
        ddp, rank, local_rank, world_size = get_dist_info()
        adamw_scale = (self.config.n_embd / 768) ** -0.5

        matrix_params = []
        router_params = []
        norm_params = list(self.norm.parameters())

        for layer in self.layers:
            for name, p in layer.named_parameters():
                if 'router.gate' in name:
                    router_params.append(p)
                elif p.dim() == 1:
                    norm_params.append(p)
                else:
                    matrix_params.append(p)

        param_groups = []
        # Output head
        param_groups.append(dict(kind='adamw', params=list(self.output.parameters()),
                                 lr=unembedding_lr * adamw_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # Embeddings
        param_groups.append(dict(kind='adamw', params=list(self.tok_embeddings.parameters()),
                                 lr=embedding_lr * adamw_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # Norms
        param_groups.append(dict(kind='adamw', params=norm_params, lr=unembedding_lr * adamw_scale,
                                 betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # Router: lower LR, higher beta for stability
        if router_params:
            param_groups.append(dict(kind='adamw', params=router_params,
                                     lr=0.001 * adamw_scale, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01))
        # Matrix params via Muon
        for shape in sorted({p.shape for p in matrix_params}):
            group = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group, lr=matrix_lr,
                                     momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay))

        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(param_groups)
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]
        return optimizer

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                kv_cache=None, loss_reduction: str = 'mean'):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.tok_embeddings(idx)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i, layer in enumerate(self.layers):
            x, aux_loss = layer(x, cos_sin, self.window_sizes[i], kv_cache)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)

        # Attention clamping on output logits (Grok-1 style)
        logits = self.output(x)[..., :self.config.vocab_size].float()
        if self.config.attn_clamp > 0:
            c = self.config.attn_clamp
            logits = c * torch.tanh(logits / c)

        if targets is not None:
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                      ignore_index=-1, reduction=loss_reduction)
            if loss_reduction == 'mean':
                return lm_loss + total_aux_loss
            return lm_loss
        return logits
