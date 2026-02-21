"""
Grok MoE training script.

Uses nanollama's data loading, optimizer, and training infrastructure
but with the Grok MoE architecture.

Usage:
    python train.py --model-size grok-nano
    torchrun --nproc_per_node=4 train.py --model-size grok-small
"""

import os
import sys
import time
import argparse
from contextlib import nullcontext

sys.path.insert(0, '/home/ubuntu/nanollama')
sys.path.insert(0, '/home/ubuntu/grok')

import torch
import torch.distributed as dist

from grok_arch import Grok, GrokConfig, get_grok_config, GROK_CONFIGS
from nanollama.common import (
    compute_init, compute_cleanup, get_dist_info, print0, print_banner,
    autodetect_device_type, get_peak_flops, DummyWandb
)
from nanollama.dataloader import DistributedDataLoader
from nanollama.checkpoint_manager import save_checkpoint
from nanollama.tokenizer import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Grok MoE model")
    # Model
    parser.add_argument("--model-size", type=str, default="grok-nano",
                        choices=list(GROK_CONFIGS.keys()),
                        help="Grok model size")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    # Data
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--personality-dir", type=str, default=None)
    parser.add_argument("--personality-ratio", type=float, default=0.0)
    # Training
    parser.add_argument("--total-batch-size", type=int, default=524288)
    parser.add_argument("--device-batch-size", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=-1)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    # Logging & Checkpoints
    parser.add_argument("--run", type=str, default="grok")
    parser.add_argument("--model-tag", type=str, default="grok-base")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def get_lr_schedule(step, warmup_iters, max_iters, max_lr):
    """WSD schedule (same as nanollama)."""
    decay_start = int(max_iters * 0.50)
    if step < warmup_iters:
        return max_lr * (step + 1) / warmup_iters
    elif step < decay_start:
        return max_lr
    else:
        progress = (step - decay_start) / (max_iters - decay_start)
        return max_lr * (1 - progress)


def main():
    args = parse_args()
    device_type = autodetect_device_type()
    ddp, rank, local_rank, world_size, device = compute_init(device_type)

    if rank == 0:
        print_banner()

    # Config
    config = get_grok_config(args.model_size)
    config.vocab_size = args.vocab_size
    config.sequence_len = args.max_seq_len

    print0(f"\n{'='*60}")
    print0(f"Grok MoE Training — {args.model_size}")
    print0(f"  Layers: {config.n_layer}, Dim: {config.n_embd}, Heads: {config.n_head}, KV: {config.n_kv_head}")
    print0(f"  Experts: {config.num_experts}, Top-k: {config.num_experts_per_tok}, Shared: {config.shared_expert}")
    print0(f"  GELU: {config.use_gelu}, Double prenorm: {config.use_double_prenorm}, Attn clamp: {config.attn_clamp}")
    print0(f"{'='*60}\n")

    # Create model
    print0("Creating Grok model...")
    with torch.device('meta'):
        model = Grok(config)
    model.to_empty(device=device)
    model.init_weights()

    params = model.num_params()
    print0(f"Total params: {params['total']:,} ({params['total']/1e6:.1f}M)")
    print0(f"Active per token: {params['active_per_token']:,} ({params['active_per_token']/1e6:.1f}M)")

    # Auto iterations (Chinchilla 10x on active params)
    if args.num_iterations == -1:
        target_tokens = 10 * params['active_per_token']
        args.num_iterations = max(1000, target_tokens // args.total_batch_size)
        print0(f"Auto iterations: {args.num_iterations} ({target_tokens/1e9:.1f}B tokens)")

    # Compile
    if device_type == "cuda":
        print0("Compiling model...")
        model = torch.compile(model)

    # Optimizer
    print0("Setting up optimizer...")
    optimizer = model.setup_optimizer(weight_decay=args.weight_decay)

    # Data
    print0("Setting up data loader...")
    if args.data_dir is None:
        from nanollama.common import get_base_dir
        args.data_dir = os.path.join(get_base_dir(), "data", "fineweb")

    tokens_per_batch = args.device_batch_size * args.max_seq_len
    assert args.total_batch_size % (tokens_per_batch * world_size) == 0
    grad_accum_steps = args.total_batch_size // (tokens_per_batch * world_size)
    print0(f"Gradient accumulation: {grad_accum_steps}")

    try:
        data_loader = DistributedDataLoader(
            data_dir=args.data_dir,
            sequence_length=args.max_seq_len,
            batch_size=args.device_batch_size,
            rank=rank, world_size=world_size,
            personality_dir=args.personality_dir,
            personality_ratio=args.personality_ratio,
        )
    except Exception as e:
        print0(f"Warning: data loader failed: {e}")
        print0("Using dummy data for testing...")
        data_loader = None

    # Wandb
    if args.wandb and rank == 0:
        import wandb
        wandb.init(project="grok", name=args.run, config=vars(args))
        logger = wandb
    else:
        logger = DummyWandb()

    # Training
    print0("\nStarting training...")
    model.train()
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    peak_flops = get_peak_flops(torch.cuda.get_device_name() if device_type == "cuda" else "cpu")
    model_flops = model.estimate_flops()

    t0 = time.time()
    total_tokens = 0

    for step in range(args.num_iterations):
        lr = get_lr_schedule(step, args.warmup_iters, args.num_iterations, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr * (pg.get('initial_lr', lr) / args.lr)

        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            if data_loader:
                x, y = data_loader.next_batch()
            else:
                x = torch.randint(0, config.vocab_size, (args.device_batch_size, args.max_seq_len))
                y = torch.randint(0, config.vocab_size, (args.device_batch_size, args.max_seq_len))
            x, y = x.to(device), y.to(device)

            with autocast_ctx:
                loss = model(x, targets=y)
                loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += loss.item()
            total_tokens += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 and rank == 0:
            dt = time.time() - t0
            tok_per_sec = total_tokens / dt if dt > 0 else 0
            mfu = model_flops * tok_per_sec / peak_flops * 100 if peak_flops < float('inf') else 0
            print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | {tok_per_sec:.0f} tok/s | {mfu:.1f}% MFU")
            logger.log({
                'train/loss': loss_accum, 'train/lr': lr,
                'train/tok_per_sec': tok_per_sec, 'train/mfu': mfu,
                'train/total_tokens': total_tokens,
            })

        if args.save_every > 0 and step > 0 and step % args.save_every == 0 and rank == 0:
            from nanollama.common import get_base_dir
            checkpoint_dir = os.path.join(get_base_dir(), "checkpoints", args.model_tag)
            save_checkpoint(model=model, optimizer=optimizer, step=step,
                            config=vars(config), checkpoint_dir=checkpoint_dir)

        if ddp:
            dist.barrier()

    # Final save
    if rank == 0:
        from nanollama.common import get_base_dir
        checkpoint_dir = os.path.join(get_base_dir(), "checkpoints", args.model_tag)
        save_checkpoint(model=model, optimizer=optimizer, step=args.num_iterations,
                        config=vars(config), checkpoint_dir=checkpoint_dir)

    logger.finish()
    compute_cleanup()
    print0("\nGrok training complete!")


if __name__ == "__main__":
    main()
