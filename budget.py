#!/usr/bin/env python3
"""Parameter budget calculator for Parameter Golf submissions.

Run: python3 budget.py
Or after training: python3 budget.py --check final_model.int8.ptz train_gpt.py
"""
import argparse
import os
import sys


TOTAL_BUDGET = 16_000_000
# Empirical bytes/param ratios (after quantization + zlib)
COMPRESSION_RATIOS = {
    "int8+zlib": 0.927,
    "int6+zlib": 0.68,
    "int4+zlib": 0.45,
    "fp16": 2.0,
}


def calc_params(dim, layers, vocab, num_heads, num_kv_heads, mlp_mult, embed_rank=0):
    """Calculate parameter count for a GPT model configuration."""
    head_dim = dim // num_heads
    kv_dim = num_kv_heads * head_dim

    # Embedding (tied)
    if embed_rank > 0:
        emb_params = vocab * embed_rank + embed_rank * dim
    else:
        emb_params = vocab * dim

    # Per block: attention (Q, K, V, proj) + MLP (fc, proj) + scalars
    attn = dim * dim + dim * kv_dim + dim * kv_dim + dim * dim
    mlp = dim * (mlp_mult * dim) + (mlp_mult * dim) * dim
    scalars = dim + dim + 2 * dim + num_heads  # attn_scale, mlp_scale, resid_mix, q_gain
    block = attn + mlp + scalars

    # Skip weights
    num_enc = layers // 2
    num_dec = layers - num_enc
    skip = min(num_enc, num_dec) * dim

    total = emb_params + layers * block + skip
    return {
        "total": total,
        "embedding": emb_params,
        "per_block": block,
        "blocks_total": layers * block,
        "skip": skip,
        "layers": layers,
    }


def estimate_size(params, quant="int8+zlib", fp16_embed=False):
    """Estimate compressed artifact size in bytes."""
    if fp16_embed:
        non_emb = params["total"] - params["embedding"]
        return params["embedding"] * COMPRESSION_RATIOS["fp16"] + non_emb * COMPRESSION_RATIOS[quant]
    return params["total"] * COMPRESSION_RATIOS[quant]


def check_actual_files(model_path, code_path):
    """Check actual file sizes on disk."""
    model_size = os.path.getsize(model_path)
    code_size = os.path.getsize(code_path)
    total = model_size + code_size
    fits = total <= TOTAL_BUDGET
    margin = TOTAL_BUDGET - total
    print(f"Model:  {model_path} = {model_size:,} bytes")
    print(f"Code:   {code_path} = {code_size:,} bytes")
    print(f"Total:  {total:,} bytes")
    print(f"Budget: {TOTAL_BUDGET:,} bytes")
    print(f"Margin: {margin:+,} bytes")
    print(f"Status: {'FITS' if fits else 'OVER BUDGET'}")
    return fits


def print_config_table():
    """Print comparison table of candidate configurations."""
    configs = [
        # (name, dim, layers, vocab, heads, kv, mlp_m, embed_rank, unique_L, loops, quant, fp16e)
        ("Baseline (current)",           512, 9,  1024, 8,  4, 2, 0,  9, 1, "int8+zlib", False),
        ("+ FP16 embed (over budget!)",   512, 9,  1024, 8,  4, 2, 0,  9, 1, "int8+zlib", True),
        ("SP4096 fact E=64",             512, 9,  4096, 8,  4, 2, 64, 9, 1, "int8+zlib", True),
        ("Recur 3x3 d=768",             768, 3,  1024, 8,  4, 2, 0,  3, 3, "int8+zlib", True),
        ("Recur 3x3 SP4096 d=768",      768, 3,  4096, 8,  4, 2, 64, 3, 3, "int8+zlib", True),
        ("INT4 12L x 576",              576, 12, 1024, 8,  4, 2, 0,  12,1, "int4+zlib", True),
        ("INT4 Recur 3x3 d=960",        960, 3,  4096, 10, 4, 2, 64, 3, 3, "int4+zlib", True),
    ]

    hdr = f"{'Config':<35} {'Params':>10} {'Emb':>8} {'Eff.L':>5} {'Est.Size':>10} {'Fits':>5} {'Margin':>9}"
    print(hdr)
    print("-" * len(hdr))

    for name, dim, layers, vocab, heads, kv, mlp_m, er, unique_l, loops, quant, fp16e in configs:
        p = calc_params(dim, unique_l, vocab, heads, kv, mlp_m, er)
        eff_layers = unique_l * loops
        code_est = 50_000
        size = estimate_size(p, quant, fp16e) + code_est
        fits = size <= TOTAL_BUDGET
        margin = TOTAL_BUDGET - size
        print(f"{name:<35} {p['total']:>10,} {p['embedding']:>8,} {eff_layers:>5} {size:>10,.0f} {'YES' if fits else 'NO':>5} {margin:>+9,.0f}")

    print()
    print("Notes:")
    print("  Recurrent: unique params shown; effective layers = unique x loops")
    print("  FP16 embed: embedding at 2 bytes/param, rest at specified quant")
    print("  Factored E=64: embed = vocab*64 + 64*dim")
    print("  INT4 assumes ~0.45 bytes/param after packing + zlib")


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf budget calculator")
    parser.add_argument("--check", nargs=2, metavar=("MODEL", "CODE"),
                        help="Check actual file sizes: --check model.ptz train_gpt.py")
    args = parser.parse_args()

    if args.check:
        model_path, code_path = args.check
        if not os.path.exists(model_path):
            print(f"Error: {model_path} not found")
            sys.exit(1)
        if not os.path.exists(code_path):
            print(f"Error: {code_path} not found")
            sys.exit(1)
        check_actual_files(model_path, code_path)
    else:
        print_config_table()


if __name__ == "__main__":
    main()
