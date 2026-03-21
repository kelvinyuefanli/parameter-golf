#!/usr/bin/env python3
"""Smoke test for the recurrent GPT architecture.

Validates all new codepaths without needing real data or GPU:
  1. FactoredEmbedding forward + logits
  2. GPT forward with depth recurrence (3-block loop)
  3. Per-depth scales and per-loop U-Net skips
  4. Peri-LN (post attn/mlp RMSNorm)
  5. QAT STE fake quantization
  6. Int8 quantization with FP16 embed passthrough
  7. Dequantization roundtrip
  8. Loss is finite and decreasing over 5 steps

Run: python3 smoke_test.py
"""
from __future__ import annotations

import io
import math
import sys
import zlib

import torch
import torch.nn.functional as F
from torch import Tensor

# Import from train_gpt.py
sys.path.insert(0, ".")
from train_gpt import (
    Block,
    CastedLinear,
    FactoredEmbedding,
    GPT,
    RMSNorm,
    _fake_quantize_int8,
    dequantize_state_dict_int8,
    quantize_state_dict_int8,
)


def check(condition: bool, msg: str) -> None:
    if condition:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def main() -> None:
    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float32

    print("=== 1. FactoredEmbedding ===")
    fem = FactoredEmbedding(vocab_size=64, dim=32, rank=8, init_std=0.1).to(device)
    ids = torch.randint(0, 64, (2, 16))
    emb = fem(ids)
    check(emb.shape == (2, 16, 32), f"embed shape {emb.shape} == (2, 16, 32)")
    check(torch.isfinite(emb).all().item(), "embed values are finite")

    x_flat = torch.randn(2 * 16, 32)
    logits = fem.logits(x_flat)
    check(logits.shape == (32, 64), f"logits shape {logits.shape} == (32, 64)")
    check(torch.isfinite(logits).all().item(), "logits values are finite")

    print("\n=== 2. GPT with depth recurrence ===")
    model = GPT(
        vocab_size=64,
        num_layers=3,
        num_loops=2,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        embed_rank=8,
        tie_embeddings=True,
        tied_embed_init_std=0.1,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    ).to(device).to(dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    loss = model(x, y)
    check(torch.isfinite(loss).item(), f"loss is finite: {loss.item():.4f}")
    check(loss.item() > 0, f"loss is positive: {loss.item():.4f}")

    print("\n=== 3. Per-depth scales ===")
    total_effective = 3 * 2  # num_layers * num_loops
    check(model.depth_scales.shape == (total_effective, 32), f"depth_scales shape {model.depth_scales.shape}")
    check((model.depth_scales == 1.0).all().item(), "depth_scales initialized to 1.0")

    print("\n=== 4. Per-loop skip weights ===")
    check(model.skip_weights.shape == (2, 32), f"skip_weights shape {model.skip_weights.shape}")

    print("\n=== 5. Peri-LN ===")
    block = model.blocks[0]
    check(hasattr(block, "post_attn_norm"), "Block has post_attn_norm")
    check(hasattr(block, "post_mlp_norm"), "Block has post_mlp_norm")
    check(isinstance(block.post_attn_norm, RMSNorm), "post_attn_norm is RMSNorm")

    print("\n=== 6. QAT STE fake quantization ===")
    w = torch.randn(32, 32)
    wq = _fake_quantize_int8(w)
    check(wq.shape == w.shape, "fake_quant preserves shape")
    check(torch.isfinite(wq).all().item(), "fake_quant values finite")
    # STE: gradient flows through
    w_param = torch.nn.Parameter(torch.randn(32, 32))
    w_ste = w_param + (_fake_quantize_int8(w_param) - w_param).detach()
    loss_q = w_ste.sum()
    loss_q.backward()
    check(w_param.grad is not None, "STE gradient flows through")
    check(torch.isfinite(w_param.grad).all().item(), "STE gradient is finite")

    # Enable QAT on CastedLinear
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m._qat = True
    loss_qat = model(x, y)
    check(torch.isfinite(loss_qat).item(), f"loss with QAT is finite: {loss_qat.item():.4f}")
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m._qat = False

    print("\n=== 7. Training (5 steps, verify loss decreases) ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  step {step}: loss={loss.item():.4f}")
    check(losses[-1] < losses[0], f"loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")

    print("\n=== 8. Int8 quantization with FP16 embed passthrough ===")
    state = model.state_dict()
    quant_obj, quant_stats = quantize_state_dict_int8(state)
    # Check that embedding tensors are in passthrough (fp16), not quantized
    passthrough_names = set(quant_obj["passthrough"].keys())
    embed_in_passthrough = any("tok_emb" in n or "E_low" in n or "E_up" in n for n in passthrough_names)
    check(embed_in_passthrough, "embedding tensors in fp16 passthrough")
    # Check no embed tensors in quantized
    quantized_names = set(quant_obj["quantized"].keys())
    embed_in_quantized = any("tok_emb" in n or "E_low" in n or "E_up" in n for n in quantized_names)
    check(not embed_in_quantized, "no embedding tensors in int8 quantized")

    print("\n=== 9. Serialization roundtrip ===")
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)
    print(f"  Raw size: {len(raw):,} bytes")
    print(f"  Compressed: {len(compressed):,} bytes")
    print(f"  Compression ratio: {len(raw)/len(compressed):.2f}x")

    # Dequantize and reload
    loaded_obj = torch.load(io.BytesIO(zlib.decompress(compressed)), map_location="cpu")
    restored_state = dequantize_state_dict_int8(loaded_obj)
    model.load_state_dict(restored_state, strict=True)
    loss_roundtrip = model(x, y)
    check(torch.isfinite(loss_roundtrip).item(), f"roundtrip loss is finite: {loss_roundtrip.item():.4f}")
    # Roundtrip loss should be close to pre-quant loss (within ~10% due to quantization noise)
    delta = abs(loss_roundtrip.item() - losses[-1])
    check(delta < 0.5, f"roundtrip loss delta {delta:.4f} < 0.5")

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
