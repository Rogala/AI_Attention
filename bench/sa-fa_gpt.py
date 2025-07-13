# Optimized SageAttention vs FlashAttention Benchmark with tabulate summary

import torch
import torch.nn.functional as F
import time
import os
import math
from torch.nn.attention import SDPBackend, sdpa_kernel
from tabulate import tabulate

# Check SageAttention availability
SAGEATTN_AVAILABLE = False
try:
    from sageattention import sageattn
    SAGEATTN_AVAILABLE = True
    print("SageAttention imported successfully.")
except ImportError as e:
    print(f"Warning: Could not import SageAttention. {e}")

# Check FlashAttention availability (via SDPA)
FLASHATTN_AVAILABLE = True
try:
    _ = F.scaled_dot_product_attention(
        torch.rand(1, 1, 2, 2).cuda(), torch.rand(1, 1, 2, 2).cuda(), torch.rand(1, 1, 2, 2).cuda()
    )
except Exception as e:
    FLASHATTN_AVAILABLE = False
    print(f"Warning: FlashAttention via SDPA not available: {e}")

# --- Config ---
CONFIG = {
    "BATCH_SIZE": 4,
    "HEAD_NUM": 32,
    "HEAD_DIM": 128,
    "DTYPE": torch.float16,
    "SEQ_LENS": [1024, 2048],
    "NUM_WARMUP": 20,
    "NUM_RUNS": 100
}

# --- FLOPs Estimation ---
def get_flops(batch_size, head_num, seq_len, head_dim, is_causal):
    flops = 2 * batch_size * head_num * seq_len * seq_len * head_dim
    return flops // 2 if is_causal else flops

# --- Benchmark Function ---
def run_benchmark():
    B, H, D = CONFIG["BATCH_SIZE"], CONFIG["HEAD_NUM"], CONFIG["HEAD_DIM"]
    dtype = CONFIG["DTYPE"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        print("ERROR: CUDA is required.")
        return

    results = []
    if FLASHATTN_AVAILABLE:
        torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    for is_causal in [False, True]:
        for seq_len in CONFIG["SEQ_LENS"]:
            q = torch.randn(B, H, seq_len, D, dtype=dtype, device=device)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # --- FlashAttention Benchmark ---
            fa_time = fa_flops = "N/A"
            if FLASHATTN_AVAILABLE:
                try:
                    for _ in range(CONFIG["NUM_WARMUP"]):
                        F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(CONFIG["NUM_RUNS"]):
                        F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                    torch.cuda.synchronize()
                    fa_time = (time.perf_counter() - start) / CONFIG["NUM_RUNS"] * 1000
                    fa_flops = get_flops(B, H, seq_len, D, is_causal) / (fa_time / 1000) / 1e12
                except Exception as e:
                    fa_flops = "ERR"

            # --- SageAttention Benchmark ---
            sa_time = sa_flops = "N/A"
            if SAGEATTN_AVAILABLE:
                try:
                    for _ in range(CONFIG["NUM_WARMUP"]):
                        sageattn(q, k, v, is_causal=is_causal)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(CONFIG["NUM_RUNS"]):
                        sageattn(q, k, v, is_causal=is_causal)
                    torch.cuda.synchronize()
                    sa_time = (time.perf_counter() - start) / CONFIG["NUM_RUNS"] * 1000
                    sa_flops = get_flops(B, H, seq_len, D, is_causal) / (sa_time / 1000) / 1e12
                except Exception as e:
                    sa_flops = "ERR"

            results.append([
                f"Causal={is_causal} Seq={seq_len}",
                f"{sa_flops:.3f}" if isinstance(sa_flops, float) else sa_flops,
                f"{fa_flops:.3f}" if isinstance(fa_flops, float) else fa_flops
            ])

    # --- Summary Output ---
    headers = ["Setting", "SageAttention (TFLOPS)", "FlashAttention (TFLOPS)"]
    print("\n" + tabulate(results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    run_benchmark()
