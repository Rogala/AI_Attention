import torch
import time
import xformers
import xformers.ops as xops
import math
import os
import sys
from tabulate import tabulate

# --- Benchmark Configuration ---
CONFIG = {
    "B": 4,          # Batch size
    "H": 8,          # Number of heads
    "L": 2048,       # Sequence length
    "D": 64,         # Head dimension
    "WARMUP_STEPS": 5,
    "MEASURE_STEPS": 50
}

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check for FlashAttention availability
flash_attention_available = False
try:
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_func
    flash_attention_available = True
except ImportError:
    print("Warning: 'flash_attn' package not found. Its benchmark will be skipped.", flush=True)
    print("To install FlashAttention, try: pip install flash-attn --no-build-isolation", flush=True)
except Exception as e:
    print(f"Warning: An error occurred while importing 'flash_attn': {e}. Benchmark will be skipped.", flush=True)

# Check for native torch.float8 support (requires PyTorch 2.3+ and Ampere+ GPU)
gpu_compute_capability_8_plus = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

torch_float8_e4m3fn_available = hasattr(torch, 'float8_e4m3fn') and gpu_compute_capability_8_plus
torch_float8_e5m2_available = hasattr(torch, 'float8_e5m2') and gpu_compute_capability_8_plus

# --- Benchmark Settings ---
B, H, L, D = CONFIG["B"], CONFIG["H"], CONFIG["L"], CONFIG["D"]
WARMUP_STEPS = CONFIG["WARMUP_STEPS"]
MEASURE_STEPS = CONFIG["MEASURE_STEPS"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data types for testing
dtypes = [torch.float32, torch.float16, torch.bfloat16]
if torch_float8_e4m3fn_available:
    dtypes.append(torch.float8_e4m3fn)
if torch_float8_e5m2_available:
    dtypes.append(torch.float8_e5m2)

dtypes.append("fp8_e4m3_fake")
dtypes.append("fp8_e5m2_fake")
dtypes.append("fp4_fake")

# --- Fake Quantization/Dequantization Functions ---
def fake_fp4_quant(tensor):
    scale = tensor.abs().max() / 7
    tensor_q = (tensor / scale).round().clamp(-8, 7).to(torch.int8)
    return tensor_q, scale

def fake_fp4_dequant(tensor_q, scale):
    return tensor_q.to(torch.float32) * scale

def fake_fp8_quant(tensor):
    scale = tensor.abs().max() / 127.0
    tensor_q = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    return tensor_q, scale

def fake_fp8_dequant(tensor_q, scale):
    return tensor_q.to(torch.float32) * scale

# --- Attention Methods ---
def pytorch_sdpa(q, k, v, attn_mask=None):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

def xformers_attention(q, k, v):
    return xops.memory_efficient_attention(q, k, v)

def separate_flash_attention(q, k, v, attn_mask=None):
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()
    output = flash_attn_func(q_fa, k_fa, v_fa)
    return output.transpose(1, 2).contiguous()

# --- Logger for console and file output ---
log_file = None

def print_and_log(msg):
    print(msg, flush=True)
    if log_file:
        try:
            log_file.write(msg + "\n")
            log_file.flush()
        except Exception as e:
            print(f"Error writing to log file: {e}", flush=True)

# --- Benchmark Function ---
@torch.no_grad()
def benchmark_attention(attn_fn, name, dtype, q_base, k_base, v_base, attn_mask=None):
    q = q_base.clone().to(device)
    k = k_base.clone().to(device)
    v = v_base.clone().to(device)

    if dtype == "fp4_fake":
        q_q, s_q = fake_fp4_quant(q)
        k_q, s_k = fake_fp4_quant(k)
        v_q, s_v = fake_fp4_quant(v)
        q = fake_fp4_dequant(q_q, s_q)
        k = fake_fp4_dequant(k_q, s_k)
        v = fake_fp4_dequant(v_q, s_v)
    elif dtype in ["fp8_e4m3_fake", "fp8_e5m2_fake"]:
        q_q, s_q = fake_fp8_quant(q)
        k_q, s_k = fake_fp8_quant(k)
        v_q, s_v = fake_fp8_quant(v)
        q = fake_fp8_dequant(q_q, s_q)
        k = fake_fp8_dequant(k_q, s_k)
        v = fake_fp8_dequant(v_q, s_v)
    elif not isinstance(dtype, str):
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)

    if attn_fn == separate_flash_attention:
        if dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(f"FlashAttention does not support dtype {dtype}.")

    for _ in range(WARMUP_STEPS):
        _ = attn_fn(q, k, v, attn_mask=attn_mask) if attn_fn == pytorch_sdpa else attn_fn(q, k, v)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        _ = attn_fn(q, k, v, attn_mask=attn_mask) if attn_fn == pytorch_sdpa else attn_fn(q, k, v)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / MEASURE_STEPS * 1000

# --- Main Script Execution ---
if __name__ == "__main__":
    log_filename = os.path.join(script_dir, "bench_full.txt")
    try:
        log_file = open(log_filename, "w")
        print_and_log("--- Attention Speed Benchmark ---")
    except IOError as e:
        print(f"ERROR: Failed to open log file {log_filename}: {e}", flush=True)

    print_and_log(f"Torch Version: {torch.__version__}")
    print_and_log(f"Xformers Version: {xformers.__version__}")
    print_and_log(f"Flash-attn Version: {flash_attn.__version__}" if flash_attention_available else "Flash-attn: Not available")
    print_and_log(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print_and_log(f"GPU: {torch.cuda.get_device_name(0)}")
        print_and_log(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")

    print_and_log(f"Native torch.float8_e4m3fn available: {torch_float8_e4m3fn_available}")
    print_and_log(f"Native torch.float8_e5m2 available: {torch_float8_e5m2_available}")
    print_and_log(f"Tensor Shape: B={B}, H={H}, L={L}, D={D}\n")

    q_base = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    k_base = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    v_base = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

    results_data = []
    headers = ["Dtype", "PyTorch SDPA (ms)", "Xformers MEA (ms)", "FlashAttn (ms)"]

    for dtype in dtypes:
        dtype_str = str(dtype).replace("torch.", "") if not isinstance(dtype, str) else dtype

        try:
            sdpa_time = benchmark_attention(pytorch_sdpa, "SDPA", dtype, q_base, k_base, v_base)
        except:
            sdpa_time = float('nan')

        try:
            xformers_time = benchmark_attention(xformers_attention, "Xformers", dtype, q_base, k_base, v_base)
        except:
            xformers_time = float('nan')

        try:
            flashattn_time = benchmark_attention(separate_flash_attention, "FlashAttn", dtype, q_base, k_base, v_base) \
                             if flash_attention_available and dtype in [torch.float16, torch.bfloat16] else float('nan')
        except:
            flashattn_time = float('nan')

        results_data.append([
            dtype_str,
            f"{sdpa_time:.2f}" if not math.isnan(sdpa_time) else "N/A",
            f"{xformers_time:.2f}" if not math.isnan(xformers_time) else "N/A",
            f"{flashattn_time:.2f}" if not math.isnan(flashattn_time) else "N/A"
        ])

    print_and_log("\n--- Summary (ms/iter) ---")
    print_and_log(tabulate(results_data, headers=headers, tablefmt="grid"))
    print_and_log("\nNote: 'N/A' means a benchmark was skipped or failed.")

    if log_file:
        log_file.close()
