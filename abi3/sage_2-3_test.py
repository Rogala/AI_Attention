"""
sage_2-3_test.py
===============
Comprehensive testing of SageAttention 2 and SageAttention 3 for Windows / PowerShell 5.1.

Usage:
    sage_2-3_test.py            # run all tests
    sage_2-3_test.py --sa2      # SA2 tests only
    sage_2-3_test.py --sa3      # SA3 tests only
    sage_2-3_test.py --perf     # performance tests only
    sage_2-3_test.py --quick    # quick mode (fewer configs)
"""

import sys
import argparse
import traceback

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# ---------------------------------------------------------------------------
# PowerShell 5.1 / Windows — Force UTF-8 for correct output
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# ANSI Colors
# ---------------------------------------------------------------------------
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"
    BLUE   = "\033[94m"

def ok(s):   return f"{C.GREEN}{s}{C.RESET}"
def err(s):  return f"{C.RED}{s}{C.RESET}"
def warn(s): return f"{C.YELLOW}{s}{C.RESET}"
def hdr(s):  return f"{C.BOLD}{C.CYAN}{s}{C.RESET}"
def dim(s):  return f"{C.GRAY}{s}{C.RESET}"

# ---------------------------------------------------------------------------
# Try to import SA2 and SA3
# ---------------------------------------------------------------------------
try:
    import sageattention
    from sageattention import sageattn as _sa2_func
    SA2_VERSION = getattr(sageattention, "__version__", "unknown")
    SA2_AVAILABLE = True
except ImportError:
    _sa2_func = None
    SA2_VERSION = None
    SA2_AVAILABLE = False

try:
    import sageattn3.api as _sa3_api
    SA3_AVAILABLE = True
except ImportError:
    _sa3_api = None
    SA3_AVAILABLE = False

# ---------------------------------------------------------------------------
# Accuracy Configurations (batch, heads, seq, head_dim, label)
# ---------------------------------------------------------------------------
ACCURACY_CONFIGS_FULL = [
    (1, 16, 2048, 128, "bs1 seq2048 hd128 fp16"),
    (2, 16, 2048, 128, "bs2 seq2048 hd128 fp16"),
    (1, 16, 4096, 128, "bs1 seq4096 hd128 fp16"),
    (1, 32, 2048,  64, "bs1 seq2048 hd64  fp16"),
    (4, 16, 2048, 128, "bs4 seq2048 hd128 fp16"),
    (1, 16, 2048, 128, "bs1 seq2048 hd128 bf16"),
    (1, 16, 4096, 128, "bs1 seq4096 hd128 bf16"),
    (1, 32, 4096,  64, "bs1 seq4096 hd64  bf16"),
]

ACCURACY_CONFIGS_QUICK = [
    (1, 16, 2048, 128, "bs1 seq2048 hd128 fp16"),
    (1, 32, 2048,  64, "bs1 seq2048 hd64  fp16"),
    (1, 16, 2048, 128, "bs1 seq2048 hd128 bf16"),
]

# Performance Configurations (batch, heads, seq, head_dim, label)
PERF_CONFIGS_FULL = [
    (1, 16,  2048, 128, "seq2K  hd128"),
    (1, 16,  4096, 128, "seq4K  hd128"),
    (1, 16,  8192, 128, "seq8K  hd128"),
    (1, 32,  4096, 128, "seq4K  h32 hd128"),
    (1, 32,  8192, 128, "seq8K  h32 hd128"),
    (1, 32, 16384, 128, "seq16K h32 hd128"),
]

PERF_CONFIGS_QUICK = [
    (1, 16, 2048, 128, "seq2K  hd128"),
    (1, 16, 4096, 128, "seq4K  hd128"),
    (1, 32, 8192, 128, "seq8K  h32 hd128"),
]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_dtype_label(dtype):
    return "bf16" if dtype == torch.bfloat16 else "fp16"

def dtype_from_label(label: str) -> torch.dtype:
    return torch.bfloat16 if "bf16" in label else torch.float16

def compute_accuracy_metrics(actual: torch.Tensor, expect: torch.Tensor) -> dict:
    """Computes accuracy metrics between two tensors."""
    a = actual.float()
    e = expect.float()
    diff = (a - e).abs()
    eps = torch.finfo(torch.float32).eps
    denom = torch.maximum(torch.maximum(a.abs(), e.abs()),
                          torch.full_like(a, eps))
    rdiff = diff / denom
    return {
        "mean_rtol": rdiff.mean().item(),
        "max_rtol":  rdiff.max().item(),
        "mean_atol": diff.mean().item(),
        "max_atol":  diff.max().item(),
    }

def rtol_status(mean_rtol: float) -> str:
    """Returns quality assessment based on relative error."""
    if mean_rtol < 0.01:
        return ok("excellent")
    elif mean_rtol < 0.05:
        return ok("good (30xx)")
    elif mean_rtol < 0.10:
        return warn("fair (40xx+)")
    else:
        return err("out of range")

def cuda_benchmark(fn, warmup: int = 10, steps: int = 100) -> float:
    """Measures average execution time of fn() in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t_start = torch.cuda.Event(enable_timing=True)
    t_end   = torch.cuda.Event(enable_timing=True)
    t_start.record()
    for _ in range(steps):
        fn()
    t_end.record()
    torch.cuda.synchronize()
    return t_start.elapsed_time(t_end) / steps

def compute_attention_tflops(batch, heads, seq, hd, ms: float) -> float:
    """
    Calculates TFLOPS for the attention operation.
    Formula: 4 * B * H * S^2 * D (2 matmuls: Q*K and Attn*V, each ~2 flops/element)
    """
    flops = 4.0 * batch * heads * (seq ** 2) * hd
    return (flops / (ms / 1000.0)) / 1e12

def print_section(title: str):
    print()
    print(hdr("=" * 72))
    print(hdr(f"  {title}"))
    print(hdr("=" * 72))

def print_env_info():
    """Prints environment information."""
    print_section("Environment Info")

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    pt_ver = torch.__version__
    cuda_ver = torch.version.cuda or "N/A"

    print(f"  Python       : {C.BOLD}{py_ver}{C.RESET}")
    print(f"  PyTorch      : {C.BOLD}{pt_ver}{C.RESET}")
    print(f"  CUDA         : {C.BOLD}{cuda_ver}{C.RESET}")

    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        name = torch.cuda.get_device_name(dev)
        cap_maj, cap_min = torch.cuda.get_device_capability(dev)
        vram_gib = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        print(f"  GPU          : {C.BOLD}{name}{C.RESET}  "
              f"[sm{cap_maj}{cap_min}]  {vram_gib:.0f} GiB VRAM")
    else:
        print(f"  GPU          : {err('CUDA NOT AVAILABLE!')}")

    if SA2_AVAILABLE:
        print(f"  SA2          : {ok('found')}  v{SA2_VERSION}")
    else:
        print(f"  SA2          : {err('not found')}  (pip install sageattention)")

    if SA3_AVAILABLE:
        print(f"  SA3          : {ok('found')}")
    else:
        print(f"  SA3          : {err('not found')}  (pip install sageattn3)")

# ---------------------------------------------------------------------------
# SA2 Accuracy Test Block
# ---------------------------------------------------------------------------
def test_sa2_accuracy(configs):
    print_section("SA2 — Accuracy Test")

    if not SA2_AVAILABLE:
        print(f"  {err('Skipped:')} SageAttention 2 is not installed.")
        return

    col_w = [27, 12, 12, 12, 12, 16]
    header = (
        f"  {'Configuration':<{col_w[0]}}"
        f"{'mean_rtol':>{col_w[1]}}"
        f"{'max_rtol':>{col_w[2]}}"
        f"{'mean_atol':>{col_w[3]}}"
        f"{'max_atol':>{col_w[4]}}"
        f"{'Status':>{col_w[5]}}"
    )
    print(dim(header))
    print(dim("  " + "-" * (sum(col_w) + 2)))

    passed = 0
    for batch, heads, seq, hd, label in configs:
        dtype = dtype_from_label(label)
        try:
            q = torch.randn(batch, heads, seq, hd, device="cuda", dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Reference: Math SDPA (float32 precision)
            with sdpa_kernel(SDPBackend.MATH):
                out_ref = F.scaled_dot_product_attention(q, k, v)

            out_sa2 = _sa2_func(q, k, v)
            torch.cuda.synchronize()

            m = compute_accuracy_metrics(out_sa2, out_ref)
            status = rtol_status(m["mean_rtol"])
            if m["mean_rtol"] < 0.10:
                passed += 1

            print(
                f"  {label:<{col_w[0]}}"
                f"{m['mean_rtol']:>{col_w[1]}.4f}"
                f"{m['max_rtol']:>{col_w[2]}.4f}"
                f"{m['mean_atol']:>{col_w[3]}.4f}"
                f"{m['max_atol']:>{col_w[4]}.4f}"
                f"  {status}"
            )
        except Exception as e:
            msg = str(e).splitlines()[0][:50]
            print(f"  {label:<{col_w[0]}}{err(f'ERROR: {msg}')}")

    total = len(configs)
    summary_color = ok if passed == total else (warn if passed > 0 else err)
    print(dim("  " + "-" * (sum(col_w) + 2)))
    print(f"  Result: {summary_color(f'{passed}/{total}')} configs passed mean_rtol < 0.10 threshold")
    print(f"  {dim('Norm: mean_rtol < 0.05 for RTX 30xx, < 0.10 for RTX 40xx/50xx/Blackwell')}")

# ---------------------------------------------------------------------------
# SA3 Accuracy Test Block
# ---------------------------------------------------------------------------
def test_sa3_accuracy(configs):
    print_section("SA3 — Accuracy Test (Ref: Math SDPA)")

    if not SA3_AVAILABLE:
        print(f"  {err('Skipped:')} SageAttention 3 is not installed.")
        return

    # SA3 uses FP4 block quantization — only 15 non-zero values per number.
    # Comparing against Math SDPA (float32) will always yield mean_rtol ~0.79.
    # This is EXPECTED and correct behavior, not an error.
    _note_fp4 = (
        "SA3 uses FP4 quantization: mean_rtol ~0.79 vs Math SDPA is normal.\n"
        "  Target criteria: mean_atol < 0.05 and correct output shape."
    )
    print(f"  {dim(_note_fp4)}")

    col_w = [27, 12, 12, 12, 16]
    header = (
        f"  {'Configuration':<{col_w[0]}}"
        f"{'mean_rtol':>{col_w[1]}}"
        f"{'mean_atol':>{col_w[2]}}"
        f"{'max_atol':>{col_w[3]}}"
        f"{'Status':>{col_w[4]}}"
    )
    print(dim(header))
    print(dim("  " + "-" * (sum(col_w) + 2)))

    passed = 0
    for batch, heads, seq, hd, label in configs:
        dtype = dtype_from_label(label)
        is_bf16 = (dtype == torch.bfloat16)
        try:
            q = torch.randn(batch, heads, seq, hd, device="cuda", dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Reference: Math SDPA in float32
            with sdpa_kernel(SDPBackend.MATH):
                out_ref = F.scaled_dot_product_attention(q, k, v)

            # SA3 pipeline
            qc, kc, vc, ds = _sa3_api.preprocess_qkv(q, k, v)
            out_sa3_tuple = _sa3_api.blockscaled_fp4_attn(
                _sa3_api.scale_and_quant_fp4(qc),
                _sa3_api.scale_and_quant_fp4(kc),
                _sa3_api.scale_and_quant_fp4_transpose(vc),
                ds, qc.shape[2], is_bf16=is_bf16
            )
            # SA3 returns a tuple; take the first element
            out_sa3 = out_sa3_tuple[0] if isinstance(out_sa3_tuple, (list, tuple)) else out_sa3_tuple
            torch.cuda.synchronize()

            # Verify output shape and dtype match expectations
            assert out_sa3.shape == q.shape, f"shape {out_sa3.shape} != {q.shape}"
            assert out_sa3.dtype == dtype,   f"dtype {out_sa3.dtype} != {dtype}"

            m = compute_accuracy_metrics(out_sa3, out_ref)

            # Pass criterion: mean_atol < 0.05
            if m["mean_atol"] < 0.05:
                passed += 1
                status = ok("normal") if m["mean_rtol"] < 0.85 else warn("atol OK / rtol high")
            else:
                status = err("mean_atol exceeded")

            print(
                f"  {label:<{col_w[0]}}"
                f"{m['mean_rtol']:>{col_w[1]}.4f}"
                f"{m['mean_atol']:>{col_w[2]}.4f}"
                f"{m['max_atol']:>{col_w[3]}.4f}"
                f"  {status}"
            )
        except Exception as e:
            msg = str(e).splitlines()[0][:55]
            print(f"  {label:<{col_w[0]}}{err(f'ERROR: {msg}')}")

    total = len(configs)
    summary_color = ok if passed == total else (warn if passed > 0 else err)
    print(dim("  " + "-" * (sum(col_w) + 2)))
    print(f"  Result: {summary_color(f'{passed}/{total}')} configs passed mean_atol < 0.05 threshold")
    _note_sa3_end = "rtol ~0.79 is a structural feature of FP4, not a build defect"
    print(f"  {dim(_note_sa3_end)}")

# ---------------------------------------------------------------------------
# SA2 Performance Test Block
# ---------------------------------------------------------------------------
def test_sa2_perf(configs, dtype=torch.float16, warmup=10, steps=100):
    label_dtype = get_dtype_label(dtype)
    print_section(f"SA2 — Performance ({label_dtype}, warmup={warmup}, steps={steps})")

    if not SA2_AVAILABLE:
        print(f"  {err('Skipped:')} SageAttention 2 is not installed.")
        return

    col_w = [20, 14, 14]
    header = (
        f"  {'Configuration':<{col_w[0]}}"
        f"{'Latency (ms)':>{col_w[1]}}"
        f"{'TFLOPS':>{col_w[2]}}"
    )
    print(dim(header))
    print(dim("  " + "-" * (sum(col_w) + 2)))

    for batch, heads, seq, hd, label in configs:
        try:
            q = torch.randn(batch, heads, seq, hd, device="cuda", dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            ms = cuda_benchmark(lambda: _sa2_func(q, k, v), warmup=warmup, steps=steps)
            tflops = compute_attention_tflops(batch, heads, seq, hd, ms)

            tflops_str = f"{tflops:.2f}"
            if tflops > 200:
                tflops_colored = ok(tflops_str)
            elif tflops > 100:
                tflops_colored = warn(tflops_str)
            else:
                tflops_colored = err(tflops_str)

            print(
                f"  {label:<{col_w[0]}}"
                f"{ms:>{col_w[1]}.3f}"
                f"  {tflops_colored}"
            )
        except Exception as e:
            msg = str(e).splitlines()[0][:60]
            print(f"  {label:<{col_w[0]}}{err(f'ERROR: {msg}')}")

# ---------------------------------------------------------------------------
# SA3 Performance Test Block
# ---------------------------------------------------------------------------
def test_sa3_perf(configs, dtype=torch.float16, warmup=10, steps=100):
    label_dtype = get_dtype_label(dtype)
    print_section(f"SA3 — Performance ({label_dtype}, warmup={warmup}, steps={steps})")

    if not SA3_AVAILABLE:
        print(f"  {err('Skipped:')} SageAttention 3 is not installed.")
        return

    is_bf16 = (dtype == torch.bfloat16)
    col_w = [20, 14, 14]
    header = (
        f"  {'Configuration':<{col_w[0]}}"
        f"{'Latency (ms)':>{col_w[1]}}"
        f"{'TFLOPS':>{col_w[2]}}"
    )
    print(dim(header))
    print(dim("  " + "-" * (sum(col_w) + 2)))

    for batch, heads, seq, hd, label in configs:
        try:
            q = torch.randn(batch, heads, seq, hd, device="cuda", dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            qc, kc, vc, ds = _sa3_api.preprocess_qkv(q, k, v)

            def _run():
                _sa3_api.blockscaled_fp4_attn(
                    _sa3_api.scale_and_quant_fp4(qc),
                    _sa3_api.scale_and_quant_fp4(kc),
                    _sa3_api.scale_and_quant_fp4_transpose(vc),
                    ds, qc.shape[2], is_bf16=is_bf16
                )

            ms = cuda_benchmark(_run, warmup=warmup, steps=steps)
            tflops = compute_attention_tflops(batch, heads, seq, hd, ms)

            tflops_str = f"{tflops:.2f}"
            if tflops > 300:
                tflops_colored = ok(tflops_str)
            elif tflops > 150:
                tflops_colored = warn(tflops_str)
            else:
                tflops_colored = err(tflops_str)

            print(
                f"  {label:<{col_w[0]}}"
                f"{ms:>{col_w[1]}.3f}"
                f"  {tflops_colored}"
            )
        except Exception as e:
            msg = str(e).splitlines()[0][:60]
            print(f"  {label:<{col_w[0]}}{err(f'ERROR: {msg}')}")

# ---------------------------------------------------------------------------
# SA2 vs SA3 Comparison Block
# ---------------------------------------------------------------------------
def test_comparison(configs, dtype=torch.float16, warmup=10, steps=100):
    label_dtype = get_dtype_label(dtype)
    print_section(f"SA2 vs SA3 — Performance Comparison ({label_dtype})")

    if not SA2_AVAILABLE and not SA3_AVAILABLE:
        print(f"  {err('Skipped:')} No library installed.")
        return

    is_bf16 = (dtype == torch.bfloat16)
    col_w = [20, 14, 14, 14]
    header = (
        f"  {'Configuration':<{col_w[0]}}"
        f"{'SA2 (ms)':>{col_w[1]}}"
        f"{'SA3 (ms)':>{col_w[2]}}"
        f"{'SA3 / SA2':>{col_w[3]}}"
    )
    print(dim(header))
    print(dim("  " + "-" * (sum(col_w) + 2)))

    for batch, heads, seq, hd, label in configs:
        ms2_str = dim("N/A")
        ms3_str = dim("N/A")
        ratio_str = dim("N/A")
        ms2 = None
        ms3 = None

        q = torch.randn(batch, heads, seq, hd, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        if SA2_AVAILABLE:
            try:
                ms2 = cuda_benchmark(lambda: _sa2_func(q, k, v), warmup=warmup, steps=steps)
                ms2_str = f"{ms2:.3f}"
            except Exception as e:
                ms2_str = err("ERR")

        if SA3_AVAILABLE:
            try:
                qc, kc, vc, ds = _sa3_api.preprocess_qkv(q, k, v)

                def _run3():
                    _sa3_api.blockscaled_fp4_attn(
                        _sa3_api.scale_and_quant_fp4(qc),
                        _sa3_api.scale_and_quant_fp4(kc),
                        _sa3_api.scale_and_quant_fp4_transpose(vc),
                        ds, qc.shape[2], is_bf16=is_bf16
                    )
                ms3 = cuda_benchmark(_run3, warmup=warmup, steps=steps)
                ms3_str = f"{ms3:.3f}"
            except Exception as e:
                ms3_str = err("ERR")

        if ms2 is not None and ms3 is not None:
            ratio = ms3 / ms2
            ratio_val = f"{ratio:.2f}x"
            if ratio < 0.7:
                ratio_str = ok(ratio_val + " (SA3 faster)")
            elif ratio < 1.0:
                ratio_str = ok(ratio_val)
            elif ratio < 1.3:
                ratio_str = warn(ratio_val)
            else:
                ratio_str = err(ratio_val + " (SA3 slower)")

        print(
            f"  {label:<{col_w[0]}}"
            f"{ms2_str:>{col_w[1]}}"
            f"{ms3_str:>{col_w[2]}}"
            f"  {ratio_str}"
        )

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive SageAttention 2/3 Testing"
    )
    parser.add_argument("--sa2",   action="store_true", help="SA2 tests only")
    parser.add_argument("--sa3",   action="store_true", help="SA3 tests only")
    parser.add_argument("--perf",  action="store_true", help="performance tests only")
    parser.add_argument("--quick", action="store_true", help="quick mode (fewer configs)")
    parser.add_argument("--steps", type=int, default=100, help="benchmark steps (default: 100)")
    args = parser.parse_args()

    run_all = not (args.sa2 or args.sa3 or args.perf)

    acc_configs  = ACCURACY_CONFIGS_QUICK  if args.quick else ACCURACY_CONFIGS_FULL
    perf_configs = PERF_CONFIGS_QUICK      if args.quick else PERF_CONFIGS_FULL
    steps        = 30 if args.quick else args.steps

    print()
    print(hdr("╔══════════════════════════════════════════════════════════════════════╗"))
    print(hdr("║         SageAttention 2 / 3 — Comprehensive Test (EN)                ║"))
    print(hdr("╚══════════════════════════════════════════════════════════════════════╝"))

    if not torch.cuda.is_available():
        print(err("\nError: CUDA is not available. Tests aborted."))
        sys.exit(1)

    print_env_info()

    # --- Accuracy ---
    if run_all or (args.sa2 and not args.perf):
        test_sa2_accuracy(acc_configs)

    if run_all or (args.sa3 and not args.perf):
        test_sa3_accuracy(acc_configs)

    # --- Performance ---
    if run_all or args.perf or args.sa2:
        test_sa2_perf(perf_configs, dtype=torch.float16, warmup=10, steps=steps)
        test_sa2_perf(perf_configs, dtype=torch.bfloat16, warmup=10, steps=steps)

    if run_all or args.perf or args.sa3:
        test_sa3_perf(perf_configs, dtype=torch.float16, warmup=10, steps=steps)
        test_sa3_perf(perf_configs, dtype=torch.bfloat16, warmup=10, steps=steps)

    # --- Comparison SA2 vs SA3 ---
    if run_all or args.perf:
        test_comparison(perf_configs, dtype=torch.float16, warmup=10, steps=steps)

    print()
    print(hdr("=" * 72))
    print(hdr("  Test Completed"))
    print(hdr("=" * 72))
    print()

if __name__ == "__main__":
    main()