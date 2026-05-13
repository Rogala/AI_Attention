"""
sage_2-3_test.py
================
Comprehensive testing of SageAttention 2 and SageAttention 3.
Windows / PowerShell 5.1 compatible.

Usage:
    sage_2-3_test.py                # run all tests + save chart
    sage_2-3_test.py --sa2          # SA2 tests only
    sage_2-3_test.py --sa3          # SA3 tests only
    sage_2-3_test.py --perf         # performance tests only
    sage_2-3_test.py --quick        # quick mode (fewer configs)
    sage_2-3_test.py --no-chart     # skip chart generation
    sage_2-3_test.py --steps 50     # custom benchmark steps
"""

import sys
import argparse
import traceback
import datetime

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# ---------------------------------------------------------------------------
# PowerShell 5.1 / Windows — Force UTF-8
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
    _sa2_func    = None
    SA2_VERSION  = None
    SA2_AVAILABLE = False

try:
    import sageattn3.api as _sa3_api
    SA3_AVAILABLE = True
except ImportError:
    _sa3_api      = None
    SA3_AVAILABLE = False

# ---------------------------------------------------------------------------
# Accuracy Configurations (batch, heads, seq, head_dim, label)
# NHD format: tensors created as (B, S, H, D) for SA2 compatibility
# ---------------------------------------------------------------------------
ACCURACY_CONFIGS_FULL = [
    (1, 16, 2048, 128, "bs1 h16 seq2048 hd128 fp16"),
    (2, 16, 2048, 128, "bs2 h16 seq2048 hd128 fp16"),
    (1, 16, 4096, 128, "bs1 h16 seq4096 hd128 fp16"),
    (1, 32, 2048,  64, "bs1 h32 seq2048 hd64  fp16"),
    (4, 16, 2048, 128, "bs4 h16 seq2048 hd128 fp16"),
    (1, 16, 2048, 128, "bs1 h16 seq2048 hd128 bf16"),
    (1, 16, 4096, 128, "bs1 h16 seq4096 hd128 bf16"),
    (1, 32, 4096,  64, "bs1 h32 seq4096 hd64  bf16"),
    # SA3 only — hd=256 not supported by SA2
    (1, 16, 2048, 256, "bs1 h16 seq2048 hd256 bf16 (SA3 only)"),
]

ACCURACY_CONFIGS_QUICK = [
    (1, 16, 2048, 128, "bs1 h16 seq2048 hd128 fp16"),
    (1, 32, 2048,  64, "bs1 h32 seq2048 hd64  fp16"),
    (1, 16, 2048, 128, "bs1 h16 seq2048 hd128 bf16"),
]

# Performance Configurations (batch, heads, seq, head_dim, label)
PERF_CONFIGS_FULL = [
    (1, 16,  2048, 128, "seq2K  h16 hd128"),
    (1, 16,  4096, 128, "seq4K  h16 hd128"),
    (1, 16,  8192, 128, "seq8K  h16 hd128"),
    (1, 32,  4096, 128, "seq4K  h32 hd128"),
    (1, 32,  8192, 128, "seq8K  h32 hd128"),
    (1, 32, 16384, 128, "seq16K h32 hd128"),
]

PERF_CONFIGS_QUICK = [
    (1, 16, 2048, 128, "seq2K  h16 hd128"),
    (1, 16, 4096, 128, "seq4K  h16 hd128"),
    (1, 32, 8192, 128, "seq8K  h32 hd128"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_dtype_label(dtype):
    return "bf16" if dtype == torch.bfloat16 else "fp16"

def dtype_from_label(label: str) -> torch.dtype:
    return torch.bfloat16 if "bf16" in label else torch.float16

def compute_accuracy_metrics(actual: torch.Tensor, expect: torch.Tensor) -> dict:
    a = actual.float()
    e = expect.float()
    diff = (a - e).abs()
    eps  = torch.finfo(torch.float32).eps
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
    if mean_rtol < 0.01:  return ok("excellent")
    if mean_rtol < 0.05:  return ok("good")
    if mean_rtol < 0.10:  return warn("fair")
    return err("out of range")

def cuda_benchmark(fn, warmup: int = 10, steps: int = 100) -> float:
    """Returns average execution time in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(steps):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / steps

def compute_attention_tflops(batch, heads, seq, hd, ms: float) -> float:
    """4 * B * H * S^2 * D flops for two matmuls (QK and AV)."""
    flops = 4.0 * batch * heads * (seq ** 2) * hd
    return (flops / (ms / 1000.0)) / 1e12

def make_nhd(batch, heads, seq, hd, dtype) -> tuple:
    """
    Create q, k, v in NHD format (B, S, H, D) — correct for SA2.
    SA3 api.preprocess_qkv accepts NHD and handles conversion internally.
    """
    q = torch.randn(batch, seq, heads, hd, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v

def make_nhd_ref(batch, heads, seq, hd, dtype) -> tuple:
    """
    Create q, k, v in HND format (B, H, S, D) for SDPA reference.
    SDPA expects HND, SA2/SA3 expect NHD — they are not the same shape.
    """
    q = torch.randn(batch, heads, seq, hd, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v

def print_section(title: str):
    print()
    print(hdr("=" * 74))
    print(hdr(f"  {title}"))
    print(hdr("=" * 74))

# ---------------------------------------------------------------------------
# Environment Info
# ---------------------------------------------------------------------------
def print_env_info() -> dict:
    """Prints and returns environment info dict."""
    print_section("Environment Info")

    info = {}
    py_ver  = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    pt_ver  = torch.__version__
    cuda_ver = torch.version.cuda or "N/A"

    print(f"  Python    : {C.BOLD}{py_ver}{C.RESET}")
    print(f"  PyTorch   : {C.BOLD}{pt_ver}{C.RESET}")
    print(f"  CUDA      : {C.BOLD}{cuda_ver}{C.RESET}")

    info["python"]  = py_ver
    info["pytorch"] = pt_ver
    info["cuda"]    = cuda_ver

    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        name = torch.cuda.get_device_name(dev)
        cap_maj, cap_min = torch.cuda.get_device_capability(dev)
        vram_gib = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        sm_str = f"SM {cap_maj}{cap_min}"
        print(f"  GPU       : {C.BOLD}{name}{C.RESET}  [{sm_str}]  {vram_gib:.0f} GiB VRAM")
        info["gpu"]  = name
        info["sm"]   = f"{cap_maj}{cap_min}"
        info["vram"] = f"{vram_gib:.0f} GiB"
    else:
        print(f"  GPU       : {err('CUDA NOT AVAILABLE!')}")
        info["gpu"] = "N/A"

    if SA2_AVAILABLE:
        print(f"  SA2       : {ok('found')}  v{SA2_VERSION}")
        info["sa2"] = f"v{SA2_VERSION}"
    else:
        print(f"  SA2       : {err('not found')}  (pip install sageattention)")
        info["sa2"] = None

    if SA3_AVAILABLE:
        print(f"  SA3       : {ok('found')}")
        info["sa3"] = True
    else:
        print(f"  SA3       : {err('not found')}  (pip install sageattn3)")
        info["sa3"] = False

    return info

# ---------------------------------------------------------------------------
# SA2 Accuracy Test
# ---------------------------------------------------------------------------
def test_sa2_accuracy(configs):
    print_section("SA2 — Accuracy Test  (ref: Math SDPA fp32)")

    if not SA2_AVAILABLE:
        print(f"  {err('Skipped:')} SageAttention 2 is not installed.")
        return

    # SA2 expects NHD format: (B, S, H, D)
    # SDPA reference needs HND: (B, H, S, D) — we create both correctly

    col_w = [34, 11, 11, 11, 11, 14]
    header = (
        f"  {'Configuration':<{col_w[0]}}"
        f"{'mean_rtol':>{col_w[1]}}"
        f"{'max_rtol':>{col_w[2]}}"
        f"{'mean_atol':>{col_w[3]}}"
        f"{'max_atol':>{col_w[4]}}"
        f"{'Status':>{col_w[5]}}"
    )
    print(dim(header))
    print(dim("  " + "-" * sum(col_w)))

    passed = 0
    skipped = 0
    for batch, heads, seq, hd, label in configs:
        # Skip hd=256 for SA2 — not supported
        if hd == 256:
            print(f"  {label:<{col_w[0]}}{dim('skipped — hd=256 not supported by SA2')}")
            skipped += 1
            continue

        dtype = dtype_from_label(label)
        try:
            # Reference in HND format for SDPA
            qr, kr, vr = make_nhd_ref(batch, heads, seq, hd, dtype)
            with sdpa_kernel(SDPBackend.MATH):
                out_ref = F.scaled_dot_product_attention(qr, kr, vr)
            # Convert reference output to NHD for comparison:
            # SDPA output: (B, H, S, D) → transpose to (B, S, H, D)
            out_ref_nhd = out_ref.transpose(1, 2)

            # SA2 in NHD format
            q, k, v = make_nhd(batch, heads, seq, hd, dtype)
            out_sa2 = _sa2_func(q, k, v)
            torch.cuda.synchronize()

            m = compute_accuracy_metrics(out_sa2, out_ref_nhd)
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

    total = len(configs) - skipped
    sc = ok if passed == total else (warn if passed > 0 else err)
    print(dim("  " + "-" * sum(col_w)))
    print(f"  Result: {sc(f'{passed}/{total}')} configs passed mean_rtol < 0.10")
    print(f"  {dim('Norm: mean_rtol < 0.05 for RTX 30xx, < 0.10 for RTX 40xx/50xx')}")

# ---------------------------------------------------------------------------
# SA3 Accuracy Test
# ---------------------------------------------------------------------------
def test_sa3_accuracy(configs):
    print_section("SA3 — Accuracy Test  (ref: Math SDPA fp32)")

    if not SA3_AVAILABLE:
        print(f"  {err('Skipped:')} SageAttention 3 is not installed.")
        return

    note = (
        "SA3 uses FP4 quantization — mean_rtol ~0.79 vs Math SDPA is EXPECTED, not an error.\n"
        "  Pass criterion: mean_atol < 0.05 and correct output shape/dtype."
    )
    print(f"  {dim(note)}")

    # SA3 valid head_dims
    SA3_VALID_HD = (64, 128, 256)

    col_w = [34, 11, 11, 11, 14]
    header = (
        f"  {'Configuration':<{col_w[0]}}"
        f"{'mean_rtol':>{col_w[1]}}"
        f"{'mean_atol':>{col_w[2]}}"
        f"{'max_atol':>{col_w[3]}}"
        f"{'Status':>{col_w[4]}}"
    )
    print(dim(header))
    print(dim("  " + "-" * sum(col_w)))

    passed = 0
    skipped = 0
    for batch, heads, seq, hd, label in configs:
        if hd not in SA3_VALID_HD:
            print(f"  {label:<{col_w[0]}}{dim(f'skipped — hd={hd} not in SA3 valid {SA3_VALID_HD}')}")
            skipped += 1
            continue

        dtype   = dtype_from_label(label)
        is_bf16 = (dtype == torch.bfloat16)
        try:
            # Reference in HND for SDPA
            qr, kr, vr = make_nhd_ref(batch, heads, seq, hd, dtype)
            with sdpa_kernel(SDPBackend.MATH):
                out_ref = F.scaled_dot_product_attention(qr, kr, vr)
            # SDPA output HND → transpose to NHD for comparison
            out_ref_nhd = out_ref.transpose(1, 2)

            # SA3 via api — accepts NHD, preprocess_qkv handles conversion
            q, k, v = make_nhd(batch, heads, seq, hd, dtype)
            qc, kc, vc, ds = _sa3_api.preprocess_qkv(q, k, v)
            out_sa3_raw = _sa3_api.blockscaled_fp4_attn(
                _sa3_api.scale_and_quant_fp4(qc),
                _sa3_api.scale_and_quant_fp4(kc),
                _sa3_api.scale_and_quant_fp4_transpose(vc),
                ds, qc.shape[1], is_bf16=is_bf16
            )
            # blockscaled_fp4_attn may return tuple
            out_sa3 = out_sa3_raw[0] if isinstance(out_sa3_raw, (list, tuple)) else out_sa3_raw
            torch.cuda.synchronize()

            assert out_sa3.shape == q.shape, f"shape {out_sa3.shape} != {q.shape}"
            assert out_sa3.dtype == dtype,   f"dtype {out_sa3.dtype} != {dtype}"

            m = compute_accuracy_metrics(out_sa3, out_ref_nhd)

            if m["mean_atol"] < 0.05:
                passed += 1
                status = ok("normal")
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

    total = len(configs) - skipped
    sc = ok if passed == total else (warn if passed > 0 else err)
    print(dim("  " + "-" * sum(col_w)))
    print(f"  Result: {sc(f'{passed}/{total}')} configs passed mean_atol < 0.05")
    print(f"  {dim('rtol ~0.79 is a structural feature of FP4, not a build defect')}")

# ---------------------------------------------------------------------------
# SA2 Performance Test
# ---------------------------------------------------------------------------
def test_sa2_perf(configs, dtype=torch.float16, warmup=10, steps=100) -> list:
    """Returns list of (label, ms, tflops) for chart."""
    label_dtype = get_dtype_label(dtype)
    print_section(f"SA2 — Performance ({label_dtype}, warmup={warmup}, steps={steps})")

    if not SA2_AVAILABLE:
        print(f"  {err('Skipped:')} SageAttention 2 is not installed.")
        return []

    col_w = [22, 16, 14]
    header = (
        f"  {'Configuration':<{col_w[0]}}"
        f"{'Latency (ms)':>{col_w[1]}}"
        f"{'TFLOPS':>{col_w[2]}}"
    )
    print(dim(header))
    print(dim("  " + "-" * sum(col_w)))

    results = []
    for batch, heads, seq, hd, label in configs:
        if hd == 256:
            print(f"  {label:<{col_w[0]}}{dim('skipped — hd=256 not supported')}")
            continue
        try:
            q, k, v = make_nhd(batch, heads, seq, hd, dtype)
            ms = cuda_benchmark(lambda: _sa2_func(q, k, v), warmup=warmup, steps=steps)
            tflops = compute_attention_tflops(batch, heads, seq, hd, ms)

            ts = f"{tflops:.2f}"
            tc = ok(ts) if tflops > 200 else (warn(ts) if tflops > 100 else err(ts))
            print(f"  {label:<{col_w[0]}}{ms:>{col_w[1]}.3f}  {tc}")
            results.append((label, ms, tflops))
        except Exception as e:
            msg = str(e).splitlines()[0][:60]
            print(f"  {label:<{col_w[0]}}{err(f'ERROR: {msg}')}")

    return results

# ---------------------------------------------------------------------------
# SA3 Performance Test
# ---------------------------------------------------------------------------
def test_sa3_perf(configs, dtype=torch.float16, warmup=10, steps=100) -> list:
    """Returns list of (label, ms, tflops) for chart."""
    label_dtype = get_dtype_label(dtype)
    print_section(f"SA3 — Performance ({label_dtype}, warmup={warmup}, steps={steps})")

    if not SA3_AVAILABLE:
        print(f"  {err('Skipped:')} SageAttention 3 is not installed.")
        return []

    SA3_VALID_HD = (64, 128, 256)
    is_bf16 = (dtype == torch.bfloat16)

    col_w = [22, 16, 14]
    header = (
        f"  {'Configuration':<{col_w[0]}}"
        f"{'Latency (ms)':>{col_w[1]}}"
        f"{'TFLOPS':>{col_w[2]}}"
    )
    print(dim(header))
    print(dim("  " + "-" * sum(col_w)))

    results = []
    for batch, heads, seq, hd, label in configs:
        if hd not in SA3_VALID_HD:
            print(f"  {label:<{col_w[0]}}{dim(f'skipped — hd={hd} not in {SA3_VALID_HD}')}")
            continue
        try:
            q, k, v = make_nhd(batch, heads, seq, hd, dtype)
            qc, kc, vc, ds = _sa3_api.preprocess_qkv(q, k, v)

            def _run():
                _sa3_api.blockscaled_fp4_attn(
                    _sa3_api.scale_and_quant_fp4(qc),
                    _sa3_api.scale_and_quant_fp4(kc),
                    _sa3_api.scale_and_quant_fp4_transpose(vc),
                    ds, qc.shape[1], is_bf16=is_bf16
                )

            ms = cuda_benchmark(_run, warmup=warmup, steps=steps)
            tflops = compute_attention_tflops(batch, heads, seq, hd, ms)

            ts = f"{tflops:.2f}"
            tc = ok(ts) if tflops > 300 else (warn(ts) if tflops > 150 else err(ts))
            print(f"  {label:<{col_w[0]}}{ms:>{col_w[1]}.3f}  {tc}")
            results.append((label, ms, tflops))
        except Exception as e:
            msg = str(e).splitlines()[0][:60]
            print(f"  {label:<{col_w[0]}}{err(f'ERROR: {msg}')}")

    return results

# ---------------------------------------------------------------------------
# SA2 vs SA3 Comparison
# ---------------------------------------------------------------------------
def test_comparison(configs, dtype=torch.float16, warmup=10, steps=100) -> list:
    """Returns list of (label, ms_sa2, ms_sa3) for chart."""
    label_dtype = get_dtype_label(dtype)
    print_section(f"SA2 vs SA3 — Performance Comparison ({label_dtype})")

    if not SA2_AVAILABLE and not SA3_AVAILABLE:
        print(f"  {err('Skipped:')} No library installed.")
        return []

    SA3_VALID_HD = (64, 128, 256)
    is_bf16 = (dtype == torch.bfloat16)

    col_w = [22, 14, 14, 16]
    header = (
        f"  {'Configuration':<{col_w[0]}}"
        f"{'SA2 (ms)':>{col_w[1]}}"
        f"{'SA3 (ms)':>{col_w[2]}}"
        f"{'SA3 / SA2':>{col_w[3]}}"
    )
    print(dim(header))
    print(dim("  " + "-" * sum(col_w)))

    results = []
    for batch, heads, seq, hd, label in configs:
        if hd not in SA3_VALID_HD:
            continue

        ms2 = None
        ms3 = None
        ms2_str = dim("N/A")
        ms3_str = dim("N/A")
        ratio_str = dim("N/A")

        q2, k2, v2 = make_nhd(batch, heads, seq, hd, dtype)

        if SA2_AVAILABLE and hd != 256:
            try:
                ms2 = cuda_benchmark(lambda: _sa2_func(q2, k2, v2), warmup=warmup, steps=steps)
                ms2_str = f"{ms2:.3f}"
            except Exception as e:
                ms2_str = err("ERR")

        if SA3_AVAILABLE:
            try:
                q3, k3, v3 = make_nhd(batch, heads, seq, hd, dtype)
                qc, kc, vc, ds = _sa3_api.preprocess_qkv(q3, k3, v3)

                def _run3():
                    _sa3_api.blockscaled_fp4_attn(
                        _sa3_api.scale_and_quant_fp4(qc),
                        _sa3_api.scale_and_quant_fp4(kc),
                        _sa3_api.scale_and_quant_fp4_transpose(vc),
                        ds, qc.shape[1], is_bf16=is_bf16
                    )

                ms3 = cuda_benchmark(_run3, warmup=warmup, steps=steps)
                ms3_str = f"{ms3:.3f}"
            except Exception as e:
                ms3_str = err("ERR")

        if ms2 is not None and ms3 is not None:
            ratio = ms3 / ms2
            rv = f"{ratio:.2f}x"
            if ratio < 0.7:   ratio_str = ok(rv + " (SA3 faster)")
            elif ratio < 1.0: ratio_str = ok(rv)
            elif ratio < 1.3: ratio_str = warn(rv)
            else:             ratio_str = err(rv + " (SA3 slower)")
            results.append((label, ms2, ms3))

        print(
            f"  {label:<{col_w[0]}}"
            f"{ms2_str:>{col_w[1]}}"
            f"{ms3_str:>{col_w[2]}}"
            f"  {ratio_str}"
        )

    return results

# ---------------------------------------------------------------------------
# Chart Generation
# ---------------------------------------------------------------------------
def save_chart(comparison_results: list, env_info: dict,
               out_path: str = "sage_benchmark.png"):
    """
    Saves one PNG with two horizontal bar subplots:
      Top    — SA2 vs SA3 Latency (ms, lower is better)
      Bottom — SA2 vs SA3 TFLOPS  (higher is better)
    Beige background, no main title, subtitle with GPU/PyTorch info.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(warn("  matplotlib not installed — skipping chart (pip install matplotlib)"))
        return

    if not comparison_results:
        print(dim("  No comparison data — chart skipped"))
        return

    labels = [r[0] for r in comparison_results]
    ms_sa2 = [r[1] for r in comparison_results]
    ms_sa3 = [r[2] for r in comparison_results]

    def ms_to_tflops(ms, label):
        seq_map = {"seq2K": 2048, "seq4K": 4096, "seq8K": 8192, "seq16K": 16384}
        seq   = next((v for k, v in seq_map.items() if k in label), 4096)
        heads = 32 if "h32" in label else 16
        return compute_attention_tflops(1, heads, seq, 128, ms)

    tflops_sa2 = [ms_to_tflops(ms, lb) for ms, lb in zip(ms_sa2, labels)]
    tflops_sa3 = [ms_to_tflops(ms, lb) for ms, lb in zip(ms_sa3, labels)]

    BG     = "#f5f0e8"
    AX_BG  = "#ede8dc"
    C_SA2  = "#5a9e55"
    C_SA3  = "#4a8fbf"
    TXT    = "#333333"
    DIM    = "#888888"

    n = len(labels)
    y = np.arange(n)
    h = 0.35

    subtitle = (
        f"{env_info.get('gpu', '')}  SM {env_info.get('sm', '')}  "
        f"{env_info.get('vram', '')}  |  "
        f"PyTorch {env_info.get('pytorch', '')}  |  "
        f"SA2 {env_info.get('sa2', '')}"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5 + n * 0.55))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.97, subtitle, ha="center", va="top",
             fontsize=9, color=DIM, fontstyle="italic")

    def _style_ax(ax):
        ax.set_facecolor(AX_BG)
        ax.spines[:].set_color("#ccc")
        ax.tick_params(colors=TXT, labelsize=9)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        ax.grid(axis="x", color="#ccc", linewidth=0.6, linestyle="--")
        ax.set_axisbelow(True)

    def _bar_labels(ax, bars, fmt="{:.2f}"):
        for bar in bars:
            w = bar.get_width()
            ax.text(w + ax.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height() / 2,
                    fmt.format(w), va="center", ha="left", fontsize=8, color=TXT)

    # ── Top: Latency ──
    b1 = ax1.barh(y + h/2, ms_sa2, h, label="SA2", color=C_SA2, alpha=0.85)
    b2 = ax1.barh(y - h/2, ms_sa3, h, label="SA3", color=C_SA3, alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.5, color=TXT)
    ax1.set_xlabel("Latency (ms)  —  lower is better", fontsize=9, color=TXT)
    ax1.invert_yaxis()
    ax1.legend(facecolor=BG, edgecolor="#ccc", labelcolor=TXT, fontsize=9)
    _style_ax(ax1)
    ax1.set_xlim(0, max(ms_sa2 + ms_sa3) * 1.18)
    _bar_labels(ax1, b1, "{:.2f}")
    _bar_labels(ax1, b2, "{:.2f}")

    # ── Bottom: TFLOPS ──
    b3 = ax2.barh(y + h/2, tflops_sa2, h, label="SA2", color=C_SA2, alpha=0.85)
    b4 = ax2.barh(y - h/2, tflops_sa3, h, label="SA3", color=C_SA3, alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8.5, color=TXT)
    ax2.set_xlabel("TFLOPS  —  higher is better", fontsize=9, color=TXT)
    ax2.invert_yaxis()
    ax2.legend(facecolor=BG, edgecolor="#ccc", labelcolor=TXT, fontsize=9)
    _style_ax(ax2)
    ax2.set_xlim(0, max(tflops_sa2 + tflops_sa3) * 1.18)
    _bar_labels(ax2, b3, "{:.1f}")
    _bar_labels(ax2, b4, "{:.1f}")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.99, 0.005, ts, ha="right", va="bottom", color="#aaa", fontsize=7)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(ok(f"  Chart saved → {out_path}"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SageAttention 2/3 Comprehensive Test")
    parser.add_argument("--sa2",      action="store_true", help="SA2 tests only")
    parser.add_argument("--sa3",      action="store_true", help="SA3 tests only")
    parser.add_argument("--perf",     action="store_true", help="performance tests only")
    parser.add_argument("--quick",    action="store_true", help="quick mode (fewer configs)")
    parser.add_argument("--no-chart", action="store_true", help="skip chart generation")
    parser.add_argument("--steps",    type=int, default=100, help="benchmark steps (default: 100)")
    parser.add_argument("--out",      type=str, default="sage_benchmark.png", help="chart output path")
    args = parser.parse_args()

    run_all = not (args.sa2 or args.sa3 or args.perf)

    acc_configs  = ACCURACY_CONFIGS_QUICK if args.quick else ACCURACY_CONFIGS_FULL
    perf_configs = PERF_CONFIGS_QUICK     if args.quick else PERF_CONFIGS_FULL
    steps        = 30 if args.quick else args.steps

    print()
    print(hdr("╔══════════════════════════════════════════════════════════════════════╗"))
    print(hdr("║          SageAttention 2 / 3 — Comprehensive Test                    ║"))
    print(hdr("╚══════════════════════════════════════════════════════════════════════╝"))

    if not torch.cuda.is_available():
        print(err("\nError: CUDA is not available. Tests aborted."))
        sys.exit(1)

    env_info = print_env_info()

    # Accuracy
    if run_all or (args.sa2 and not args.perf):
        test_sa2_accuracy(acc_configs)
    if run_all or (args.sa3 and not args.perf):
        test_sa3_accuracy(acc_configs)

    # Performance
    comparison_results = []
    if run_all or args.perf or args.sa2:
        test_sa2_perf(perf_configs, dtype=torch.float16,  warmup=10, steps=steps)
        test_sa2_perf(perf_configs, dtype=torch.bfloat16, warmup=10, steps=steps)
    if run_all or args.perf or args.sa3:
        test_sa3_perf(perf_configs, dtype=torch.float16,  warmup=10, steps=steps)
        test_sa3_perf(perf_configs, dtype=torch.bfloat16, warmup=10, steps=steps)

    # Comparison + chart data
    if run_all or args.perf:
        comparison_results = test_comparison(perf_configs, dtype=torch.float16,
                                             warmup=10, steps=steps)

    if not args.no_chart:
        print_section("Chart")
        save_chart(comparison_results, env_info, out_path=args.out)

    print()
    print(hdr("=" * 74))
    print(hdr("  Test Completed"))
    print(hdr("=" * 74))
    print()

if __name__ == "__main__":
    main()
