# -*- coding: utf-8 -*-
# Full system diagnostic script with improved CUDA, RAM type and SageAttention version detection

import subprocess
import sys
import platform
import importlib

# --- Install required modules silently ---
def install_if_needed(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

install_if_needed("tabulate")
install_if_needed("psutil")
install_if_needed("py-cpuinfo")

from tabulate import tabulate
import psutil
import cpuinfo

# --- CPU Info ---
def get_cpu_info():
    try:
        info = cpuinfo.get_cpu_info()
        name = info.get("brand_raw", "Unknown")
        cores = psutil.cpu_count(logical=False)
        threads = psutil.cpu_count()
        return f"{name} ({cores} cores / {threads} threads)"
    except:
        return "Unknown"

# --- RAM Type using PowerShell: SMBIOSMemoryType and MemoryType fallback ---
def get_ram_type_powershell():
    try:
        output = subprocess.check_output([
            'powershell', '-Command',
            "(Get-CimInstance -ClassName Win32_PhysicalMemory).SMBIOSMemoryType"
        ], universal_newlines=True)

        codes = set(int(line.strip()) for line in output.splitlines() if line.strip().isdigit())
        type_map = {
            20: "DDR",
            21: "DDR2",
            24: "DDR3",
            26: "DDR4",
            27: "DDR5"
        }
        for code in codes:
            return type_map.get(code, "Unknown")
    except:
        return "Unknown"

# --- RAM Info ---
def get_ram_info():
    try:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        ram_type = get_ram_type_powershell()
        return f"{ram_type}, {ram_gb:.2f} GB"
    except:
        return "Unknown"

# --- GPU Info using PyTorch ---
def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            driver_version = torch.version.cuda or "Unknown"
            return f"{props.name}, {props.total_memory / (1024 ** 3):.2f} GB VRAM, CUDA {driver_version}"
        else:
            return "GPU not available"
    except ImportError:
        return "Torch not installed"
    except:
        return "Unknown"

# --- CUDA version from nvidia-smi (compact) ---
def get_cuda_version():
    try:
        output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        for line in output.split("\n"):
            if "CUDA Version" in line:
                part = line.split("CUDA Version:")[-1]
                version = part.strip().split()[0]
                return version
        return "Unknown"
    except:
        return "nvidia-smi not available"

# --- Generic module version ---
def get_module_version(module_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", "Unknown")
    except ImportError:
        return "Not installed"
    except:
        return "Unknown"

# --- Get pip-installed version of a package ---
def get_pip_version(package_name):
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "list"],
            universal_newlines=True
        )
        for line in output.splitlines():
            if line.lower().startswith(package_name.lower()):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
        return "Not installed"
    except:
        return "Unknown"

# --- Triton ---
def get_triton_version():
    try:
        import triton
        return triton.__version__
    except ImportError:
        return "Not installed"
    except:
        return "Unknown"

# --- Flash-Attention ---
def get_flash_attention_version():
    try:
        import flash_attn
        return flash_attn.__version__
    except ImportError:
        return "Not installed"
    except:
        return "Unknown"

# --- SageAttention via pip list ---
def get_sage_attention_version():
    try:
        import sageattention
        if hasattr(sageattention, "__version__"):
            return sageattention.__version__
        else:
            return get_pip_version("sageattention")
    except ImportError:
        return "Not installed"
    except:
        return "Unknown"

# --- Main Output ---
def main():
    data = [
        ["CPU Model / Cores / Threads", get_cpu_info()],
        ["RAM Type and Size", get_ram_info()],
        ["GPU Model / VRAM / Driver", get_gpu_info()],
        ["CUDA Version (nvidia-smi)", get_cuda_version()],
        ["Python Version", platform.python_version()],
        ["Torch Version", get_module_version("torch")],
        ["Torchaudio Version", get_module_version("torchaudio")],
        ["Torchvision Version", get_module_version("torchvision")],
        ["Triton (Windows)", get_triton_version()],
        ["Xformers Version", get_module_version("xformers")],
        ["Flash-Attention Version", get_flash_attention_version()],
        ["Sage-Attention Version", get_sage_attention_version()],
    ]

    print(tabulate(data, headers=["Component", "Version / Info"], tablefmt="grid"))

if __name__ == "__main__":
    main()
