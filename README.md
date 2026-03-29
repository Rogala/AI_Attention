# ComfyUI Performance Boosters for NVIDIA RTX 5000 Series (Windows) ✨

This repository provides **pre-compiled acceleration packages** for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) users on **Windows** with **NVIDIA RTX 5000 Series (Blackwell, SM120)** GPUs.

Packages available:
- [**xFormers**](https://github.com/facebookresearch/xformers)
- [**SageAttention**](https://github.com/woct0rdho/SageAttention)
- [**Flash Attention**](https://github.com/Dao-AILab/flash-attention)

> **If you need builds for other GPU generations or Python/Torch versions, refer to the official project pages linked above.**

---

## System Requirements

| Component | Range |
|---|---|
| **GPU** | NVIDIA RTX 5000 Series (Blackwell, SM120) |
| **OS** | Windows |
| **Python** | 3.10 → 3.14 |
| **PyTorch + CUDA** | 2.7.0+cu128 → 2.11.x+cu130 *(current, changes over time)* |

> Builds are organized in folders by Torch version. Always pick the folder that matches your installed Torch.

---

## How to Install a Wheel

Download the `.whl` file from the corresponding folder and install it with:

```bash
pip install <path-to-wheel-file>.whl
```

For example:

```bash
pip install sageattention-2.2.0+cu128torch2.7.1-cp312-cp312-win_amd64.whl
```

---

## Acceleration Packages

### Choosing the Right Accelerator

| Situation | Recommendation |
|---|---|
| You have SageAttention | **Skip xFormers** — it adds no benefit when SageAttention is active |
| You want the best speed on RTX 5000 | Use **SageAttention** (requires Triton) |
| Flash Attention on Python 3.14 | ⚠️ **Not supported** — see note below |

---

## 1. Triton for Windows *(required for SageAttention)*

> **SageAttention 2 requires triton-windows — this is mandatory.**

Official repository: [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows)

*(Previously maintained as a personal fork by woct0rdho, now officially part of triton-lang)*

### PyTorch ↔ Triton Version Compatibility

| PyTorch | triton-windows |
|---|---|
| 2.7.x | 3.3.x |
| 2.8.x | 3.4.x |
| 2.9.x | 3.5.x |
| 2.10.x | 3.6.x |

### GPU Compatibility (Triton)

| GPU Architecture | Compute Capability | Triton Support |
|---|---|---|
| **Blackwell** (RTX 5000) | SM120 | ✅ Triton >= 3.3, PyTorch >= 2.7, CUDA >= 12.8 |
| **Ada Lovelace** (RTX 4000) | SM89 | ✅ Officially supported |
| **Ampere** (RTX 3000 / A100) | SM80/86/87 | ✅ Officially supported |
| **Hopper** (H100) | SM90 | ✅ Officially supported |
| **Turing** (RTX 2000 / GTX 16xx) | SM75 | ⚠️ Triton <= 3.2 only (dropped in 3.3) |
| **Volta and older** | SM70 and below | ❌ Not supported |

### Installation

Install and pin to current minor version to avoid breaking on future updates:

```bash
# For PyTorch 2.7.x
pip install -U "triton-windows<3.4"

# For PyTorch 2.8.x
pip install -U "triton-windows<3.5"

# For PyTorch 2.9.x
pip install -U "triton-windows<3.6"

# For PyTorch 2.10.x (current)
pip install -U "triton-windows<3.7"
```

Or install from a wheel file:

```bash
pip install <path-to-triton-wheel>.whl
```

> Since triton-windows 3.2.0, a minimal CUDA toolchain is bundled in the wheels — no separate CUDA installation required.

---

## 2. SageAttention

Official repository: [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention)

**Requires triton-windows** (see above). For SageAttention 2, this dependency is mandatory.

### Supported GPUs

The pre-built wheels support: **GTX 16xx, RTX 20xx/30xx/40xx/50xx, A100, H100, AGX Orin** (sm75/80/86/87/89/90/120).

> There are reports that SageAttention works with B200 (sm100) and DGX Spark (sm121), but these kernels are not bundled in the wheels — you would need to build from source.

### Installation

```bash
pip install <path-to-sageattention-wheel>.whl
```

Download the wheel from the folder matching your Torch version. For the full list of available releases, see the [SageAttention releases page](https://github.com/woct0rdho/SageAttention/releases).

### ComfyUI Launch Flag

```
--use-sage-attention
```

> SageAttention can run **in parallel with xFormers** in ComfyUI, but when SageAttention is present, xFormers provides no meaningful additional benefit.

---

## 3. xFormers

Official repository: [facebookresearch/xformers](https://github.com/facebookresearch/xformers)

### Installation

```bash
pip install <path-to-xformers-wheel>.whl
```

Or install directly from PyTorch index:

```bash
# [linux & win] cuda 12.6 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
# [linux & win] cuda 12.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu128
# [linux & win] cuda 13.0 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu130
```

### Verify Installation

```bash
python -m xformers.info
```

### ComfyUI Launch Flags

```
--disable-xformers    # to disable
```

> xFormers is enabled automatically in ComfyUI when installed. If you are using SageAttention, xFormers adds no performance benefit.

---

## 4. Flash Attention *(довідково)*

Official repository: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

> ⚠️ **Готових білдів під Windows у цьому репозиторії немає і не планується.**
>
> Flash Attention вкрай складно скомпілювати під Windows і особливо під Blackwell (SM120). Також наразі **не працює з Python 3.14**.
>
> Якщо все ж потрібен — шукайте сторонні білди на GitHub або HuggingFace. Рекомендована альтернатива для RTX 5000 Series — **SageAttention**.

### ComfyUI Launch Flag

```
--use-flash-attention --disable-xformers
```

> При увімкненому Flash Attention xFormers потрібно обов'язково вимкнути — вони не призначені для спільної роботи.

---


