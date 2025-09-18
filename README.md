# ComfyUI, Fooocus, FramePack Performance Boosters for NVIDIA RTX 5000 Series (Windows) ✨

This repository is dedicated to helping [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [Fooocus](https://github.com/lllyasviel/Fooocus), [FramePack](https://github.com/lllyasviel/FramePack) users on **Windows** significantly enhance their AI workflow efficiency. It offers **pre-compiled acceleration packages** such as [**xformers**](https://github.com/facebookresearch/xformers), [**Flash Attention**](https://github.com/Dao-AILab/flash-attention), and [**SageAttention**](https://github.com/woct0rdho/SageAttention), along with **detailed installation guides**.

***If you need to upgrade to other versions of Nvidia graphics cards or a different version of Python or Torch, use the links to the official project pages***

---

## What's Inside?

* **Optimized Attention Packages:** Directly downloadable, self-compiled versions of leading attention optimizers for ComfyUI, Fooocus, FramePack.
    * **xformers:** A library providing highly optimized attention mechanisms.
    * **Flash Attention:** Designed for ultra-fast attention computations.
    * **SageAttention:** Another powerful tool for accelerating attention.
* **Step-by-Step Installation Guides:** Clear and concise instructions to seamlessly integrate these packages into your ComfyUI environment on Windows.
* **Direct Download Links:** Convenient links to quickly access the compiled files.

---

## System Requirements & Compatibility

* **GPU:** All provided builds are specifically compiled and optimized for **NVIDIA RTX 5000 series GPUs (Blackwell architecture)**. They are targeted at **SM120** (Streaming Multiprocessor 120). Compatibility with other NVIDIA series is not guaranteed.
* **Operating System:** Windows.
* **Python:** The compilation process used **Python 3.12.10**. For optimal compatibility, it's recommended to use **Python 3.12.x**.
* **Compilation Tools:** The builds were created using **CUDA Toolkit 12.8** and **Visual Studio 2022**.
* **Added new compilation branch:** **Python 3.13.x** and **Cuda Toolkit 12.9** and **Pytorch 2.8**.
---

## Why Use This?

If you're running ComfyUI on Windows with an **NVIDIA RTX 5000 series GPU** and looking to drastically **reduce generation times** and **improve overall performance**, this repository provides the necessary tools and guidance to get you started quickly.

---

## Deployment Methods & Python Version Challenges

There are generally three straightforward ways to deploy applications like ComfyUI, Fooocus, and FramePack. While their setup processes are quite similar, a common challenge arises with their bundled Python environments: cannot change the version of Python in their `python_embedded' folders.

**Method 1:**  **Portable Version (Download & Unpack):**
    This method involves downloading and extracting a portable version of these applications. Each portable version comes with its own embedded Python environment, which might not always align with Python 3.12.x and would require installing accelerators from their respective repositories.

* **Fooocus:** Typically includes **Python 3.10.9** and **Torch 2.1.0+cu121**. This version **Torch does not work** with the NVIDIA RTX 5000 series.
* **FramePack:** Usually comes with **Python 3.10.6** and **Torch 2.6.0+cu126**. This version **Torch does not work** with the NVIDIA RTX 5000 series.
* **ComfyUI:** The latest version, **v0.3.44**, includes **Python 3.12.10** and **Torch 2.7.1+cu128**, which **fully meets our requirements** for NVIDIA RTX 5000 series compatibility.

---

## Updating Torch for Fooocus and FramePack (NVIDIA RTX 5000 Series Compatibility)

To ensure **Fooocus** and **FramePack** are compatible with the **NVIDIA RTX 5000 series GPUs**, you'll need to update their **Torch** libraries. According to the official [PyTorch website](https://pytorch.org/get-started/locally/), it supports Python versions 3.9 and higher, making it compatible with the embedded Python versions in these applications.

### Accessing the Console in `python_embedded`

First, you need to open a command prompt (cmd) directly within the `python_embedded` folder of each application.

* **For Fooocus:** Navigate to this path: `X:\Fooocus_win64_2-5-0\python_embeded\`
* **For FramePack:** Navigate to this path: `X:\framepack_cu126_torch26\system\python\`

**To open the console in these folders:**

1.  Open **File Explorer** (Windows Explorer).
2.  Navigate to the specific `python_embeded` or `python` folder for your application.
3.  In the address bar at the top of the File Explorer window, type `cmd` and press **Enter**. [This will open a command prompt window directly in that directory](https://www.youtube.com/watch?v=bgSSJQolR0E).

### Updating Torch

For the 5000 series, you need to install Torch version 2.7.0 or higher, compiled with CUDA 128.
Once the console is open in the correct folder, execute the following two commands in sequence:

1.  **Upgrade pip (the package installer):**
    ```bash
    .\python.exe -m pip install --upgrade pip
    ```
    This ensures you have the latest version of pip, which is necessary for reliable package installations.

2.  **Upgrade Torch, torchvision, and torchaudio for CUDA 12.8:**
    ```bash
    .\python.exe -m pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
    ```
    This command will download and install the latest versions (currently it is 2.7.1, to install another version, you need to register its revision **torch==2.7.0**) of **Torch**, **torchvision** (for computer vision utilities), and **torchaudio** (for audio utilities) that are compiled for **CUDA 12.8**. The `--extra-index-url` specifies the official PyTorch wheel index for CUDA 12.8 builds, ensuring you get the correct versions for your RTX 5000 series GPU.

After successfully running these commands, your Fooocus and FramePack installations should have updated Torch versions, enabling proper functionality with your NVIDIA RTX 5000 series graphics card.

---

## Deployment Methods: Global Python vs. Virtual Environments 🚀

We go to method 2 and 3.

### Initial Setup: Git and Python 3.12.10

Before proceeding with either method, you need to install the core tools:

1.  **Install Git:** Download and install Git from its [official website](https://git-scm.com/downloads).
2.  **Install Python 3.12.10:** Download and install Python 3.12.10 from [python.org](https://www.python.org/downloads/release/python-31210/). **Crucially, remember to check the "Add Python to PATH" option during installation.** This ensures Python is accessible from your command line.

---

### Method 2: Global Python Installation 🌍

This method involves using a **globally installed Python** (e.g., Python 3.12.10) for all your applications.

* **How it works:** When you install packages, they are placed in the global Python environment and become accessible to any application configured to use that Python installation.
* **Pros:** Simple to set up initially, as you don't manage separate environments.
* **Cons:** If you have multiple AI applications installed (like Fooocus, FramePack, and ComfyUI), they might require different versions of the same packages. This can easily lead to **package conflicts**, causing one or more applications to break. This approach is generally **not recommended** if you plan to run multiple Python-based applications simultaneously. If you only run one application, it might be acceptable.

---

### Method 3: Virtual Environments (Recommended) 🏞️

This is the **recommended approach** for managing multiple AI applications. It involves using a **virtual environment** to isolate each application's dependencies, even while using a single global Python version.

* **How it works:** A virtual environment creates an isolated copy of a Python environment for each project. All packages installed within this virtual environment are specific to that project and won't interfere with other applications or your global Python installation. This ensures each application runs with its required package versions, specified in its `requirements.txt`.
* **Pros:** Prevents package conflicts between different applications, making your development and deployment more stable and reliable. Each application gets its own clean set of dependencies.
* **Cons:** Requires an extra step to create and activate the virtual environment for each project.

#### Steps for Method 3 (Virtual Environment Setup):

1.  **Create a Project Folder:**
    First, create a new folder for your application (e.g., `Fooocus_Git`, `FramePack_Git`, `ComfyUI_Git`). It's best to place this folder in the root of a fast drive (e.g., `D:\ComfyUI_Git`).
2.  **Clone the Application Repository:**
    Open the console (cmd) in the newly created application folder (for example, X:\ComfyUI_Git), and then clone the desired application from its official GitHub repository:
    * For Fooocus:
        ```bash
        git clone https://github.com/lllyasviel/Fooocus.git
        ```
    * For FramePack:
        ```bash
        git clone https://github.com/lllyasviel/FramePack.git
        ```
    * For ComfyUI:
        ```bash
        git clone https://github.com/comfyanonymous/ComfyUI.git
        ```
3.  **Create and Activate Virtual Environment:**
    Then, create and activate your virtual environment:
    * **Create the virtual environment:**
        ```bash
        python -m venv venv
        ```
    * **Activate it:**
        ```bash
        .\venv\Scripts\activate
        ```
    After activation, your console prompt will show `(venv)` before the drive letter and path, like this:
    ```
    (venv) X:\ComfyUI_Git>
    ```
    This indicates that all subsequent package installations will be isolated to this virtual environment, specifically for the `ComfyUI` application in this example. The same logic applies to other applications like Fooocus or FramePack, with their own dedicated virtual environments.

---

## Launch Files

In the portable version, after unpacking, there are *.bat files for launching programs. When cloning, these are absent, so you need to create your own `run.bat` file next to the `venv` folder and the cloned program's folder, in the following format:

**Fooocus:**
```bat
@echo off
call .\venv\Scripts\activate.bat
python .\Fooocus\entry_with_update.py --theme dark
pause
```
**FramePack:**
```bat
@echo off
call .\venv\Scripts\activate.bat
python .\FramePack\demo_gradio_f1.py --server 127.0.0.1 --inbrowser
pause
```
**ComfyUI:**
```bat
@echo off
call .\venv\Scripts\activate.bat
python .\ComfyUI\main.py --auto-launch
pause
```

 You can add all other launch arguments yourself according to your needs.
 
---

## Installing xformers 🚀

To install `xformers` for Fooocus, FramePack, or ComfyUI, execute the following commands in the console, depending on your version:

### For the portable version:

Open the console in your program's `python_embedded` folder and execute:

```bash
.\python.exe -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
````

### For the cloned version:

Ensure you have **activated your virtual environment**, then execute:

```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
```

### Important notes regarding Torch and xformers versions:

  * If your **Torch version is 2.7.0**, install `xformers` with a specific version:
    ```bash
    pip install -U xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
    ```
* For **Torch 2.7.1 and higher**, you can omit the `xformers` version (as shown in the commands above), as a compatible latest version will be installed.
* To download my version of `xformers', make sure you get the link from the folder corresponding to your version of Torch.

---

## Installing FlashAttention (for FramePack and ComfyUI only) 🚀

**Note:** Fooocus does not support FlashAttention or SageAttention, so these instructions are only for FramePack and ComfyUI.

To install FlashAttention for FramePack or ComfyUI, execute the following commands in the console:

### For the portable version:

```bash
.\python.exe -m pip install https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl
````

### For the cloned version:

```bash
pip install https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl
```

* This version works with both Torch 2.7.0 and 2.7.1.
* To download my version of `FlashAttention', make sure you get the link from the folder corresponding to your version of Torch.

---

## Installing Triton (Prerequisite for SageAttention) 🛠️

Before installing SageAttention, you must install [Triton](https://github.com/woct0rdho/triton-windows). The aim of Triton is to provide an open-source environment to write fast code with higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

### For the portable version:

```bash
.\python.exe -m pip install -U triton-windows==3.3.1
```

### For the cloned version:

```bash
pip install -U triton-windows==3.3.1
```

### For the PyTorch 2.8.0:

```bash
.\python.exe -m pip install -U triton-windows==3.4.0
```
or
```bash
pip install -U triton-windows==3.4.0
```
---

## Installing SageAttention (for FramePack and ComfyUI only) 🚀

To install SageAttention for FramePack or ComfyUI, execute the following commands in the console:

### For the portable version:

```bash
.\python.exe -m pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows/sageattention-2.2.0+cu128torch2.7.1-cp312-cp312-win_amd64.whl
```

### For the cloned version:

```bash
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows/sageattention-2.2.0+cu128torch2.7.1-cp312-cp312-win_amd64.whl
```

* [If you need a different version, you can find it in the list here:](https://github.com/woct0rdho/SageAttention/releases)
* To download my version of `SageAttention', make sure you get the link from the folder corresponding to your version of Torch.
---

## Launch Flags for Fooocus, FramePack, and ComfyUI 🚀

This section details how different attention mechanisms are handled by each application and any specific flags you might need to use in their respective `run.bat` files.

### Fooocus

**Fooocus automatically works with an installed `xformers` library.** You don't need any special flags to enable it. However, if you wish to **disable `xformers`**, you can do so by adding the following flag to your `run.bat` file:

```bat
--disable-xformers
```

### FramePack

**FramePack also automatically utilizes installed `xformers`, FlashAttention, and SageAttention.** No specific flags are required in your `run.bat` file for these to function.

### ComfyUI

**ComfyUI automatically works with `xformers` when installed.** If you need to **disable `xformers`** for ComfyUI, add this flag to its `run.bat` file:

```bat
--disable-xformers
```

For **FlashAttention**, you need to **explicitly activate it** in ComfyUI's `run.bat` file. Crucially, if you use FlashAttention, you **must disable `xformers`**, as `xformers` has its own built-in FlashAttention code, and they are not designed to work concurrently.

To activate FlashAttention:

```bat
--use-flash-attention
```

Conversely, **SageAttention can operate in parallel with `xformers`** in ComfyUI. To activate SageAttention, add this to your `run.bat` file:

```bat
--use-sage-attention
```
---
## Benchmarking Tools 📊

The `bench` folder contains three tests that allow you to verify the functionality of all installed components:

* **`environment.py`**: Displays information about your system and installed packages.
* **`fp32-16-8-4_bf16_gpt-bench.py`**: Runs a system benchmark with various parameters for Torch, xformers, and FlashAttention.
* **`sa-fa_gpt.py`**: Compares SageAttention and FlashAttention where applicable.

---
## Speed of operation with a standard workflow in seconds with ComfyUI
* SDXL sd_xl_base_1.0_0.9vae.safetensors, steps 20, 1024x1024, euler+normal
* Flux flux1-dev-fp8.safetensors, steps 20, 1024x1024, euler+simple
* Qwen qwen-image-Q5_K_M.gguf + qwen_2.5_vl_7b_fp8_scaled.safetensors, steps 20, 1328x1328, euler+simple
<img width="1337" height="574" alt="Speed of operation with a standard workflow in seconds" src="https://github.com/user-attachments/assets/393e9078-65ee-4fdd-a1f4-672ca4da2c4d" />

