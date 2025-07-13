# ComfyUI Performance Boosters for NVIDIA RTX 5000 Series (Windows) ✨

This repository is dedicated to helping [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [Fooocus](https://github.com/lllyasviel/Fooocus), [FramePack](https://github.com/lllyasviel/FramePack) users on **Windows** significantly enhance their AI workflow efficiency. It offers **pre-compiled acceleration packages** such as **xformers**, **Flash Attention**, and **SageAttention**, along with **detailed installation guides**.

---

## What's Inside?

* **Optimized Attention Packages:** Directly downloadable, self-compiled versions of leading attention optimizers for ComfyUI.
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

---

## Why Use This?

If you're running ComfyUI on Windows with an **NVIDIA RTX 5000 series GPU** and looking to drastically **reduce generation times** and **improve overall performance**, this repository provides the necessary tools and guidance to get you started quickly.

---

## Deployment Methods & Python Version Challenges

There are generally three straightforward ways to deploy applications like ComfyUI, Fooocus, and FramePack. While their setup processes are quite similar, a common challenge arises with their bundled Python environments: cannot change the version of Python in their `python_embedded' folders.

1.  **Portable Version (Download & Unpack):**
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
    This command will download and install the latest versions of **Torch**, **torchvision** (for computer vision utilities), and **torchaudio** (for audio utilities) that are compiled for **CUDA 12.8**. The `--extra-index-url` specifies the official PyTorch wheel index for CUDA 12.8 builds, ensuring you get the correct versions for your RTX 5000 series GPU.

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
    Open your console (cmd) within the newly cloned application folder (e.g., `X:\ComfyUI_Git`). Navigate into this newly created folder using your command line, and then clone the desired application from its official GitHub repository:
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
