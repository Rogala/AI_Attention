# VERSION 3.10 (COMFYUI RUNTIME ENVIRONMENT MANAGER - UI OPTIMIZED)
$ProgressPreference = 'Continue'

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell.exe "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs -WorkingDirectory "$PSScriptRoot"
    exit
}

Set-Location -Path "$PSScriptRoot"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$targetFolder = "ComfyUI"
$venvFolder = "venv"
$verFile = "python_version.txt"
$report = @()

# --- UI OFFSET FOR PROGRESS BAR ---
Write-Host "`n`n`n`n`n" # 5 empty lines to stay below the blue progress bar

# --- HEADER & INFO BLOCK ---
Write-Host "=== ComfyUI Runtime Environment Manager ===" -ForegroundColor Cyan
Write-Host "A system for deploying and updating the ComfyUI runtime environment.`n" -ForegroundColor Gray

Write-Host "[*] HOW IT WORKS:" -ForegroundColor White
Write-Host "  - First run: Installs Python Manager, Git, VC++, Python" -ForegroundColor Gray
Write-Host "    (from the selected version), and a venv for it," -ForegroundColor Gray
Write-Host "    and clones ComfyUI." -ForegroundColor Gray

Write-Host "  - Second run (updates): Check for Git updates," -ForegroundColor Gray
Write-Host "    Py Manager updates, and minor Python updates" -ForegroundColor Gray
Write-Host "    for the installed version." -ForegroundColor Gray

Write-Host "  - Version Swap: To change Python branch," -ForegroundColor Gray
Write-Host "    delete the '$venvFolder' folder." -ForegroundColor Gray

Write-Host "    (Note: This will remove all pip packages installed" -ForegroundColor DarkYellow
Write-Host "    that are required for ComfyUI to work; you will need" -ForegroundColor DarkYellow
Write-Host "    to reinstall them in the virtual environment!)`n" -ForegroundColor DarkYellow

# --- 1. PYTHON INSTALL MANAGER ---
if (-not (Get-Command "py" -ErrorAction SilentlyContinue)) {
    Write-Host "[*] Installing Python Manager..." -ForegroundColor Yellow
    $baseUrl = "https://www.python.org/ftp/python/pymanager/"
    $msiFile = ((Invoke-WebRequest -Uri $baseUrl -UseBasicParsing).Links | Where-Object { $_.href -like "python-manager-*.msi" } | Select-Object -Last 1).href
    $msiPath = "$env:TEMP\pymanager.msi"
    Invoke-WebRequest -Uri ($baseUrl + $msiFile) -OutFile $msiPath
    Start-Process msiexec.exe -ArgumentList "/i `"$msiPath`" /quiet /norestart" -Wait
    Remove-Item $msiPath
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    $report += "[+] Python Launcher: Installed"
} else {
    $pyManagerRaw = py --version 2>$null | Select-Object -Last 1
    $pyManagerVer = $pyManagerRaw -replace '.*version\s+', ''
    $report += "[V] Python Launcher: Up to date ($pyManagerVer)"
}

# --- 2. GIT INSTALLATION & UPDATES (STABLE VERSION CHECK) ---
$gitInfo = Invoke-WebRequest -Uri "https://api.github.com/repos/git-for-windows/git/releases/latest" -UseBasicParsing | ConvertFrom-Json
$latestGitTag = ($gitInfo.tag_name -replace 'v', '').Trim()

if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
    Write-Host "[*] Downloading Git $latestGitTag..." -ForegroundColor Yellow
    $gitAsset = $gitInfo.assets | Where-Object { $_.name -like "Git-*-64-bit.exe" } | Select-Object -First 1
    Invoke-WebRequest -Uri $gitAsset.browser_download_url -OutFile "$env:TEMP\git.exe"
    Write-Host "[*] Installing Git..." -ForegroundColor Yellow
    Start-Process "$env:TEMP\git.exe" -ArgumentList "/VERYSILENT /NORESTART" -Wait
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    $report += "[+] Git: Installed ($latestGitTag)"
} else {
    $currentGitRaw = (git --version) -replace '.*version\s+', ''
    $currentGitVer = $currentGitRaw.Trim()
    
    if ($latestGitTag -ne $currentGitVer) {
        Write-Host "[*] Updating Git to $latestGitTag..." -ForegroundColor Yellow
        $gitAsset = $gitInfo.assets | Where-Object { $_.name -like "Git-*-64-bit.exe" } | Select-Object -First 1
        Invoke-WebRequest -Uri $gitAsset.browser_download_url -OutFile "$env:TEMP\git.exe"
        Start-Process "$env:TEMP\git.exe" -ArgumentList "/VERYSILENT /NORESTART" -Wait
        $report += "[+] Git: Updated to $latestGitTag"
    } else { $report += "[V] Git: Up to date ($currentGitVer)" }
}

# --- 3. MICROSOFT VISUAL C++ RUNTIME ---
if (-not (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue)) {
    Write-Host "[*] Installing Visual C++ Runtime..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "$env:TEMP\vc.exe"
    Start-Process "$env:TEMP\vc.exe" -ArgumentList "/install /quiet /norestart" -Wait
    $report += "[+] Visual C++: Installed"
} else { $report += "[V] Visual C++: Verified" }

# --- 4. PYTHON VERSION MANAGEMENT ---
if (-not (Test-Path $venvFolder)) {
    if (Test-Path $verFile) {
        $oldBranch = (Get-Content $verFile).Trim()
        Write-Host "[*] Removing legacy Python $oldBranch..." -ForegroundColor Gray
        py uninstall $oldBranch --yes *>$null
    }

    Write-Host "--- Python Version Selection ---" -ForegroundColor Cyan
    $pyList = py list --online | ForEach-Object { if ($_ -match "^\s*(3\.\d+\[-64\])") { $Matches[1] } } | Select-Object -First 3
    for ($i=0; $i -lt $pyList.Count; $i++) { Write-Host "$($i+1). $($pyList[$i])" }
    
    $choice = Read-Host "Select Python branch (1-3)"
    $selectedBranch = if ($choice -match '^[1-3]$') { $pyList[$choice-1].Replace('[-64]', '') } else { "3.12" }
    $selectedBranch | Out-File $verFile
    Write-Host "[*] Installing Python $selectedBranch..." -ForegroundColor Yellow
    py install $selectedBranch *>$null
    $report += "[+] Python: Selected branch $selectedBranch"
} else {
    $savedBranch = if (Test-Path $verFile) { (Get-Content $verFile).Trim() } else { "3.12" }
    Write-Host "[*] Checking for minor Python updates ($savedBranch)..." -ForegroundColor Gray
    py install -u $savedBranch *>$null 
    
    if (Test-Path "$venvFolder\pyvenv.cfg") {
        $vNum = (Get-Content "$venvFolder\pyvenv.cfg" | Select-String "version =" | ForEach-Object { ($_ -split "=")[1].Trim() })
        $report += "[V] Python: v$vNum (Branch $savedBranch)"
    }
}

# --- 5. COMFYUI REPOSITORY ---
if (-not (Test-Path $targetFolder)) {
    Write-Host "[*] Cloning ComfyUI repository..." -ForegroundColor Yellow
    git clone https://github.com/Comfy-Org/ComfyUI.git --quiet
    if ($LASTEXITCODE -eq 0) { $report += "[+] ComfyUI: Cloned" }
} else { $report += "[V] ComfyUI: Folder exists" }

# --- 6. VIRTUAL ENVIRONMENT (VENV) ---
if (-not (Test-Path $venvFolder)) {
    Write-Host "[*] Creating virtual environment..." -ForegroundColor Yellow
    py -m venv $venvFolder *>$null
    $report += "[+] Venv: Created"
} else { $report += "[V] Venv: Exists" }

# --- 7. PIP PACKAGE MANAGER ---
if (Test-Path "$venvFolder\Scripts\python.exe") {
    Write-Host "[*] Upgrading Pip..." -ForegroundColor Gray
    & "$PSScriptRoot\$venvFolder\Scripts\python.exe" -m pip install --upgrade pip --quiet *>$null
    $report += "[V] Pip: Up to date"
}

# --- FINAL SUMMARY REPORT ---
Write-Host "`n=== FINAL SUMMARY REPORT ===" -ForegroundColor Yellow
foreach ($line in $report) { Write-Host $line -ForegroundColor Yellow }
Write-Host ""
pause