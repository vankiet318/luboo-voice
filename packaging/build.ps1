# Build dist\Luboo-Setup-<version>.exe  (run from anywhere; outputs go to <repo>\dist)
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1             public build, no API key inside
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -EmbedKey   build for trusted testers:
#       embeds your Groq key (env GROQ_API_KEY, else Luboo's Settings) -> Luboo-Setup-<v>-PRIVATE.exe
#       The key is only obfuscated: anyone determined can extract it. Revoke it at console.groq.com if leaked.
# Requires: Python 3.11 (py launcher), Inno Setup 6.
# (ASCII only: Windows PowerShell 5.1 misreads UTF-8 files without BOM.)
param([switch]$EmbedKey)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# 1. Clean build environment with only the runtime packages
if (-not (Test-Path ".venv-build")) {
    py -3.11 -m venv .venv-build
}
$py = ".\.venv-build\Scripts\python.exe"
& $py -m pip install -q --upgrade pip
& $py -m pip install -q -r packaging\requirements.txt

# 2. Icon, wake word models (pretrained ones only if no custom models in wakewords\)
& $py packaging\make_icon.py
& $py -c "from luboo import config, wakeword; wakeword.download_models(config.MODELS_DIR, [] if config.USE_CUSTOM_WAKE_WORDS else list(config.WAKE_WORDS))"

# 2b. Built-in Groq key (only with -EmbedKey). _embedded_key.py is git-ignored and removed after the build.
$embedded = Join-Path $root "luboo\_embedded_key.py"
if (Test-Path $embedded) { Remove-Item $embedded }
$suffix = ""
if ($EmbedKey) {
    $key = $env:GROQ_API_KEY
    $settingsPath = Join-Path $env:APPDATA "Luboo\settings.json"
    if (-not $key -and (Test-Path $settingsPath)) {
        $key = (Get-Content $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json).api_key
    }
    $env:LUBOO_EMBED_KEY = $key
    & $py packaging\embed_key.py
    $ok = $LASTEXITCODE -eq 0
    Remove-Item Env:\LUBOO_EMBED_KEY -ErrorAction SilentlyContinue
    if (-not $ok) { throw "No Groq API key found: set GROQ_API_KEY or enter a key in Luboo Settings first" }
    $suffix = "-PRIVATE"
}

# 3. PyInstaller -> dist\Luboo\
& $py -m PyInstaller packaging\luboo.spec --noconfirm --clean --log-level WARN
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# 4. Inno Setup -> dist\Luboo-Setup-<version>.exe
$version = & $py -c "from luboo import config; print(config.APP_VERSION)"
$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found - install it with: winget install JRSoftware.InnoSetup" }
& $iscc /Q "/DAppVersion=$version" "/DSuffix=$suffix" packaging\installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
if (Test-Path $embedded) { Remove-Item $embedded }

$setup = Get-Item "dist\Luboo-Setup-$version$suffix.exe"
$appSize = (Get-ChildItem "dist\Luboo" -Recurse -File | Measure-Object Length -Sum).Sum
"{0}  -  setup: {1:N1} MB, installed: {2:N1} MB" -f $setup.Name, ($setup.Length / 1MB), ($appSize / 1MB)
