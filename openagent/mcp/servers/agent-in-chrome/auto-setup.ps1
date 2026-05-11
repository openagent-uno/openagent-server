# PowerShell auto-setup script for OpenAgent in Chrome (Windows)
# Installs native messaging manifests for all detected Chromium browsers.
# The extension ID is deterministic (baked into manifest.json) so no user input needed.
#
# Can be called by OpenAgent on startup, or run manually:
#   powershell -ExecutionPolicy Bypass -File auto-setup.ps1

$ErrorActionPreference = "Continue"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$EXTENSION_DIR = Join-Path $SCRIPT_DIR "extension"
$HOST_DIR = Join-Path $SCRIPT_DIR "host"
$HOST_NAME = "com.openagent.agent_in_chrome"

# Read deterministic extension ID
$IdFile = Join-Path $EXTENSION_DIR "extension-id.txt"
if (-not (Test-Path $IdFile)) {
    Write-Error "extension-id.txt not found. Run generate-key.py first."
    exit 1
}
$EXTENSION_ID = (Get-Content $IdFile).Trim()
$ORIGIN = "chrome-extension://$EXTENSION_ID/"

Write-Host "OpenAgent in Chrome auto-setup"
Write-Host "Extension ID: $EXTENSION_ID"
Write-Host ""

# Create Windows batch wrapper for native messaging
$BatchPath = Join-Path $HOST_DIR "native-host-wrapper.bat"
$BatchContent = @"
@echo off
node "$HOST_DIR\native-host.js"
"@
Set-Content -Path $BatchPath -Value $BatchContent

# Generate native messaging manifest
$ManifestJson = @"
{
  "name": "$HOST_NAME",
  "description": "OpenAgent in Chrome Native Messaging Host",
  "path": "$($BatchPath -replace '\\', '\\')",
  "type": "stdio",
  "allowed_origins": [
    "$ORIGIN"
  ]
}
"@

function Install-NativeHost {
    param($Name, $Dir)
    
    $parent = Split-Path -Parent $Dir
    if (-not (Test-Path $parent)) { return }
    
    New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    $ManifestFile = Join-Path $Dir "$HOST_NAME.json"
    Set-Content -Path $ManifestFile -Value $ManifestJson
    Write-Host "  [$Name] Installed: $ManifestFile"
}

Write-Host "Installing native messaging hosts..."

$LocalAppData = [Environment]::GetFolderPath("LocalApplicationData")

Install-NativeHost "Chrome" (Join-Path $LocalAppData "Google\Chrome\User Data\NativeMessagingHosts")
Install-NativeHost "Chrome Beta" (Join-Path $LocalAppData "Google\Chrome Beta\User Data\NativeMessagingHosts")
Install-NativeHost "Chromium" (Join-Path $LocalAppData "Chromium\User Data\NativeMessagingHosts")
Install-NativeHost "Edge" (Join-Path $LocalAppData "Microsoft\Edge\User Data\NativeMessagingHosts")
Install-NativeHost "Brave" (Join-Path $LocalAppData "BraveSoftware\Brave-Browser\User Data\NativeMessagingHosts")

Write-Host ""
Write-Host "Native messaging hosts installed."
Write-Host ""

# Create launch wrappers
function New-Launcher {
    param($Name, $ExePath)
    
    if (-not (Test-Path $ExePath)) { return }
    
    $LauncherPath = Join-Path $HOST_DIR "launch-$Name.bat"
    $Content = @"
@echo off
echo Launching $Name with OpenAgent extension...
taskkill /F /IM "$(Split-Path $ExePath -Leaf)" 2>nul
timeout /t 2 /nobreak >nul
start "" "$ExePath" --load-extension="$EXTENSION_DIR" --no-first-run --no-default-browser-check
"@
    Set-Content -Path $LauncherPath -Value $Content
    Write-Host "  Created launcher: $LauncherPath"
}

Write-Host "Creating browser launchers..."
Write-Host "  (Use these to launch your browser with the extension auto-loaded)"
Write-Host ""

$ProgramFiles = [Environment]::GetFolderPath("ProgramFiles")
$ProgramFilesX86 = [Environment]::GetFolderPath("ProgramFilesX86")

New-Launcher "chrome" (Join-Path $ProgramFiles "Google\Chrome\Application\chrome.exe")
New-Launcher "edge" (Join-Path $ProgramFilesX86 "Microsoft\Edge\Application\msedge.exe")
New-Launcher "brave" (Join-Path $ProgramFiles "BraveSoftware\Brave-Browser\Application\brave.exe")

Write-Host ""
Write-Host "Done! To start using OpenAgent in Chrome:"
Write-Host ""
Write-Host "  1. Launch your browser using one of the launchers above:"
Write-Host "     $HOST_DIR\launch-chrome.bat"
Write-Host ""
Write-Host "  2. Or manually launch your browser from cmd/PowerShell with:"
Write-Host "     chrome.exe --load-extension=$EXTENSION_DIR"
Write-Host ""
Write-Host "  3. Restart OpenAgent (or it will connect on next startup)"
Write-Host ""
