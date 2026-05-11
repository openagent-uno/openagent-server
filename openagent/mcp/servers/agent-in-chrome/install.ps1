# PowerShell install script for OpenAgent in Chrome extension (Windows)
# Registers the native messaging host for Chrome, Edge, and Brave on Windows.
#
# Usage: .\install.ps1 <extension-id> [extension-id-2] [extension-id-3] ...
#
# Run from PowerShell as Administrator for HKLM registration,
# or as regular user for HKCU registration.

param(
    [Parameter(Mandatory=$true)]
    [string[]]$ExtensionIds
)

$ErrorActionPreference = "Stop"
$HOST_NAME = "com.openagent.agent_in_chrome"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$HOST_DIR = Join-Path $SCRIPT_DIR "host"
$NATIVE_HOST_PATH = Join-Path $HOST_DIR "native-host-wrapper.bat"

# Install npm dependencies
$NodeModules = Join-Path $HOST_DIR "node_modules"
if (-not (Test-Path $NodeModules)) {
    Write-Host "Installing npm dependencies..."
    Push-Location $HOST_DIR
    npm install
    Pop-Location
}

# Create Windows batch wrapper (Chrome needs .bat on Windows)
$BatchContent = @"
@echo off
node "$HOST_DIR\native-host.js"
"@
Set-Content -Path $NATIVE_HOST_PATH -Value $BatchContent

# Build allowed_origins
$Origins = ($ExtensionIds | ForEach-Object { "`"chrome-extension://$_/`"" }) -join ",`n    "

# Generate manifest
$Manifest = @"
{
  "name": "$HOST_NAME",
  "description": "OpenAgent in Chrome Native Messaging Host",
  "path": "$($NATIVE_HOST_PATH -replace '\\', '\\')",
  "type": "stdio",
  "allowed_origins": [
    $Origins
  ]
}
"@

# Install per-browser
function Install-Host {
    param($BrowserName, $RegKey)
    
    $RegPath = "HKCU:\Software\Google\Chrome\NativeMessagingHosts\$HOST_NAME"
    $BrowserRegPath = $RegKey -replace 'Chrome', $BrowserName
    
    # Try user-level first
    try {
        $parent = Split-Path -Parent $BrowserRegPath
        if (-not (Test-Path $parent)) {
            New-Item -Path $parent -Force | Out-Null
        }
        Set-ItemProperty -Path $parent -Name "(Default)" -Value $NATIVE_HOST_PATH -ErrorAction Stop
        Write-Host "  Installed for $BrowserName (user-level)"
    } catch {
        Write-Host "  Skipping $BrowserName: $($_.Exception.Message)"
    }
}

Write-Host ""
Write-Host "Installing native messaging host for extension(s): $($ExtensionIds -join ', ')"
Write-Host ""

# Install manifest files per browser
$ManifestJson = $Manifest
$ChromeDir = "$env:LOCALAPPDATA\Google\Chrome\User Data\NativeMessagingHosts"
$EdgeDir = "$env:LOCALAPPDATA\Microsoft\Edge\User Data\NativeMessagingHosts"
$BraveDir = "$env:LOCALAPPDATA\BraveSoftware\Brave-Browser\User Data\NativeMessagingHosts"

foreach ($dir in @($ChromeDir, $EdgeDir, $BraveDir)) {
    $parent = Split-Path -Parent $dir
    if (Test-Path $parent) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        $ManifestFile = Join-Path $dir "$HOST_NAME.json"
        Set-Content -Path $ManifestFile -Value $ManifestJson
        Write-Host "  Installed manifest: $ManifestFile"
    }
}

Write-Host ""
Write-Host "Done! Next steps:"
Write-Host ""
Write-Host "  1. Restart your browser (close all windows and reopen)"
Write-Host "  2. OpenAgent will now be able to control your browser"
Write-Host ""
Write-Host "  To test: ask your OpenAgent to 'navigate to example.com and take a screenshot'"
Write-Host ""
