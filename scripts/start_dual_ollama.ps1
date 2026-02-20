<#
.SYNOPSIS
  Start two Ollama servers pinned to different NVIDIA GPUs.

.DESCRIPTION
  Useful for dual-GPU routing:
  - default model (8b) -> one Ollama endpoint
  - escalation model (30b) -> another Ollama endpoint

  This script launches two new PowerShell windows:
  1) Primary server:   CUDA_VISIBLE_DEVICES=<PrimaryGpuId>, host <PrimaryHost>
  2) Secondary server: CUDA_VISIBLE_DEVICES=<SecondaryGpuId>, host <SecondaryHost>

  Then update config.yaml:
    model_router.default_api_base
    model_router.escalation_api_base

.EXAMPLE
  .\scripts\start_dual_ollama.ps1

.EXAMPLE
  .\scripts\start_dual_ollama.ps1 -PrimaryGpuId 1 -SecondaryGpuId 0 -PrimaryPort 11434 -SecondaryPort 11435
#>

[CmdletBinding()]
param(
    [int]$PrimaryGpuId = 0,
    [int]$SecondaryGpuId = 1,
    [string]$PrimaryBindIp = "127.0.0.1",
    [int]$PrimaryPort = 11434,
    [string]$SecondaryBindIp = "127.0.0.1",
    [int]$SecondaryPort = 11435
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
    Write-Host ("  " + $Title) -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
}

function Assert-Command {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Command not found: $Name"
    }
}

function Start-OllamaServerWindow {
    param(
        [string]$Label,
        [int]$GpuId,
        [string]$Host
    )
    $psCommand = @"
`$env:CUDA_VISIBLE_DEVICES='$GpuId'
`$env:OLLAMA_HOST='$Host'
Write-Host '[${Label}] CUDA_VISIBLE_DEVICES=' `$env:CUDA_VISIBLE_DEVICES -ForegroundColor Yellow
Write-Host '[${Label}] OLLAMA_HOST=' `$env:OLLAMA_HOST -ForegroundColor Yellow
ollama serve
"@

    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        $psCommand
    ) | Out-Null
}

Assert-Command -Name "ollama"

$primaryHost = "$PrimaryBindIp`:$PrimaryPort"
$secondaryHost = "$SecondaryBindIp`:$SecondaryPort"

Write-Section "Dual Ollama Launch"
Write-Host ("Primary   : GPU {0} -> {1}" -f $PrimaryGpuId, $primaryHost)
Write-Host ("Secondary : GPU {0} -> {1}" -f $SecondaryGpuId, $secondaryHost)

Start-OllamaServerWindow -Label "PRIMARY" -GpuId $PrimaryGpuId -Host $primaryHost
Start-OllamaServerWindow -Label "SECONDARY" -GpuId $SecondaryGpuId -Host $secondaryHost

Write-Section "Next Step (config.yaml)"
Write-Host "Set routing endpoints:"
Write-Host ("  model_router.default_api_base:    http://{0}/v1" -f $primaryHost)
Write-Host ("  model_router.escalation_api_base: http://{0}/v1" -f $secondaryHost)

Write-Section "Sanity Checks"
Write-Host ("  curl http://{0}/api/tags" -f $primaryHost)
Write-Host ("  curl http://{0}/api/tags" -f $secondaryHost)
Write-Host "  ollama list"
Write-Host "  ollama ps"

