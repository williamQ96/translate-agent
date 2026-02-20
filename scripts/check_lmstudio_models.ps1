<#
.SYNOPSIS
  Validate LM Studio OpenAI-compatible server and list available model IDs.

.DESCRIPTION
  Queries /v1/models and optionally checks whether required model IDs exist.
  Use this before running pipeline to avoid "model not found" runtime errors.

.EXAMPLE
  .\scripts\check_lmstudio_models.ps1

.EXAMPLE
  .\scripts\check_lmstudio_models.ps1 -ApiBase "http://127.0.0.1:1234/v1" `
    -RequireModels "qwen3-8b,qwen3-32b"
#>

[CmdletBinding()]
param(
    [string]$ApiBase = "http://127.0.0.1:1234/v1",
    [string]$RequireModels = "qwen3-8b,qwen3-32b"
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

$url = ($ApiBase.TrimEnd("/") + "/models")
Write-Section "LM Studio Server Check"
Write-Host ("Endpoint: {0}" -f $url)

try {
    $resp = Invoke-RestMethod -Method Get -Uri $url -TimeoutSec 10
}
catch {
    Write-Host ("ERROR: Cannot reach LM Studio server: {0}" -f $_.Exception.Message) -ForegroundColor Red
    Write-Host "Hint: Start LM Studio server (headless or GUI) and verify port/path."
    exit 1
}

$models = @()
if ($resp -and $resp.data) {
    foreach ($m in $resp.data) {
        if ($m.id) { $models += [string]$m.id }
    }
}
$models = @($models | Sort-Object -Unique)

Write-Section "Available Models"
if ($models.Count -eq 0) {
    Write-Host "No models returned by /v1/models" -ForegroundColor Yellow
}
else {
    foreach ($id in $models) {
        Write-Host ("  - {0}" -f $id)
    }
}

$required = @()
foreach ($x in ($RequireModels -split ",")) {
    $t = $x.Trim()
    if ($t) { $required += $t }
}

if ($required.Count -gt 0) {
    Write-Section "Requirement Check"
    $missing = @()
    foreach ($id in $required) {
        if ($models -contains $id) {
            Write-Host ("[OK ] {0}" -f $id) -ForegroundColor Green
        }
        else {
            Write-Host ("[MISS] {0}" -f $id) -ForegroundColor Yellow
            $missing += $id
        }
    }

    if ($missing.Count -gt 0) {
        Write-Host ""
        Write-Host "Action needed: update config.yaml model IDs to exact /v1/models names." -ForegroundColor Yellow
        exit 2
    }
}

Write-Host ""
Write-Host "LM Studio model check passed." -ForegroundColor Green
