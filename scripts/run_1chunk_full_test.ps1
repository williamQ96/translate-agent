<#
.SYNOPSIS
  Run a 1-chunk full pipeline test with GPU telemetry sampling.

.DESCRIPTION
  1) Runs full pipeline on a small OCR directory expected to produce ~1 chunk.
  2) Captures terminal output to a pipeline log.
  3) Samples `nvidia-smi` every 2s into CSV.

.EXAMPLE
  .\scripts\run_1chunk_full_test.ps1 -SourceDir "data/input/_bench_1chunk_xxx"
#>

[CmdletBinding()]
param(
    [string]$SourceDir = "",
    [int]$LoopTargetScore = 10,
    [int]$LoopMaxLoops = 3,
    [switch]$ForceFresh = $true
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

if (-not $SourceDir) {
    $latest = Get-ChildItem -Path "data/input" -Directory |
        Where-Object { $_.Name -like "_bench_1chunk_*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "No _bench_1chunk_* source dir found. Pass -SourceDir explicitly."
    }
    $SourceDir = $latest.FullName
}

if (-not (Test-Path -LiteralPath $SourceDir)) {
    throw "Source directory not found: $SourceDir"
}

$python = ".\translateagent\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outRoot = "data/output/legacy_data"
New-Item -ItemType Directory -Path $outRoot -Force | Out-Null

$root = (Get-Location).Path
$outRootAbs = Join-Path $root $outRoot
$pipeLog = Join-Path $outRootAbs ("_pipeline_1chunk_{0}.log" -f $ts)
$gpuLog = Join-Path $outRootAbs ("_gpu_1chunk_{0}.csv" -f $ts)
$stopFile = Join-Path $outRootAbs ("_gpu_stop_{0}.flag" -f $ts)

"timestamp,index,name,utilization.gpu,memory.used,memory.total" | Out-File -FilePath $gpuLog -Encoding utf8
if (Test-Path -LiteralPath $stopFile) {
    Remove-Item -LiteralPath $stopFile -Force
}

Write-Section "Start GPU Sampler"
$job = Start-Job -ScriptBlock {
    param($GpuLogPath, $StopPath)
    while (-not (Test-Path -LiteralPath $StopPath)) {
        $t = (Get-Date).ToString("s")
        $rows = & nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
        foreach ($r in $rows) {
            "{0},{1}" -f $t, $r | Add-Content -Path $GpuLogPath -Encoding utf8
        }
        Start-Sleep -Seconds 2
    }
} -ArgumentList $gpuLog, $stopFile

try {
    if ($ForceFresh) {
        $baseName = Split-Path -Path $SourceDir -Leaf
        $progressPath = Join-Path $root ("data/output/{0}_progress.json" -f $baseName)
        if (Test-Path -LiteralPath $progressPath) {
            Remove-Item -LiteralPath $progressPath -Force
            Write-Host ("Removed progress file: {0}" -f $progressPath) -ForegroundColor Yellow
        }
    }

    Write-Section "Run Pipeline"
    Write-Host ("Source      : {0}" -f $SourceDir)
    Write-Host ("Loop target : {0}" -f $LoopTargetScore)
    Write-Host ("Loop max    : {0}" -f $LoopMaxLoops)
    Write-Host ("Pipeline log: {0}" -f $pipeLog)
    Write-Host ("GPU log     : {0}" -f $gpuLog)

    & $python -m src.pipeline `
        --source $SourceDir `
        --no-style-prompt `
        --loop-target-score $LoopTargetScore `
        --loop-max-loops $LoopMaxLoops 2>&1 | Tee-Object -FilePath $pipeLog

    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline failed with exit code $LASTEXITCODE"
    }
}
finally {
    New-Item -ItemType File -Path $stopFile -Force | Out-Null
    $done = Wait-Job $job -Timeout 20
    if (-not $done) {
        Stop-Job $job -ErrorAction SilentlyContinue | Out-Null
    }
    Receive-Job $job -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $job
    Remove-Item -LiteralPath $stopFile -Force
}

Write-Section "Complete"
Write-Host ("PIPE_LOG={0}" -f $pipeLog)
Write-Host ("GPU_LOG={0}" -f $gpuLog)
