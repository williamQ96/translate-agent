<#
.SYNOPSIS
  Run a small-scale translation quality loop smoke test on 3-4 chunks.

.DESCRIPTION
  This script creates an isolated mini workspace with selected chunks, then runs:
  1) chunk-level audit
  2) rewrite_audit_loop

  It is intended for quick validation after code/config changes.

.EXAMPLE
  .\scripts\smoke_test_4chunks.ps1

.EXAMPLE
  .\scripts\smoke_test_4chunks.ps1 -ChunkIds "1,2,3" -MaxLoops 2
#>

[CmdletBinding()]
param(
    [string]$SourceChunksDir = "data/output/source_chunks",
    [string]$ChunksDir = "data/output/chunks",
    [string]$ChunkIds = "1,2,3,4",
    [int]$TargetScore = 9,
    [int]$MaxLoops = 3,
    [string]$WorkRoot = "data/output/_smoke_tests"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    # Priority:
    # 1) active venv python
    # 2) local project venv ./translateagent/Scripts/python.exe
    # 3) system python from PATH
    if ($env:VIRTUAL_ENV) {
        $venvPy = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (Test-Path -LiteralPath $venvPy) { return $venvPy }
    }
    $localPy = ".\translateagent\Scripts\python.exe"
    if (Test-Path -LiteralPath $localPy) { return $localPy }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return "python" }
    throw "Python executable not found. Activate venv or install python."
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
    Write-Host ("  " + $Title) -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
}

function Parse-ChunkIds {
    param([string]$Raw)
    $nums = @()
    foreach ($p in ($Raw -split ",")) {
        $t = $p.Trim()
        if (-not $t) { continue }
        $n = 0
        if (-not [int]::TryParse($t, [ref]$n)) {
            throw "Invalid chunk id: $t"
        }
        if ($n -le 0) {
            throw "Chunk id must be >= 1: $n"
        }
        $nums += $n
    }
    $nums = @($nums | Sort-Object -Unique)
    if ($nums.Count -lt 1) {
        throw "No chunk ids provided."
    }
    if ($nums.Count -gt 6) {
        Write-Host "WARN: more than 6 chunks selected; smoke test may take longer." -ForegroundColor Yellow
    }
    return $nums
}

function Copy-SelectedChunks {
    param(
        [int[]]$Ids,
        [string]$FromDir,
        [string]$ToDir
    )
    New-Item -ItemType Directory -Path $ToDir -Force | Out-Null
    foreach ($id in $Ids) {
        $name = ("chunk_{0:000}.md" -f $id)
        $src = Join-Path $FromDir $name
        $dst = Join-Path $ToDir $name
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Missing chunk file: $src"
        }
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
}

try {
    $swTotal = [System.Diagnostics.Stopwatch]::StartNew()
    $pythonExe = Resolve-PythonExe
    $ids = Parse-ChunkIds -Raw $ChunkIds
    $runId = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDir = Join-Path $WorkRoot ("smoke_{0}" -f $runId)
    $srcMini = Join-Path $runDir "source_chunks"
    $trMini = Join-Path $runDir "chunks"
    $auditsDir = Join-Path $runDir "audits"
    $rewritesRoot = Join-Path $runDir "rewrites"

    Write-Section "Prepare Mini Workspace"
    Write-Host ("RunDir      : {0}" -f $runDir)
    Write-Host ("Chunk IDs   : {0}" -f (($ids | ForEach-Object { $_.ToString("000") }) -join ", "))
    Write-Host ("Source dir  : {0}" -f $SourceChunksDir)
    Write-Host ("Chunks dir  : {0}" -f $ChunksDir)

    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    Copy-SelectedChunks -Ids $ids -FromDir $SourceChunksDir -ToDir $srcMini
    Copy-SelectedChunks -Ids $ids -FromDir $ChunksDir -ToDir $trMini

    Write-Section "Stage 1 - Audit"
    $swAudit = [System.Diagnostics.Stopwatch]::StartNew()
    $auditArgs = @(
        "-m", "src.audit",
        "--source-chunks-dir", $srcMini,
        "--chunks-dir", $trMini
    )
    Write-Host ("Python      : {0}" -f $pythonExe)
    Write-Host ("Command     : {0} {1}" -f $pythonExe, ($auditArgs -join " "))
    & $pythonExe @auditArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Audit failed with exit code $LASTEXITCODE"
    }
    $swAudit.Stop()
    Write-Host ("Audit done  : {0:N1}s" -f $swAudit.Elapsed.TotalSeconds) -ForegroundColor Green

    Write-Section "Stage 2 - Rewrite Loop"
    $swLoop = [System.Diagnostics.Stopwatch]::StartNew()
    $loopArgs = @(
        "-m", "src.rewrite_audit_loop",
        "--source-chunks-dir", $srcMini,
        "--chunks-dir", $trMini,
        "--audit-dir", $auditsDir,
        "--output-root", $rewritesRoot,
        "--target-score", $TargetScore.ToString(),
        "--max-loops", $MaxLoops.ToString()
    )
    Write-Host ("Command     : {0} {1}" -f $pythonExe, ($loopArgs -join " "))
    & $pythonExe @loopArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Rewrite loop failed with exit code $LASTEXITCODE"
    }
    $swLoop.Stop()
    Write-Host ("Loop done   : {0:N1}s" -f $swLoop.Elapsed.TotalSeconds) -ForegroundColor Green

    $swTotal.Stop()
    Write-Section "Smoke Test Complete"
    Write-Host ("Total time  : {0:N1}s" -f $swTotal.Elapsed.TotalSeconds) -ForegroundColor Green
    Write-Host ("Mini run dir: {0}" -f $runDir)
    Write-Host ("Audits      : {0}" -f $auditsDir)
    Write-Host ("Rewrites    : {0}" -f $rewritesRoot)
    Write-Host ""
    Write-Host "Tip: if successful, send me the terminal output + run dir, I will analyze pass/fail and tuning next."
}
catch {
    Write-Host ""
    Write-Host ("FATAL: " + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
