<#
.SYNOPSIS
  Windows permission/access diagnostic with optional fix modes.

.DESCRIPTION
  Designed to investigate "Access is denied" issues across projects.
  Default mode is read-only diagnostics.

  Features:
  - Environment and execution policy checks
  - ACL owner/inheritance/deny-rule checks
  - Full write-cycle test (create/append/rename/copy/delete)
  - Sampled subdirectory writability checks
  - Defender Controlled Folder Access (CFA) status
  - Optional safe fix and aggressive fix modes
  - Optional JSON report export

.EXAMPLE
  .\scripts\check_permissions.ps1

.EXAMPLE
  .\scripts\check_permissions.ps1 -TargetPath "C:\dev\myproj" -ExportJson

.EXAMPLE
  .\scripts\check_permissions.ps1 -TargetPath "C:\dev\myproj" -Fix -ExportJson

.EXAMPLE
  .\scripts\check_permissions.ps1 -TargetPath "C:\dev\myproj" -AggressiveFix -ExportJson
#>

[CmdletBinding()]
param(
    [string]$TargetPath = (Get-Location).Path,
    [switch]$Fix,
    [switch]$AggressiveFix,
    [switch]$ExportJson,
    [string]$ReportPath = "",
    [int]$SampleCount = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host ("  " + $Title) -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
}

function Write-Result {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Detail = ""
    )
    $icon = if ($Passed) { "OK  " } else { "FAIL" }
    $color = if ($Passed) { "Green" } else { "Red" }
    Write-Host ("[{0}] {1,-34} {2}" -f $icon, $Name, $Detail) -ForegroundColor $color
}

function Get-IsAdmin {
    try {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    catch {
        return $false
    }
}

function Try-GetRegValue {
    param(
        [string]$Path,
        [string]$Name
    )
    try {
        return (Get-ItemProperty -Path $Path -ErrorAction Stop).$Name
    }
    catch {
        return $null
    }
}

function Get-CFAStatus {
    $status = [ordered]@{
        source = "registry"
        enabled = "Unknown"
        raw = $null
        protected_folders = @()
        allowed_apps = @()
    }

    $regPath = "HKLM:\SOFTWARE\Microsoft\Windows Defender\Windows Defender Exploit Guard\Controlled Folder Access"
    $enabled = Try-GetRegValue -Path $regPath -Name "EnableControlledFolderAccess"
    $status.raw = $enabled
    if ($null -ne $enabled) {
        $status.enabled = switch ([int]$enabled) {
            1 { "Enabled" }
            2 { "AuditMode" }
            default { "Disabled" }
        }
    }

    try {
        if (Get-Command Get-MpPreference -ErrorAction SilentlyContinue) {
            $mp = Get-MpPreference -ErrorAction Stop
            $status.source = "Get-MpPreference"
            $status.protected_folders = @($mp.ControlledFolderAccessProtectedFolders)
            $status.allowed_apps = @($mp.ControlledFolderAccessAllowedApplications)
        }
    }
    catch {
        # keep registry fallback
    }

    return $status
}

function Get-ExecutionPolicies {
    try {
        return @(Get-ExecutionPolicy -List | ForEach-Object {
            [ordered]@{
                scope = $_.Scope
                policy = $_.ExecutionPolicy
            }
        })
    }
    catch {
        return @()
    }
}

function Test-WriteCycle {
    param([string]$BasePath)

    $result = [ordered]@{
        base_path = $BasePath
        create_dir = $false
        create_file = $false
        append_file = $false
        rename_file = $false
        copy_file = $false
        delete_file = $false
        delete_dir = $false
        error = ""
    }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $tmpDir = Join-Path $BasePath ".perm_check_$stamp"
    $a = Join-Path $tmpDir "a.txt"
    $b = Join-Path $tmpDir "b.txt"
    $c = Join-Path $tmpDir "c.txt"

    try {
        New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
        $result.create_dir = $true

        Set-Content -Path $a -Value "perm-check" -Encoding UTF8
        $result.create_file = $true

        Add-Content -Path $a -Value "append" -Encoding UTF8
        $result.append_file = $true

        Rename-Item -Path $a -NewName "b.txt" -Force
        $result.rename_file = $true

        Copy-Item -Path $b -Destination $c -Force
        $result.copy_file = $true

        Remove-Item -Path $b -Force
        Remove-Item -Path $c -Force
        $result.delete_file = $true

        Remove-Item -Path $tmpDir -Force
        $result.delete_dir = $true
    }
    catch {
        $result.error = $_.Exception.Message
        try { Remove-Item -Path $tmpDir -Recurse -Force -ErrorAction SilentlyContinue } catch {}
    }

    return $result
}

function Test-SampledDirs {
    param(
        [string]$BasePath,
        [int]$MaxSamples = 30
    )

    $rows = @()
    try {
        $dirs = @(Get-ChildItem -Path $BasePath -Directory -Recurse -ErrorAction SilentlyContinue | Select-Object -First $MaxSamples)
        foreach ($dir in $dirs) {
            $ok = $true
            $err = ""
            try {
                $probe = Join-Path $dir.FullName ".perm_probe.tmp"
                Set-Content -Path $probe -Value "x" -Encoding UTF8 -ErrorAction Stop
                Remove-Item -Path $probe -Force -ErrorAction Stop
            }
            catch {
                $ok = $false
                $err = $_.Exception.Message
            }
            $rows += [ordered]@{
                path = $dir.FullName
                writable = $ok
                error = $err
            }
        }
    }
    catch {
        $rows += [ordered]@{
            path = $BasePath
            writable = $false
            error = $_.Exception.Message
        }
    }
    return $rows
}

function Get-FocusedAclRules {
    param([System.Security.AccessControl.DirectorySecurity]$Acl)
    $who = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $focus = @($who, $env:USERNAME, "BUILTIN\Users", "Everyone", "BUILTIN\Administrators")
    $out = @()

    foreach ($rule in @($Acl.Access)) {
        foreach ($f in $focus) {
            if ($rule.IdentityReference.Value -like "*$f*") {
                $out += [ordered]@{
                    identity = $rule.IdentityReference.Value
                    type = $rule.AccessControlType.ToString()
                    rights = $rule.FileSystemRights.ToString()
                    inherited = [bool]$rule.IsInherited
                }
                break
            }
        }
    }
    return $out
}

function Invoke-SafeFix {
    param([string]$Path)
    $actions = @()

    try {
        cmd /c "attrib -R `"$Path\*`" /S /D" | Out-Null
        $actions += "Removed read-only attributes."
    }
    catch { $actions += "attrib -R failed: $($_.Exception.Message)" }

    try {
        cmd /c "icacls `"$Path`" /inheritance:e /T /C" | Out-Null
        $actions += "Enabled ACL inheritance."
    }
    catch { $actions += "icacls inheritance failed: $($_.Exception.Message)" }

    try {
        cmd /c "icacls `"$Path`" /grant `"$env:USERNAME`":(OI)(CI)M /T /C" | Out-Null
        $actions += "Granted Modify to current user."
    }
    catch { $actions += "icacls grant Modify failed: $($_.Exception.Message)" }

    return $actions
}

function Invoke-AggressiveFix {
    param([string]$Path)
    $actions = @()

    try {
        cmd /c "takeown /f `"$Path`" /r /d y" | Out-Null
        $actions += "takeown completed."
    }
    catch { $actions += "takeown failed: $($_.Exception.Message)" }

    try {
        $owner = "$env:USERDOMAIN\$env:USERNAME"
        cmd /c "icacls `"$Path`" /setowner `"$owner`" /T /C" | Out-Null
        $actions += "Owner set to current user."
    }
    catch { $actions += "setowner failed: $($_.Exception.Message)" }

    try {
        cmd /c "icacls `"$Path`" /grant `"$env:USERNAME`":(OI)(CI)F /T /C" | Out-Null
        $actions += "Granted FullControl to current user."
    }
    catch { $actions += "grant FullControl failed: $($_.Exception.Message)" }

    return $actions
}

function New-Recommendations {
    param([hashtable]$Ctx)
    $list = New-Object System.Collections.Generic.List[string]

    if (-not $Ctx.is_admin) {
        $list.Add("Run PowerShell/VS Code as Administrator and re-test.")
    }
    if ($Ctx.onedrive_path) {
        $list.Add("Project is under OneDrive. Move to a local non-sync path (for example C:\dev).")
    }
    if ($Ctx.cfa_enabled -eq "Enabled") {
        $list.Add("Controlled Folder Access is enabled. Allow Code.exe, powershell.exe, and python.exe.")
    }
    if (-not $Ctx.write_cycle_ok) {
        $list.Add("Write cycle failed. Try -Fix first, then -AggressiveFix if needed.")
    }
    if ($Ctx.long_paths_enabled -eq 0) {
        $list.Add("LongPathsEnabled=0. Enable long path support to avoid deep-path failures.")
    }
    if ($Ctx.has_deny_rule) {
        $list.Add("Explicit Deny ACL rules exist. Review and remove unnecessary Deny entries.")
    }
    return $list
}

try {
    Write-Section "Resolve Target"
    $resolved = (Resolve-Path -Path $TargetPath).Path
    $target = Get-Item -LiteralPath $resolved
    if (-not $target.PSIsContainer) {
        throw "TargetPath must be a directory: $resolved"
    }
    Write-Host "TargetPath : $resolved"

    Write-Section "Environment"
    $isAdmin = Get-IsAdmin
    $os = Get-CimInstance Win32_OperatingSystem
    $policies = Get-ExecutionPolicies
    $longPathsEnabled = Try-GetRegValue -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled"
    $cfa = Get-CFAStatus
    $isOneDrivePath = ($resolved -match "OneDrive")

    Write-Host ("Now        : {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ("Machine    : {0}" -f $env:COMPUTERNAME)
    Write-Host ("User       : {0}\{1}" -f $env:USERDOMAIN, $env:USERNAME)
    Write-Host ("IsAdmin    : {0}" -f $isAdmin)
    Write-Host ("PSVersion  : {0}" -f $PSVersionTable.PSVersion.ToString())
    Write-Host ("OS         : {0}" -f $os.Caption)
    Write-Host ("Build      : {0}" -f $os.BuildNumber)
    Write-Host ("LongPaths  : {0}" -f $(if ($null -eq $longPathsEnabled) { "Unknown" } else { $longPathsEnabled }))
    Write-Host ("OneDrive   : {0}" -f $isOneDrivePath)
    Write-Host ("CFA        : {0} (source={1}, raw={2})" -f $cfa.enabled, $cfa.source, $cfa.raw)

    Write-Host ""
    Write-Host "ExecutionPolicy:"
    foreach ($p in $policies) {
        Write-Host ("  - {0,-18} {1}" -f $p.scope, $p.policy)
    }

    Write-Section "Path and ACL"
    $acl = Get-Acl -LiteralPath $resolved
    $parent = Split-Path -Parent $resolved
    $parentAcl = Get-Acl -LiteralPath $parent
    $focusedAcl = Get-FocusedAclRules -Acl $acl
    $denyRules = @($acl.Access | Where-Object { $_.AccessControlType -eq "Deny" })
    $readOnly = (($target.Attributes -band [IO.FileAttributes]::ReadOnly) -ne 0)

    Write-Host ("Path             : {0}" -f $resolved)
    Write-Host ("Parent           : {0}" -f $parent)
    Write-Host ("Owner            : {0}" -f $acl.Owner)
    Write-Host ("Inheritance      : {0}" -f $(if ($acl.AreAccessRulesProtected) { "Disabled(Protected)" } else { "Enabled" }))
    Write-Host ("DenyRuleCount    : {0}" -f $denyRules.Count)
    Write-Host ("ReadOnlyAttr     : {0}" -f $readOnly)
    Write-Host ("ParentOwner      : {0}" -f $parentAcl.Owner)
    Write-Host ("ParentInherit    : {0}" -f $(if ($parentAcl.AreAccessRulesProtected) { "Disabled(Protected)" } else { "Enabled" }))

    Write-Host ""
    Write-Host "Focused ACL rules:"
    if ($focusedAcl.Count -eq 0) {
        Write-Host "  (none)"
    } else {
        foreach ($r in $focusedAcl) {
            Write-Host ("  - {0} | {1} | {2} | inherited={3}" -f $r.identity, $r.type, $r.rights, $r.inherited)
        }
    }

    Write-Section "Write Tests"
    $mainWrite = Test-WriteCycle -BasePath $resolved
    $parentWrite = Test-WriteCycle -BasePath $parent
    $mainOk = $mainWrite.create_dir -and $mainWrite.create_file -and $mainWrite.append_file -and $mainWrite.rename_file -and $mainWrite.copy_file -and $mainWrite.delete_file -and $mainWrite.delete_dir
    $parentOk = $parentWrite.create_dir -and $parentWrite.create_file -and $parentWrite.append_file -and $parentWrite.rename_file -and $parentWrite.copy_file -and $parentWrite.delete_file -and $parentWrite.delete_dir

    Write-Result -Name "Target write cycle" -Passed $mainOk -Detail $(if ($mainOk) { "All operations passed." } else { $mainWrite.error })
    Write-Result -Name "Parent write cycle" -Passed $parentOk -Detail $(if ($parentOk) { "All operations passed." } else { $parentWrite.error })

    Write-Host ""
    Write-Host "Target write-cycle detail:"
    foreach ($name in @("create_dir", "create_file", "append_file", "rename_file", "copy_file", "delete_file", "delete_dir")) {
        Write-Result -Name ("  " + $name) -Passed ([bool]$mainWrite.$name)
    }

    Write-Section "Sample Subdirectory Writability"
    $sample = Test-SampledDirs -BasePath $resolved -MaxSamples $SampleCount
    $sampleFailures = @($sample | Where-Object { -not $_.writable })
    Write-Host ("Sampled dirs      : {0}" -f $sample.Count)
    Write-Host ("Writable failures : {0}" -f $sampleFailures.Count)
    if ($sampleFailures.Count -gt 0) {
        Write-Host ""
        Write-Host "Top failures:"
        foreach ($row in ($sampleFailures | Select-Object -First 10)) {
            Write-Host ("  - {0}" -f $row.path)
            if ($row.error) { Write-Host ("    Error: {0}" -f $row.error) }
        }
    }

    $fixActions = @()
    if ($Fix -or $AggressiveFix) {
        Write-Section "Fix Mode"
        Write-Host ("Fix requested: Fix={0}, AggressiveFix={1}" -f $Fix, $AggressiveFix) -ForegroundColor Yellow

        if ($Fix) {
            $fixActions += Invoke-SafeFix -Path $resolved
        }
        if ($AggressiveFix) {
            $fixActions += Invoke-AggressiveFix -Path $resolved
        }

        foreach ($a in $fixActions) {
            Write-Host ("  - " + $a)
        }

        Write-Host ""
        Write-Host "Post-fix write test..." -ForegroundColor Cyan
        $after = Test-WriteCycle -BasePath $resolved
        $afterOk = $after.create_dir -and $after.create_file -and $after.append_file -and $after.rename_file -and $after.copy_file -and $after.delete_file -and $after.delete_dir
        Write-Result -Name "Post-fix write cycle" -Passed $afterOk -Detail $(if ($afterOk) { "All operations passed." } else { $after.error })
    }

    $ctx = @{
        is_admin = $isAdmin
        onedrive_path = $isOneDrivePath
        cfa_enabled = $cfa.enabled
        write_cycle_ok = $mainOk
        long_paths_enabled = $(if ($null -eq $longPathsEnabled) { -1 } else { [int]$longPathsEnabled })
        has_deny_rule = ($denyRules.Count -gt 0)
    }
    $reco = @(New-Recommendations -Ctx $ctx)

    Write-Section "Summary"
    Write-Result -Name "Target path writable" -Passed $mainOk -Detail $resolved
    Write-Result -Name "Has explicit deny rule" -Passed (-not $ctx.has_deny_rule) -Detail ("count=" + $denyRules.Count)
    Write-Result -Name "Run as admin" -Passed $isAdmin
    Write-Result -Name "OneDrive path" -Passed (-not $isOneDrivePath)
    Write-Result -Name "CFA not enforcing" -Passed ($cfa.enabled -ne "Enabled") -Detail ("status=" + $cfa.enabled)

    Write-Host ""
    Write-Host "Recommendations:"
    if ($reco.Count -eq 0) {
        Write-Host "  - No critical blocker detected."
    } else {
        foreach ($r in $reco) {
            Write-Host ("  - " + $r)
        }
    }

    $report = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        target_path = $resolved
        environment = [ordered]@{
            machine = $env:COMPUTERNAME
            user = "$env:USERDOMAIN\$env:USERNAME"
            is_admin = $isAdmin
            ps_version = $PSVersionTable.PSVersion.ToString()
            os = $os.Caption
            build = $os.BuildNumber
            long_paths_enabled = $longPathsEnabled
            onedrive_path = $isOneDrivePath
            execution_policy = $policies
        }
        defender = [ordered]@{
            cfa_status = $cfa.enabled
            cfa_source = $cfa.source
            cfa_raw = $cfa.raw
            protected_folders = $cfa.protected_folders
            allowed_apps = $cfa.allowed_apps
        }
        acl = [ordered]@{
            owner = $acl.Owner
            inheritance_enabled = (-not $acl.AreAccessRulesProtected)
            deny_rule_count = $denyRules.Count
            focused_rules = $focusedAcl
        }
        tests = [ordered]@{
            write_cycle_target = $mainWrite
            write_cycle_parent = $parentWrite
            sampled_children = $sample
        }
        fix = [ordered]@{
            requested_fix = [bool]$Fix
            requested_aggressive_fix = [bool]$AggressiveFix
            actions = $fixActions
        }
        recommendations = $reco
    }

    if ($ExportJson -or $ReportPath) {
        if (-not $ReportPath) {
            $reportDir = Join-Path $resolved "data\output\logs\permission_reports"
            New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
            $ReportPath = Join-Path $reportDir ("permission_report_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
        }
        $report | ConvertTo-Json -Depth 8 | Set-Content -Path $ReportPath -Encoding UTF8
        Write-Host ""
        Write-Host ("JSON report saved: {0}" -f $ReportPath) -ForegroundColor Green
    }
}
catch {
    Write-Host ""
    Write-Host ("FATAL: " + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
