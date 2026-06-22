<#
.SYNOPSIS
    Registers (or removes) a daily Windows Task Scheduler job that keeps
    S&P 500 price data fresh for the Signal Dashboard.

.PARAMETER Uninstall
    Remove the scheduled task instead of creating it.

.PARAMETER RunAt
    Time of day to trigger the refresh (default: 17:00 = 5:00 PM local time).
    Run after 4:15 PM ET so Yahoo Finance has published final closing prices.

.EXAMPLE
    .\setup_scheduler.ps1                   # Install, runs at 5:00 PM daily
    .\setup_scheduler.ps1 -RunAt "18:00"    # Install, runs at 6:00 PM daily
    .\setup_scheduler.ps1 -Uninstall        # Remove the task
#>

param(
    [switch]$Uninstall,
    [string]$RunAt = "17:00"
)

$TaskName    = "SP500-Signal-DataRefresh"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Locate uv
$UvPath = try { (Get-Command uv -ErrorAction Stop).Source } catch { $null }
if (-not $UvPath) {
    Write-Error "uv not found in PATH. Install it from https://docs.astral.sh/uv/getting-started/installation/ then re-run this script."
    exit 1
}

# ── Uninstall ──────────────────────────────────────────────────────────────────

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Task '$TaskName' removed."
    } else {
        Write-Host "Task '$TaskName' not found — nothing to remove."
    }
    exit 0
}

# ── Install ────────────────────────────────────────────────────────────────────

$action = New-ScheduledTaskAction `
    -Execute $UvPath `
    -Argument "run python webapp\update_data.py" `
    -WorkingDirectory $ProjectRoot

$trigger = New-ScheduledTaskTrigger -Daily -At $RunAt

# StartWhenAvailable: catches up if the machine was off at trigger time
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

# Replace any existing registration silently
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName    $TaskName `
    -Action      $action `
    -Trigger     $trigger `
    -Settings    $settings `
    -Description "Downloads latest S&P 500 OHLCV data daily so the Signal Dashboard shows fresh signals." |
    Out-Null

Write-Host ""
Write-Host "Task '$TaskName' registered successfully."
Write-Host "  Runs daily at : $RunAt (local time)"
Write-Host "  Working dir   : $ProjectRoot"
Write-Host "  Command       : $UvPath run python webapp\update_data.py"
Write-Host "  Log file      : $ProjectRoot\webapp\update_data.log"
Write-Host ""
Write-Host "To run immediately : Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To remove          : .\setup_scheduler.ps1 -Uninstall"
