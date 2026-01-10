# PowerShell script to kill any running agent processes

Write-Host "Checking for running agent processes..."

# Find processes using the agent ports
$port9009 = Get-NetTCPConnection -LocalPort 9009 -ErrorAction SilentlyContinue
$port9019 = Get-NetTCPConnection -LocalPort 9019 -ErrorAction SilentlyContinue

if ($port9009) {
    $pid9009 = $port9009.OwningProcess
    Write-Host "Found process on port 9009: PID $pid9009"
    Stop-Process -Id $pid9009 -Force -ErrorAction SilentlyContinue
    Write-Host "Killed process on port 9009"
}

if ($port9019) {
    $pid9019 = $port9019.OwningProcess
    Write-Host "Found process on port 9019: PID $pid9019"
    Stop-Process -Id $pid9019 -Force -ErrorAction SilentlyContinue
    Write-Host "Killed process on port 9019"
}

# Also check for Python processes that might be running agents
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*rca_judge*" -or $_.CommandLine -like "*rca_finder*"
}

if ($pythonProcs) {
    Write-Host "Found Python agent processes:"
    $pythonProcs | ForEach-Object {
        Write-Host "  PID $($_.Id): $($_.ProcessName)"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Killed Python agent processes"
}

if (-not $port9009 -and -not $port9019 -and -not $pythonProcs) {
    Write-Host "No agent processes found running."
}

Write-Host "Done."
