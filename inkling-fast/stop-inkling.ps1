# Stops the Inkling-2-Mini server started by run-inkling.ps1
#   .\stop-inkling.ps1
#   .\stop-inkling.ps1 -Engine C:\llama\llama-server.exe   # if run-inkling.ps1 was started with -Engine
param([string]$Engine = (Join-Path $PSScriptRoot "..\build-cuda\bin\llama-server.exe"))

$pidFile = Join-Path $PSScriptRoot "inkling-server.pid"
$exes    = @([IO.Path]::GetFullPath($Engine))
$stopped = 0
$id = -1

if (Test-Path $pidFile) {
    $id = [int](Get-Content $pidFile -Raw).Trim()
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq 'llama-server') { Stop-Process -Id $id -Force; $stopped++ }
    Remove-Item $pidFile -Force
}

# Fallback: any other llama-server started from the same engine
Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $exes -contains $_.Path -and $_.Id -ne $id } | ForEach-Object {
    Stop-Process -Id $_.Id -Force; $stopped++
}

if ($stopped) { Write-Host "Inkling server stopped ($stopped process(es)). VRAM is free again." -ForegroundColor Green }
else          { Write-Host "No Inkling server was running." -ForegroundColor Yellow }
