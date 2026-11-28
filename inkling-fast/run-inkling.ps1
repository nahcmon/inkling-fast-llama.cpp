# Inkling-2-Mini Uncensored launcher for one 12 GB NVIDIA GPU (tested: RTX 3080 Ti + Ryzen 5 5600X)
# Starts an OpenAI-compatible server + web UI on port 8080 in the background, reachable from every
# device on your local network (phone, laptop, ...). Use -LocalOnly to keep it on this PC only.
# Image input works in the web UI and the API (send images as image_url / base64 data URLs).
# Stop it with .\stop-inkling.ps1
#
#   .\run-inkling.ps1                  # whole model + vision in VRAM, 128k context, 8 chats/agents at once
#   .\run-inkling.ps1 -Ctx 65536       # different context size (smaller = more VRAM left for other apps)
#   .\run-inkling.ps1 -ImageTokens 2048  # more detail per image (default 1024; larger limits need more VRAM)
#   .\run-inkling.ps1 -Parallel 2      # how many requests run at the same time (default 8)
#   .\run-inkling.ps1 -NoVision        # text only (frees ~0.4 GB VRAM)
#   .\run-inkling.ps1 -Foreground      # run in this window (Ctrl+C to stop) instead of the background
#   .\run-inkling.ps1 -ModelDir D:\models -Engine C:\llama\llama-server.exe   # other locations
#   .\run-inkling.ps1 -CacheRam 4096   # keep up to 4 GB of older chats' states in RAM for instant switching (default 0 = off)
#
# Notes:
#  - The context is ONE pool shared by all parallel requests (-kvu): any single request can use all of it,
#    and concurrent requests share it. Each extra slot reserves its own sliding-window KV and recurrent state.
#  - Requests that start with the same text (e.g. OpenCode subagents sharing a system prompt) share it in the
#    cache: it is stored once and not processed again, so subagents start in milliseconds.
#  - The model name for API clients (OpenCode etc.) is "inkling-2-mini".
#  - VRAM is used up to ~11.9 of 12 GB (model + vision, 8 chats, 128k, desktop apps). If it overflows, Windows silently
#    spills into shared memory and speed drops to a third: close other GPU-heavy apps (games, video editors, other AI
#    tools) before starting. NVIDIA Broadcast alone takes ~0.25 GB and some GPU time.
#  - RAM: the model is read straight into VRAM (-lm none) and the server's RAM prompt cache is off,
#    so the server itself uses ~1 GB of system RAM. Each chat keeps only its 2 newest state snapshots
#    (--ctx-checkpoints 2) so RAM stays flat in long chats.
#  - Network: there is no password. Anyone on your Wi-Fi/LAN can use the chat while it runs; use -LocalOnly on
#    untrusted networks. If other devices can't connect, allow llama-server.exe in the Windows Firewall prompt
#    (Private networks) - it appears the first time.
#  - Uncensored model: it follows requests the original refuses. You are responsible for what you do with it.
#  - Log file: inkling-server.log next to this script.
param(
    [int]$Ctx = 131072,
    [string]$Mode = '',      # retired (Fast / Long modes were removed); accepted so old shortcuts still work
    [switch]$Long,           # retired
    [int]$ImageTokens = 1024,
    [switch]$Foreground,
    [int]$Port = 8080,
    [int]$CacheRam = 0,
    [int]$Parallel = 8,
    [switch]$NoVision,
    [switch]$LocalOnly,
    [string]$ModelDir = $PSScriptRoot,
    [string]$Engine = (Join-Path $PSScriptRoot "..\build-cuda\bin\llama-server.exe")
)

$exe = (Resolve-Path $Engine).Path   # custom build: fused Inkling kernels
$pidFile = Join-Path $PSScriptRoot "inkling-server.pid"
$logFile = Join-Path $PSScriptRoot "inkling-server.log"

if (Test-Path $pidFile) {
    $old = Get-Process -Id ([int](Get-Content $pidFile -Raw).Trim()) -ErrorAction SilentlyContinue
    if ($old -and $old.Path -eq $exe) { Write-Host "Server already running (PID $($old.Id)). Run .\stop-inkling.ps1 first." -ForegroundColor Yellow; exit 1 }
}
if (($Mode -and $Mode -ne 'Standard') -or $Long) {
    Write-Host "Fast and Long modes were removed; starting the 128k model (use -Ctx for another context size)." -ForegroundColor Yellow
}

# Custom quant of AtomicChat's uncensored Inkling-2-Mini, repacked ("folded") so the custom engine runs fewer GPU
# kernels per token (same math): bits per layer by measured sensitivity, GPTQ error compensation and learned block
# scales on the expert gate/up, Q5_K attention (9.24 GB, KLD 0.038 vs its Q8 reference with an f16 cache).
# Hadamard-rotated q4_0 KV cache; -bs samples on the GPU.
Remove-Item Env:LLAMA_KV_TYPE_GLOBAL -ErrorAction SilentlyContinue
$model = Join-Path $ModelDir "Inkling-2-Mini-Uncensored-fast.gguf"
if (-not (Test-Path $model)) { Write-Host "Model not found: $model" -ForegroundColor Red; exit 1 }
$common = @('--jinja', '-ngl', '99', '--fit', 'off', '-fa', 'on', '-np', "$Parallel", '--ctx-checkpoints', '2',
            '-lm', 'none', '--cache-ram', "$CacheRam", '--alias', 'inkling-2-mini',
            '--temp', '1.0', '--top-p', '1.0', '--top-k', '0', '--min-p', '0.0',
            '--host', $(if ($LocalOnly) { '127.0.0.1' } else { '0.0.0.0' }), '--port', "$Port",
            '--image-max-tokens', "$ImageTokens")

if ($Parallel -gt 1) { $common += '-kvu' }   # one shared global KV cache for all chats
if (-not $NoVision) { $common += @('--mmproj', (Join-Path $ModelDir "mmproj-Inkling-2-Mini-Q8_0.gguf")) }
$args2 = @('-m', $model, '-c', "$Ctx", '-ub', '512', '-t', '2', '-ctk', 'q4_0', '-ctv', 'q4_0', '-bs') + $common

if ($Foreground) { & $exe @args2; exit $LASTEXITCODE }

$p = Start-Process -FilePath $exe -ArgumentList $args2 -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
$p.Id | Set-Content $pidFile
Write-Host "Starting Inkling-2-Mini Uncensored ($Ctx context, $Parallel parallel, PID $($p.Id))..." -NoNewline
while ($true) {
    if ($p.HasExited) { Write-Host " FAILED. See $logFile.err" -ForegroundColor Red; Remove-Item $pidFile -Force; exit 1 }
    try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq 'ok') { break } } catch {}
    Start-Sleep -Milliseconds 500; Write-Host "." -NoNewline
}
Write-Host " ready." -ForegroundColor Green
Write-Host "Web UI / API: http://127.0.0.1:$Port   (OpenAI API base: http://127.0.0.1:$Port/v1)"
if (-not $LocalOnly) {
    $lan = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
           Where-Object { $_.PrefixOrigin -in 'Dhcp', 'Manual' -and $_.IPAddress -notlike '169.254.*' -and $_.IPAddress -ne '127.0.0.1' -and $_.InterfaceAlias -notlike 'vEthernet*' }
    foreach ($a in $lan) { Write-Host "From other devices on your network: http://$($a.IPAddress):$Port   ($($a.InterfaceAlias))" -ForegroundColor Cyan }
}
Write-Host "Stop with: .\stop-inkling.ps1"
