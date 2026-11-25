# Inkling-2-Mini launcher for one 12 GB NVIDIA GPU (tested: RTX 3080 Ti + Ryzen 5 5600X)
# Starts an OpenAI-compatible server + web UI on port 8080 in the background, reachable from every
# device on your local network (phone, laptop, ...). Use -LocalOnly to keep it on this PC only.
# Image input works in the web UI and the API (send images as image_url / base64 data URLs).
# Stop it with .\stop-inkling.ps1
#
#   .\run-inkling.ps1                  # asks which mode: Standard (128k), Fast (64k, faster) or Long (256k)
#   .\run-inkling.ps1 -Mode Standard   # no question: whole model + vision in VRAM, 128k context, 8 chats/agents at once
#   .\run-inkling.ps1 -Mode Fast       # no question: 64k context with a 4-bit cache
#   .\run-inkling.ps1 -Mode Long       # no question: 256k context with the smaller long-context model (Inkling-2-Mini-fast-long.gguf)
#   .\run-inkling.ps1 -Ctx 65536       # different context size
#   .\run-inkling.ps1 -ImageTokens 2048  # more detail per image (default 1024; more = more VRAM on the first big image)
#   .\run-inkling.ps1 -Parallel 2      # how many requests run at the same time (default 8)
#   .\run-inkling.ps1 -NoVision        # text only (frees ~0.4 GB VRAM)
#   .\run-inkling.ps1 -Foreground      # run in this window (Ctrl+C to stop) instead of the background
#   .\run-inkling.ps1 -ModelDir D:\models -Engine C:\llama\llama-server.exe   # other locations
#   .\run-inkling.ps1 -CacheRam 4096   # keep up to 4 GB of older chats' states in RAM for instant switching (default 0 = off)
#
# Notes:
#  - The context is ONE pool shared by all parallel requests (-kvu): any single request can use all of it,
#    and 8 requests at once share it. Extra parallel slots cost no extra VRAM.
#  - Requests that start with the same text (e.g. OpenCode subagents sharing a system prompt) share it in the
#    cache: it is stored once and not processed again, so subagents start in milliseconds.
#  - The model name for API clients (OpenCode etc.) is "inkling-2-mini".
#  - VRAM is used up to ~11.9 of 12 GB (model + vision, 8 chats, desktop apps). If it overflows, Windows silently spills
#    into shared memory and speed drops to a third: close other GPU-heavy apps (games, video editors, other AI tools)
#    before starting. NVIDIA Broadcast alone takes ~0.25 GB and some GPU time.
#  - RAM: the model is read straight into VRAM (-lm none) and the server's RAM prompt cache is off,
#    so the server itself uses ~1 GB of system RAM. Each chat keeps only its 2 newest state snapshots
#    (--ctx-checkpoints 2) so RAM stays flat in long chats.
#  - Network: there is no password. Anyone on your Wi-Fi/LAN can use the chat while it runs; use -LocalOnly on
#    untrusted networks. If other devices can't connect, allow llama-server.exe in the Windows Firewall prompt
#    (Private networks) - it appears the first time.
#  - Log file: inkling-server.log next to this script.
param(
    [int]$Ctx = 0,
    [ValidateSet('', 'Standard', 'Fast', 'Long')][string]$Mode = '',
    [switch]$Long,
    [switch]$Quality,
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

$common = @('--jinja', '-ngl', '99', '--fit', 'off', '-fa', 'on', '-np', "$Parallel", '--ctx-checkpoints', '2',
            '-lm', 'none', '--cache-ram', "$CacheRam", '--alias', 'inkling-2-mini',
            '--temp', '1.0', '--top-p', '1.0', '--top-k', '0', '--min-p', '0.0',
            '--host', $(if ($LocalOnly) { '127.0.0.1' } else { '0.0.0.0' }), '--port', "$Port",
            '--image-max-tokens', "$ImageTokens")

if ($Parallel -gt 1) { $common += '-kvu' }   # one shared KV cache for all chats: same VRAM as -Parallel 1
if (-not $NoVision) { $common += @('--mmproj', (Join-Path $ModelDir "mmproj-Inkling-2-Mini-Q8_0.gguf")) }

# Custom quants, repacked ("folded") so the custom engine runs fewer GPU kernels per token (same math).
# 4-bit KV cache (Hadamard-rotated); -bs samples on the GPU.
# Numbers below: realtest.py (hi, 4032x3024 photo, 2560x1440 screenshot, 5k-token paste, follow-up; 8 slots, vision on,
# Discord/Chrome/NVIDIA Broadcast open on an RTX 3080 Ti). Writes = decode tok/s of the first reply -> reply at ~9k
# context. Top-1 = next-token agreement with the Q8 reference (kld.py).
$longFile = Join-Path $ModelDir "Inkling-2-Mini-fast-long.gguf"
if ($Long) { $Mode = 'Long' }
if (-not $Mode -and -not $Quality) {
    if ([Console]::IsInputRedirected) { $Mode = 'Standard' }
    else {
        Write-Host "Which mode?" -ForegroundColor Cyan
        Write-Host "  [1] Standard  128k context | writes 84 -> 74 tok/s | reads prompts ~1842 tok/s | 92.8% top-1 (default)"
        Write-Host "  [2] Fast       64k context | writes 84 -> 74 tok/s | reads prompts ~1827 tok/s | 92.8% top-1"
        Write-Host "  [3] Long      256k context | writes 85 -> 75 tok/s | reads prompts ~1757 tok/s | 92.4% top-1"
        $choice = (Read-Host "Enter 1, 2 or 3 (Enter = 1)").Trim()
        $Mode = switch ($choice) { '2' { 'Fast' } '3' { 'Long' } default { 'Standard' } }
    }
}
$Long = $Mode -eq 'Long'

$ub = 512
Remove-Item Env:LLAMA_KV_TYPE_GLOBAL -ErrorAction SilentlyContinue
# bits per layer by measured sensitivity, GPTQ error compensation on expert gate/up, MoE scales tuned to match the
# Q8 reference on real activations, Q5_K attention (8.6 GB, KLD 0.039 with an f16 cache)
$model = Join-Path $ModelDir "Inkling-2-Mini-fast-9.3-tuned.gguf"
if ($Quality) {
    # legacy: IQ2_XS expert gate/up, original NVFP4 expert down, Q6_K attention (10.9 GB)
    $Mode = 'Quality (legacy)'
    if (-not $Ctx) { $Ctx = 49152 }
    $model = Join-Path $ModelDir "Inkling-2-Mini-fast-folded.gguf"
    $ub = 256
    Remove-Item Env:LLAMA_KV_TYPE_GLOBAL -ErrorAction SilentlyContinue
} elseif ($Long) {
    # 9.3-tuned with the least sensitive expert layers cut further: 256k context with vision and 8 chats
    if (-not $Ctx) { $Ctx = 262144 }
    $model = $longFile
    $ub = 512
} elseif ($Mode -eq 'Fast') {
    # 64k context keeps more VRAM available for other GPU workloads
    if (-not $Ctx) { $Ctx = 65536 }
    $ub = 512
    Remove-Item Env:LLAMA_KV_TYPE_GLOBAL -ErrorAction SilentlyContinue
} else {
    if (-not $Ctx) { $Ctx = 131072 }
}
if (-not (Test-Path $model)) { Write-Host "Model not found: $model" -ForegroundColor Red; exit 1 }
$args2 = @('-m', $model, '-c', "$Ctx", '-ub', "$ub", '-t', '2', '-ctk', 'q4_0', '-ctv', 'q4_0', '-bs') + $common

if ($Foreground) { & $exe @args2; exit $LASTEXITCODE }

$p = Start-Process -FilePath $exe -ArgumentList $args2 -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
$p.Id | Set-Content $pidFile
Write-Host "Starting Inkling-2-Mini ($Mode mode, $Ctx context, $Parallel parallel, PID $($p.Id))..." -NoNewline
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
