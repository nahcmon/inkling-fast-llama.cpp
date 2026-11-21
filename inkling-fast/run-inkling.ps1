# Inkling-2-Mini launcher for RTX 3080 Ti (12 GB) + Ryzen 5 5600X
# Starts an OpenAI-compatible server + web UI on port 8080 in the background, reachable from every
# device on your local network (phone, laptop, ...). Use -LocalOnly to keep it on this PC only.
# Stop it with .\stop-inkling.ps1
#
#   .\run-inkling.ps1                  # FAST mode (default): whole model in VRAM, 64k context
#   .\run-inkling.ps1 -Quality         # QUALITY mode: original Q4_K_M, part of the experts in system RAM (slower)
#   .\run-inkling.ps1 -Ctx 32768       # different context size (fast mode max ~64k on 12 GB)
#   .\run-inkling.ps1 -Foreground      # run in this window (Ctrl+C to stop) instead of the background
#   .\run-inkling.ps1 -ModelDir D:\models -Engine C:\llama\llama-server.exe   # other locations
#   .\run-inkling.ps1 -CacheRam 4096   # keep up to 4 GB of older chats' states in RAM for instant switching (default 0 = off)
#   .\run-inkling.ps1 -Parallel 4      # serve 4 chats at once; they share one KV cache (-kvu), so no extra VRAM
#
# Notes:
#  - If VRAM overflows, Windows silently spills into shared memory and speed halves:
#    close other GPU-heavy apps (games, video editors, other AI tools) before starting.
#  - RAM: the model is read straight into VRAM (-lm none, no memory-mapped copy of the 11 GB file)
#    and the server's RAM prompt cache is off, so the server itself uses ~1 GB of system RAM.
#    Each chat keeps only its 2 newest state snapshots (--ctx-checkpoints 2, ~50 MB each) instead of 32,
#    so RAM stays flat in long chats and with -Parallel (enough to resume the next turn without reprocessing).
#  - Network: there is no password. Anyone on your Wi-Fi/LAN can use the chat while it runs; use -LocalOnly on
#    untrusted networks. If other devices can't connect, allow llama-server.exe in the Windows Firewall prompt
#    (Private networks) - it appears the first time.
#  - Log file: inkling-server.log next to this script.
param(
    [int]$Ctx = 0,
    [switch]$Quality,
    [switch]$Foreground,
    [int]$Port = 8080,
    [int]$CacheRam = 0,
    [int]$Parallel = 1,
    [switch]$LocalOnly
)

$exe = (Resolve-Path $Engine).Path   # custom build: fused Inkling kernels
$pidFile = Join-Path $PSScriptRoot "inkling-server.pid"
$logFile = Join-Path $PSScriptRoot "inkling-server.log"

if (Test-Path $pidFile) {
    $old = Get-Process -Id ([int](Get-Content $pidFile -Raw).Trim()) -ErrorAction SilentlyContinue
    if ($old -and $old.Path -eq $exe) { Write-Host "Server already running (PID $($old.Id)). Run .\stop-inkling.ps1 first." -ForegroundColor Yellow; exit 1 }
}

$common = @('--jinja', '-ngl', '99', '--fit', 'off', '-fa', 'on', '-np', "$Parallel", '--ctx-checkpoints', '2',
            '-lm', 'none', '--cache-ram', "$CacheRam",
            '--temp', '1.0', '--top-p', '1.0', '--top-k', '0', '--min-p', '0.0',
            '--host', $(if ($LocalOnly) { '127.0.0.1' } else { '0.0.0.0' }), '--port', "$Port")

if ($Parallel -gt 1) { $common += '-kvu' }   # one shared KV cache for all chats: same VRAM as -Parallel 1

if ($Quality) {
    if (-not $Ctx) { $Ctx = 16384 }
    $model = Join-Path $ModelDir "Inkling-2-Mini-UD-Q4_K_M.gguf"
    $args2 = @('-m', $model, '-c', "$Ctx", '-ub', '256', '-t', '6',
               '-ot', 'blk\.([0-4]|2[0-3])\.ffn_(gate_up|down)_exps=CPU') + $common
} else {
    # Custom quant: IQ2_XS expert gate/up, original NVFP4 expert down, Q6_K attention,
    # repacked ("folded") so the custom engine runs fewer GPU kernels per token (same math).
    # 4-bit KV cache (Hadamard-rotated) keeps 64k context inside 12 GB; -ub 512 = prompt
    # processed in 512-token batches (~1500 tok/s prefill, ~650 MB scratch); -bs samples on the GPU (+3%).
    if (-not $Ctx) { $Ctx = 65536 }
    $model = Join-Path $ModelDir "Inkling-2-Mini-fast-folded.gguf"
    $args2 = @('-m', $model, '-c', "$Ctx", '-ub', '512', '-t', '2',
               '-ctk', 'q4_0', '-ctv', 'q4_0', '-bs') + $common
}

if ($Foreground) { & $exe @args2; exit $LASTEXITCODE }

$p = Start-Process -FilePath $exe -ArgumentList $args2 -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
$p.Id | Set-Content $pidFile
Write-Host "Starting Inkling-2-Mini ($(if ($Quality) {'quality'} else {'fast'}) mode, PID $($p.Id))..." -NoNewline
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
