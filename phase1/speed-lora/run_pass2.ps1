# Pass 2: balanced interleaving (off,on,on,off,off,on) per base, 3 pairs each, after check2 finishes.
# Adds peak commit (immune to working-set trimming) and CPU-load logging to diagnose run-to-run noise.
$env:PYTHONIOENCODING="utf-8"; Set-Location C:\Users\dhruv\quant-study
while (-not (Select-String -Path speed-lora\check2.status -Pattern "ALLDONE" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 15 }
$st = "speed-lora\status2.txt"
"START $(Get-Date -Format s) $(powercfg /getactivescheme)" | Out-File -Append $st
Start-Sleep 20
$order = @("off","on","on","off","off","on")
foreach ($m in "fp16","rtn-int8","rtn-int4") {
  for ($i = 0; $i -lt 6; $i++) {
    $mode = $order[$i]; $rnd = 3 + $i
    if (!(Test-Path speed-lora\speed-$m-$mode-r$rnd.json)) {
      & .\.venv\Scripts\python.exe speed-lora\speed_lora.py $m $mode $rnd *> speed-lora\speed-$m-$mode-r$rnd.log
      "$m $mode r$rnd exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append $st
      Start-Sleep 10
    }
  }
}
"ALLDONE $(Get-Date -Format s)" | Out-File -Append $st
