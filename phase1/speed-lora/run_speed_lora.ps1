# Detached runner: unpack adapters, sanity checks, then speed runs in two rounds with opposite order.
$env:PYTHONIOENCODING="utf-8"
Set-Location C:\Users\dhruv\quant-study
$st = "speed-lora\status.txt"
"START $(Get-Date -Format s)" | Out-File -Append $st
& .\.venv\Scripts\python.exe speed-lora\check_ov.py *> speed-lora\check_ov.log
"check_ov exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append $st
& .\.venv\Scripts\python.exe speed-lora\check_ref.py *> speed-lora\check_ref.log
"check_ref exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append $st
Start-Sleep 20
$r1 = @("fp16 off","fp16 on","rtn-int8 off","rtn-int8 on","rtn-int4 off","rtn-int4 on")
$r2 = @("rtn-int4 on","rtn-int4 off","rtn-int8 on","rtn-int8 off","fp16 on","fp16 off")
foreach ($pair in @(@(1,$r1),@(2,$r2))) {
  $rnd = $pair[0]
  foreach ($cell in $pair[1]) {
    $m,$mode = $cell.Split(" ")
    if (!(Test-Path speed-lora\speed-$m-$mode-r$rnd.json)) {
      & .\.venv\Scripts\python.exe speed-lora\speed_lora.py $m $mode $rnd *> speed-lora\speed-$m-$mode-r$rnd.log
      "$m $mode r$rnd exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append $st
      Start-Sleep 10
    }
  }
}
"ALLDONE $(Get-Date -Format s)" | Out-File -Append $st
