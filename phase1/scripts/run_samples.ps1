$env:PYTHONIOENCODING="utf-8"
Set-Location $PSScriptRoot
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState(0x80000001) | Out-Null
New-Item -ItemType Directory -Force results\samples | Out-Null
foreach ($m in "fp16","rtn-int8","rtn-int4","gptq-int4","awq-int4") {
  if (!(Test-Path results\samples\$m-full.DONE)) {
    .\.venv\Scripts\lm_eval --model openvino --model_args "pretrained=models/ov-$m" --device gpu --tasks arc_easy,arc_challenge,hellaswag,winogrande --batch_size 1 --seed 42 --log_samples --output_path results\samples\$m-full *> results\samples\$m-full.log
    if ($LASTEXITCODE -eq 0) { "ok $(Get-Date -Format o)" | Out-File results\samples\$m-full.DONE }
  }
  if (!(Test-Path results\samples\$m-qa.DONE)) {
    .\.venv\Scripts\lm_eval --model openvino --model_args "pretrained=models/ov-$m,max_length=2048" --device gpu --tasks piqa,boolq --batch_size 1 --seed 42 --log_samples --output_path results\samples\$m-qa *> results\samples\$m-qa.log
    if ($LASTEXITCODE -eq 0) { "ok $(Get-Date -Format o)" | Out-File results\samples\$m-qa.DONE }
  }
}
"done $(Get-Date -Format o)" | Out-File results\samples\ALLDONE.txt
[W.P]::SetThreadExecutionState(0x80000000) | Out-Null
