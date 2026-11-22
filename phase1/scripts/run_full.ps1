$env:PYTHONIOENCODING="utf-8"
Set-Location $PSScriptRoot
foreach ($m in "fp16","rtn-int8","rtn-int4","gptq-int4","awq-int4") {
  if (Test-Path results\$m-full.DONE) { continue }
  .\.venv\Scripts\lm_eval --model openvino --model_args "pretrained=models/ov-$m" --device gpu --tasks arc_easy,arc_challenge,hellaswag,winogrande --batch_size 1 --seed 42 --output_path results\$m-full *> results\$m-full.log
  if ($LASTEXITCODE -eq 0) { "ok" | Out-File results\$m-full.DONE }
}
"done" | Out-File results\full-ALLDONE.txt
