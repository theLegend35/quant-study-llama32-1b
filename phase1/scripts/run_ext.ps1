$env:PYTHONIOENCODING="utf-8"
Set-Location $PSScriptRoot
foreach ($m in "fp16","rtn-int8","rtn-int4","gptq-int4","awq-int4") {
  if (!(Test-Path results\$m-qa.DONE)) {
    .\.venv\Scripts\lm_eval --model openvino --model_args "pretrained=models/ov-$m,max_length=2048" --device gpu --tasks piqa,boolq --batch_size 1 --seed 42 --output_path results\$m-qa *> results\$m-qa.log
    if ($LASTEXITCODE -eq 0) { "ok" | Out-File results\$m-qa.DONE }
  }
  if (!(Test-Path results\$m-ppl.DONE)) {
    .\.venv\Scripts\lm_eval --model openvino --model_args "pretrained=models/ov-$m,max_length=2048" --device gpu --tasks wikitext --batch_size 1 --seed 42 --output_path results\$m-ppl *> results\$m-ppl.log
    if ($LASTEXITCODE -eq 0) { "ok" | Out-File results\$m-ppl.DONE }
  }
}
"done" | Out-File results\ext-ALLDONE.txt
