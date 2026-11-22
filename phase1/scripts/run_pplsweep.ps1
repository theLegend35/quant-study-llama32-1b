$env:PYTHONIOENCODING="utf-8"
Set-Location $PSScriptRoot
foreach ($L in 256,512,1024) { foreach ($m in "gptq-int4","awq-int4") {
  .\.venv\Scripts\lm_eval --model openvino --model_args "pretrained=models/ov-$m,max_length=$L" --device gpu --tasks wikitext --batch_size 1 --seed 42 --output_path results\pplsweep-$m-$L *> results\pplsweep-$m-$L.log
}}
"done" | Out-File results\pplsweep-DONE.txt
