$env:PYTHONIOENCODING="utf-8"; Set-Location C:\Users\dhruv\quant-study
"START $(Get-Date)" | Out-File diag\se_run.status
& .\.venv\Scripts\python.exe diag\se_bisect.py *> diag\se_bisect.log
"bisect exit=$LASTEXITCODE $(Get-Date)" | Out-File -Append diag\se_run.status
& .\.venv\Scripts\python.exe diag\act_check.py *> diag\act_check.log
"act exit=$LASTEXITCODE $(Get-Date)" | Out-File -Append diag\se_run.status
if (Test-Path models\ov-gptq-int4-se-patched\openvino_model.xml) {
  & .\.venv\Scripts\lm_eval.exe --model openvino --model_args "pretrained=models/ov-gptq-int4-se-patched,max_length=2048" --device gpu --tasks wikitext --batch_size 1 --seed 42 --output_path diag\results\ov-gptq-int4-se-patched *> diag\ppl-ov-gptq-int4-se-patched.log
  "lmeval exit=$LASTEXITCODE $(Get-Date)" | Out-File -Append diag\se_run.status }
Get-ChildItem models -Directory -Filter tmp-se-patch* | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
"ALLDONE $(Get-Date)" | Out-File -Append diag\se_run.status

