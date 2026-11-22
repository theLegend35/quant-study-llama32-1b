$env:PYTHONIOENCODING="utf-8"; Set-Location C:\Users\dhruv\quant-study
& .\.venv\Scripts\python.exe diag\patch_se_l1down.py *> diag\patch_se_l1down.log
& .\.venv\Scripts\lm_eval.exe --model openvino --model_args "pretrained=models/ov-gptq-int4-se-l1down-rtn,max_length=2048" --device gpu --tasks wikitext --batch_size 1 --seed 42 --output_path diag\results\ov-gptq-int4-se-l1down-rtn *> diag\ppl-ov-gptq-int4-se-l1down-rtn.log
"SEL1DONE $(Get-Date)" | Out-File -Append diag\se_run.status
