cd C:\Users\dhruv\quant-study
$env:PYTHONIOENCODING = "utf-8"
while (-not (Select-String -Path diag\ablation-status.txt -Pattern "ALLDONE" -Quiet)) { Start-Sleep 60 }
.\.venv\Scripts\python diag\maxerr_check.py models\ov-gptq-int4-asym models\ov-gptq-int4-se *> diag\maxerr_ablations.log
.\.venv\Scripts\python diag\bos_window_ppl.py ov-fp16 ov-rtn-int8 ov-rtn-int4 ov-gptq-int4 ov-awq-int4 ov-gptq-int4-l1down-rtn *> diag\bos_window_ppl.log
"DONE" | Out-File diag\post.done
