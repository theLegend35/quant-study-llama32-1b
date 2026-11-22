$env:PYTHONIOENCODING="utf-8"; Set-Location C:\Users\dhruv\quant-study
while (-not (Select-String -Path diag\se_run.status -Pattern "ALLDONE 10/16/2026 (19:3|19:4|19:5|2)" -Quiet)) { Start-Sleep 20 }
& .\.venv\Scripts\python.exe diag\se_extra.py *> diag\se_extra.log
Get-ChildItem models -Directory -Filter tmp-* | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
"EXTRADONE $(Get-Date)" | Out-File -Append diag\se_run.status
