$env:PYTHONIOENCODING="utf-8"
Set-Location $PSScriptRoot
foreach ($m in "fp16","rtn-int8","rtn-int4","gptq-int4","awq-int4") {
  if (!(Test-Path results\speed-$m.json)) { .\.venv\Scripts\python speed.py $m *> results\speed-$m.log }
}
.\run_ext.ps1
