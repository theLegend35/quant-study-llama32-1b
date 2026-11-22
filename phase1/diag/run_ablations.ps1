$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location C:\Users\dhruv\quant-study
$base = @("export","openvino","--model","models\fp16-src","--task","text-generation-with-past","--weight-format","int4","--group-size","128","--ratio","1.0","--gptq","--num-samples","128")
$runs = @(
  @{n="ov-gptq-int4-c4";   a=@("--sym","--dataset","c4")},
  @{n="ov-gptq-int4-asym"; a=@("--dataset","wikitext2")},
  @{n="ov-gptq-int4-se";   a=@("--sym","--dataset","wikitext2","--scale-estimation")}
)
foreach ($r in $runs) {
  $out = "models\" + $r.n
  if (Test-Path $out) { "SKIP exists $out" | Out-File -Append diag\ablation-status.txt; continue }
  "START build $($r.n) $(Get-Date)" | Out-File -Append diag\ablation-status.txt
  & .\.venv\Scripts\optimum-cli.exe @base @($r.a) $out *> "diag\build-$($r.n).log"
  "END build $($r.n) exit=$LASTEXITCODE $(Get-Date)" | Out-File -Append diag\ablation-status.txt
  if (Test-Path "$out\openvino_model.xml") {
    & .\.venv\Scripts\lm_eval.exe --model openvino --model_args "pretrained=models/$($r.n),max_length=2048" --device gpu --tasks wikitext --batch_size 1 --seed 42 --output_path "diag\results\$($r.n)" *> "diag\ppl-$($r.n).log"
    "END ppl $($r.n) exit=$LASTEXITCODE $(Get-Date)" | Out-File -Append diag\ablation-status.txt
  }
}
"ALLDONE $(Get-Date)" | Out-File -Append diag\ablation-status.txt
