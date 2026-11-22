$env:PYTHONIOENCODING="utf-8"
Set-Location $PSScriptRoot
.\.venv\Scripts\optimum-cli export openvino --model models\fp16-src --task text-generation-with-past --weight-format int4 --sym --group-size 128 --ratio 1.0 --gptq --dataset wikitext2 --num-samples 128 models\ov-gptq-int4-v2 *> results\gptq-v2-build.log
if ($LASTEXITCODE -eq 0) {
  .\.venv\Scripts\lm_eval --model openvino --model_args "pretrained=models/ov-gptq-int4-v2,max_length=2048" --device gpu --tasks wikitext --batch_size 1 --seed 42 --output_path results\gptq-v2-ppl *> results\gptq-v2-ppl.log
}
"done" | Out-File results\gptq-v2-DONE.txt
