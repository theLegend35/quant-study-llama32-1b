import numpy as np, torch
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained("models/fp16-src")
print("bos", tok.bos_token_id, "eos", tok.eos_token_id)
text = "\n\n".join(load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")["text"])
ids = tok(text, add_special_tokens=False).input_ids
L=2048; starts=[50000,120000]
for m in ["ov-fp16","ov-awq-int4","ov-gptq-int4"]:
    mdl = OVModelForCausalLM.from_pretrained("models/"+m, device="GPU")
    for name,pre in [("BOS",[tok.bos_token_id]),("EOS",[tok.eos_token_id]),("none",[])]:
        r=[]; r0=[]
        for s in starts:
            x = np.array(pre+ids[s:s+L-len(pre)])
            lg = mdl(input_ids=torch.tensor(x[None]),attention_mask=torch.ones(1,len(x),dtype=torch.long)).logits[0].float()
            nll = -torch.log_softmax(lg,-1)[:-1].gather(1,torch.tensor(x[1:])[:,None])[:,0]
            r.append(nll[len(pre):].mean().item()); r0.append(nll[len(pre):len(pre)+32].mean().item())
        print(m, name, "ppl", round(float(np.exp(np.mean(r))),2), "firstwin32 meanNLL", round(float(np.mean(r0)),2), flush=True)
    del mdl
