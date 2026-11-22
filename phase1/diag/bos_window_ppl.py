import numpy as np, torch, json, sys, math, os
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
from datasets import load_dataset
tok = AutoTokenizer.from_pretrained("models/fp16-src")
text = "\n\n".join(load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")["text"])
ids = tok(text, add_special_tokens=False).input_ids
nwords = len(text.split())
W = 2047  # text tokens per window; +BOS = 2048
chunks = [ids[i:i+W] for i in range(0, len(ids), W)]
chunks = chunks[:int(os.environ.get("NWIN", len(chunks)))]; out_path = os.environ.get("OUTJ","diag/bos_window_ppl.json")
print(f"tokens {len(ids)} words {nwords} windows {len(chunks)}", flush=True)
out = {}
for m in sys.argv[1:]:
    mdl = OVModelForCausalLM.from_pretrained("models/"+m, device="GPU")
    tot = 0.0; n = 0
    for c in chunks:
        x = torch.tensor([[tok.bos_token_id]+c])
        lg = mdl(input_ids=x, attention_mask=torch.ones_like(x)).logits[0].float()
        nll = -torch.log_softmax(lg,-1)[:-1].gather(1, x[0,1:,None])[:,0]
        tot += float(nll.double().sum()); n += len(c)
    out[m] = dict(token_ppl=math.exp(tot/n), word_ppl=math.exp(tot/nwords), nll_sum=tot, ntok=n)
    print(m, out[m], flush=True)
    del mdl
    json.dump(out, open(out_path,"w"), indent=1)
