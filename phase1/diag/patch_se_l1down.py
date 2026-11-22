src=open("diag/se_bisect.py",encoding="utf-8-sig").read().split("tok=AutoTokenizer")[0]; exec(src)
patch(["layers.1.mlp.down_proj"],"models/ov-gptq-int4-se-l1down-rtn"); print("saved")
