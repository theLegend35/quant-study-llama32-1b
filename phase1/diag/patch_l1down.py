import openvino as ov, numpy as np, re, shutil, os
from openvino import opset13 as ops
core=ov.Core(); dst="models/ov-gptq-int4-l1down-rtn"
assert not os.path.exists(dst)
def find(m):
    for op in m.get_ordered_ops():
        if op.get_type_name()=="MatMul" and "layers.1.mlp.down_proj" in op.get_friendly_name(): return op
r=core.read_model("models/ov-rtn-int4/openvino_model.xml"); ro=find(r)
w=list(core.compile_model(ov.Model([ro.input_value(1)],[]),"CPU").create_infer_request().infer({}).values())[0]
g=core.read_model("models/ov-gptq-int4/openvino_model.xml"); go=find(g)
print("rtn weight", w.shape, w.dtype, "gptq input type", go.input_value(1).get_element_type())
c=ops.convert(ops.constant(np.asarray(w,np.float16)), go.input_value(1).get_element_type())
go.input(1).replace_source_output(c.output(0))
shutil.copytree("models/ov-gptq-int4", dst, ignore=shutil.ignore_patterns("openvino_model.xml","openvino_model.bin"))
ov.save_model(g, dst+"/openvino_model.xml", compress_to_fp16=False)
print("saved", dst)
