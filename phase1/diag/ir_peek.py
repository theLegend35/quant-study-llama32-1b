import openvino as ov
m = ov.Core().read_model("models/ov-gptq-int4/openvino_model.xml")
n=0
for op in m.get_ordered_ops():
    if op.get_type_name()=="MatMul":
        w = op.input_value(1).get_node()
        chain=[]; cur=w
        for _ in range(6):
            chain.append(f"{cur.get_type_name()}:{cur.get_friendly_name()}:{cur.get_output_element_type(0)}:{cur.get_output_partial_shape(0)}")
            if cur.get_input_size()==0: break
            cur = cur.input_value(0).get_node()
        print(op.get_friendly_name(), "|", " <- ".join(chain)); n+=1
        if n>3: break
