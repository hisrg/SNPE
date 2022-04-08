
"""
1.Reference: https://pytorch.org/docs/stable/onnx.html
2.Install related packages:
      1).pip install onnxruntime -i https://pypi.douban.com/simple or pip install onnxruntime-gpu -i https://pypi.douban.com/simple
      2).pip install torch -i https://pypi.douban.com/simple
      3).pip install onnx -i https://pypi.douban.com/simple
"""
import torch
import torchvision

"""
1. Download PreTrain model from official website: Alexnet, then Write in file as alexnet.onnx
"""
dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
model = torchvision.models.alexnet(pretrained=True).cuda()
"""
# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
"""
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

# Write in file 
torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

"""
2. Verify Onnx Model
"""
import onnx
# Load the ONNX model
model = onnx.load("alexnet.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

"""
3. Run Onnx Model
"""

import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("alexnet.onnx")

outputs = ort_session.run(
    None,
    {"actual_input_1": np.random.randn(10, 3, 224, 224).astype(np.float32)},
)
print(outputs[0])