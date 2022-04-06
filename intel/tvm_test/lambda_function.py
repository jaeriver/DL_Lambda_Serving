import time
import tvm 
from tvm import relay
import numpy as np 
from tvm.contrib import graph_executor
from tvm.contrib import graph_runtime
import tvm.contrib.graph_executor as runtime
import os
from io import BytesIO
import base64
from PIL import Image
from requests_toolbelt.multipart import decoder
import torch

print('test1')

model_name = os.environ['model_name']
batch_size = int(os.environ['batch_size'])
workload = os.environ['workload']
framework = os.environ['framework']
dtype = "float32"
def load_model(model_name):

    PATH = model_name + '/'
    model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장
    
    return model

efs_path = '/mnt/efs/'
model_path = f'/var/task/{model_name}'


image_size = 224
if "inception_v3" in model_name:
    image_size = 299
channel = 3
image_classification_shape_type = {
    "mxnet" : (channel, image_size, image_size),
    "tf" : (image_size, image_size, channel)
}

arch_type = 'llvm -mcpu=core-avx2' 
# ctx = tvm.cpu()
if arch_type == 'arm':
    target = tvm.target.arm_cpu()
else:
    target = arch_type
ctx = tvm.device(target, 0)

input_shape = (batch_size, 3, image_size, image_size)

data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
torch_data = torch.tensor(data_array)
print('test2')

torch_model = load_model(model_path)
print('test3')

torch_model.eval()
traced_model = torch.jit.trace(torch_model, torch_data)

print('test4')
mod, params = relay.frontend.from_pytorch(traced_model, input_infos=[('input0', input_shape)],default_dtype=dtype)
print('test5')

build_time = time.time()
with tvm.transform.PassContext(opt_level=3):
    mod = relay.transform.InferType()(mod)
    graph, lib, params = relay.build_module.build(mod, target=target, params=params)
print('build time:', time.time() - build_time)
load_start = time.time()
module = graph_runtime.create(graph, lib, ctx)
load_time = time.time() - load_start
print('test6')

def make_dataset(multipart_data, workload, framework):
    if workload == "image_classification":
        mx_start = time.time()
        image_shape = (3, 224, 224)
        if "inception_v3" in model_name:
            image_shape = (3, 299, 299)
        data_shape = (batch_size,) + image_shape
        img = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        data = tvm.nd.array(img, ctx)

        print(time.time() - mx_start)
        return data
    # case bert
    else:
        mx_start = time.time()
        dtype = "float32"
        seq_length = 128
        inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
        token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
        valid_length = np.asarray([seq_length] * batch_size).astype(dtype)
        if "lstm" in model_name:
            inputs = np.transpose(inputs)
            token_types = np.transpose(token_types)
        dtype = 'float32'
        valid_length = np.asarray([seq_length] * batch_size).astype(dtype)
  
        inputs_nd = tvm.nd.array(inputs, ctx)
        token_types_nd = tvm.nd.array(token_types, ctx)
        valid_length_nd = tvm.nd.array(valid_length, ctx)
        print(time.time() - mx_start)
        return inputs_nd, token_types_nd, valid_length_nd


def lambda_handler(event, context):
    handler_start = time.time()
    compiler = 'tvm'
    multipart_data = ""
    if workload == "image_classification":

        data_start = time.time()
        data = make_dataset(multipart_data, workload, framework)
        print(time.time() - data_start)
        input_name = "input0"
        if "mxnet" in framework:
            input_name = "data"
        module.set_input("input_1", data)
        module.set_input(**params)
#         module.set_input(input_name, data)
    #case bert
    elif "bert_base" in model_name:
        data, token_types, valid_length = make_dataset(multipart_data, workload, framework)
        module.set_input(data0=data, data1=token_types, data2=valid_length)
    else:
        data, token_types, valid_length = make_dataset(multipart_data, workload, framework)
        module.set_input(data0=data, data1=valid_length)
    
    print('test7')
    
    start_time = time.time()
    module.run(data=data)
    running_time = time.time() - start_time
    print(f"TVM {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
    handler_time = time.time() - handler_start
    print(f"TVM {model_name}-{batch_size} handler latency : ",(handler_time)*1000,"ms")
    return load_time, handler_time
