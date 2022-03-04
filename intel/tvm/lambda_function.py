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

model_name = os.environ['model_name']
batch_size = 1
workload = os.environ['workload']

efs_path = '/mnt/efs/'
model_path = efs_path + f'mxnet/tvm/intel/{model_name}'

image_size = 224
if "inception_v3" in model_name:
    image_size = 299
channel = 3
image_classification_shape_type = {
    "mxnet" : (channel, image_size, image_size),
    "tf" : (image_size, image_size, channel)
}

ctx = tvm.cpu()

load_start = time.time()
loaded_lib = tvm.runtime.load_module(model_path)
module = runtime.GraphModule(loaded_lib["default"](ctx))
load_time = time.time() - load_start

def make_dataset(multipart_data, workload, framework):
    if workload == "image_classification":
        binary_content = []
        for part in multipart_data.parts:
            binary_content.append(part.content)
        print(binary_content)
        img = BytesIO(binary_content[0])
        print(img)
        img = Image.open(img)
        if "inception_v3" in model_name:
            img = img.resize((299,299), Image.ANTIALIAS)
        else:
            img = img.resize((224,224), Image.ANTIALIAS)
        img = np.array(img).astype('float32')
        data = img.reshape(batch_size, channel, image_size, image_size)
        data = tvm.nd.array(data)
        
        return data
    # case bert
    else:
        binary_content = []
        for part in multipart_data.parts:
            binary_content.append(part.content)
        d = binary_content[0].split(b'\n\r')[0].decode('utf-8')
        inputs = np.array([d.split(" ")]).astype('float32')
        if "lstm" in model_name:
            inputs = np.transpose(inputs)
        seq_length = 128
        dtype = 'float32'
        valid_length = np.asarray([seq_length] * batch_size).astype(dtype)
        inputs = tvm.nd.array(inputs)
        valid_length = tvm.nd.array(valid_length)
        return inputs, inputs, valid_length


def lambda_handler(event, context):
    handler_start = time.time()
    body = event['body-json']
    body = base64.b64decode(body)
    boundary = body.split(b'\r\n')[0]
    boundary = boundary.decode('utf-8')
    content_type = f"multipart/form-data; boundary={boundary}"
    multipart_data = decoder.MultipartDecoder(body, content_type)
    compiler = 'tvm'
    framework = 'mxnet'   
    arch_type = 'llvm -mcpu=core-avx2' 
    if arch_type == 'arm':
        target = tvm.target.arm_cpu()
    else:
        target = arch_type
    
    if workload == "image_classification":
        data_start = time.time()
        data = make_dataset(multipart_data, workload, framework)
        print(time.time() - data_start)
        input_name = "data"
        module.set_input(input_name, data)
    #case bert
    elif "bert_base" in model_name:
        data, token_types, valid_length = make_dataset(multipart_data, workload, framework)
        module.set_input(data0=data, data1=token_types, data2=valid_length)
    else:
        data, token_types, valid_length = make_dataset(multipart_data, workload, framework)
        module.set_input(data0=data, data1=valid_length)
    
    start_time = time.time()
    module.run(data=data)
    running_time = time.time() - start_time
    print(f"TVM {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
    handler_time = time.time() - handler_start
    return load_time, handler_time
