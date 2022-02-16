import time
from json import load
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, gluon
import numpy as np
import os
from io import BytesIO
import base64
from PIL import Image
from requests_toolbelt.multipart import decoder

ctx = mx.cpu()

model_name = os.environ['model_name']
batch_size = 1
workload = os.environ['workload']

efs_path = '/mnt/efs/'
model_path = efs_path + f'mxnet/base/{model_name}'

image_size = 224
if model_name == "inception_v3":
    image_size = 299
channel = 3

image_classification_shape_type = {
    "mxnet" : (channel, image_size, image_size),
    "tf" : (image_size, image_size, channel)
}

load_start = time.time()
model_json, model_params = model_path + '/model.json', model_path + '/model.params'
model = gluon.nn.SymbolBlock.imports(model_json, ['data'], model_params, ctx=ctx)
load_time = time.time() - load_start

def make_dataset(multipart_data, workload, framework):
    if workload == "image_classification":
        binary_content = []
        for part in multipart_data.parts:
            binary_content.append(part.content)
        print(binary_content)
        img = BytesIO(binary_content[0])
        img = Image.open(img)
        img = img.resize((224,224), Image.ANTIALIAS)
        img = np.array(img)
        img = img.reshape(batch_size, channel, image_size, image_size)
        data = mx.nd.array(img, ctx=ctx)

        return data
    # case bert
    else:
        seq_length = 128
        shape_dict = {
            "data0": (batch_size, seq_length),
            "data1": (batch_size, seq_length),
            "data2": (batch_size,),
        }
        dtype = "float32"
        inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
        token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
        valid_length = np.asarray([seq_length] * batch_size).astype(dtype)
        
        return inputs, token_types, valid_length


def lambda_handler(event, context):
    handler_start = time.time()
    
    body = event['body-json']
    body = base64.b64decode(body)
    boundary = body.split(b'\r\n')[0]
    boundary = boundary.decode('utf-8')
    content_type = f"multipart/form-data; boundary={boundary}"
    multipart_data = decoder.MultipartDecoder(body, content_type)
    
    framework = 'mxnet'
    if workload == "image_classification":
        data = make_dataset(multipart_data, workload, framework)
    #case bert
#     else:
#         data, token_types, valid_length = make_dataset(batch_size, workload, framework)

    start_time = time.time()
    model(data)
    running_time = time.time() - start_time
    print(f"MXNet {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
    handler_time = time.time() - handler_start
    return load_time, handler_time
