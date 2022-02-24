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
if "bert_base" in model_name:
    model = gluon.nn.SymbolBlock.imports(model_json, ['data','token_types','valid_length'] , model_params, ctx=ctx)
elif "distilbert" in model_name:
    model = gluon.nn.SymbolBlock.imports(model_json, ['data','valid_length'], model_params, ctx=ctx)
elif "lstm" in model_name:
    model = gluon.nn.SymbolBlock.imports(model_json, ['data0','data1'], model_params, ctx=ctx)
else:
    model = gluon.nn.SymbolBlock.imports(model_json, ['data'], model_params, ctx=ctx)
load_time = time.time() - load_start

def make_dataset(multipart_data, workload, framework):
    if workload == "image_classification":
        binary_content = []
        for part in multipart_data.parts:
            binary_content.append(part.content)
        img = BytesIO(binary_content[0])
        img = Image.open(img)
        if model_name == "inception_v3":
            img = img.resize((299,299), Image.ANTIALIAS)
        else:
            img = img.resize((224,224), Image.ANTIALIAS)
        img = np.array(img)
        img = img.reshape(batch_size, channel, image_size, image_size)
        data = mx.nd.array(img, ctx=ctx)

        return data
    # case bert
    else:
        binary_content = []
        print('multipart_data: ', multipart_data)
        for part in multipart_data.parts:
            binary_content.append(part.content)
        print(binary_content)
        d = binary_content[0].split('\n\r')[0].encode('utf-8').astype('float32')
        print(d)
        inputs = np.array(binary_content[0])
        print(inputs)
        valid_length = 128
        valid_length = np.array(valid_length)
        inputs_nd = mx.nd.array(inputs, ctx=ctx)
        token_types_nd = mx.nd.array(inputs, ctx=ctx)
        valid_length_nd = mx.nd.array(valid_length, ctx=ctx)
        
        return inputs_nd, token_types_nd, valid_length_nd


def lambda_handler(event, context):
    handler_start = time.time()
    
    body = event['body-json']
    body = base64.b64decode(body)
    boundary = body.split(b'\r\n')[0]
    print('body_splited',body.split(b'\r\n'))
    boundary = boundary.decode('utf-8')
    content_type = f"multipart/form-data; boundary={boundary}"
    multipart_data = decoder.MultipartDecoder(body, content_type)
    framework = 'mxnet'

    if workload == "image_classification":
        data = make_dataset(multipart_data, workload, framework)
    #case bert
    else:
        data, token_types, valid_length = make_dataset(multipart_data, workload, framework)

    start_time = time.time()
    if workload == "image_classification":
        model(data)
    elif "bert_base" in model_name:
        model(data, token_types, valid_length)
    else:
        model(data, valid_length)
    running_time = time.time() - start_time
    print(f"MXNet {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
    handler_time = time.time() - handler_start
    return load_time, handler_time
