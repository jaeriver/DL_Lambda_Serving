import time
from json import load
import numpy as np
import torch
import os
from io import BytesIO
import base64
from PIL import Image
from requests_toolbelt.multipart import decoder

model_name = os.environ['model_name']
batch_size = int(os.environ['batch_size'])
workload = os.environ['workload']
framework = 'torch'

efs_path = '/mnt/efs/'
model_path = efs_path + f'{framework}/base/{model_name}'

image_size = 224
if model_name == "inception_v3":
    image_size = 299
channel = 3

image_classification_shape_type = {
    "mxnet" : (channel, image_size, image_size),
    "tf" : (image_size, image_size, channel)
}

load_start = time.time()
model = model.load_state_dict(torch.load(model_path + '/model_state_dict.pt'))
model.eval()
load_time = time.time() - load_start

def make_dataset(multipart_data, workload, framework):
    if workload == "image_classification":
        mx_start = time.time()
#         binary_content = []
#         for part in multipart_data.parts:
#             binary_content.append(part.content)
#         img = BytesIO(binary_content[0])
#         img = Image.open(img)
#         if model_name == "inception_v3":
#             img = img.resize((299,299), Image.ANTIALIAS)
#         else:
#             img = img.resize((224,224), Image.ANTIALIAS)
        image_shape = (3, 224, 224)
        if "inception_v3" in model_name:
            image_shape = (3, 299, 299)
        data_shape = (batch_size,) + image_shape
        img = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        print(time.time() - mx_start)
        return img
    # case bert
    else:
        mx_start = time.time()
#         binary_content = []
#         for part in multipart_data.parts:
#             binary_content.append(part.content)
#         d = binary_content[0].split(b'\n\r')[0].decode('utf-8')
#         inputs = np.array([d.split(" ")]).astype('float32')
        seq_length = 128
        dtype = "float32"
        inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
        token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
        valid_length = np.asarray([seq_length] * batch_size).astype(dtype)
        if "lstm" in model_name:
            inputs = np.transpose(inputs)
            token_types = np.transpose(token_types)
        dtype = 'float32'
        valid_length = np.asarray([seq_length] * batch_size).astype(dtype)
  
        print(time.time() - mx_start)
        return inputs_nd, token_types_nd, valid_length_nd


def lambda_handler(event, context):
    handler_start = time.time()
    
#     body = event['body-json']
#     body = base64.b64decode(body)
#     boundary = body.split(b'\r\n')[0]
#     boundary = boundary.decode('utf-8')
#     content_type = f"multipart/form-data; boundary={boundary}"
#     multipart_data = decoder.MultipartDecoder(body, content_type)
    multipart_data = ""
    if workload == "image_classification":
        data = make_dataset(multipart_data, workload, framework)
    #case bert
    else:
        data, token_types, valid_length = make_dataset(multipart_data, workload, framework)

    start_time = time.time()
    if workload == "image_classification":
        model(data)
    elif "bert_base" in model_name:
        model.hybridize(static_alloc=True)
        model(data, valid_length, token_types)
    else:
        model.hybridize(static_alloc=True)
        model(data, valid_length,)
    running_time = time.time() - start_time
    print(f"Torch {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
    handler_time = time.time() - handler_start
    return load_time, handler_time
