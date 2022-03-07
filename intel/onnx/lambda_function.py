import time
import numpy as np
import onnxruntime as ort
import os
from io import BytesIO
import base64
from PIL import Image
from requests_toolbelt.multipart import decoder

model_name = os.environ['model_name']
batch_size = int(os.environ['batch_size'])
workload = os.environ['workload']

efs_path = '/mnt/efs/'
model_path = efs_path + f'mxnet/onnx/{model_name}'

image_size = 224
if "inception_v3" in model_name:
    image_size = 299
channel = 3
image_classification_shape_type = {
    "mxnet" : (channel, image_size, image_size),
    "tf" : (image_size, image_size, channel)
}

load_start = time.time()
session = ort.InferenceSession(model_path)
session.get_modelmeta()
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
        data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        print(time.time() - mx_start)
        return data
    # case bert
    else:
        mx_start = time.time()
#         binary_content = []
#         for part in multipart_data.parts:
#             binary_content.append(part.content)
#         d = binary_content[0].split(b'\n\r')[0].decode('utf-8')
#         inputs = np.array([d.split(" ")]).astype('float32')
        dtype = "float32"
        inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
        token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
        valid_length = np.asarray([seq_length] * batch_size).astype(dtype)
        if "lstm" in model_name:
            inputs = np.transpose(inputs)
            token_types = np.transpose(token_types)
        seq_length = 128
        dtype = 'float32'
        valid_length = np.asarray([seq_length] * batch_size).astype(dtype)
  
        print(time.time() - mx_start)
        return inputs, token_types, valid_length


def lambda_handler(event, context):
    handler_start = time.time()
    
#     body = event['body-json']
#     body = base64.b64decode(body)
#     boundary = body.split(b'\r\n')[0]
#     boundary = boundary.decode('utf-8')
#     content_type = f"multipart/form-data; boundary={boundary}"
#     multipart_data = decoder.MultipartDecoder(body, content_type)
    
    framework = 'mxnet'
    multipart_data = ""
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
    if workload == "image_classification":
        data = make_dataset(multipart_data, workload, framework)
    #case bert
    else:
        data, token_types, valid_length = make_dataset(multipart_data, workload, framework)
    start_time = time.time()
    if workload == "image_classification":
        session.run(outname, {inname[0]: data})
    # case : bert
    elif "bert_base" in model_name:
        session.run(outname, {inname[0]: data,inname[1]:token_types,inname[2]:valid_length})
    else:
        session.run(outname, {inname[0]: data,inname[1]:valid_length})
        
    running_time = time.time() - start_time
    handler_time = time.time() - handler_start
    return load_time, handler_time
