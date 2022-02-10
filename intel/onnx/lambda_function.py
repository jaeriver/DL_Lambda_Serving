import time
total_start = time.time()
import numpy as np
import onnxruntime as ort
import argparse
import os

image_size = 224
channel = 3
image_classification_shape_type = {
    "mxnet" : (channel, image_size, image_size),
    "tf" : (image_size, image_size, channel)
}

model_name = os.environ['model_name']
efs_path = '/mnt/efs/'
model_path = efs_path + f'mxnet/base/{model_name}'

load_start = time.time()
session = ort.InferenceSession(model_path)
session.get_modelmeta()
load_time = time.time() - load_start

def make_dataset(batch_size, workload, framework):
    if workload == "image_classification":
        image_shape = image_classification_shape_type[framework]
        data_shape = (batch_size,) + image_shape

        data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

        return data, image_shape
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
    event = event['body-json']
    bucket_name = os.environ['bucket_name']
    batch_size = event['batch_size']
    workload = event['workload']
    
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
    data_start = time.time()
    if workload == "image_classification":
        data, image_shape = make_dataset(batch_size, workload, framework)
        input_name = "data"
    #case bert
    else:
        data, token_types, valid_length = make_dataset(batch_size, workload, framework)
    data_time = time.time() - data_start
    start_time = time.time()
    if workload == "image_classification":
        session.run(outname, {inname[0]: data})
    # case : bert
    else:
        session.run(outname, {inname[0]: data,inname[1]:token_types,inname[2]:valid_length})
    running_time = time.time() - start_time
    print(f"ONNX {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
    total_time = time.time() - total_start
    handler_time = time.time() - handler_start
    return {'running_time': running_time, 'total_time': total_time, 'data_time':data_time, 'load_time':load_time, 'handler_time': handler_time}
