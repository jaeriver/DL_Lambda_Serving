from json import load
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, gluon
import time
import numpy as np
import boto3

ctx = mx.cpu()

image_size = 224
channel = 3
image_classification_shape_type = {
    "mxnet" : (channel, image_size, image_size),
    "tf" : (image_size, image_size, channel)
}

s3_client = boto3.client('s3')    

def get_model(bucket_name, model_path, model_name):
    s3_client.download_file(bucket_name, model_path + '/model.json', '/tmp/'+ 'model.json')
    s3_client.download_file(bucket_name, model_path + '/model.params', '/tmp/'+ 'model.params')
    return '/tmp/'+ 'model.json', '/tmp/'+ 'model.params'

def make_dataset(batch_size, workload, framework):
    if workload == "image_classification":
        image_shape = image_classification_shape_type[framework]
        data_shape = (batch_size,) + image_shape

        data = np.random.uniform(size=data_shape)
        data = mx.nd.array(data, ctx=ctx)

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
    event = event['body-json']
    bucket_name = event['bucket_name']
    batch_size = event['batch_size']
    arch_type = event['arch_type']
    framework = event['framework']
    model_name = event['model_name']
    compiler = 'base'
    model_path = f'{framework}/{compiler}/{model_name}'
    workload = event['workload']

    model_json, model_params = get_model(bucket_name, model_path, model_name)
    model = gluon.nn.SymbolBlock.imports(model_json, ['data'], model_params, ctx=ctx)
    
    if workload == "image_classification":
        data, image_shape = make_dataset(batch_size, workload, framework)
        input_name = "data"
    #case bert
    else:
        data, token_types, valid_length = make_dataset(batch_size, workload, framework)
    

    start_time = time.time()
    model(data)
    running_time = time.time() - start_time
    print(f"MXNet {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
    return running_time
