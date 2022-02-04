import numpy as np
import onnxruntime as ort
import argparse
import time
import boto3

image_size = 224
channel = 3
image_classification_shape_type = {
    "mxnet" : (channel, image_size, image_size),
    "tf" : (image_size, image_size, channel)
}


def get_model(bucket_name, model_path, model_name):
    s3_client = boto3.client('s3')    
    s3_client.download_file(bucket_name, model_path, '/tmp/'+ model_name)
    return '/tmp/' + model_name

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
        inputs = np.random.randint(0, 2000, size=(batch, seq_length)).astype(dtype)
        token_types = np.random.uniform(size=(batch, seq_length)).astype(dtype)
        valid_length = np.asarray([seq_length] * batch).astype(dtype)
        
        return inputs, token_types, valid_length


def lambda_handler(event, context):
    bucket_name = event['bucket_name']
    batch_size = event['batch_size']
    arch_type = event['arch_type']
    frame_work = event['frame_work']
    model_name = event['model_name']
    compiler = 'onnx'
    model_path = f'{frame_work}/{compiler}/{model_name}'
    workload = event['workload']
    is_build = event['is_build']
    count = event['count']
    
    session = ort.InferenceSession(get_model(bucket_name, model_path, model_name))
    session.get_modelmeta()
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
    
    if workload == "image_classification":
        data, image_shape = make_dataset(batch_size, workload, framework)
        input_name = "data"
        module.set_input(input_name, data)
    #case bert
    else:
        data, token_types, valid_length = make_dataset(batch_size, workload)
        module.set_input(data0=data, data1=token_types, data2=valid_length)
    
    time_list = []
    for i in range(count):
        start_time = time.time()
        if workload == "image_classification":
            session.run(outname, {inname[0]: data})
        # case : bert
        else:
            session.run(outname, {inname[0]: data,inname[1]:token_types,inname[2]:valid_length})
        running_time = time.time() - start_time
        print(f"VM {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        time_list.append(running_time)
    time_medium = np.median(np.array(time_list))
    return time_medium
