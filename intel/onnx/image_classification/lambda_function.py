import numpy as np
import onnxruntime as ort
import argparse
import time
import boto3

def get_model(model_name, bucket_name):
    s3_client = boto3.client('s3')    
    s3_client.download_file(bucket_name, 'onnx/'+ model_name, '/tmp/'+ model_name)
    
    return '/tmp/' + model_name

def make_dataset(batch_size,size):
    image_shape = (size, size, 3)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape


def lambda_handler(event, context):
    bucket_name = event['bucket_name']
    batch_size = event['batch_size']
    model_name = event['model_name']
    count = event['count']
    size = 224
    arch_type = 'intel'
    
    model_path = get_model(model_name, bucket_name)
    
    session = ort.InferenceSession(model_path)
    session.get_modelmeta()
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
    
    data, image_shape = make_dataset(batch_size,size)
    
    time_list = []
    for i in range(count):
        start_time = time.time()
        session.run(outname, {inname[0]: data})
        running_time = time.time() - start_time
        print(f"VM {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        time_list.append(running_time)
    time_medium = np.median(np.array(time_list))
    return time_medium
