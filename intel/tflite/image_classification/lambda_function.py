import tflite_runtime.interpreter as tflite
import numpy as np
import argparse
import time
import boto3

def get_model(model_name, bucket_name):
    s3_client = boto3.client('s3')    
    s3_client.download_file(bucket_name, 'tflite/'+ model_name, '/tmp/'+ model_name)
    
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
    
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    
    data, image_shape = make_dataset(batch_size,size)
    
    
    time_list = []
    for i in range(count):
        interpreter.set_tensor(input_details[0]['index'], data)
        start_time = time.time()
        interpreter.invoke()
        running_time = time.time() - start_time
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"TFLite {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        print(output_data)
        time_list.append(running_time)
    time_medium = np.median(np.array(time_list))
    return time_medium
