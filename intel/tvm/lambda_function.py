import time
import_start_time = time.time()
from tvm import relay
import numpy as np 
import tvm 
from tvm.contrib import graph_executor
from tvm.contrib import graph_runtime
import onnx
import boto3
print('import time: ', time.time() - import_start_time)

image_size = 224

def get_model(model_name, bucket_name):
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, model_name, '/tmp/'+ model_name)
    return '/tmp/' + model_name


def make_dataset(batch_size, workload):
    if workload == 'image_classification":
        image_shape = (image_size, image_size, 3)
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
        input_shape = (shape_dict["data0"], shape_dict["data1"], shape_dict["data2"])
        data = np.random.uniform(size=input_shape[0])
        token_types = np.random.uniform(size=input_shape[1])
        valid_length = np.array([seq_length] * batch_size)
        
        return data, token_types, valid_length

load_model = time.time()


def lambda_handler(event, context):
    bucket_name = event['bucket_name']
    batch_size = event['batch_size']
    arch_type = event['arch_type']
    frame_work = event['frame_work']
    model_name = event['model_name']
    model_path = frame_work + arch_type + event['model_name']
    is_build = event['is_build']
    count = event['count']
    workload = event['workload']
    
    if arch_type == 'arm':
        target = tvm.target.arm_cpu()
    else:
        target = arch_type
    ctx = tvm.cpu()
    
    loaded_lib = tvm.runtime.load_module(get_model(model_path))
    module = runtime.GraphModule(loaded_lib["default"](ctx))
    
    if workload == "image_classification":
        data, image_shape = make_dataset(batch_size, workload)
        input_name = "data"
        module.set_input(input_name, data)
    #case bert
    else:
        data, token_types, valid_length = make_dataset(batch_size, workload)
        module.set_input(data0=data, data1=token_types, data2=valid_length)
    
    
    time_list = []
    for i in range(count):
        start_time = time.time()
        module.run(data=data)
        running_time = time.time() - start_time
        print(f"VM {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
        time_list.append(running_time)
    time_medium = np.median(np.array(time_list))
    return time_medium
