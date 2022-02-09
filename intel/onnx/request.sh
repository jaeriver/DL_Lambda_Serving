API_URL=https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/onnx
Body='{"bucket_name": "dl-converted-models", "batch_size": 1, "arch_type": "intel", "framework": "mxnet", "model_name": "mobilenet_v2", "workload": "image_classification"}'

curl --request POST --url $API_URL -H 'Content-Type: application/json' -d $Body
