API_URL="https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/onnx"
Body='{"bucket_name": "dl-converted-models", "batch_size": 1, "arch_type": "intel", "framework": "mxnet", "model_name": "mobilenet_v2", "workload": "image_classification"}'

SET=$(seq 0 4)
for i in $SET
do
    curl -X POST -H 'Content-Type: application/json' \
        -d '{ "bucket_name" : "dl-converted-models", "batch_size": 1, "arch_type": "intel", "framework": "mxnet", "model_name": "mobilenet_v2.onnx", "workload": "image_classification" }' \
        $API_URL
done
