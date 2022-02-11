API_URL="https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/onnx"
function_name='jg-onnx_serving'

models="mobilenet.onnx mobilenet_v2.onnx inception_v3.onnx resnet50.onnx alexnet.onnx vgg16.onnx vgg19.onnx"
memorys="512 1024 2048 4096 8192"

for mem in $memorys
do
    echo "Memory:"$mem >> onnx.txt
    echo "---------------------" >> onnx.txt
    for m in $models
    do
        aws lambda update-function-configuration \
            --function-name $function_name \
            --environment Variables="{model_name=$m}" \
            --memory-size $mem
        sleep 60
        
        echo $m "performance" >> onnx.txt

        SET=$(seq 0 4)
        for i in $SET
        do
        start=$(($(date +%s%N)/1000000))
        response=$(curl -X POST -H 'Content-Type: application/json' \
            -d '{"batch_size": 1, "workload": "image_classification" }' \
            $API_URL)
        echo $response >> onnx.txt
        end=$(($(date +%s%N)/1000000))
        runtime=$((end - start))
        echo "API runtime" $((runtime / 1000)).$((runtime % 1000)) >> onnx.txt
        done
    echo "--------------------------------" >> onnx.txt
    done
done
