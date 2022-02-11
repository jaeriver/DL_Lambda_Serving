API_URL="https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/onnx"
function_name='jg-onnx_serving'

models=("mobilenet.onnx" "mobilenet_v2.onnx" "inception_v3.onnx" "resnet50.onnx" "alexnet.onnx" "vgg16.onnx" "vgg19.onnx")
memorys=("512" "1024" "2048" "4096" "8192")

for mem in $memorys
do
    for m in $models
    do
        aws lambda update-function-configuration \
            --function-name $function_name \
            --environment Variables="{model_name=$m}"
            --memory-size $mem
        sleep 60

        SET=$(seq 0 4)
        for i in $SET
        do
        echo $m "performance"
        echo "----------------"
        start=`date +%s.%N`
        curl -X POST -H 'Content-Type: application/json' \
            -d '{"batch_size": 1, "workload": "image_classification" }' \
            $API_URL
            echo ""
        end=`date +%s.%N`
        runtime=$((end-start))
        echo "API runtime" $runtime
        done
    done
done
