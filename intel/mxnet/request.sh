API_URL="https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/mxnet"
function_name='jg-mxnet-serving'

models="mobilenet mobilenet_v2 inception_v3 resnet50 alexnet vgg16 vgg19"
memorys="512 1024 2048 4096 8192"

for mem in $memorys
do
    for m in $models
    do
        aws lambda update-function-configuration \
            --function-name $function_name \
            --environment Variables="{model_name=$m}" \
            --memory-size $mem
        sleep 60
        
        echo $m "performance" >> mxnet.txt
        
        SET=$(seq 0 4)
        for i in $SET
        do

        start=$(($(date +%s%N)/1000000))
        response=$(curl -X POST -H 'Content-Type: application/json' \
            -d '{"batch_size": 1, "workload": "image_classification" }' \
            $API_URL)
        echo $response >> mxnet.txt
        end=$(($(date +%s%N)/1000000))
        runtime=$((end - start))
        echo "API runtime" $((runtime / 1000)).$((runtime % 1000)) >> mxnet.txt
        done
    echo "--------------------------------" >> mxnet.txt
    done
done
