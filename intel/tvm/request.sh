framework="tvm"

API_URL="https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/"$framework
function_name='jg-'$framework'-serving'

models="mobilenet_1.tar mobilenet_v2_1.tar inception_v3_1.tar resnet50_1.tar alexnet_1.tar vgg16_1.tar vgg19_1.tar"
models="vgg19_1.tar vgg16_1.tar alexnet_1.tar resnet50_1.tar inception_v3_1.tar mobilenet_v2_1.tar mobilenet_1.tar"
memorys="512 1024 2048 4096 8192"

echo "lambda_memory, model_name, hardware, framework, total_time, lambda_time, load_time" >> $framework'.csv'
for mem in $memorys
do
    for m in $models
    do
        aws lambda update-function-configuration \
            --function-name $function_name \
            --environment Variables="{model_name=$m}" \
            --memory-size $mem
        sleep 60
        
        SET=$(seq 0 4)
        for i in $SET
        do

        start=$(($(date +%s%N)/1000000))
        response=$(curl -X POST -H 'Content-Type: application/json' \
            -d '{"batch_size": 1, "workload": "image_classification" }' \
            $API_URL)
        end=$(($(date +%s%N)/1000000))
        runtime=$((end - start))
        
        echo $mem, $m, "intel", $framework, $((runtime / 1000)).$((runtime % 1000)), $response >> $framework'.csv'
        done
    done
done
