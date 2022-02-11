API_URL="https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/tvm"
function_name='jg-tvm'

models="mobilenet_1.tar mobilenet_v2_1.tar inception_v3_1.tar resnet50_1.tar alexnet_1.tar vgg16_1.tar vgg19_1.tar"
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

        SET=$(seq 0 4)
        for i in $SET
        do
        echo $m "performance" >> tvm.txt
        echo "----------------" >> tvm.txt
        start=`date +%s.%N`
        response=$(curl -X POST -H 'Content-Type: application/json' \
            -d '{"batch_size": 1, "workload": "image_classification" }' \
            $API_URL)
        echo response >> tvm.txt
        end=`date +%s.%N`
        runtime=$(($end-$start))
        echo "API runtime" $runtime >> tvm.txt
        done
    done
done
