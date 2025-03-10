framework="mxnet"

API_URL="https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/"$framework
function_name='jg-'$framework'-serving'

models="mobilenet mobilenet_v2 inception_v3 resnet50 alexnet vgg16 vgg19"
models="vgg19 vgg16 alexnet resnet50 inception_v3 mobilenet_v2 mobilenet"
nlp="bert_base distilbert"
memorys="512 1024 2048 4096 8192"

echo "lambda_memory,model_name,hardware,framework,total_time,load_time,lambda_time" >> $framework'.csv'
for mem in $memorys
do
    for m in $models
    do
        aws lambda update-function-configuration \
            --function-name $function_name \
            --environment Variables="{model_name=$m, workload=image_classification}" \
            --memory-size $mem
        sleep 900
        
        SET=$(seq 1 100)
        for i in $SET
        do

        start=$(($(date +%s%N)/1000000))
        response=$(curl -X POST -H 'Content-Type: multipart/form-data' \
            -F "data=@png/test$i.jpeg" \
            $API_URL)
        end=$(($(date +%s%N)/1000000))
        runtime=$((end - start))
        
        echo $mem, $m, "intel", $framework, $((runtime / 1000)).$((runtime % 1000)), $response >> $framework'.csv'
        done
    done
    
    for m in $nlp
    do
        aws lambda update-function-configuration \
            --function-name $function_name \
            --environment Variables="{model_name=$m, workload=bert}" \
            --memory-size $mem
        sleep 900
        
        SET=$(seq 1 100)
        for i in $SET
        do

        start=$(($(date +%s%N)/1000000))
        response=$(curl -X POST -H 'Content-Type: multipart/form-data' \
            -F "data=@png/test$i.txt" \
            $API_URL)
        end=$(($(date +%s%N)/1000000))
        runtime=$((end - start))
        
        echo $mem, $m, "intel", $framework, $((runtime / 1000)).$((runtime % 1000)), $response >> $framework'.csv'
        done
    done
done
