API_URL="https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/mxnet"
function_name='jg-mxnet-serving'

aws lambda update-function-configuration \
    --function-name $function_name \
    --environment Variables="{model_name=$1}"
SET=$(seq 0 4)
for i in $SET
do
start=`date +%s`
curl -X POST -H 'Content-Type: application/json' \
    -d '{"batch_size": 1, "workload": "image_classification" }' \
    $API_URL
    echo ""
end=`date +%s`
runtime=$((end-start))
done
