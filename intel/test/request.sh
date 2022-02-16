framework="test"

API_URL="https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/"$framework

response=$(curl -X POST -H 'Content-Type: multipart/form-data' \
    -F "batch_size=1" \
    $API_URL)
#     -F "data=@test.jpeg" \
