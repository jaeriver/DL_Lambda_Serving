framework="test"

API_URL="https://jbu3pcymu6.execute-api.us-west-2.amazonaws.com/stage1/"$framework

response=$(curl -v -H 'Content-Type: multipart/form-data' \
    -F "data=@test.jpeg;type=application/json" \
    $API_URL)
