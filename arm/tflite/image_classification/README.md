# TFLite_Serving
DL inference serving with TFLite in AWS Lambda
- Lambda Hardware Type:  ARM64 (AWS Graviton2)

### Push docker image to AWS ECR
```
export IMAGE_NAME="tflite_lambda_container_arm"

docker build -t $IMAGE_NAME . --no-cache

export ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

docker tag $IMAGE_NAME $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com

docker push $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME
