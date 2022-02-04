### Push docker image to AWS ECR
```
export IMAGE_NAME="tvm_lambda_container"

docker build -t $IMAGE_NAME . --no-cache

export ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

docker tag $IMAGE_NAME $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com

docker push $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME
```

### Git clone origin TVM repo
```
git clone -b v0.8 --recursive https://github.com/apache/tvm tvm
```

### Lambda Parameters
```
{
  "bucket_name": "your s3 bucket name",
  "batch_size": 1,
  "arch_type": "intel",
  "framework" : "mxnet",
  "model_name": "mobilenet_v2",
  "workload" : "bert",
  "count": 5
}
```
