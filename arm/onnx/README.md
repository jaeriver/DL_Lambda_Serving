### Push docker image to AWS ECR
```
export IMAGE_NAME="onnx_lambda_container_arm"

docker build -t $IMAGE_NAME . --no-cache

export ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

docker tag $IMAGE_NAME $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com

docker push $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME
```

### Lambda Parameters
```
{
  "bucket_name": "your s3 bucket name",
  "batch_size": 1,
  "arch_type": "intel",
  "framework" : "mxnet",
  "model_name": "bert_base.onnx",
  "workload" : "bert",
  "count": 5
}
```

### Remark
Now(2022/02/07) some issue in onnxruntime package with AWS Graviton2 Hardware.

So, replaced requirements.txt
onnxruntime -> https://test.pypi.org/simple/ ort-nightly 
reference:
- https://github.com/microsoft/onnxruntime/issues/10038
