# DL_Lambda_Serving
Deep Learning Inference Serving with variety Frameworks in AWS Lambda

## Environments
- python3.8
- Use Docker Container Image to build AWS Lambda Container

## Hardware Types
- ARM64 (ARM Architecture based on AWS Graviton2)
- X86 (Intel Architecture based on AMD)

## Frameworks
- Tensorflow (v2.7.0)
- Tensorflow Lite (use tflite runtime v2.4.0)
- Pytorch (preparing)
- ONNX (use onnxruntime v1.10.0) [intel-imageclassification](https://github.com/jaeriver/DL_Lambda_Serving/tree/main/intel/onnx/image_classification) [arm-imageclassification](https://github.com/jaeriver/DL_Lambda_Serving/tree/main/arm/onnx/image_classification)
- TVM (v0.8)

## Workloads
- Image Classification with CNN models
- NLP (Natural Language Processing) with Bert, Transformer (preparing)

## To Do
- All NLP scenario
- Pytorch, Tensorflow Lite, TF(in ARM)

## Future Work
- Use AWS SAM to build Serverless System
