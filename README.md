# DL_Lambda_Serving
Deep Learning Inference Serving with variety Frameworks in AWS Lambda

## Environments
- python3.8
- Use Docker Container Image to build AWS Lambda Container

## Hardware Types
- ARM64 (ARM Architecture based on AWS Graviton2)
- X86 (Intel Architecture based on AMD)

## Frameworks
### Main
- MXNet (v1.8.0)
- ONNX (use onnxruntime v1.10.0) ([intel](https://github.com/jaeriver/DL_Lambda_Serving/tree/main/intel/onnx)) 
- TVM (v0.8) ([intel](https://github.com/jaeriver/DL_Lambda_Serving/tree/main/intel/tvm)) 

### Preparing
- Tensorflow (v2.7.0)
- Tensorflow Lite (use tflite runtime v2.4.0)
- Pytorch (preparing)
## Workloads
- Image Classification with CNN models
- NLP (Natural Language Processing) with Bert, Transformer (preparing)

## To Do
- ARM architecture

## Future Work
- Use AWS SAM to build Serverless System
