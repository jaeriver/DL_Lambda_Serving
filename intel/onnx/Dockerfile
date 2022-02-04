FROM amazon/aws-lambda-python:3.8

# install essential library
RUN yum update && yum clean all
RUN yum -y install python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential git

# git clone
RUN git clone https://github.com/jaeriver/DL_Lambda_Serving.git
WORKDIR DL_Lambda_Serving/intel/onnx
RUN pip3 install -r requirements.txt

RUN cp lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
