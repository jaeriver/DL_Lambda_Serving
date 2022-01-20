FROM public.ecr.aws/lambda/python:3.8-arm64

# install essential library
RUN yum update && yum install -y wget && yum clean all
RUN yum -y install cmake3 gcc gcc-c++ make && ln -s /usr/bin/cmake3 /usr/bin/cmake
RUN yum -y install python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential libedit-dev llvm llvm-devel libxml2-dev git tar wget gcc gcc-c++

# git clone
RUN git clone https://github.com/jaeriver/Onnx_Serving.git
RUN pip3 install -r /var/task/Onnx_Serving/requirements.txt

RUN cp /var/task/Onnx_Serving/lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
