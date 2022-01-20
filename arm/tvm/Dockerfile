FROM public.ecr.aws/lambda/python:3.8-arm64

# install essential library
RUN yum update && yum install -y wget && yum clean all
RUN yum -y install cmake3 gcc gcc-c++ make && ln -s /usr/bin/cmake3 /usr/bin/cmake
RUN yum -y install python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential libedit-dev llvm llvm-devel libxml2-dev git tar wget gcc gcc-c++

# git clone
RUN git clone https://github.com/jaeriver/TVM_Lambda_Container_Arm.git

# ENV PYTHONPATH "/var/lang/lib/python3.8/site-packages:/var/task"
ENV TVM_HOME=/var/task/TVM_Lambda_Container_Arm/tvm
ENV PATH=$PATH:$TVM_HOME/bin
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
ENV PATH=$TVM_HOME/python:$PATH

RUN pip3 install -r /var/task/TVM_Lambda_Container_Arm/requirements.txt
RUN pip3 install onnx

# install packages
WORKDIR TVM_Lambda_Container_Arm
RUN mkdir tvm/build
RUN cp /var/task/TVM_Lambda_Container_Arm/config.cmake tvm/build
RUN env CC=cc CXX=CC

WORKDIR tvm/build
RUN cmake ..
RUN make -j3

RUN cp /var/task/TVM_Lambda_Container_Arm/lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
