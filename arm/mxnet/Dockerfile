FROM public.ecr.aws/lambda/python:3.8-arm64

# install essential library
RUN yum -y install wget tar python38-devel python3-setuptools libtinfo-dev zlib1g-dev build-essential git libgomp gcc-gfortran libgfortran blas lapack atlas-sse3-devel
RUN yum -y install gcc gcc-c++


RUN mkdir -p /var/task/lib
# git clone
RUN git clone https://github.com/jaeriver/DL_Lambda_Serving.git
WORKDIR DL_Lambda_Serving/arm/mxnet
RUN pip3 install -r requirements.txt
RUN pip3 install gluonnlp
RUN yum -y install environment-modules

WORKDIR /tmp/
RUN wget https://developer.arm.com/-/media/Files/downloads/hpc/arm-performance-libraries/22-0-2/RHEL7/arm-performance-libraries_22.0.2_RHEL-7_gcc-7.5.tar?revision=986003c5-4e15-445d-908a-c0b3d7c5a39d
RUN tar -xvf arm-performance-libraries_22.0.2_RHEL-7_gcc-7.5.tar\?revision\=986003c5-4e15-445d-908a-c0b3d7c5a39d
RUN arm-performance-libraries_22.0.2_RHEL-7/arm-performance-libraries_22.0.2_RHEL-7.sh -a
RUN export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/arm/armpl_22.0.2_gcc-7.5/lib
# RUN wget https://repo.mxnet.io/dist/python/cpu/mxnet-1.9.0-py3-none-manylinux2014_aarch64.whl

WORKDIR ${LAMBDA_TASK_ROOT}/DL_Lambda_Serving/arm/mxnet
RUN cp lambda_function.py ${LAMBDA_TASK_ROOT}

WORKDIR ${LAMBDA_TASK_ROOT}
CMD ["lambda_function.lambda_handler"]
