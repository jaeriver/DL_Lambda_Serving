FROM amazon/aws-lambda-python:3.8

# install essential library
RUN yum update && yum clean all
RUN yum -y install libtinfo-dev zlib1g-dev build-essential git

# git clone
RUN git clone https://github.com/jaeriver/DL_Lambda_Serving.git
WORKDIR DL_Lambda_Serving/intel/tflite/image_classification
RUN pip install -r requirements.txt
RUN pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_x86_64.whl

RUN cp lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
