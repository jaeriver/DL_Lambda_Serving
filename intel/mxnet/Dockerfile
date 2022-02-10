FROM amazon/aws-lambda-python:3.8

# install essential library
RUN yum -y install python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential git libgomp

RUN mkdir -p /var/task/lib
RUN cp /usr/lib64/atlas-sse3/liblapack.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libptf77blas.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libf77blas.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libptcblas.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libcblas.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libatlas.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libptf77blas.so.3 /var/task/lib/. & cp /usr/lib64/libgfortran.so.3 /var/task/lib/. & cp /usr/lib64/libquadmath.so.0 /var/task/lib/.
# git clone
RUN git clone https://github.com/jaeriver/DL_Lambda_Serving.git
WORKDIR DL_Lambda_Serving/intel/mxnet
RUN pip3 install -r requirements.txt

RUN cp lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
