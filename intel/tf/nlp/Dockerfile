FROM public.ecr.aws/lambda/python:3.8

# install essential library
RUN yum -y install libtinfo-dev zlib1g-dev build-essential git

RUN git clone https://github.com/jaeriver/DL_Lambda_Serving.git
WORKDIR DL_Lambda_Serving/intel/tf/nlp
RUN pip install -r requirements.txt

RUN cp lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
