FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt 

RUN yum -y install tar gzip
RUN mkdir ./model
RUN curl -L "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4?tf-hub-format=compressed" | tar -zxvC ./model
RUN chmod 774 ./model/*

COPY ./app/app.py   ./

CMD ["app.handler"]
